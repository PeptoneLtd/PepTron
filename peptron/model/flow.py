# Copyright 2025 Peptone Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Optional, Dict
import torch
import time
import numpy as np
import json

from collections import defaultdict
from peptron.utils.util import rmsdalign, NumpyEncoder
from peptron.data import protein

from openfold.utils.loss import lddt_ca
from openfold.np import residue_constants
from openfold.utils.superimposition import superimpose
from openfold.utils.validation_metrics import (
    drmsd,
    gdt_ts,
    gdt_ha,
)
from openfold.utils.feats import pseudo_beta_fn

from peptron.utils.logger_config import logger
from megatron.core import parallel_state
from peptron.model import model
from bionemo.moco.schedules.inference_time_schedules import LinearInferenceSchedule
from bionemo.moco.schedules.utils import TimeDirection
from pathlib import Path


class HarmonicPrior:
    def __init__(self, N = 256, a =3/(3.8**2)):
        J = torch.zeros(N, N)
        for i, j in zip(np.arange(N-1), np.arange(1, N)):
            J[i,i] += a
            J[j,j] += a
            J[i,j] = J[j,i] = -a
        D, P = torch.linalg.eigh(J)
        D_inv = 1/D
        D_inv[0] = 0
        self.P, self.D_inv = P, D_inv
        self.N = N

    def to(self, device):
        self.P = self.P.to(device)
        self.D_inv = self.D_inv.to(device)
        
    def sample(self, batch_dims=()):
        # batch_dims should be a tuple: e.g. (batch_size,) for B samples
        batch_size = batch_dims[0] if batch_dims else 1
        # Draw B independent noise samples -- shape (B, N, 3)
        z = torch.randn(batch_size, self.N, 3, device=self.P.device)
        # Apply per-mode scaling
        scales = torch.sqrt(self.D_inv)[None, :, None]
        z = z * scales
        # Expand P for batched matmul
        P_batched = self.P.unsqueeze(0).expand(batch_size, -1, -1)
        return torch.matmul(P_batched, z)


class FlowSteps:
    def __init__(
        self, 
        model: Optional["ESMFoldSeqModel"] = None, 
        cfg=None, 
        **kwargs):
        # Store any other shared parameters/state
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Store parameters
        self.cfg = cfg
        self.model = model

        # training state
        self._log = defaultdict(list)

        if torch.cuda.is_available():
            self.generator = torch.Generator("cuda")
            self.generator.manual_seed(137)
        else:
            self.generator = torch.Generator()
            self.generator.manual_seed(137)

        self.last_log_time = time.time()
        self.iter_step = 0
        self.gen_device = self.generator.device if hasattr(self.generator, 'device') else 'cpu'


    def interpolants(self, device):
        # Continuous Flow Matching
        self.harmonic_prior = HarmonicPrior(512) # TODO: take it from the confs
        self.harmonic_prior.to(device)

    def _add_noise(self, batch):
        device = batch['aatype'].device
        batch_dims = batch['seq_length'].shape

        self.interpolants(device)

        mask = batch['pseudo_beta_mask']
        x1 = batch['pseudo_beta']  # DATA

        batch_size = batch['aatype'].shape[0]
        noisy = self.harmonic_prior.sample((batch_size,))

        try:
            noisy = rmsdalign(x1, noisy, weights=mask).detach()
        except Exception as e:
            logger.warning(f'SVD failed to converge: {type(e).__name__}: {e}')
            batch['t'] = torch.ones(batch_dims, device=device)
            return

        t = torch.rand(batch_dims, device=device)
        
        # Use MoCo's interpolate method - it implements standard convention correctly
        noisy_beta = (1 - t[:,None,None]) * x1 + t[:,None,None] * noisy

        pseudo_beta_dists = torch.sum((noisy_beta.unsqueeze(-2) - noisy_beta.unsqueeze(-3)) ** 2, dim=-1)**0.5
        batch['noised_pseudo_beta_dists'] = pseudo_beta_dists
        batch['t'] = t

    
    def _compute_validation_metrics(self, batch, outputs, superimposition_metrics=False):
        metrics = {}
        
        gt_coords = batch["all_atom_positions"]
        pred_coords = outputs["final_atom_positions"]
        all_atom_mask = batch["all_atom_mask"]
    
        # This is super janky for superimposition. Fix later
        gt_coords_masked = gt_coords * all_atom_mask[..., None]
        pred_coords_masked = pred_coords * all_atom_mask[..., None]
        ca_pos = residue_constants.atom_order["CA"]
        gt_coords_masked_ca = gt_coords_masked[..., ca_pos, :]
        pred_coords_masked_ca = pred_coords_masked[..., ca_pos, :]
        all_atom_mask_ca = all_atom_mask[..., ca_pos]
    
        lddt_ca_score = lddt_ca(
            pred_coords,
            gt_coords,
            all_atom_mask,
            eps=1e-6,
            per_residue=False,
        )
   
        metrics["lddt_ca"] = lddt_ca_score
   
        drmsd_ca_score = drmsd(
            pred_coords_masked_ca,
            gt_coords_masked_ca,
            mask=all_atom_mask_ca, # still required here to compute n
        )
   
        metrics["drmsd_ca"] = drmsd_ca_score
    
        if(superimposition_metrics):
            superimposed_pred, alignment_rmsd = superimpose(
                gt_coords_masked_ca, pred_coords_masked_ca, all_atom_mask_ca,
            )
            gdt_ts_score = gdt_ts(
                superimposed_pred, gt_coords_masked_ca, all_atom_mask_ca
            )
            gdt_ha_score = gdt_ha(
                superimposed_pred, gt_coords_masked_ca, all_atom_mask_ca
            )

            metrics["alignment_rmsd"] = alignment_rmsd
            metrics["gdt_ts"] = gdt_ts_score
            metrics["gdt_ha"] = gdt_ha_score

        return metrics

    def log(self, key, data):
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        if self.stage == 'train':
            self._log["iter_" + key].extend(data)
        self._log[self.stage + "_" + key].extend(data)

    def save_log_to_json(self, filename="forward_struct_metrics.json"):
        """Saves the self._log dictionary to a JSON file."""
        output_directory = Path(__file__).resolve().parent.parent.parent
        file_path = output_directory / filename

        # Ensure the output directory exists
        output_directory.mkdir(parents=True, exist_ok=True)

        try:
            with open(file_path, 'w') as f:
                json.dump(dict(self._log), f, indent=2, cls=NumpyEncoder)
        except:
            pass

    def linear_interpolation(
        self, 
        batch, 
        as_protein=False, 
        no_diffusion=False, 
        self_cond=True, 
        noisy_first=False, 
        steps=10,
        schedule=None
        ):

        N = batch['aatype'].shape[1]
        batch_size = batch['aatype'].shape[0]
        device = batch['aatype'].device

        self.interpolants(device)

        prior = HarmonicPrior(N)
        prior.to(device)
        noisy = prior.sample((batch_size,))

        batch_dims = batch['seq_length'].shape
        
        if noisy_first:
            batch['noised_pseudo_beta_dists'] = torch.sum((noisy.unsqueeze(-2) - noisy.unsqueeze(-3)) ** 2, dim=-1)**0.5
            batch['t'] = torch.ones(1, device=noisy.device)
            
        if no_diffusion:
            output = self.model(batch)
            outputs = output["structure_output"]
            if as_protein:
                return protein.output_to_protein({**output, **batch})
            else:
                return [{**output, **batch}]

        if schedule is None:
            schedule = np.array([1.0, 0.75, 0.5, 0.25, 0.1, 0]) 

        outputs = []
        prev_outputs = None
        for t, s in zip(schedule[:-1], schedule[1:]):
            output = self.model(batch, prev_outputs=prev_outputs)
            output = output["structure_output"]
            pseudo_beta = pseudo_beta_fn(batch['aatype'], output['final_atom_positions'], None)
            outputs.append({**output, **batch})
            noisy = rmsdalign(pseudo_beta, noisy)
            noisy = (s / t) * noisy + (1 - s / t) * pseudo_beta
            batch['noised_pseudo_beta_dists'] = torch.sum((noisy.unsqueeze(-2) - noisy.unsqueeze(-3)) ** 2, dim=-1)**0.5
            batch['t'] = torch.ones(1, device=noisy.device) * s 
            if self_cond:
                prev_outputs = output

        del batch['noised_pseudo_beta_dists'], batch['t']
        if as_protein:
            prots = []
            for output in outputs:
                prots.extend(protein.output_to_protein(output))
            return prots[-1]
        else:
            return {"interpolations": outputs[-1]} 


_FLOW_STEPS: FlowSteps | None = None


def init_flow_steps(cfg, **kwargs):             # ← NEW helper
    """Create the singleton FlowSteps object once, before training."""
    global _FLOW_STEPS
    if _FLOW_STEPS is None:
        _FLOW_STEPS = FlowSteps(cfg=cfg, **kwargs)
    else:
        raise RuntimeError("FlowSteps already initialised")


def peptron_forward_step(                       # signature Lightning expects
    model: "ESMFoldSeqModel",
    batch: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Custom forward step that handles structure prediction."""

    if _FLOW_STEPS is None:                     # safety guard
        raise RuntimeError(
            "init_flow_steps() was never called before training started"
        )

    fs = _FLOW_STEPS                            # short alias

    # bind the runner’s model once so helpers can use fs.model if needed
    if fs.model is None:
        fs.model = model
    elif fs.model is not model:
        raise ValueError("FlowSteps already bound to a different model")

    fs.iter_step += 1
    fs.stage = "train" if model.training else "val"

    device = batch["aatype"].device

    rand   = lambda p=(): torch.rand(1, device=device, generator=fs.generator).item()

    # noise injection
    if rand() < fs.cfg.flow_matching.noise_prob:
        fs._add_noise(batch)
        fs.log("time", [batch["t"].mean().item()])
    else:
        fs.log("time", [1])

    # optional extra input removal
    if fs.cfg.flow_matching.extra_input and rand() >= fs.cfg.flow_matching.extra_input_prob:
        batch.pop("extra_all_atom_positions", None)

    # self-conditioning pass
    prev = None
    if rand() < fs.cfg.flow_matching.self_cond_prob:
        with torch.no_grad():
            prev = model(batch)["structure_output"]

    outputs = model(batch, prev_outputs=prev)["structure_output"]

    # metrics
    with torch.no_grad():
        for k, v in fs._compute_validation_metrics(
            batch, outputs, superimposition_metrics=False
        ).items():
            fs.log(k, [v.cpu().numpy()])

    fs.log("dur", [time.time() - fs.last_log_time])
    fs.last_log_time = time.time()
    fs.save_log_to_json()

    return outputs
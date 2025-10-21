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
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Type
import typing as T
import torch
from torch import Tensor

from esm2.api import ESM2GenericConfig, ESM2Model
from esm2.data import tokenizer
from bionemo.llm.utils import iomixin_utils as iom

from peptron.model.loss import ESMFoldLossReduction, get_loss_reduction_class, LossConfig, loss_config_from_configdict
from peptron.model.layers import GaussianFourierProjection
from peptron.model.input_stack import InputPairStack

from torch import nn
from torch.nn import LayerNorm
from openfold.model.primitives import Linear
from peptron.model.trunk import FoldingTrunk
from openfold.np import residue_constants
from openfold.data.data_transforms import make_atom14_masks
from openfold.model.heads import PerResidueLDDTCaPredictor
from ml_collections import ConfigDict

from peptron.data.misc import (
    categorical_lddt,
)

from openfold.utils.feats import (
    atom14_to_atom37,
    pseudo_beta_fn,
)

from openfold.utils.loss import compute_predicted_aligned_error, compute_tm
from bionemo.esm2.data import tokenizer

__all__: Sequence[str] = (
    "StructureHead",
    "ESMFoldSeqModel",
    "ESMFoldSeqConfig",
)


class ESMFoldSeqModel(ESM2Model):
    """Fold model that is suitable for training on structure generation tasks."""

    def __init__(self, config, *args, post_process: bool = True, include_hiddens: bool = False, **kwargs):
        """Constructs an instance of the ESMFold model suitable for training."""
        super().__init__(config, *args, post_process=post_process, include_hiddens=True, **kwargs)

        # freeze encoder parameters
        if config.training.encoder_frozen:
            for _, param in self.named_parameters():
                param.requires_grad = False

        self.include_hiddens_structure = (
            include_hiddens  # this include_hiddens is for the final output of the structure head
        )
        # If post_process is True that means that we are at the last megatron parallelism stage and we can
        #   apply the head.

        if post_process:
            # if we are doing post process (eg pipeline last stage) then we need to add the output layers
            self.include_hiddens = True

            self.structure_head = StructureHead(
                config,
                compute_lm_repr=self._compute_language_model_representations
            )
            if config.training.structure_frozen:
                for key, param in self.structure_head.named_parameters():
                    fold_layers = {
                        "esm_s_combine",
                        "af2_to_esm",
                        "positional_encoding",
                        "esm_s_mlp",
                        "embedding",
                        "trunk",
                        "distogram_head",
                        "ptm_head",
                        "lm_head",
                        "lddt_head"
                    }
                    segments = key.split('.')
                    if fold_layers.intersection(segments):
                        param.requires_grad = False

    def _compute_language_model_representations(
            self,
            tokenizer: tokenizer.BioNeMoESMTokenizer,
            esmaa: torch.Tensor,
    ) -> torch.Tensor:
        """Adds bos/eos tokens for the language model, since the structure module doesn't use these."""
        batch_size = esmaa.size(0)

        bosi = tokenizer.cls_token_id
        eosi = tokenizer.eos_token_id
        bos = esmaa.new_full((batch_size, 1), bosi)
        eos = esmaa.new_full((batch_size, 1), tokenizer.pad_token_id)
        esmaa = torch.cat([bos, esmaa, eos], dim=1)
        # Use the first padding index as eos during inference.
        esmaa[range(batch_size), (esmaa != 1).sum(1)] = eosi
        attention_mask = torch.ones_like(esmaa, dtype=torch.int64)

        output = super().forward(
            esmaa,
            attention_mask=attention_mask,
        )
        # Stop early if we are not in post_process mode (for example if we are in the middle of model parallelism)
        if not self.post_process:
            return output  # we are not at the last pipeline stage so just return what the parent has
        # Double check that the output from the parent has everything we need to do prediction in this head.

        if not isinstance(output, dict) or "all_hidden_states" not in output:
            raise ValueError(
                f"Expected to find 'all_hidden_states' in the output, and output to be dictionary-like, found {output},\n"
                "Make sure include_hiddens=True in the call to super().__init__"
            )
        # Get the hidden_states from the parent output, and pull out the [CLS] token for this task
        all_hidden_states: Tensor = output["all_hidden_states"]

        if not self.include_hiddens_structure:
            del output["hidden_states"]

        esm_s = all_hidden_states.permute(1, 2, 0, 3)  # Shape → [B, L, Layers, C]

        last = esm_s[:, :, -1:, :].clone()
        esm_s = torch.cat([esm_s, last], dim=2)  # add last layer to make reprs compatible with HF esm ckpt

        esm_s = esm_s[:, 1:-1]
        return esm_s

    def forward(self, *args, **kwargs) -> Tensor:
        batch = args[0]
        prev_outputs = kwargs.get("prev_outputs", None)
        structure_output = self.structure_head(batch=batch, prev_outputs=prev_outputs)
        return {"structure_output": structure_output}


class StructureHead(nn.Module):
    """ESM-based protein structure prediction model."""

    def __init__(
            self,
            cfg,
            compute_lm_repr: T.Callable[[Tensor, Tensor], Tensor],
    ):
        super().__init__()

        # Add dummy parameter for nemo ckpt
        # self.dummy_module = nn.Linear(1, 1)

        self._compute_lm_repr = compute_lm_repr
        self.bionemo_esm_tokenizer = tokenizer.get_tokenizer()
        self.register_buffer("af2_to_esm", self._af2_to_esm(self.bionemo_esm_tokenizer))

        self.cfg: ESMFoldSeqConfig = cfg

        # logger.info("The StructureHead configs are:")
        # for key, value in self.cfg.__dict__.items():
        #     logger.info(f"  {key}: {value}")

        self.distogram_bins = self.cfg.distogram_bins
        self.esm_feats = self.cfg.esm2.feats
        self.esm_num_layers = self.cfg.esm2.num_layers
        self.esm_attention_heads = self.cfg.esm2.attention_heads
        self.esm_attns = self.esm_num_layers * self.esm_attention_heads

        self.esm_s_combine = nn.Parameter(torch.zeros(self.esm_num_layers + 1))

        c_s = self.cfg.trunk.sequence_state_dim
        c_z = self.cfg.trunk.pairwise_state_dim

        self.esm_s_mlp = nn.Sequential(
            LayerNorm(self.esm_feats),
            nn.Linear(self.esm_feats, c_s),
            nn.ReLU(),
            nn.Linear(c_s, c_s),
        )

        ######################
        self.input_pair_embedding = Linear(
            self.cfg.input_pair_embedder.no_bins,
            self.cfg.trunk.pairwise_state_dim,
            init="final",
        )
        self.input_time_projection = GaussianFourierProjection(
            embedding_size=self.cfg.input_pair_embedder.time_emb_dim
        )
        self.input_time_embedding = Linear(
            self.cfg.input_pair_embedder.time_emb_dim,
            self.cfg.trunk.pairwise_state_dim,
            init="final",
        )
        # torch.nn.init.zeros_(self.input_pair_embedding.weight)
        # torch.nn.init.zeros_(self.input_pair_embedding.bias)
        self.input_pair_stack = InputPairStack(**self.cfg.input_pair_stack)

        if self.cfg.flow_matching.extra_input:
            self.extra_input_pair_embedding = Linear(
                self.cfg.input_pair_embedder.no_bins,
                self.cfg.evoformer_stack.c_z,
                init="final",
            )
            self.extra_input_pair_stack = InputPairStack(**self.cfg.input_pair_stack)

        #######################

        # 0 is padding, N is unknown residues, N + 1 is mask.
        # self.n_tokens_embed = self.cfg.tokenizer.vocab_size
        self.n_tokens_embed = residue_constants.restype_num + 3
        self.pad_idx = 0
        self.unk_idx = self.n_tokens_embed - 2
        self.mask_idx = self.n_tokens_embed - 1
        self.embedding = nn.Embedding(self.n_tokens_embed, c_s, padding_idx=0)

        self.trunk = FoldingTrunk(self.cfg.trunk)

        self.distogram_head = nn.Linear(c_z, self.distogram_bins)
        self.ptm_head = nn.Linear(c_z, self.distogram_bins)
        self.lm_head = nn.Linear(c_s, self.n_tokens_embed)
        self.lddt_bins = self.cfg.lddt_bins

        self.lddt_head = PerResidueLDDTCaPredictor(
            no_bins=self.lddt_bins,
            c_in=self.cfg.trunk.structure_module["c_s"],
            c_hidden=self.cfg.lddt_head_hid_dim
        )

        # self.lddt_head = nn.Sequential(
        #     nn.LayerNorm(self.cfg.trunk.structure_module.c_s),
        #     nn.Linear(self.cfg.trunk.structure_module.c_s, self.cfg.lddt_head_hid_dim),
        #     nn.Linear(self.cfg.lddt_head_hid_dim, self.cfg.lddt_head_hid_dim),
        #     nn.Linear(self.cfg.lddt_head_hid_dim, 37 * self.lddt_bins),
        # )


    @staticmethod
    def _af2_to_esm(d: tokenizer.BioNeMoESMTokenizer):
        # Remember that t is shifted from residue_constants by 1 (0 is padding).
        esm_reorder = [d.pad_token_id] + [
            d.convert_tokens_to_ids(v) for v in residue_constants.restypes_with_x
        ]
        return torch.tensor(esm_reorder)

    def _af2_idx_to_esm_idx(self, aa, mask):
        aa = (aa + 1).masked_fill(mask != 1, 0)
        return self.af2_to_esm[aa]

    def _mask_inputs_to_esm(self, esmaa, pattern):
        new_esmaa = esmaa.clone()
        new_esmaa[pattern == 1] = self.esm_dict.mask_idx
        return new_esmaa

    def _get_input_pair_embeddings(self, dists, mask):

        mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)

        lower = torch.linspace(
            self.cfg.input_pair_embedder.min_bin,
            self.cfg.input_pair_embedder.max_bin,
            self.cfg.input_pair_embedder.no_bins,
            device=dists.device
        )
        dists = dists.unsqueeze(-1)
        inf = self.cfg.input_pair_embedder.inf
        upper = torch.cat([lower[1:], lower.new_tensor([inf])], dim=-1)
        dgram = ((dists > lower) * (dists < upper)).type(dists.dtype)

        # Ensure dgram matches the dtype of the embedding layer
        dgram = dgram.to(self.input_pair_embedding.weight.dtype)
        mask = mask.to(self.input_pair_embedding.weight.dtype)

        inp_z = self.input_pair_embedding(dgram * mask.unsqueeze(-1))
        inp_z = self.input_pair_stack(inp_z, mask, chunk_size=None)
        return inp_z

    def _get_extra_input_pair_embeddings(self, dists, mask):

        mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)

        lower = torch.linspace(
            self.cfg.input_pair_embedder.min_bin,
            self.cfg.input_pair_embedder.max_bin,
            self.cfg.input_pair_embedder.no_bins,
            device=dists.device
        )
        dists = dists.unsqueeze(-1)
        inf = self.cfg.input_pair_embedder.inf
        upper = torch.cat([lower[1:], lower.new_tensor([inf])], dim=-1)
        dgram = ((dists > lower) * (dists < upper)).type(dists.dtype)

        # Ensure dgram matches the dtype of the embedding layer
        dgram = dgram.to(self.input_pair_embedding.weight.dtype)
        mask = mask.to(self.input_pair_embedding.weight.dtype)

        inp_z = self.extra_input_pair_embedding(dgram * mask.unsqueeze(-1))
        inp_z = self.extra_input_pair_stack(inp_z, mask, chunk_size=None)
        return inp_z

    def forward(
            self,
            num_recycles: T.Optional[int] = None,
            masking_pattern: T.Optional[torch.Tensor] = None,
            **kwargs
    ):
        """Runs a forward pass given input tokens. Use `model.infer` to
        run inference from a sequence.

        Args:
            aa (torch.Tensor): Tensor containing indices corresponding to amino acids. Indices match
                openfold.np.residue_constants.restype_order_with_x.
            mask (torch.Tensor): Binary tensor with 1 meaning position is unmasked and 0 meaning position is masked.
            residx (torch.Tensor): Residue indices of amino acids. Will assume contiguous if not provided.
            masking_pattern (torch.Tensor): Optional masking to pass to the input. Binary tensor of the same size
                as `aa`. Positions with 1 will be masked. ESMFold sometimes produces different samples when
                different masks are provided.
            num_recycles (int): How many recycle iterations to perform. If None, defaults to training max
                recycles, which is 3.
        """
        batch = kwargs["batch"]
        prev_outputs = kwargs["prev_outputs"]

        aa = batch['aatype']
        mask = batch['seq_mask']
        residx = batch['residue_index']

        if mask is None:
            mask = torch.ones_like(aa)

        B = aa.shape[0]
        L = aa.shape[1]
        device = aa.device

        if residx is None:
            residx = torch.arange(L, device=device).expand_as(aa)

        # === ESM ===
        esmaa = self._af2_idx_to_esm_idx(aa, mask)
        esm_s = self._compute_lm_repr(self.bionemo_esm_tokenizer, esmaa)

        esm_s = esm_s.to(self.esm_s_combine.dtype)
        esm_s = esm_s.detach()

        # === preprocessing ===
        esm_s = (self.esm_s_combine.softmax(0).unsqueeze(0) @ esm_s).squeeze(2)

        s_s_0 = self.esm_s_mlp(esm_s)
        mask = mask.to(s_s_0.dtype)

        s_s_0 += self.embedding(aa)

        #######################
        if 'noised_pseudo_beta_dists' in batch:
            inp_z = self._get_input_pair_embeddings(
                batch['noised_pseudo_beta_dists'],
                batch['pseudo_beta_mask']
            )
            inp_z = inp_z + self.input_time_embedding(self.input_time_projection(batch['t'].to(s_s_0.dtype)))[:, None,
                            None].to(s_s_0.dtype)
        else:  # have to run the module, else DDP wont work
            B, L = batch['aatype'].shape
            inp_z = self._get_input_pair_embeddings(
                s_s_0.new_zeros(B, L, L),
                batch['pseudo_beta_mask'] * 0.0
            )
            inp_z = inp_z + self.input_time_embedding(self.input_time_projection(inp_z.new_zeros(B).to(s_s_0.dtype)))[:,
                            None, None].to(s_s_0.dtype)
        ##########################
        #############################
        if self.cfg.flow_matching.extra_input:
            if 'extra_all_atom_positions' in batch:
                extra_pseudo_beta = pseudo_beta_fn(batch['aatype'], batch['extra_all_atom_positions'], None)
                extra_pseudo_beta_dists = torch.sum(
                    (extra_pseudo_beta.unsqueeze(-2) - extra_pseudo_beta.unsqueeze(-3)) ** 2, dim=-1) ** 0.5
                extra_inp_z = self._get_extra_input_pair_embeddings(
                    extra_pseudo_beta_dists.to(s_s_0.dtype),
                    batch['pseudo_beta_mask'].to(s_s_0.dtype),
                )

            else:  # otherwise DDP complains
                B, L = batch['aatype'].shape
                extra_inp_z = self._get_extra_input_pair_embeddings(
                    inp_z.new_zeros(B, L, L),
                    inp_z.new_zeros(B, L),
                ) * 0.0

            inp_z = inp_z + extra_inp_z
        ########################

        s_z_0 = inp_z
        if prev_outputs is not None:
            s_s_0 = s_s_0 + self.trunk.recycle_s_norm(prev_outputs['s_s'])
            s_z_0 = s_z_0 + self.trunk.recycle_z_norm(prev_outputs['s_z'])
            s_z_0 = s_z_0 + self.trunk.recycle_disto(FoldingTrunk.distogram(
                prev_outputs['sm']["positions"][-1][:, :, :3],
                3.375,
                21.375,
                self.trunk.recycle_bins,
            ))

        else:
            s_s_0 = s_s_0 + self.trunk.recycle_s_norm(torch.zeros_like(s_s_0)) * 0.0
            s_z_0 = s_z_0 + self.trunk.recycle_z_norm(torch.zeros_like(s_z_0)) * 0.0

            # Manual broadcast workaround for custom PyTorch build broadcasting issue
            zero_disto_input = s_z_0.new_zeros(s_z_0.shape[:-2], dtype=torch.long)
            recycled_disto_component = self.trunk.recycle_disto(zero_disto_input)
            term_to_add = recycled_disto_component * 0.0
            term_to_add_expanded = term_to_add.unsqueeze(2).expand_as(s_z_0)
            s_z_0 = s_z_0 + term_to_add_expanded

        structure: dict = self.trunk(
            s_s_0, s_z_0, aa, residx, mask, no_recycles=0
        )

        disto_logits = self.distogram_head(structure["s_z"])
        disto_logits = (disto_logits + disto_logits.transpose(1, 2)) / 2
        structure["distogram_logits"] = disto_logits

        lm_logits = self.lm_head(structure["s_s"])
        structure["lm_logits"] = lm_logits

        structure["aatype"] = aa
        make_atom14_masks(structure)

        for k in [
            "atom14_atom_exists",
            "atom37_atom_exists",
        ]:
            structure[k] *= mask.unsqueeze(-1)
        structure["residue_index"] = residx

        # lddt_head = self.lddt_head(structure['sm']["states"]).reshape(
        #     structure['sm']["states"].shape[0], B, L, -1, self.lddt_bins
        # )
        # structure["lddt_logits"] = lddt_head[-1].mean(dim=2)
        # plddt = categorical_lddt(lddt_head[-1], bins=self.lddt_bins)
        # we predict plDDT between 0 and 1, scale to be between 0 and 100.
        # plddt = 100 * plddt
        # extract only Ca plddt
        # structure["plddt"] = collapse_to_ca(plddt)

        lddt_head = self.lddt_head(structure['sm']["single"])
        structure["lddt_logits"] = lddt_head
        plddt = categorical_lddt(lddt_head, bins=self.lddt_bins)
        structure["plddt"] = 100 * plddt

        ptm_logits = self.ptm_head(structure["s_z"])
        seqlen = mask.type(torch.int64).sum(1)
        structure["tm_logits"] = ptm_logits
        structure["ptm"] = torch.stack(
            [
                compute_tm(
                    batch_ptm_logits[None, :sl, :sl],
                    max_bins=31,
                    no_bins=self.distogram_bins,
                )
                for batch_ptm_logits, sl in zip(ptm_logits, seqlen)
            ]
        )

        structure.update(
            compute_predicted_aligned_error(
                ptm_logits, max_bin=31, no_bins=self.distogram_bins
            )
        )

        structure["final_atom_positions"] = atom14_to_atom37(structure["sm"]["positions"][-1], batch)
        structure["final_affine_tensor"] = structure["sm"]["frames"][-1]

        """
        pseudo_beta, new_pb_mask = pseudo_beta_fn(
            aa,
            structure["final_atom_positions"],      # [B, L, 37, 3]
            structure["atom37_atom_exists"],        # [B, L, 37]
        )

        # stash them on the structure if you need to inspect later
        structure["pseudo_beta"]      = pseudo_beta       # [B, L, 3]
        structure["pseudo_beta_mask"] = new_pb_mask       # [B, L]

        # overwrite your batch so that the loss sees the cropped L×L mask
        batch["pseudo_beta"]          = pseudo_beta
        batch["pseudo_beta_mask"]     = new_pb_mask
        """

        if "name" in batch: structure["name"] = batch["name"]
        if "seq_mask" in batch: structure["seq_mask"] = batch["seq_mask"]
        if "seq_length" in batch: structure["seq_length"] = batch["seq_length"]

        return structure

    @property
    def device(self):
        return self.esm_s_combine.device


@dataclass
class StructureModuleConfig(iom.IOMixinWithGettersSetters):
    c_s: int = 384
    c_z: int = 128
    c_ipa: int = 16
    c_resnet: int = 128
    no_heads_ipa: int = 12
    no_qk_points: int = 4
    no_v_points: int = 8
    dropout_rate: float = 0.1
    no_blocks: int = 8
    no_transition_layers: int = 1
    no_resnet_blocks: int = 2
    no_angles: int = 7
    trans_scale_factor: int = 10
    epsilon: float = 1e-8
    inf: float = 1e5


@dataclass
class TrunkConfig(iom.IOMixinWithGettersSetters):
    num_blocks: int = 48
    sequence_state_dim: int = 1024
    pairwise_state_dim: int = 128
    sequence_head_width: int = 32
    pairwise_head_width: int = 32
    position_bins: int = 32
    dropout: float = 0.0
    layer_drop: float = 0.0
    cpu_grad_checkpoint: bool = False
    chunk_size: T.Optional[int] = None
    use_cuequivariance_attention: bool = True
    use_cuequivariance_multiplicative_update: bool = False
    structure_module: Dict[str, T.Any] = field(default_factory=lambda: {
        "c_s": 384,
        "c_z": 128,
        "c_ipa": 16,
        "c_resnet": 128,
        "no_heads_ipa": 12,
        "no_qk_points": 4,
        "no_v_points": 8,
        "dropout_rate": 0.1,
        "no_blocks": 8,
        "no_transition_layers": 1,
        "no_resnet_blocks": 2,
        "no_angles": 7,
        "trans_scale_factor": 10,
        "epsilon": 1e-8,
        "inf": 1e5,
    })


@dataclass
class InputPairEmbedderConfig(iom.IOMixinWithGettersSetters):
    min_bin: float = 3.25
    max_bin: float = 50.75
    no_bins: int = 39
    time_emb_dim: int = 256
    inf: float = 1e8


@dataclass
class EvoformerStackConfig(iom.IOMixinWithGettersSetters):
    c_m: int = 256
    c_z: int = 128
    c_hidden_msa_att: int = 32
    c_hidden_opm: int = 32
    c_hidden_mul: int = 128
    c_hidden_pair_att: int = 32
    c_s: int = 384
    no_heads_msa: int = 8
    no_heads_pair: int = 4
    no_blocks: int = 48
    transition_n: int = 4
    msa_dropout: float = 0.15
    pair_dropout: float = 0.25
    blocks_per_ckpt: int = 1
    clear_cache_between_blocks: bool = False
    tune_chunk_size: T.Optional[int] = None
    inf: float = 1e9
    eps: float = 1e-10


@dataclass
class DataConfig(iom.IOMixinWithGettersSetters):
    fixed_size: bool = True
    subsample_templates: bool = False
    masked_msa_replace_fraction: float = 0.15
    max_msa_clusters: int = 128
    max_extra_msa: int = 1024
    max_template_hits: int = 4
    max_templates: int = 4
    crop: bool = False
    crop_size: T.Optional[int] = None
    supervised: bool = False
    uniform_recycling: bool = False


@dataclass
class PredictDataConfig(DataConfig):
    max_msa_clusters: int = 512


@dataclass
class EvalDataConfig(DataConfig):
    crop: bool = True
    crop_size: int = 512
    supervised: bool = True


@dataclass
class TrainDataConfig(DataConfig):
    subsample_templates: bool = True
    max_msa_clusters: int = 128
    crop: bool = True
    crop_size: int = 256
    supervised: bool = True
    shuffle_top_k_prefiltered: int = 20
    clamp_prob: float = 0.9
    max_distillation_msa_clusters: int = 1000
    uniform_recycling: bool = True
    distillation_prob: float = 0.75


@dataclass
class ESM2Config(iom.IOMixinWithGettersSetters):
    feats: int = 2560
    num_layers: int = 36
    attention_heads: int = 40


@dataclass
class TrainingConfig(iom.IOMixinWithGettersSetters):
    experiment_dir: str = ""
    wandb_project: str = ""
    experiment_name: str = ""
    initial_nemo_ckpt_path: str = ""
    train_data_dir_pdb: str = ""
    val_data_dir_pdb: str = ""
    val_data_dir_idp: str = ""
    train_msa_dir_pdb: str = ""
    val_msa_dir_pdb: str = ""
    val_msa_dir_idp: str = ""
    train_chains_pdb: str = ""
    valid_chains_pdb: str = ""
    valid_chains_idp: str = ""
    train_data_dir_idp: str = ""
    train_msa_dir_idp: str = ""
    train_chains_idp: str = ""
    mmcif_dir: str = ""
    pretrained_structure_head_path: str = ""
    train_clusters: str = ""
    valid_clusters: str = ""
    train_cutoff: str = ""
    n_steps_train: int = 6250
    warmup_steps_percentage: float = 0.10
    train_epoch_len: int = 80
    val_epoch_len: int = 80
    micro_batch_size: int = 16
    num_nodes: int = 1
    devices: int = 8
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    accumulate_grad_batches: int = 1
    steps_to_save_ckpt: int = 10
    val_check_interval: int = 10
    limit_val_batches: int = 2
    precision: str = "fp16"
    sample_train_confs_pdb: bool = False
    sample_train_confs_idp: bool = True
    sample_val_confs_pdb: bool = False
    sample_val_confs_idp: bool = False
    first_as_template: bool = False
    num_val_confs_pdb: T.Optional[int] = None
    num_val_confs_idp: T.Optional[int] = 10
    filter_chains: bool = True
    encoder_frozen: bool = True
    structure_frozen: bool = False
    dataset_prob_pdb: float = 0.5
    dataset_prob_idp: float = 0.5


@dataclass
class FlowMatchingConfig(iom.IOMixinWithGettersSetters):
    noise_prob: float = 0.5
    extra_input: bool = False
    extra_input_prob: float = 0.5
    self_cond_prob: float = 0.5


@dataclass
class ESMFoldSeqConfig(
    ESM2GenericConfig[ESMFoldSeqModel, ESMFoldLossReduction], iom.IOMixinWithGettersSetters
):
    """ExampleConfig is a dataclass that is used to configure the model.

    Timers from ModelParallelConfig are required for megatron forward compatibility.
    """
    trunk: TrunkConfig = field(default_factory=TrunkConfig)
    input_pair_embedder: InputPairEmbedderConfig = field(default_factory=InputPairEmbedderConfig)
    evoformer_stack: EvoformerStackConfig = field(default_factory=EvoformerStackConfig)
    data: DataConfig = field(default_factory=DataConfig)
    esm2: ESM2Config = field(default_factory=ESM2Config)
    input_pair_stack: Dict[str, T.Any] = field(default_factory=lambda: {})
    model_cls: Type[ESMFoldSeqModel] = ESMFoldSeqModel
    # If you are loading a checkpoint that has this new head and want to keep using
    # these weights, please drop this next line or set to []
    initial_ckpt_skip_keys_with_these_prefixes: List[str] = field(default_factory=lambda: [])
    lddt_head_hid_dim: int = 128
    training: TrainingConfig = field(default_factory=TrainingConfig)
    flow_matching: FlowMatchingConfig = field(default_factory=FlowMatchingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    distogram_bins: int = 64
    lddt_bins: int = 50
    pretrained_structure_head_path: str = ""

    tokenizer = tokenizer.get_tokenizer()

    def get_loss_reduction_class(self) -> Type[ESMFoldLossReduction]:
        """Returns ESMFoldLossReduction class."""
        return get_loss_reduction_class(self.loss)


def get_esmfoldconfig(config: ConfigDict,
                      **kwargs
                      ) -> ESMFoldSeqConfig:
    structure_module_config = dict(
        **config.model.trunk.structure_module
    )
    trunk_config_dict = dict(**config.model.trunk)
    del trunk_config_dict["structure_module"]
    trunk_config = TrunkConfig(
        structure_module=structure_module_config,
        **trunk_config_dict
    )
    input_pair_embedder_config = InputPairEmbedderConfig(
        **config.model.input_pair_embedder
    )
    input_pair_stack_config = dict(**config.model.input_pair_stack)
    evoformer_stack_config = EvoformerStackConfig(
        **config.model.evoformer_stack
    )
    if config.mode == "train":
        data_config = TrainDataConfig(**config.data.train)
    elif config.mode == "predict":
        data_config = PredictDataConfig(**config.data.predict)
    else:
        data_config = DataConfig()
    esm2_config = ESM2Config(
        **config.model.esm2
    )
    training_config = TrainingConfig(**config.training)
    flow_matching_config = FlowMatchingConfig(**config.model.flow_matching)
    loss_config = loss_config_from_configdict(config.loss)

    return ESMFoldSeqConfig(
        trunk=trunk_config,
        input_pair_embedder=input_pair_embedder_config,
        input_pair_stack=input_pair_stack_config,
        evoformer_stack=evoformer_stack_config,
        data=data_config,
        esm2=esm2_config,
        training=training_config,
        flow_matching=flow_matching_config,
        lddt_head_hid_dim=config.model.lddt_head_hid_dim,
        distogram_bins=config.model.heads.distogram.no_bins,
        lddt_bins=config.model.heads.lddt.no_bins,
        loss=loss_config,
        **kwargs
    )

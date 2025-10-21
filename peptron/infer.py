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
import os
from pathlib import Path
from typing import List, Sequence, Type, Optional
import torch
import numpy as np
from nemo import lightning as nl
from peptron.utils.callbacks import StreamingPredictionWriter

from bionemo.core.utils.dtypes import PrecisionTypes, get_autocast_dtype
from esm2.api import ESM2Config
from esm2.data.tokenizer import get_tokenizer
from bionemo.llm.model.biobert.model import BioBertConfig
from bionemo.llm.utils.datamodule_utils import infer_global_batch_size
from peptron.model.model import ESMFoldSeqConfig, get_esmfoldconfig
from peptron.data.datamodule import ESMFoldDataModule, structure_data_step
from peptron.data.data import CSVDataset
from peptron.utils.util import _repeat_item
from peptron.utils.tensor_utils import tensor_tree_map
from peptron.model import flowmoco as flow
from bionemo.llm.model.biobert.lightning import biobert_lightning_module
from types import MethodType


# Import and apply monkey patches to fix tensor collation issues
from peptron.monkey_patch import apply_monkey_patches
from ml_collections import config_flags, ConfigDict
from absl import app
import uuid

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Apply the patches at module import time
apply_monkey_patches()

EXEC_CONFIG = config_flags.DEFINE_config_file('config', 'peptron/model/config.py:peptron_o_inference_cueq')


__all__: Sequence[str] = ("infer_model",)


SUPPORTED_CONFIGS = {
    "ESM2Config": ESM2Config,
    "ESMFoldSeqConfig": ESMFoldSeqConfig,
}


def infer_model(
    runtime_config: ConfigDict,
    chains_path: Path,
    checkpoint_path: Path,
    results_path: Path,
    msa_dir: Path = None,
    micro_batch_size: int = 64,
    max_batch_size: Optional[int] = 1,
    num_workers: int = 8,
    precision: PrecisionTypes = "bf16-mixed",
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    devices: int = 1,
    num_nodes: int = 1,
    prediction_interval: str = "batch",
    config_class: Type[BioBertConfig] = ESMFoldSeqConfig,
    as_protein: bool = False,
    pdb_id: List = None,
    no_diffusion: bool = False,
    self_cond: bool = True,
    noisy_first: bool = False,
    tmax: float = 1.0,
    steps: int = 10,
    samples: int = 10,
    runtime_json: Path = None,
) -> None:
    """Runs inference on a BioNeMo ESMFold model using PyTorch Lightning.

    Args:
        data_path (Path): Path to the input data.
        checkpoint_path (Path): Path to the model checkpoint.
        results_path (Path): Path to save the inference results.
        min_seq_length (int): minimum sequence length to be padded. This should be at least equal to the length of largest sequence in the dataset
        micro_batch_size (int, optional): Micro batch size for inference. Defaults to 64.
        precision (PrecisionTypes, optional): Precision type for inference. Defaults to "bf16-mixed".
        tensor_model_parallel_size (int, optional): Tensor model parallel size for distributed inference. Defaults to 1.
        pipeline_model_parallel_size (int, optional): Pipeline model parallel size for distributed inference. Defaults to 1.
        devices (int, optional): Number of devices to use for inference. Defaults to 1.
        num_nodes (int, optional): Number of nodes to use for distributed inference. Defaults to 1.
        prediction_interval (IntervalT, optional): Intervals to write predict method output into disck for DDP inference. Defaults to epoch.
        config_class (Type[BioBertConfig]): The config class for configuring the model using checkpoint provided
        as_protein (bool, optional): Treat input as protein sequence. Defaults to False.
        no_diffusion (bool, optional): Disable diffusion. Defaults to False.
        self_cond (bool, optional): Enable self-conditioning. Defaults to True.
        noisy_first (bool, optional): Apply noise first. Defaults to False.
        schedule (str, optional): Sampling schedule. Defaults to None.
    """
    # create the directory to save the inference results
    os.makedirs(results_path, exist_ok=True)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Setup the strategy and trainer
    global_batch_size = infer_global_batch_size(
        micro_batch_size=micro_batch_size,
        num_nodes=num_nodes,
        devices=devices,
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
    )

    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        ddp="megatron",
        find_unused_parameters=True,
    )

    prediction_writer = StreamingPredictionWriter(output_dir=results_path, write_interval=prediction_interval)


    trainer = nl.Trainer(
        accelerator="gpu",
        devices=devices,
        strategy=strategy,
        num_nodes=num_nodes,
        callbacks=[prediction_writer],
        plugins=nl.MegatronMixedPrecision(precision=precision),
    )

    dataset = CSVDataset(
        runtime_config.data,
        chains_path,
    )

    data_module = ESMFoldDataModule(
        predict_dataset=dataset, 
        micro_batch_size=micro_batch_size,   # size of a batch to be processed in a device
        global_batch_size=global_batch_size,  # size of batch across all devices. Should be multiple of micro_batch_size
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )


    esmfold_config = get_esmfoldconfig(
        config=runtime_config,
        initial_ckpt_path=str(checkpoint_path),
        params_dtype=get_autocast_dtype(precision),
        pipeline_dtype=get_autocast_dtype(precision),
        autocast_dtype=get_autocast_dtype(precision),  # setting this speeds things up a lot
        initial_ckpt_skip_keys_with_these_prefixes=[],
        pretrained_structure_head_path=""
    )


    tokenizer = get_tokenizer()

    flow.init_flow_steps(cfg=esmfold_config, generator_seed=137)
    structure_forward_step = flow.peptron_forward_step 

    if tmax == 1.0:
        schedule = np.linspace(tmax, 0, steps + 1)
    elif 0.0 < tmax < 1.0:
        schedule = np.linspace(tmax, 0, steps + 1)
        schedule = np.array([1.0] + list(schedule))
    else:
        raise ValueError("tmax must be 1.0 or between 0.0 and 1.0 (exclusive).")

    predict_cfg = dict(
        samples        = samples,
        as_protein     = as_protein,
        no_diffusion   = no_diffusion,
        self_cond      = self_cond,
        noisy_first    = noisy_first,
        steps          = steps,
        schedule       = schedule,
        max_batch_size = max_batch_size,
    )

    def peptron_predict_step(self, batch, *_, **__):
        """
        Called with a *batch dict*. This version generates samples in chunks
        and writes them to temporary files to prevent OOM errors. It returns
        a list of paths to these temporary files.
        """
        cfg = self._predict_cfg
        total_samples: int = cfg["samples"]
        # Process in small, memory-safe chunks. Adjust chunk_size if needed.
        chunk_size: int = min(cfg["max_batch_size"], total_samples)

        # Ensure the model is available to the flow steps
        if flow._FLOW_STEPS.model is None:
            if hasattr(self, "model"):
                flow._FLOW_STEPS.model = self.model
            elif hasattr(self, "module"):
                flow._FLOW_STEPS.model = self.module
            else:
                flow._FLOW_STEPS.model = self

        temp_file_paths = []
        rank = self.trainer.global_rank
        
        # --- The Core of the New Solution ---
        # Create a single, unique ID for this entire prediction call (i.e., for this one sequence).
        prediction_uuid = uuid.uuid4()

        with torch.no_grad():
            for i, start in enumerate(range(0, total_samples, chunk_size)):
                n = min(chunk_size, total_samples - start)
                repeated_batch = {k: _repeat_item(v, n) for k, v in batch.items()}

                out = flow._FLOW_STEPS.linear_interpolation(
                    repeated_batch,
                    as_protein=cfg["as_protein"],
                    no_diffusion=cfg["no_diffusion"],
                    self_cond=cfg["self_cond"],
                    noisy_first=cfg["noisy_first"],
                    steps=cfg["steps"],
                    schedule=cfg["schedule"],
                )

                cpu_out = tensor_tree_map(lambda x: x.cpu() if isinstance(x, torch.Tensor) else x, out)
                
                # Tag the temporary file with the unique ID
                temp_path = os.path.join(
                    self.trainer.callbacks[0].output_dir,
                    f"_tmp_rank{rank}_predictid_{prediction_uuid}_part{i}.pt"
                )
                torch.save(cpu_out, temp_path)
                temp_file_paths.append(temp_path)

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # The returned dictionary now contains the unique ID for this prediction
        return {"temp_files": temp_file_paths, "prediction_uuid": str(prediction_uuid)}



    module = biobert_lightning_module(
        config=esmfold_config,
        tokenizer=tokenizer, 
        data_step=structure_data_step,
        forward_step=structure_forward_step,
    )

    module.predict_step = MethodType(peptron_predict_step, module)
    module._predict_cfg = { **predict_cfg, "samples": samples }    

    if torch.cuda.is_available():
        module = module.to("cuda")
    module.eval()

    if hasattr(module, "gradient_checkpointing_enable"):
        module.gradient_checkpointing_enable()

    trainer.predict(module, datamodule=data_module)


def main(_):
    """Entrypoint for running inference on a geneformer checkpoint and data."""
    config = EXEC_CONFIG.value
    config.mode="predict"
    config.data.common.use_templates = False
    config.data.common.max_recycling_iters = 0

    # hack
    config.globals.blocks_per_ckpt = 1
    config.globals.chunk_size = None
    config.globals.use_lma = False
    config.globals.offload_inference = False
    config.globals.use_cuequivariance_attention = True
    config.globals.use_cuequivariance_multiplicative_update = False
    config.model.template.average_templates = False
    config.model.template.offload_templates = False

    os.makedirs(config.inference.results_path, exist_ok=True)
    with open(os.path.join(config.inference.results_path, "params.json"), "w") as f:
        f.write(config.to_json())

    infer_model(
        runtime_config=config,
        chains_path=config.inference.chains_path,
        checkpoint_path=config.inference.checkpoint_path,
        results_path=config.inference.results_path,
        msa_dir=config.inference.msa_dir,
        micro_batch_size=config.inference.micro_batch_size,
        max_batch_size=config.inference.max_batch_size,
        num_workers=config.inference.num_workers,
        precision=config.inference.precision,
        tensor_model_parallel_size=config.inference.tensor_model_parallel_size,
        pipeline_model_parallel_size=config.inference.pipeline_model_parallel_size,
        devices=config.inference.num_gpus,
        num_nodes=config.inference.num_nodes,
        config_class=config.inference.config_class,
        as_protein=config.inference.as_protein,
        pdb_id=config.inference.pdb_id,
        no_diffusion=config.inference.no_diffusion,
        self_cond=config.inference.self_cond,
        noisy_first=config.inference.noisy_first,
        tmax=config.inference.tmax,
        steps=config.inference.steps,
        samples=config.inference.samples,
        runtime_json=config.inference.runtime_json,
    )

if __name__ == "__main__":
    app.run(main)

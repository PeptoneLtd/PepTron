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
from pathlib import Path
from typing import Sequence, Tuple, Optional

import pytorch_lightning as pl
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from nemo import lightning as nl
from nemo.collections import llm as nllm
from nemo.lightning import resume
from nemo.lightning.pytorch import callbacks as nl_callbacks
from nemo.lightning.pytorch.callbacks.model_transform import ModelTransform
from nemo.lightning.pytorch.callbacks.peft import PEFT
from nemo.lightning.pytorch.optim.megatron import MegatronOptimizerModule
from lightning.pytorch.callbacks import RichModelSummary, LearningRateMonitor
from pytorch_lightning.callbacks import Callback

from esm2.api import ESM2GenericConfig
from esm2.data.tokenizer import BioNeMoESMTokenizer, get_tokenizer
from bionemo.llm.model.lr_scheduler import WarmupAnnealDecayHoldScheduler

import pandas as pd

from peptron.model import flow
from bionemo.llm.model.biobert.lightning import biobert_lightning_module
from peptron.data.datamodule import ESMFoldDataModule, structure_data_step

from peptron.data.data import OpenFoldSingleDataset, OpenFoldDataset
from peptron.model.model import get_esmfoldconfig
from bionemo.llm.utils.logger_utils import WandbConfig, setup_nemo_lightning_logger
from bionemo.llm.utils.datamodule_utils import float_or_int_or_none, infer_global_batch_size
from bionemo.core.utils.dtypes import get_autocast_dtype
from nemo.utils.exp_manager import TimingCallback
from peptron.utils.logger_config import logger

from ml_collections import config_flags, config_dict
from absl import app
import os

EXEC_CONFIG = config_flags.DEFINE_config_file('config', 'peptron/model/config.py:peptron_o_mixed')

__all__: Sequence[str] = ("train_model",)

def load_clusters(path):
    cluster_size = []
    with open(path) as f:
        for line in f:
            names = line.split()
            for name in names:
                cluster_size.append({'name': name, 'cluster_size': len(names)})
    return pd.DataFrame(cluster_size).set_index('name')


def train_model(
    experiment_name: str,
    experiment_dir: Path,
    config: ESM2GenericConfig,
    data_module: pl.LightningDataModule,
    n_steps_train: int,
    steps_to_save_ckpt: int,
    val_check_interval: int,
    wandb_project: str,
    metric_tracker: Callback | None = None,
    tokenizer: BioNeMoESMTokenizer = get_tokenizer(),
    peft: PEFT | None = None,
    metric_to_monitor_for_checkpoints: str = "val_loss",
    precision: str = "bf16-mixed",
    warmup_steps_percentage: float = 0.10,
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    num_nodes: int = 1,
    devices: int = 8,
    limit_val_batches: int = 5,
    accelerator: str = "gpu",
    wandb_entity: str = "Peptone"
) -> Tuple[Path, Callback | None, nl.Trainer]:
    """Trains a BioNeMo ESM2 model using PyTorch Lightning.

    Parameters:
        experiment_name: The name of the experiment.
        experiment_dir: The directory where the experiment will be saved.
        config: The configuration for the ESM2 model.
        data_module: The data module for training and validation.
        n_steps_train: The number of training steps.
        metric_tracker: Optional callback to track metrics
        tokenizer: The tokenizer to use. Defaults to `get_tokenizer()`.
        peft: The PEFT (Parameter-Efficient Fine-Tuning) module. Defaults to None.
        _use_rich_model_summary: Whether to use the RichModelSummary callback, omitted in our test suite until
            https://nvbugspro.nvidia.com/bug/4959776 is resolved. Defaults to True.

    Returns:
        A tuple containing the path to the saved checkpoint, a MetricTracker
        object, and the PyTorch Lightning Trainer object.
    """

    if not isinstance(experiment_dir, Path):
        experiment_dir = Path(experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    warmup_steps = int(n_steps_train) * warmup_steps_percentage

    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=tensor_model_parallel_size, # size of a batch to be processed in a device
        pipeline_model_parallel_size=pipeline_model_parallel_size, # size of batch across all devices. Should be multiple of micro_batch_size
        ddp="megatron",
        find_unused_parameters=True,
        enable_nemo_ckpt_io=True,
    )

    wandb_config: Optional[WandbConfig] = (
        None
        if wandb_project is None
        else WandbConfig(
            offline=False,
            project=wandb_project,
            entity=wandb_entity,
            tags=None,
            group=None,
            job_type=None,
            id=None,
            anonymous=False,
            log_model=False,
        )
    )

    callbacks = [
        RichModelSummary(max_depth=4),
        LearningRateMonitor(),
        nl_callbacks.PreemptionCallback(),
        TimingCallback(),
    ]

    trainer = nl.Trainer(
        accelerator="gpu",
        devices="auto",
        strategy=strategy,
        limit_val_batches=limit_val_batches,
        val_check_interval=val_check_interval,
        max_steps=n_steps_train,
        num_nodes=num_nodes,
        log_every_n_steps=val_check_interval,
        callbacks=callbacks,
        plugins=nl.MegatronMixedPrecision(precision=precision),
    )

    # Needed so that the trainer can find an output directory for the profiler
    # ckpt_path needs to be a string for SerDe
    optimizer = MegatronOptimizerModule(
        config=OptimizerConfig(
            lr=1e-4,
            optimizer="adam",
            use_distributed_optimizer=True,
            weight_decay=0.01,
            adam_beta1=0.9,
            adam_beta2=0.95,
            fp16=config.fp16,
            bf16=config.bf16,
        ),
        lr_scheduler=WarmupAnnealDecayHoldScheduler(
            warmup_steps=warmup_steps,
            max_steps=n_steps_train,
            max_lr=1e-4,  # Match OptimizerConfig.lr
            min_lr=1e-6,
            anneal_percentage=0.15
        )
    )

    flow.init_flow_steps(cfg=config, generator_seed=137)
    structure_forward_step = flow.peptron_forward_step

    module = biobert_lightning_module(
        config=config,
        tokenizer=tokenizer,
        optimizer=optimizer,
        model_transform=peft,
        data_step=structure_data_step,
        forward_step=structure_forward_step,
    )

    if peft is not None:
        callbacks.append(
            ModelTransform()
        )  # Callback needed for PEFT fine-tuning using NeMo2, i.e. biobert_lightning_module(model_transform=peft).

    checkpoint_callback = nl_callbacks.ModelCheckpoint(
        save_last=True,
        monitor=metric_to_monitor_for_checkpoints,
        save_top_k=20,
        every_n_train_steps=steps_to_save_ckpt,
        always_save_context=True,  # Enables the .nemo file-like checkpointing where all IOMixins are under SerDe
        filename="{epoch}-{step}-{consumed_samples}",  # Including step and consumed_samples in the checkpoint filename prevents duplicate filenames and bugs related to this.
    )

    nemo_logger = setup_nemo_lightning_logger(
        root_dir=experiment_dir,
        name=experiment_name,
        initialize_tensorboard_logger=False,
        wandb_config=wandb_config,
        ckpt_callback=checkpoint_callback,
    )


    nllm.train(
        model=module,
        data=data_module,
        trainer=trainer,
        log=nemo_logger,
        resume=resume.AutoResume(
            resume_if_exists=True,  # Looks for the -last checkpoint to continue training.
            resume_ignore_no_checkpoint=True,  # When false this will throw an error with no existing checkpoint.
        ),
    )
    ckpt_path = Path(checkpoint_callback.last_model_path.replace(".ckpt", ""))
    return ckpt_path, trainer



def main(_):

    config = EXEC_CONFIG.value
    config.mode="train"
    config.data.common.use_templates = False
    config.data.common.max_recycling_iters = 0

    logger.info(f"config is: {config}")
    os.makedirs(os.path.join(config.training.experiment_dir, config.training.experiment_name), exist_ok=True)
    with open(os.path.join(config.training.experiment_dir, config.training.experiment_name, "params.json"), "w") as f:
        f.write(config.to_json())


    precision = config.training["precision"]
    micro_batch_size = config.training["micro_batch_size"]
    filter_chains = config.training["filter_chains"]

    global_batch_size = infer_global_batch_size(
        micro_batch_size=micro_batch_size,
        num_nodes=config.training["num_nodes"],
        devices=config.training["devices"],
        accumulate_grad_batches=config.training["accumulate_grad_batches"],
        tensor_model_parallel_size=config.training["tensor_model_parallel_size"],
        pipeline_model_parallel_size=config.training["pipeline_model_parallel_size"],
    )

    logger.info("Loading the chains dataframe")
    pdb_chains = pd.read_csv(config.training["train_chains_pdb"], index_col='name')
    idp_chains = pd.read_csv(config.training["train_chains_idp"], index_col='name')
    val_pdb_chains = pd.read_csv(config.training["valid_chains_pdb"], index_col='name')
    val_idp_chains = pd.read_csv(config.training["valid_chains_idp"], index_col='name')

    if filter_chains:
        clusters = load_clusters(config.training["train_clusters"])
        pdb_chains = pdb_chains.join(clusters)
        pdb_chains = pdb_chains[pdb_chains.release_date < config.training["train_cutoff"]]

    pdb_trainset = OpenFoldSingleDataset(
        data_dir=config.training["train_data_dir_pdb"],
        alignment_dir=config.training["train_msa_dir_pdb"],
        pdb_chains=pdb_chains,
        config=config.data,
        mode='train',
        subsample_pos=config.training["sample_train_confs_pdb"],
        first_as_template=config.training["first_as_template"],
    )

    idp_trainset = OpenFoldSingleDataset(
        data_dir=config.training["train_data_dir_idp"],
        alignment_dir=config.training["train_msa_dir_idp"],
        pdb_chains=idp_chains,
        config=config.data,
        mode='train',
        subsample_pos=config.training["sample_train_confs_idp"],
        first_as_template=config.training["first_as_template"],
    )

    trainset = OpenFoldDataset(
        [pdb_trainset, idp_trainset],
        [config.training["dataset_prob_pdb"], config.training["dataset_prob_idp"]],
        config.training["train_epoch_len"]
    )

    valset = OpenFoldSingleDataset(
        data_dir=config.training["val_data_dir_pdb"],
        alignment_dir=config.training["val_msa_dir_pdb"],
        pdb_chains=val_pdb_chains,
        config=config.data,
        mode='train',
        subsample_pos=config.training["sample_val_confs_pdb"],
        num_confs=config.training["num_val_confs_pdb"],
        first_as_template=config.training["first_as_template"],
    )

    """
    idp_valset = OpenFoldSingleDataset(
        data_dir=config.training["val_data_dir_idp"],
        alignment_dir=config.training["val_msa_dir_idp"],
        pdb_chains=val_idp_chains,
        config=config.data,
        mode='train',
        subsample_pos=config.training["sample_val_confs_idp"],
        num_confs=config.training["num_val_confs_idp"],
        first_as_template=config.training["first_as_template"],
    )
    """

    data_module = ESMFoldDataModule(
        train_dataset=trainset,
        valid_dataset=valset,
        micro_batch_size=micro_batch_size,  # size of a batch to be processed in a device
        global_batch_size=global_batch_size,  # size of batch across all devices. Should be multiple of micro_batch_size
    )

    esmfold_config = get_esmfoldconfig(
        config=config,
        initial_ckpt_path=str(config.training["initial_nemo_ckpt_path"]),
        params_dtype=get_autocast_dtype(precision),
        pipeline_dtype=get_autocast_dtype(precision),
        autocast_dtype=get_autocast_dtype(precision),
        # setting this speeds things up a lot
        initial_ckpt_skip_keys_with_these_prefixes=[
           #"structure_head",
           #"structure_head.lddt_head",
           #"structure_head.input_pair_embedding",
           #"structure_head.input_time_projection",
           #"structure_head.input_time_embedding",
           #"structure_head.input_pair_stack",
           #"structure_head.extra_input_pair_embedding",
           #"structure_head.extra_input_pair_stack",
        ],
        pretrained_structure_head_path=config.training.pretrained_structure_head_path)

    print(esmfold_config)

    # TODO(fabio): this config needs to become an ml_collections config
    checkpoint, trainer = train_model(
        experiment_name=config.training.experiment_name,
        experiment_dir=config.training.experiment_dir,  # new checkpoint will land in a subdir of this
        config=esmfold_config,  # same config as before since we are just continuing training
        data_module=data_module,
        n_steps_train=config.training.n_steps_train,
        limit_val_batches=config.training.limit_val_batches,
        steps_to_save_ckpt=config.training.steps_to_save_ckpt,
        val_check_interval=config.training.val_check_interval,
        warmup_steps_percentage=config.training.warmup_steps_percentage,
        metric_to_monitor_for_checkpoints="val_loss",
        tensor_model_parallel_size=config.training.tensor_model_parallel_size,
        pipeline_model_parallel_size=config.training.pipeline_model_parallel_size,
        num_nodes=config.training.num_nodes,
        devices=config.training.devices,
        wandb_project=config.training.wandb_project,
    )

if __name__ == "__main__":
    app.run(main)

# In peptron/callbacks.py
import os
import logging
import glob
from typing import Any, Sequence
import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import BasePredictionWriter
from peptron.utils.util import expand_tensors_in_list
from peptron.data import protein
from collections import defaultdict

class StreamingPredictionWriter(BasePredictionWriter):
    """
    This writer merges temporary files identified by a unique prediction ID (UUID)
    to create a single final output file per sequence, avoiding OOM errors.
    """
    def __init__(self, output_dir: str | os.PathLike, write_interval: str = "batch"):
        super().__init__(write_interval)
        self.output_dir = str(output_dir)

    def write_on_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        prediction: Any,
        batch_indices: Sequence[int],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """
        Merges temporary prediction files based on their unique prediction_uuid.
        """
        prediction_uuid_val = prediction.get("prediction_uuid")
        if not prediction_uuid_val:
            logging.warning("StreamingPredictionWriter: No prediction_uuid found in prediction object.")
            return

        # --- THIS IS THE FIX ---
        # Ensure prediction_uuid is a string, not a list containing a string.
        if isinstance(prediction_uuid_val, list):
            prediction_uuid = prediction_uuid_val[0]
        else:
            prediction_uuid = str(prediction_uuid_val)
        # --- END OF FIX ---

        rank = trainer.global_rank

        # This pattern will now correctly match the files on disk.
        temp_file_pattern = os.path.join(self.output_dir, f"_tmp_rank{rank}_predictid_{prediction_uuid}_part*.pt")
        temp_files = sorted(glob.glob(temp_file_pattern))

        if not temp_files:
            # This warning should no longer appear.
            logging.warning(f"No temporary files found for pattern: {temp_file_pattern}")
            return

        first_chunk = torch.load(temp_files[0])
        keys = list(first_chunk.keys())
        final_prediction = {k: [] for k in keys}

        final_prediction["batch_idx"] = torch.tensor([batch_idx], dtype=torch.int64)

        prot_frame_idx = defaultdict(int)
        for idx, temp_path in enumerate(temp_files):
            chunk = torch.load(temp_path)
            for sample in expand_tensors_in_list([chunk['interpolations']]):
                pdb, prot_name = protein.get_prot_pdb(sample)
                i = prot_frame_idx[prot_name]
                os.makedirs(os.path.join(self.output_dir, prot_name), exist_ok=True)
                pdb_path = os.path.join(self.output_dir, prot_name, f"predictions_rank_{rank}_batch_{batch_idx}_{i}.pt")
                with open(pdb_path, "w") as f:
                    f.write(protein.prots_to_pdb(pdb))
                prot_frame_idx[prot_name] += 1
            del chunk
            os.remove(temp_path)

        logging.info(f"Finalized memory-safe prediction for batch {batch_idx}")


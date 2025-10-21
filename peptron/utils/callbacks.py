# In peptron/callbacks.py
import os
import logging
import glob
from typing import Any, Sequence
import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import BasePredictionWriter
from peptron.utils.util import expand_tensors_in_list

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

        # Load all data chunks from the temporary files
        all_chunks = []
        for temp_path in temp_files:
            all_chunks.append(torch.load(temp_path))
            os.remove(temp_path)

        # Collate the chunks into one final prediction object
        final_prediction = {}
        for key in all_chunks[0].keys():
            vals = [chunk[key] for chunk in all_chunks]
            final_prediction[key] = expand_tensors_in_list(vals)

        result_path = os.path.join(self.output_dir, f"predictions__rank_{rank}__batch_{batch_idx}.pt")
        final_prediction["batch_idx"] = torch.tensor([batch_idx], dtype=torch.int64)

        torch.save(final_prediction, result_path)
        logging.info(f"Finalized memory-safe prediction for batch {batch_idx} to {result_path}")


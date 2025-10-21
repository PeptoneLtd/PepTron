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
import torch
import logging
from typing import List, Dict, Any, Union, Optional, Tuple, Sequence

# Improved function based on the new BioNeMo implementation
def safe_batch_collator(
    batches: Optional[Union[Tuple[Any], List[Any]]],
    batch_dim: int = 0,
    seq_dim: int = 1,
    batch_dim_key_defaults: dict[str, int] = {"token_logits": 1},
    seq_dim_key_defaults: dict[str, int] = {"token_logits": 0},
) -> Optional[Any]:
    """
    A safe version of the batch_collator function that properly handles scalar tensors and sequences of different lengths.
    
    Args:
        batches: A list of tensors or other objects to collate
        batch_dim: The dimension along which to concatenate tensors
        seq_dim: The sequence dimension for padding
        batch_dim_key_defaults: Dictionary of keys to batch dimensions
        seq_dim_key_defaults: Dictionary of keys to sequence dimensions
        
    Returns:
        A concatenated tensor or the original list if concatenation is not possible
    """
    # Handle None case
    if batches is None or len(batches) == 0:
        return batches
        
    # Handle None values in the list
    if batches[0] is None:
        return None
        
    # Handle tensor case
    if isinstance(batches[0], torch.Tensor):
        # First shortcut if all tensors are 1D
        if batches[0].ndim == 1:
            try:
                return torch.cat(batches, dim=0)
            except RuntimeError as e:
                logging.warning(f"Could not concatenate 1D tensors: {e}. Returning as list.")
                return batches
                
        # Handle scalar tensors
        if batches[0].ndim == 0:
            try:
                return torch.stack(batches)
            except RuntimeError as e:
                logging.warning(f"Could not stack scalar tensors: {e}. Returning as list.")
                return batches
                
        # For sequence tensors, pad to max length
        try:
            # Find max sequence length across all tensors
            max_seq_len = max(batch.size(seq_dim) for batch in batches if isinstance(batch, torch.Tensor))
            
            # Pad each tensor to the max length
            padded_batches = []
            for batch in batches:
                if not isinstance(batch, torch.Tensor):
                    continue
                    
                if batch.size(seq_dim) < max_seq_len:
                    # Initialize padding tuple
                    pad_size = [0] * (2 * batch.ndim)
                    # Calculate padding needed at end of sequence dimension
                    pad_amount = max_seq_len - batch.size(seq_dim)
                    # Pad end of sequence dimension
                    pad_size[2 * (batch.ndim - 1 - seq_dim) + 1] = pad_amount
                    padded_batch = torch.nn.functional.pad(batch, tuple(pad_size))
                    padded_batches.append(padded_batch)
                else:
                    padded_batches.append(batch)
                    
            return torch.cat(padded_batches, dim=batch_dim)
        except Exception as e:
            logging.warning(f"Error during tensor padding/concatenation: {e}. Returning as list.")
            return batches
            
    # Handle dictionary case
    elif isinstance(batches[0], dict):
        return {
            key: safe_batch_collator(
                [batch[key] for batch in batches if key in batch],
                batch_dim=batch_dim_key_defaults.get(key, batch_dim),
                seq_dim=seq_dim_key_defaults.get(key, seq_dim),
                batch_dim_key_defaults=batch_dim_key_defaults,
                seq_dim_key_defaults=seq_dim_key_defaults,
            )
            for key in batches[0]
        }
        
    # Handle tuple case
    elif isinstance(batches[0], tuple):
        return tuple(
            safe_batch_collator(
                [batch[i] for batch in batches if i < len(batch)],
                batch_dim=batch_dim,
                seq_dim=seq_dim,
                batch_dim_key_defaults=batch_dim_key_defaults,
                seq_dim_key_defaults=seq_dim_key_defaults,
            )
            for i in range(len(batches[0]))
        )
        
    # Handle list case
    elif isinstance(batches[0], list):
        return [
            safe_batch_collator(
                [batch[i] for batch in batches if i < len(batch)],
                batch_dim=batch_dim,
                seq_dim=seq_dim,
                batch_dim_key_defaults=batch_dim_key_defaults,
                seq_dim_key_defaults=seq_dim_key_defaults,
            )
            for i in range(len(batches[0]))
        ]
        
    # Default case - return as is
    return batches

def recursive_batch_collator(batch_dict: Dict[str, Any], batch_dim: int = 0) -> Dict[str, Any]:
    """
    A recursive version of the batch_collator that handles nested dictionaries.
    
    Args:
        batch_dict: A dictionary of tensors or nested dictionaries to collate
        batch_dim: The dimension along which to concatenate tensors
        
    Returns:
        A dictionary with collated values
    """
    result = {}
    for key, value in batch_dict.items():
        if isinstance(value, dict):
            # Recursively handle nested dictionaries
            result[key] = recursive_batch_collator(value, batch_dim)
        elif isinstance(value, list) and value and isinstance(value[0], dict):
            # Handle list of dictionaries
            result[key] = [recursive_batch_collator({k: [d[k] for d in value] for k in value[0]}, batch_dim)]
        else:
            # Handle leaf values (tensors or other objects)
            result[key] = safe_batch_collator(value, batch_dim)
    return result

def apply_monkey_patches():
    """Apply monkey patches to fix tensor collation issues in BioNeMo."""
    try:
        # Import the module containing the function we want to patch
        import bionemo.llm.lightning
        
        # Save original function for reference
        original_batch_collator = bionemo.llm.lightning.batch_collator
        
        # Replace with our safe version
        bionemo.llm.lightning.batch_collator = safe_batch_collator
        
        # Also patch the PassthroughLossReduction.reduce method if it exists
        try:
            from bionemo.llm.lightning import PassthroughLossReduction
            original_reduce = PassthroughLossReduction.reduce
            
            # Create a patched version of the reduce method
            def patched_reduce(self, forward_out):
                return safe_batch_collator(forward_out)
                
            # Apply the patch
            PassthroughLossReduction.reduce = patched_reduce
            logging.info("Successfully applied monkey patch to PassthroughLossReduction.reduce")
        except (ImportError, AttributeError):
            logging.warning("Could not patch PassthroughLossReduction.reduce (not found)")
        
        logging.info("Successfully applied monkey patch to bionemo.llm.lightning.batch_collator")
        return True
    except ImportError:
        logging.error("Could not import bionemo.llm.lightning for monkey patching")
        return False
    except AttributeError:
        logging.error("Could not find batch_collator function in bionemo.llm.lightning")
        return False
    except Exception as e:
        logging.error(f"Error applying monkey patch: {e}")
        return False
import torch
import pandas as pd
from glob import glob
from bionemo.llm.lightning import batch_collator
import numpy as np
from datetime import datetime
from peptron.utils.logger_config import logger
from esm2.data import tokenizer
from collections import defaultdict
from pathlib import Path
import os
from typing import Dict, List, Sequence, Type, get_args, Any, Literal, Optional
from openfold.np.residue_constants import atom_types, atom_order
import json


needed_keys = [
    "aatype",
    "final_atom_positions",
    "atom37_atom_exists",
    "residue_index",
    "name",
    "plddt",
]


class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert arrays to Python lists
        elif isinstance(obj, np.integer): # Handle numpy integers
            return int(obj)
        elif isinstance(obj, np.floating): # Handle numpy floats
            return float(obj)
        elif isinstance(obj, np.bool_): # Handle numpy booleans
            return bool(obj)
        # Let the base class default method raise the TypeError for other types
        return super(NumpyEncoder, self).default(obj)

def rmsdalign(a, b, weights=None): # alignes B to A  # [*, N, 3]
    B = a.shape[:-2]
    N = a.shape[-2]
    if weights == None:
        weights = a.new_ones(*B, N)
    weights = weights.unsqueeze(-1)
    a_mean = (a * weights).sum(-2, keepdims=True) / weights.sum(-2, keepdims=True)
    a = a - a_mean
    b_mean = (b * weights).sum(-2, keepdims=True) / weights.sum(-2, keepdims=True)
    b = b - b_mean
    B = torch.einsum('...ji,...jk->...ik', weights * a, b)
    u, s, vh = torch.linalg.svd(B)

    # Correct improper rotation if necessary (as in Kabsch algorithm)
    '''
    if torch.linalg.det(u @ vh) < 0:
        s[-1] = -s[-1]
        u[:, -1] = -u[:, -1]
    '''
    sgn = torch.sign(torch.linalg.det(u @ vh))
    s[...,-1] *= sgn
    u[...,:,-1] *= sgn.unsqueeze(-1)
    C = u @ vh # c rotates B to A
    return b @ C.mT + a_mean

def pad_tensor_list(tensor_list, pad_value=0):
    """
    Pad a list of tensors with different shapes to the maximum dimension.
    
    Args:
        tensor_list (List[torch.Tensor]): List of tensors to pad
        pad_value (int/float): Value to use for padding
    
    Returns:
        torch.Tensor: Padded and stacked tensor
    """
    # Find maximum dimensions
    max_dims = []
    num_dims = max(tensor.dim() for tensor in tensor_list)
    
    # Ensure all tensors have the same number of dimensions
    padded_tensors = []
    for tensor in tensor_list:
        if tensor.dim() < num_dims:
            # Add dimensions at the end until reaching num_dims
            tensor = tensor.view(*tensor.shape, *([1] * (num_dims - tensor.dim())))
        padded_tensors.append(tensor)
    
    # Find max length for each dimension
    for dim in range(num_dims):
        max_dim_length = max(tensor.shape[dim] for tensor in padded_tensors)
        max_dims.append(max_dim_length)
    
    # Pad each tensor to max dimensions
    padded_tensors = []
    for tensor in tensor_list:
        pad_sizes = []
        current_shape = tensor.shape
        
        # Calculate padding sizes for each dimension
        for i, (current_dim, max_dim) in enumerate(zip(current_shape, max_dims)):
            pad_size = max_dim - current_dim
            # Padding needs to be specified from last dim to first, with both sides
            pad_sizes = [0, pad_size] + pad_sizes
            
        # Pad the tensor
        padded_tensor = torch.nn.functional.pad(tensor, pad_sizes, value=pad_value)
        padded_tensors.append(padded_tensor)
    
    # Stack all padded tensors
    return torch.stack(padded_tensors, dim=0)

# Modified batch processing code
def process_batch(_batch, required_keys, logger=None):
    """
    Process a batch by moving tensors to CUDA and handling different sized tensors.
    
    Args:
        _batch (dict): Batch dictionary
        required_keys (list): List of required keys to process
        logger (Logger, optional): Logger instance for debugging
    
    Returns:
        dict: Processed batch
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    for key, val in _batch.items():
        if key in required_keys and val is not None:
            if isinstance(val, torch.Tensor):
                _batch[key] = val.cuda(non_blocking=True)
            else:
                # Check if val is a list of tensors
                if isinstance(val, list) and all(isinstance(x, torch.Tensor) for x in val):
                    try:
                        # Try to pad and stack tensors
                        padded_tensor = pad_tensor_list(val)
                        _batch[key] = padded_tensor.cuda(non_blocking=True)
                        logger.debug(f"Successfully padded tensors for key '{key}'")
                    except Exception as e:
                        logger.error(f"Error padding tensors for key '{key}': {str(e)}")
                        _batch[key] = None
                else:
                    _batch[key] = None
        else:
            _batch[key] = None
    
    return _batch

    # Use it as:  
    # _batch = process_batch(_batch, required_keys, logger)


def load_and_collate_predictions(work_dir):
    """
    Load and collate prediction files from multiple ranks in the specified directory.
    
    Args:
        work_dir (str): Directory containing the prediction .pt files.
    
    Returns:
        dict: Collated predictions with structure output.
    """
    
    prediction_files = glob(f"{work_dir}/predictions__rank_*.pt")
    #predictions = [torch.load(path) for path in prediction_files]
    predictions = [torch.load(path, map_location="cpu") for path in prediction_files]
    collated_predictions = batch_collator(predictions)
    return collated_predictions


def convert_tensor(x):
    # Keep aatype as long (int64)
    if x.dtype in [torch.int32, torch.int64, torch.long] and not isinstance(x, str):
        return x.long().cpu().numpy()
    # Convert other tensors (positions, masks, etc) to float32
    elif isinstance(x, torch.Tensor):
        return x.float().cpu().numpy()
    # Return non-tensor values unchanged
    return x

def str_to_ascii(val):
    if isinstance(val, str):
        # Convert string to list of ASCII/Unicode values
        char_codes = [ord(c) for c in val]
        val = torch.tensor(char_codes, dtype=torch.long)
    elif isinstance(val, list) and all(isinstance(s, str) for s in val):
        # Handle list of strings
        char_codes_list = [[ord(c) for c in s] for s in val]
        # Pad to same length if necessary
        max_len = max(len(codes) for codes in char_codes_list)
        padded_codes = [codes + [0] * (max_len - len(codes)) for codes in char_codes_list]
        val = torch.tensor(padded_codes, dtype=torch.long)
    else:
        val = None
    return val


def get_batch_size(dict_item):
    """Find the batch size from the first valid tensor in the dict"""
    for value in dict_item.values():
        if hasattr(value, 'shape') and len(value.shape) >= 2:
            return value.shape[0]
    return None

def unbatch_item(item, batch_idx):
    """Unbatch a single item if it's a valid tensor, otherwise return as-is"""
    if hasattr(item, 'shape') and len(item.shape) >= 2:
        # This is a tensor with at least 2 dimensions - unbatch it
        return item[batch_idx:batch_idx+1]
    else:
        # This is not a tensor or has < 2 dimensions - return as-is
        return item

def unbatch_tensor_dicts(k):
    """
    Transform a list of dicts containing batched tensors into individual tensor dicts.
    Skips non-tensor values and tensors with dimension < 2.
    
    Args:
        k: List of dicts, where each dict contains tensors with batch dimension
    
    Returns:
        l: List of dicts, where each dict contains tensors with batch dimension = 1
    """
    l = []
    
    for dict_item in k:
        # Get the batch size from the first valid tensor found
        batch_size = get_batch_size(dict_item)
        
        if batch_size is None:
            # If no valid tensors found, just append the dict as-is
            l.append(dict_item)
            continue
        
        # For each item in the batch, create a new dict
        for batch_idx in range(batch_size):
            new_dict = {}
            for key, value in dict_item.items():
                new_dict[key] = unbatch_item(value, batch_idx)
            l.append(new_dict)
    
    return l


def _repeat_item(item: Any, n: int):
    """Repeat *one* element of a batch *n* times along its leading axis."""
    if isinstance(item, torch.Tensor):
        # Always convert to at least 1-D, then repeat_interleave on dim 0.
        if item.dim() == 0:
            # Scalar → shape [1] before interleaving
            item = item.unsqueeze(0)
        return item.repeat_interleave(n, dim=0)

    if isinstance(item, str):
        return [item] * n

    if isinstance(item, (list, tuple)):
        repeated = itertools.chain.from_iterable(
            itertools.repeat(elem, n) for elem in item
        )
        return type(item)(repeated)

    if isinstance(item, dict):
        return {k: _repeat_item(v, n) for k, v in item.items()}

    # Otherwise leave it unchanged
    return item


def expand_tensors_in_list(data):
    """
    Expands list of dicts so that tensors with first dim > 1 are split into shape [1, ...],
    keeping only keys in needed_keys.
    Non-tensor values are copied as-is.
    """
    expanded_list = []

    for d in data:
        # Find the max length to expand from tensor values (only for needed keys)
        lengths = [
            v.shape[0]
            for k, v in d.items()
            if k in needed_keys and isinstance(v, torch.Tensor) and v.ndim > 0
        ]
        if not lengths:
            # No tensors — just keep filtered dict as-is
            expanded_list.append({k: v for k, v in d.items() if k in needed_keys})
            continue

        expand_len = lengths[0]  # assume same first dim for all relevant tensors

        for i in range(expand_len):
            new_dict = {}
            for k, v in d.items():
                if k not in needed_keys:
                    continue
                if isinstance(v, torch.Tensor) and v.ndim > 0:
                    if v.shape[0] == expand_len:
                        new_dict[k] = v[i:i+1]  # slice to keep shape [1, ...]
                    elif v.shape[0] == 1:
                        new_dict[k] = v.clone()  # already a single row
                    else:
                        raise ValueError(
                            f"Tensor at key '{k}' has first dim {v.shape[0]} inconsistent with others ({expand_len})"
                        )
                else:
                    new_dict[k] = v  # non-tensor value
            expanded_list.append(new_dict)

    return expanded_list

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
from peptron.data import protein
import os
import argparse
import glob
from tqdm.auto import tqdm
from collections import defaultdict
from peptron.utils.logger_config import logger
from multiprocessing import Pool


def process_path(path_and_output):
    path, output_dir = path_and_output
    try:
        prediction = torch.load(path, map_location="cpu")
        all_interps = prediction["interpolations"]

        processed_struct_output = defaultdict(list)
        # You may want to disable the inner tqdm for cleaner logs in multiprocessing:
        for sample in all_interps:
            prot_list, prot_name = protein.get_prot_pdb(sample)
            processed_struct_output[prot_name].extend(prot_list)

        protein.write_pdb(output_dir, processed_struct_output)
        return True
    except Exception as e:
        logger.info(f"❌ {path} raised the error {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Convert PyTorch model output to PDB files')
    parser.add_argument('--input', '-i', default=None, help='Directory containing the prediction .pt files')
    parser.add_argument('--output_dir', '-o', default=None, help='Directory to save output PDB files')
    parser.add_argument('--num_processes', '-p', default=16, help='Number of concurrent processes in Pool')
    args = parser.parse_args()
    
    # Use provided paths or defaults
    if args.input is None:
        work_dir = '/mnt/data/carlo-workdir/peptron_preds'
    else:
        work_dir = args.input
        
    if args.output_dir is None:
        output_dir = work_dir
    else:
        output_dir = args.output_dir
        
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all prediction files
    file_paths = glob.glob(f"{work_dir}/predictions__rank_*.pt")
    paths_and_output = [(path, output_dir) for path in file_paths]

    print(f"Loading prediction files from: {work_dir}")
    with Pool(processes=int(args.num_processes)) as pool:
        for _ in tqdm(pool.imap_unordered(process_path, paths_and_output), total=len(paths_and_output), desc="Prediction files"):
            pass  # tqdm shows how many have completed

if __name__ == "__main__":
    main()

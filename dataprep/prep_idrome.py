import argparse
import dataclasses
import io
import os
import tempfile
from multiprocessing import Pool
from typing import Any, Mapping, Optional

import mdtraj
import numpy as np
import pandas as pd
import tqdm
from Bio.PDB import PDBParser
from openfold.data.data_pipeline import make_protein_features
from openfold.np import residue_constants
from peptron.utils.logger_config import logger

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--split', type=str, default='splits/IDRome_DB-val.csv',
                    help='Path to the CSV file with the names column.')
parser.add_argument('--ensembles_dir', type=str, required=True,
                    help='Directory containing .pdb topology and .xtc trajectory files.')
parser.add_argument('--outdir', type=str, default='./IDRome_DB-clustered-val',
                    help='Output directory to save .npz files.')
parser.add_argument('--num_workers', type=int, default=1, help='Number of worker processes.')
args = parser.parse_args()

# Complete sequence of chain IDs supported by the PDB format.
PDB_CHAIN_IDS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
PDB_MAX_CHAINS = len(PDB_CHAIN_IDS)  # := 62.


@dataclasses.dataclass(frozen=True)
class Protein:
    """Protein structure representation."""

    # Cartesian coordinates of atoms in angstroms. The atom types correspond to
    # residue_constants.atom_types, i.e. the first three are N, CA, CB.
    atom_positions: np.ndarray  # [num_res, num_atom_type, 3]

    # Amino-acid type for each residue represented as an integer between 0 and
    # 20, where 20 is 'X'.
    aatype: np.ndarray  # [num_res]

    # Binary float mask to indicate presence of a particular atom. 1.0 if an atom
    # is present and 0.0 if not. This should be used for loss masking.
    atom_mask: np.ndarray  # [num_res, num_atom_type]

    # Residue index as used in PDB. It is not necessarily continuous or 0-indexed.
    residue_index: np.ndarray  # [num_res]

    # 0-indexed number corresponding to the chain in the protein that this residue
    # belongs to.
    chain_index: np.ndarray  # [num_res]

    # B-factors, or temperature factors, of each residue (in sq. angstroms units),
    # representing the displacement of the residue from its ground truth mean
    # value.
    b_factors: np.ndarray  # [num_res, num_atom_type]

    def __post_init__(self):
        if len(np.unique(self.chain_index)) > PDB_MAX_CHAINS:
            raise ValueError(
                f'Cannot build an instance with more than {PDB_MAX_CHAINS} chains '
                'because these cannot be written to PDB format.')


def from_pdb_md_string(pdb_str: str, chain_id: Optional[str] = None) -> Protein:
    """Takes a PDB string and constructs a Protein object.

    WARNING: All non-standard residue types will be converted into UNK. All
    non-standard atoms will be ignored.

    Args:
    pdb_str: The contents of the pdb file
    chain_id: If chain_id is specified (e.g. A), then only that chain
        is parsed. Otherwise all chains are parsed.

    Returns:
    A new `Protein` parsed from the pdb contents.
    """
    pdb_fh = io.StringIO(pdb_str)
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('none', pdb_fh)
    models = list(structure.get_models())
    if len(models) != 1:
        raise ValueError(
            f'Only single model PDBs are supported. Found {len(models)} models.')
    model = models[0]

    atom_positions = []
    aatype = []
    atom_mask = []
    residue_index = []
    chain_ids = []
    b_factors = []

    for chain in model:
        if chain_id is not None and chain.id != chain_id:
            continue
        for res in chain:
            if res.id[2] != ' ':
                raise ValueError(
                    f'PDB contains an insertion code at chain {chain.id} and residue '
                    f'index {res.id[1]}. These are not supported.')
            res_shortname = residue_constants.restype_3to1.get(res.resname, 'X')
            restype_idx = residue_constants.restype_order.get(
                res_shortname, residue_constants.restype_num)
            pos = np.zeros((residue_constants.atom_type_num, 3))
            mask = np.zeros((residue_constants.atom_type_num,))
            res_b_factors = np.zeros((residue_constants.atom_type_num,))
            for atom in res:
                if atom.name not in residue_constants.atom_types:
                    continue
                pos[residue_constants.atom_order[atom.name]] = atom.coord
                mask[residue_constants.atom_order[atom.name]] = 1.
                res_b_factors[residue_constants.atom_order[atom.name]] = atom.bfactor
            if np.sum(mask) < 0.5:
                # If no known atom positions are reported for the residue then skip it.
                continue
            aatype.append(restype_idx)
            atom_positions.append(pos)
            atom_mask.append(mask)
            residue_index.append(res.id[1])
            chain_ids.append(chain.id)
            b_factors.append(res_b_factors)

    # Chain IDs are usually characters so map these to ints.
    unique_chain_ids = np.unique(chain_ids)
    chain_id_mapping = {cid: n for n, cid in enumerate(unique_chain_ids)}
    chain_index = np.array([chain_id_mapping[cid] for cid in chain_ids])

    return Protein(
        atom_positions=np.array(atom_positions),
        atom_mask=np.array(atom_mask),
        aatype=np.array(aatype),
        residue_index=np.array(residue_index),
        chain_index=chain_index,
        b_factors=np.array(b_factors))


os.makedirs(args.outdir, exist_ok=True)

df = pd.read_csv(args.split, index_col='name')


def do_job(name):
    """
    Loads trajectory and topology for a given name, processes each frame,
    and saves the features as a .npz file.
    """
    try:
        traj_path = f'{args.ensembles_dir}/{name}.xtc'
        top_path = f'{args.ensembles_dir}/{name}.pdb'

        # Load trajectory from .xtc file with topology from .pdb file
        traj = mdtraj.load(traj_path, top=top_path)

        f, temp_path = tempfile.mkstemp(suffix=".pdb");
        os.close(f)
        positions_stacked = []

        # Process each frame in the trajectory
        for i in tqdm.trange(len(traj), desc=f'Processing {name}', leave=False):
            traj[i].save_pdb(temp_path)

            with open(temp_path) as f_temp:
                prot = from_pdb_md_string(f_temp.read())
                pdb_feats = make_protein_features(prot, name)
                positions_stacked.append(pdb_feats['all_atom_positions'])

        # Stack features from all frames
        pdb_feats['all_atom_positions'] = np.stack(positions_stacked)

        # Save the combined features to a compressed .npz file
        np.savez_compressed(f"{args.outdir}/{name}.npz", **pdb_feats)
        os.unlink(temp_path)

    except Exception as e:
        logger.error(f'Could not process {name}. Error: {e}. Skipping.')
        pass


def main():
    """
    Main function to set up multiprocessing and run jobs.
    """
    jobs = [name for name in df.index]

    if args.num_workers > 1:
        with Pool(args.num_workers) as p:
            list(tqdm.tqdm(p.imap(do_job, jobs), total=len(jobs), desc="Overall Progress"))
    else:
        # Use a simple map for single-threaded execution for easier debugging
        list(tqdm.tqdm(map(do_job, jobs), total=len(jobs), desc="Overall Progress"))


if __name__ == '__main__':
    main()


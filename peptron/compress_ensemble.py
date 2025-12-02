import argparse
import logging
import glob
import os
import MDAanlysis as mda
import mdtraj as md
from peptron.utils.filter_unphysical_traj import filter_unphysical_traj

logger = logging.getLogger(__name__)


def pdb_dir_to_xtc(pdb_dir, filter_unphysical_frames: bool = False):
    """
    Convert directory of PDB files to PDB+XTC without loading all frames in memory.
    Creates 2 files in the specified directory:
     - topology.pdb
     - trajectory.xtc
    removes original pdb files
    """

    pdb_files = sorted(glob.glob(os.path.join(pdb_dir, "*.pdb")))
    if not pdb_files:
        raise ValueError(f"No PDB files found in {pdb_dir}")

    logger.info(f"Found {len(pdb_files)} PDB files")

    u = mda.Universe(pdb_files[0])
    topology_file = os.path.join(pdb_dir, "topology.pdb")
    u.atoms.write(topology_file)

    # Write XTC incrementally, one PDB at a time
    output_xtc = os.path.join(pdb_dir, "trajectory.xtc")
    valid_frame_count = 0
    with mda.Writer(output_xtc, u.atoms.n_atoms) as W:
        for i, pdb_file in enumerate(pdb_files):
            if filter_unphysical_frames:
                t = md.load(pdb_file)
                t = filter_unphysical_traj(t, strict=False)
                if len(t) == 0:
                    # frame not valid, skipping
                    continue
            valid_frame_count += 1
            frame = mda.Universe(pdb_file)
            W.write(frame.atoms)
            os.remove(pdb_file)
            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1}/{len(pdb_files)} frames")

    logger.info(f"Saved trajectory: {output_xtc}")
    logger.info(f"Total frames: {len(pdb_files)}, valid frames: {valid_frame_count}")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--pdb-dir", required=True, type=str)
    args.add_argument("--filter-unphysical", required=False, action="store_true", default=False)
    args = args.parse_args()

    pdb_dir_to_xtc(args.pdb_dir, args.filter_unphysical)
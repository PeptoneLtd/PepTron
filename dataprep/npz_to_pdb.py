#!/usr/bin/env python
# NPZ to PDB converter for protein structure data
# This script converts protein structure data from NPZ to PDB format

import os
import numpy as np
import argparse

# Define atom names for standard amino acids
# These correspond to the 37 atom positions in the all_atom_positions array
ATOM_TYPES = [
    "N", "CA", "C", "O",                     # Backbone atoms
    "CB", "CG", "CG1", "CG2", "OG", "OG1",   # Common side chain atoms
    "SG", "CD", "CD1", "CD2", "ND1", "ND2", 
    "OD1", "OD2", "CE", "CE1", "CE2", "CE3", 
    "NE", "NE1", "NE2", "OE1", "OE2", 
    "CZ", "CZ2", "CZ3", "NZ", 
    "CH2", "OH", "NH1", "NH2",
    "OXT", "SE"                              # Terminal oxygen and selenomethionine
]

# Mapping from residue type indices to three-letter codes
# This is based on standard AlphaFold/protein models indexing
RESTYPES = [
    "ALA", "ARG", "ASN", "ASP", "CYS", 
    "GLN", "GLU", "GLY", "HIS", "ILE", 
    "LEU", "LYS", "MET", "PHE", "PRO", 
    "SER", "THR", "TRP", "TYR", "VAL",
    "UNK"  # Unknown/other residue type
]

# The standard 20 amino acids in one-letter code
ONE_LETTER_MAP = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "UNK": "X"
}

# Mapping from one-letter to three-letter codes
ONE_TO_THREE = {v: k for k, v in ONE_LETTER_MAP.items()}

def get_restypes_from_aatype(aatype):
    """Convert numerical aatype to three-letter residue codes."""
    return [RESTYPES[aa] if aa < len(RESTYPES) else "UNK" for aa in np.argmax(aatype, axis=1)]

def get_restypes_from_sequence(sequence):
    """Convert a sequence string to a list of three-letter residue codes."""
    return [ONE_TO_THREE.get(aa.upper(), "UNK") for aa in sequence]

def npz_to_pdb(npz_path, output_path=None, chain_id="A"):
    """
    Convert protein structure data from NPZ format to PDB.
    
    Parameters:
    -----------
    npz_path : str
        Path to the NPZ file containing protein structure data
    output_path : str, optional
        Path to save the PDB file (default: same as input with .pdb extension)
    chain_id : str, optional
        Chain identifier to use in the PDB file (default: "A")
    
    Returns:
    --------
    str
        Path to the created PDB file
    """
    # Default output path if none provided
    if output_path is None:
        output_path = os.path.splitext(npz_path)[0] + ".pdb"
    
    # Load the NPZ file
    print(f"Loading NPZ file: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    
    # Print available keys (useful for debugging)
    print(f"Available keys in NPZ file: {list(data.keys())}")
    
    # Extract required data
    atom_positions = data['all_atom_positions']  # Shape: (num_residues, num_atoms, 3)
    atom_mask = data['all_atom_mask']           # Shape: (num_residues, num_atoms)
    
    # Determine residue types
    if 'aatype' in data:
        # If one-hot encoded aatype is available
        if data['aatype'].shape[1] == 21:  # One-hot encoded
            residue_types = get_restypes_from_aatype(data['aatype'])
        else:  # Direct indices
            residue_types = [RESTYPES[aa] if aa < len(RESTYPES) else "UNK" for aa in data['aatype']]
    elif 'sequence' in data:
        # If direct sequence is available
        sequence = data['sequence'][0] if isinstance(data['sequence'], np.ndarray) else data['sequence']
        residue_types = get_restypes_from_sequence(sequence)
    else:
        raise ValueError("Neither 'aatype' nor 'sequence' found in NPZ file")
    
    # Extract or generate residue indices
    if 'residue_index' in data:
        residue_indices = data['residue_index']
    else:
        residue_indices = np.arange(1, len(residue_types) + 1)
    
    # Open the output file
    with open(output_path, 'w') as f:
        # Write HEADER line
        domain_name = data['domain_name'][0] if 'domain_name' in data else "UNKNOWN"
        f.write(f"HEADER    {domain_name}\n")
        
        # Write TITLE line
        f.write(f"TITLE     STRUCTURE CONVERTED FROM NPZ\n")
        
        # Add resolution if available
        if 'resolution' in data:
            resolution = float(data['resolution'][0])
            f.write(f"REMARK   2 RESOLUTION. {resolution:6.2f} ANGSTROMS\n")
        
        # Write ATOM records
        atom_serial = 1
        for i in range(len(residue_types)):
            resname = residue_types[i]
            resnum = int(residue_indices[i])
            
            # For each atom in this residue
            for j in range(min(len(ATOM_TYPES), atom_positions.shape[1])):
                # Skip atoms that are masked out (not present)
                if atom_mask[i, j] < 0.5:
                    continue
                
                # Get coordinates
                x, y, z = atom_positions[i, j]
                
                # Get atom name
                atom_name = ATOM_TYPES[j] if j < len(ATOM_TYPES) else f"A{j+1}"
                
                # Format atom name - right justify for most, left justify for two-character elements with two chars
                if len(atom_name) >= 4:
                    atom_name_fmt = atom_name[:4]
                elif len(atom_name) == 1:
                    atom_name_fmt = f" {atom_name}  "
                else:
                    # Handle two-character atom names (like CD, CG, etc.)
                    atom_name_fmt = f" {atom_name} "
                
                # Write ATOM line in PDB format
                f.write(f"ATOM  {atom_serial:5d} {atom_name_fmt} {resname:3s} {chain_id:1s}{resnum:4d}    "
                        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {atom_name[0]:1s}  \n")
                
                atom_serial += 1
        
        # Write END marker
        f.write("END\n")
    
    print(f"Created PDB file: {output_path}")
    return output_path

def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(description="Convert protein structure data from NPZ to PDB format")
    parser.add_argument("npz_file", help="Path to the NPZ file containing protein structure data")
    parser.add_argument("--output", "-o", help="Path to save the PDB file (default: same as input with .pdb extension)")
    parser.add_argument("--chain", "-c", default="A", help="Chain identifier to use in the PDB file (default: A)")
    
    args = parser.parse_args()
    
    # Convert the file
    output_file = npz_to_pdb(args.npz_file, args.output, args.chain)
    print(f"Successfully converted {args.npz_file} to {output_file}")

if __name__ == "__main__":
    main()

"""
computeAveragedFeatures.py: Compute surface vertex features based on neighbor atom averaging.

This module implements the neighbor-based feature averaging approach where:
- Surface vertices search for protein atoms within a configurable radius (default 2.5Å)
- If N>=1 neighbor atoms are found, compute the average of their features
- If N=0 neighbor atoms, fall back to the original (nearest atom) feature value
"""

import numpy as np
from scipy.spatial import KDTree
from Bio.PDB import PDBParser, Selection

from .atom_params import get_atom_partial_charge, get_atom_hydrophobicity
from .computeHydrophobicity import kd_scale


def compute_averaged_vertex_features(
    vertices: np.ndarray,
    names: list,
    pdb_file: str,
    neighbor_radius: float = 2.5,
    use_atom_charges: bool = True,
    use_atom_hphob: bool = True,
) -> tuple:
    """
    Compute averaged vertex features based on neighbor atoms within the specified radius.
    
    Args:
        vertices: Surface vertices coordinates, shape (N, 3)
        names: Vertex names from MSMS output, format "chain_resid_ins_resname_atomname_color"
        pdb_file: Path to the PDB file (with or without .pdb extension)
        neighbor_radius: Radius in Angstroms to search for neighbor atoms (default 2.5Å)
        use_atom_charges: Whether to compute atom-level partial charges
        use_atom_hphob: Whether to compute atom-level hydrophobicity
    
    Returns:
        Tuple of (vertex_charges, vertex_hphobicity):
            - vertex_charges: Array of averaged partial charges per vertex
            - vertex_hphobicity: Array of averaged hydrophobicity per vertex
    """
    # Parse PDB to get atom coordinates and information
    parser = PDBParser(QUIET=True)
    if not pdb_file.endswith('.pdb'):
        pdb_file = pdb_file + '.pdb'
    
    struct = parser.get_structure('protein', pdb_file)
    atoms = list(struct.get_atoms())
    
    if len(atoms) == 0:
        # Return zeros if no atoms found
        return np.zeros(len(vertices)), np.zeros(len(vertices))
    
    # Build atom coordinate array and info list
    atom_coords = np.array([atom.get_coord() for atom in atoms])
    atom_info = []  # List of (res_name, atom_name) tuples
    for atom in atoms:
        res = atom.get_parent()
        atom_info.append((res.get_resname(), atom.get_name()))
    
    # Build KDTree for efficient neighbor search
    atom_kdtree = KDTree(atom_coords)
    
    # Initialize output arrays
    vertex_charges = np.zeros(len(vertices))
    vertex_hphobicity = np.zeros(len(vertices))
    
    # Compute original (fallback) values from MSMS names
    original_charges = np.zeros(len(vertices))
    original_hphob = np.zeros(len(vertices))
    
    for ix, name in enumerate(names):
        fields = name.split("_")
        if len(fields) >= 5:
            res_name = fields[3]
            atom_name = fields[4]
            
            # Get original atom-level values
            original_charges[ix] = get_atom_partial_charge(res_name, atom_name, default=0.0)
            original_hphob[ix] = get_atom_hydrophobicity(res_name, atom_name, default=0.0)
    
    # For each vertex, find neighbor atoms and compute averaged features
    for vi in range(len(vertices)):
        vertex_coord = vertices[vi]
        
        # Query all atoms within neighbor_radius
        neighbor_indices = atom_kdtree.query_ball_point(vertex_coord, neighbor_radius)
        
        if len(neighbor_indices) == 0:
            # No neighbors found: use original values
            vertex_charges[vi] = original_charges[vi]
            vertex_hphobicity[vi] = original_hphob[vi]
        else:
            # Compute average over neighbor atoms
            charges_sum = 0.0
            hphob_sum = 0.0
            
            for atom_idx in neighbor_indices:
                res_name, atom_name = atom_info[atom_idx]
                charges_sum += get_atom_partial_charge(res_name, atom_name, default=0.0)
                hphob_sum += get_atom_hydrophobicity(res_name, atom_name, default=0.0)
            
            n_neighbors = len(neighbor_indices)
            vertex_charges[vi] = charges_sum / n_neighbors
            vertex_hphobicity[vi] = hphob_sum / n_neighbors
    
    return vertex_charges, vertex_hphobicity


def compute_averaged_charges_only(
    vertices: np.ndarray,
    names: list,
    pdb_file: str,
    neighbor_radius: float = 2.5,
) -> np.ndarray:
    """
    Compute only averaged vertex charges based on neighbor atoms.
    
    Args:
        vertices: Surface vertices coordinates, shape (N, 3)
        names: Vertex names from MSMS output
        pdb_file: Path to the PDB file
        neighbor_radius: Radius in Angstroms to search for neighbor atoms
    
    Returns:
        Array of averaged partial charges per vertex
    """
    charges, _ = compute_averaged_vertex_features(
        vertices, names, pdb_file, neighbor_radius,
        use_atom_charges=True, use_atom_hphob=False
    )
    return charges


def compute_averaged_hphobicity_only(
    vertices: np.ndarray,
    names: list,
    pdb_file: str,
    neighbor_radius: float = 2.5,
) -> np.ndarray:
    """
    Compute only averaged vertex hydrophobicity based on neighbor atoms.
    
    Args:
        vertices: Surface vertices coordinates, shape (N, 3)
        names: Vertex names from MSMS output
        pdb_file: Path to the PDB file
        neighbor_radius: Radius in Angstroms to search for neighbor atoms
    
    Returns:
        Array of averaged hydrophobicity per vertex
    """
    _, hphob = compute_averaged_vertex_features(
        vertices, names, pdb_file, neighbor_radius,
        use_atom_charges=False, use_atom_hphob=True
    )
    return hphob


def assign_averaged_features_to_new_mesh(
    new_vertices: np.ndarray,
    old_vertices: np.ndarray,
    old_charges: np.ndarray,
    old_hphob: np.ndarray,
    feature_interpolation: bool = True,
) -> tuple:
    """
    Assign averaged features from old mesh to new mesh vertices.
    
    Uses the same interpolation logic as assignChargesToNewMesh from computeCharges.py.
    
    Args:
        new_vertices: New mesh vertices, shape (M, 3)
        old_vertices: Old mesh vertices, shape (N, 3)
        old_charges: Charges on old vertices, shape (N,)
        old_hphob: Hydrophobicity on old vertices, shape (N,)
        feature_interpolation: If True, use weighted average of 4 nearest neighbors
    
    Returns:
        Tuple of (new_charges, new_hphob)
    """
    new_charges = np.zeros(len(new_vertices))
    new_hphob = np.zeros(len(new_vertices))
    
    kdt = KDTree(old_vertices)
    
    if feature_interpolation:
        num_inter = 4
        dists, indices = kdt.query(new_vertices, k=num_inter)
        dists = np.square(dists)  # Square distances for weighting
        
        for vi_new in range(len(new_vertices)):
            vi_old = indices[vi_new]
            dist_old = dists[vi_new]
            
            # If one vertex is exactly on top, use its value directly
            if dist_old[0] == 0.0:
                new_charges[vi_new] = old_charges[vi_old[0]]
                new_hphob[vi_new] = old_hphob[vi_old[0]]
                continue
            
            # Weighted average based on inverse distance
            total_weight = np.sum(1.0 / dist_old)
            for i in range(num_inter):
                weight = (1.0 / dist_old[i]) / total_weight
                new_charges[vi_new] += old_charges[vi_old[i]] * weight
                new_hphob[vi_new] += old_hphob[vi_old[i]] * weight
    else:
        # Nearest neighbor only
        dists, indices = kdt.query(new_vertices)
        new_charges = old_charges[indices]
        new_hphob = old_hphob[indices]
    
    return new_charges, new_hphob


# python surface_maker_test.py --check_software yes
import argparse
import os
import numpy as np
import subprocess
import pymesh
import tempfile, shutil
#import Bio.PDB
from Bio.PDB import PDBParser, PDBIO, Select
from Bio.PDB import NeighborSearch, Selection
from rdkit import Chem
from scipy.spatial import distance, KDTree
from IPython.utils import io
from joblib import Parallel, delayed
import sys
sys.path.append('./utils/masif')
from compute_normal import compute_normal
from computeAPBS import computeAPBS
from computeCharges import computeCharges, assignChargesToNewMesh
from computeHydrophobicity import computeHydrophobicity
from computeMSMS import computeMSMS
from fixmesh import fix_mesh
from save_ply import save_ply


arg_parser = argparse.ArgumentParser(description="Generates a protein surface mesh (PLY format) for a binding pocket.") # Renamed parser
arg_parser.add_argument(
    '--pdb_file', action='store',required=False,type=str,default='./3cl.pdb',
    help='Input protein PDB file path.'
)

arg_parser.add_argument(
    '--lig_file', action='store',required=False,type=str,default='./3cl_ligand.sdf',
    help='Input ligand file path (SDF, MOL, MOL2).'
)

arg_parser.add_argument(
    '--check_software', action='store',required=True,type=str,default='no', # Keep as string if original script expects 'yes'/'no'
    help='Flag to indicate if dependent software locations are manually specified and should be checked (not actively used in this version for checking, but kept for compatibility).'
)

arg_parser.add_argument(
    '--outdir', action='store',required=False,type=str,default='.',
    help='Output directory for generated files (surface.ply and intermediate files).'
)

arg_parser.add_argument(
    '--dist_threshold', action='store',required=False,type=float,default=8.0,
    help='Distance threshold in Angstroms for defining the binding pocket around the ligand.'
)


args = arg_parser.parse_args()

print("--- Initializing Surface Maker ---")
print(f"Arguments: {args}")

# --- Environment Setup ---
apbs_lib_path = '/home/weizg/weizg/wangzz/APBS-3.0.0.Linux/lib'
conda_lib_path = '/home/weizg/weizg/softwares/miniconda3/envs/surf_maker/lib' # Assuming this is your conda env lib

new_ld_path_parts = []
if os.path.exists(apbs_lib_path):
    new_ld_path_parts.append(apbs_lib_path)
if os.path.exists(conda_lib_path):
    new_ld_path_parts.append(conda_lib_path)

current_ld_path = os.environ.get('LD_LIBRARY_PATH')
if current_ld_path:
    new_ld_path_parts.extend(current_ld_path.split(':'))

# Remove duplicates while preserving order (important for library loading precedence)
seen_paths = set()
unique_ordered_paths = []
for path_part in new_ld_path_parts:
    if path_part not in seen_paths:
        unique_ordered_paths.append(path_part)
        seen_paths.add(path_part)

os.environ['LD_LIBRARY_PATH'] = ":".join(filter(None, unique_ordered_paths))
print(f"DEBUG: LD_LIBRARY_PATH set to: {os.environ.get('LD_LIBRARY_PATH', 'Not Set')}")


# --- Software Binary Paths ---
# Ensure these paths are correct and the binaries are executable
msms_bin="/home/weizg/weizg/wangzz/APBS-3.0.0.Linux/bin/msms"
apbs_bin = '/home/weizg/weizg/wangzz/APBS-3.0.0.Linux/bin/apbs'
pdb2pqr_bin="/home/weizg/weizg/wangzz/pdb2pqr-linux-bin64-2.1.1/pdb2pqr"
multivalue_bin="/home/weizg/weizg/wangzz/APBS-3.0.0.Linux/share/apbs/tools/bin/multivalue"


# --- Input Parameters ---
prot_path = args.pdb_file
lig_path = args.lig_file
outdir = args.outdir
dist_threshold = args.dist_threshold

print(f"DEBUG: Protein PDB: {prot_path}")
print(f"DEBUG: Ligand File: {lig_path}")
print(f"DEBUG: Output Directory: {outdir}")

# --- Feature Calculation Flags ---
use_hbond=True
use_hphob=True
use_apbs=True
mesh_res=1.0 # Mesh resolution for fix_mesh
epsilon=1.0e-6 # Epsilon for normal computation
feature_interpolation=True

# --- Setup Working Directory ---
workdir = outdir # Intermediate files will also go into the outdir
os.makedirs(workdir, exist_ok=True)
print(f"DEBUG: Working directory: {workdir}")

protname = os.path.basename(prot_path).replace(".pdb","")
print(f"DEBUG: Protein Name (basename): {protname}")

# --- Load Ligand ---
print("DEBUG: Loading ligand...")
mol = None
if not os.path.exists(lig_path):
    print(f"DEBUG: FATAL ERROR - Ligand file not found: {lig_path}")
    sys.exit(f"Ligand file not found: {lig_path}")

lig_suffix = lig_path.split('.')[-1].lower()
try:
    if lig_suffix == 'mol':
        mol = Chem.MolFromMolFile(lig_path, sanitize=False)
    elif lig_suffix == 'mol2':
        mol = Chem.MolFromMol2File(lig_path, sanitize=False)
    elif lig_suffix == 'sdf':
        suppl = Chem.SDMolSupplier(lig_path, sanitize=False)
        mols = [m for m in suppl if m]
        if mols:
            mol = mols[0] # Use the first molecule from SDF
        else:
            print(f"DEBUG: FATAL ERROR - No molecules found in SDF file: {lig_path}")
            sys.exit(f"No molecules found in SDF file: {lig_path}")
    else:
        print(f"DEBUG: FATAL ERROR - Unsupported ligand file format: {lig_suffix}")
        sys.exit(f"Unsupported ligand file format: {lig_suffix}")

    if mol is None:
        print(f"DEBUG: FATAL ERROR - RDKit could not parse ligand file: {lig_path}")
        sys.exit(f"RDKit could not parse ligand file: {lig_path}")

    atomCoords = mol.GetConformers()[0].GetPositions()
    print(f"DEBUG: Ligand loaded successfully. Number of ligand atom coordinates: {len(atomCoords)}")
except Exception as e:
    print(f"DEBUG: FATAL ERROR - Failed to load or process ligand file {lig_path}: {e}")
    sys.exit(f"Failed to load or process ligand file: {e}")


# --- Load Protein and Define Binding Pocket ---
print("DEBUG: Loading protein and defining binding pocket...")
if not os.path.exists(prot_path):
    print(f"DEBUG: FATAL ERROR - Protein PDB file not found: {prot_path}")
    sys.exit(f"Protein PDB file not found: {prot_path}")

pdb_bio_parser = PDBParser(QUIET=True) # Renamed to avoid conflict
try:
    structures = pdb_bio_parser.get_structure('target_protein', prot_path)
    structure = structures[0] # Assuming a single model/structure in the PDB
except Exception as e:
    print(f"DEBUG: FATAL ERROR - Failed to parse protein PDB file {prot_path}: {e}")
    sys.exit(f"Failed to parse protein PDB file: {e}")

all_protein_atoms  = Selection.unfold_entities(structure, 'A') # 'A' for atoms
if not all_protein_atoms:
    print(f"DEBUG: FATAL ERROR - No atoms found in protein PDB file: {prot_path}")
    sys.exit(f"No atoms found in protein PDB file: {prot_path}")

neighbor_search = NeighborSearch(all_protein_atoms)

close_residues = []
pocket_dist_cutoff = dist_threshold + 5.0 # As per original logic (dist_threshold for interface, +5 for pocket PDB)

for lig_atom_idx, lig_atom_coord in enumerate(atomCoords):
    found_residues = neighbor_search.search(lig_atom_coord, pocket_dist_cutoff, level='R') # 'R' for residues
    close_residues.extend(found_residues)
close_residues = Selection.uniqueify(close_residues)
print(f"DEBUG: Found {len(close_residues)} unique residues near the ligand.")

if not close_residues:
    print(f"DEBUG: WARNING - No residues found within {pocket_dist_cutoff} Å of the ligand. The pocket PDB might be empty or too small.")

class SelectNeighbors(Select):
    def accept_residue(self, residue):
        if residue in close_residues:
            # Original check: if all(a.id in ['N', 'CA', 'C', 'O'] for a in residue.get_unpacked_list() if a.level == 'A') or residue.resname == 'HOH':
            # A slightly more robust check for backbone atoms, ensuring they are actual atoms.
            # And that residue.get_unpacked_list() returns Atom objects.
            backbone_atom_ids = {'N', 'CA', 'C', 'O'}
            present_atoms = {atom.get_name() for atom in residue.get_atoms() if atom.get_name() in backbone_atom_ids}

            if residue.resname == 'HOH' or backbone_atom_ids.issubset(present_atoms):
                # print(f"DEBUG SelectNeighbors: Accepting residue {residue.get_resname()}{residue.id[1]}")
                return True
            else:
                # print(f"DEBUG SelectNeighbors: Rejecting residue {residue.get_resname()}{residue.id[1]} (incomplete backbone or not HOH). Present: {present_atoms}")
                return False
        else:
            return False

pdbio = PDBIO()
pdbio.set_structure(structure)

pocket_pdb_filename = f"{protname}_pocket_{pocket_dist_cutoff}.pdb"
pocket_pdb_path_generated = os.path.join(workdir, pocket_pdb_filename)
print(f"DEBUG: Attempting to save pocket PDB to: {pocket_pdb_path_generated}")
try:
    pdbio.save(pocket_pdb_path_generated, SelectNeighbors())
except Exception as e:
    print(f"DEBUG: FATAL ERROR - Failed to save pocket PDB file {pocket_pdb_path_generated}: {e}")
    sys.exit(f"Failed to save pocket PDB file: {e}")


if not os.path.exists(pocket_pdb_path_generated):
    print(f"DEBUG: FATAL ERROR - Pocket PDB file was NOT created at: {pocket_pdb_path_generated}")
    sys.exit("Pocket PDB file not created.")
elif os.path.getsize(pocket_pdb_path_generated) == 0:
    print(f"DEBUG: FATAL ERROR - Pocket PDB file IS EMPTY: {pocket_pdb_path_generated}. This will cause downstream tools to fail.")
    sys.exit("Pocket PDB file is empty.")
else:
    print(f"DEBUG: Pocket PDB file successfully created: {pocket_pdb_path_generated}, Size: {os.path.getsize(pocket_pdb_path_generated)} bytes")

# Use the path of the file that was actually generated for all downstream processes
pocket_pdb_for_processing = pocket_pdb_path_generated

# --- Determine Closest Atom in Pocket PDB for MSMS one_cavity ---
print("DEBUG: Determining closest atom in pocket PDB for MSMS one_cavity selection...")
atom_idx_for_msms = 0 # Default
try:
    pocket_structures = pdb_bio_parser.get_structure('pocket_target', pocket_pdb_for_processing)
    pocket_structure_model = pocket_structures[0]
    pocket_atoms_list = Selection.unfold_entities(pocket_structure_model, 'A')

    if not pocket_atoms_list:
        print(f"DEBUG: WARNING - No atoms found in the parsed pocket PDB: {pocket_pdb_for_processing}. MSMS might not center correctly.")
    else:
        ligand_centroid = atomCoords.mean(axis=0)
        atom_distances = [distance.euclidean(ligand_centroid, pa.get_coord()) for pa in pocket_atoms_list]
        if not atom_distances:
            print(f"DEBUG: WARNING - Could not calculate distances to pocket atoms. MSMS might not center correctly.")
        else:
            atom_idx_for_msms = np.argmin(atom_distances)
            print(f"DEBUG: Closest atom index in pocket PDB for MSMS one_cavity: {atom_idx_for_msms} (0-indexed)")
except Exception as e:
    print(f"DEBUG: WARNING - Error processing pocket PDB for MSMS atom index: {e}. Using default atom_idx=0.")


# --- Compute MSMS Surface ---
print(f"DEBUG: Calling computeMSMS with PDB: {pocket_pdb_for_processing}, one_cavity index: {atom_idx_for_msms}")
try:
    vertices1, faces1, normals1, names1, areas1 = computeMSMS(
        pocket_pdb_for_processing,
        protonate=True,
        one_cavity=atom_idx_for_msms, # Ensure this index is valid for the atoms in pocket_pdb_for_processing
        msms_bin=msms_bin,
        workdir=workdir
    )
    print(f"DEBUG: computeMSMS completed. Vertices: {len(vertices1)}, Faces: {len(faces1)}")
    if len(vertices1) == 0 or len(faces1) == 0:
        print(f"DEBUG: FATAL ERROR - MSMS did not generate any vertices or faces. Check MSMS logs in {workdir} if any.")
        sys.exit("MSMS output is empty.")
except Exception as e:
    print(f"DEBUG: FATAL ERROR - computeMSMS failed: {e}")
    # Potentially look for MSMS error files in workdir if computeMSMS creates them
    sys.exit(f"computeMSMS failed: {e}")


# --- Filter Surface Vertices near Ligand (Interface) ---
print("DEBUG: Filtering surface vertices for interface...")
kdt = KDTree(atomCoords) # KDTree from original ligand coordinates
distances_to_ligand, _ = kdt.query(vertices1) # Distances from MSMS vertices to ligand atoms
assert(len(distances_to_ligand) == len(vertices1))

# Vertices within the user-defined dist_threshold of the ligand
iface_v_indices = np.where(distances_to_ligand <= dist_threshold)[0]

if len(iface_v_indices) == 0:
    print(f"DEBUG: WARNING - No surface vertices found within {dist_threshold} Å of the ligand. The final mesh might be empty or very small.")

# Keep faces where all vertices are part of the interface
faces_to_keep_indices = [face_idx for face_idx, face_vertices in enumerate(faces1) if all(v_idx in iface_v_indices for v_idx in face_vertices)]
print(f"DEBUG: Keeping {len(faces_to_keep_indices)} faces for the interface mesh.")


# --- Compute Physicochemical Features ---
if use_hbond:
    print("DEBUG: Computing Hbond charges...")
    try:
        # computeCharges expects the original full protein PDB path (or a version of it)
        # and the MSMS vertices + atom names from the pocket surface
        vertex_hbond = computeCharges(prot_path, vertices1, names1) # Using full prot_path as per original logic
        print(f"DEBUG: Hbond charges computed. Length: {len(vertex_hbond)}")
    except Exception as e:
        print(f"DEBUG: ERROR - Failed to compute Hbond charges: {e}")
        vertex_hbond = np.zeros(len(vertices1)) # Fallback
if use_hphob:
    print("DEBUG: Computing hydrophobicity...")
    try:
        vertex_hphobicity = computeHydrophobicity(names1) # names1 are from MSMS output
        print(f"DEBUG: Hydrophobicity computed. Length: {len(vertex_hphobicity)}")
    except Exception as e:
        print(f"DEBUG: ERROR - Failed to compute hydrophobicity: {e}")
        vertex_hphobicity = np.zeros(len(vertices1)) # Fallback


# --- Mesh Processing with PyMesh ---
print("DEBUG: Processing mesh with PyMesh...")
# Initial mesh from MSMS output (vertices1, faces1)
mesh = pymesh.form_mesh(vertices1, faces1)

# Submesh to keep only interface faces
if faces_to_keep_indices: # Only submesh if there are faces to keep
    mesh = pymesh.submesh(mesh, np.array(faces_to_keep_indices, dtype=int), 0) # 0 for face index
    print(f"DEBUG: Submesh created. Vertices: {mesh.num_vertices}, Faces: {mesh.num_faces}")
else:
    print(f"DEBUG: WARNING - No faces to keep for submesh. The resulting mesh will be empty.")
    # Create an empty mesh or handle this case appropriately
    # For now, proceeding with potentially empty mesh which might cause errors later or an empty PLY

if mesh.num_vertices == 0 or mesh.num_faces == 0:
    print(f"DEBUG: FATAL ERROR - Mesh is empty after submeshing (or MSMS output was insufficient). Cannot proceed.")
    sys.exit("Mesh is empty after submeshing.")


print("DEBUG: Fixing mesh (regularization)...")
with io.capture_output() as captured: # Suppress verbose output from fix_mesh
    try:
        regular_mesh = fix_mesh(mesh, mesh_res)
        print(f"DEBUG: Mesh fixed. Vertices: {regular_mesh.num_vertices}, Faces: {regular_mesh.num_faces}")
    except Exception as e:
        print(f"DEBUG: FATAL ERROR - fix_mesh failed: {e}")
        print(f"DEBUG: Captured output from fix_mesh:\n{captured}")
        sys.exit(f"fix_mesh failed: {e}")

if regular_mesh.num_vertices == 0 or regular_mesh.num_faces == 0:
    print(f"DEBUG: FATAL ERROR - Mesh is empty after fix_mesh. Cannot proceed.")
    sys.exit("Mesh is empty after fix_mesh.")

# Remove degenerated triangles
try:
    regular_mesh, _ = pymesh.remove_degenerated_triangles(regular_mesh, 100) # Default iterations
    print(f"DEBUG: Degenerated triangles removed. Vertices: {regular_mesh.num_vertices}, Faces: {regular_mesh.num_faces}")
except Exception as e:
    print(f"DEBUG: WARNING - remove_degenerated_triangles failed: {e}. Proceeding with mesh.")


if regular_mesh.num_vertices == 0:
    print(f"DEBUG: FATAL ERROR - Mesh has no vertices after removing degenerated triangles.")
    sys.exit("Mesh has no vertices after removing degenerated triangles.")


# --- Compute Normals ---
print("DEBUG: Computing vertex normals...")
try:
    vertex_normal = compute_normal(regular_mesh.vertices, regular_mesh.faces, eps=epsilon)
    print(f"DEBUG: Vertex normals computed. Shape: {vertex_normal.shape}")
except Exception as e:
    print(f"DEBUG: FATAL ERROR - Failed to compute normals: {e}")
    sys.exit(f"Failed to compute normals: {e}")


# --- Interpolate Features to Regular Mesh ---
if use_hbond:
    print("DEBUG: Interpolating Hbond charges to regular mesh...")
    try:
        vertex_hbond = assignChargesToNewMesh(regular_mesh.vertices, vertices1, vertex_hbond, feature_interpolation)
        print(f"DEBUG: Hbond charges interpolated. Length: {len(vertex_hbond)}")
    except Exception as e:
        print(f"DEBUG: ERROR - Failed to interpolate Hbond charges: {e}")
        vertex_hbond = np.zeros(regular_mesh.num_vertices) # Fallback
if use_hphob:
    print("DEBUG: Interpolating hydrophobicity to regular mesh...")
    try:
        vertex_hphobicity = assignChargesToNewMesh(regular_mesh.vertices, vertices1, vertex_hphobicity, feature_interpolation)
        print(f"DEBUG: Hydrophobicity interpolated. Length: {len(vertex_hphobicity)}")
    except Exception as e:
        print(f"DEBUG: ERROR - Failed to interpolate hydrophobicity: {e}")
        vertex_hphobicity = np.zeros(regular_mesh.num_vertices) # Fallback

# --- Compute APBS Electrostatics ---
vertex_charges = np.zeros(regular_mesh.num_vertices) # Default if APBS not used or fails
if use_apbs:
    print(f"DEBUG: Calling computeAPBS with PDB: {pocket_pdb_for_processing} for {regular_mesh.num_vertices} vertices.")
    try:
        # computeAPBS needs the vertices of the *regular_mesh* and the *pocket_pdb_for_processing*
        vertex_charges = computeAPBS(
            regular_mesh.vertices,
            pocket_pdb_for_processing,
            apbs_bin,
            pdb2pqr_bin,
            multivalue_bin,
            workdir # Pass workdir for APBS intermediate files
        )
        print(f"DEBUG: APBS charges computed. Length: {len(vertex_charges)}")
        if len(vertex_charges) != regular_mesh.num_vertices:
            print(f"DEBUG: WARNING - APBS returned {len(vertex_charges)} charges, but mesh has {regular_mesh.num_vertices} vertices. Using zeros.")
            vertex_charges = np.zeros(regular_mesh.num_vertices)
    except Exception as e:
        print(f"DEBUG: ERROR - computeAPBS failed: {e}. Electrostatic charges will be zero.")
        # Look for APBS/PDB2PQR error files in workdir (e.g., temp1.pqr.log, apbs_error.log if created by computeAPBS)
        vertex_charges = np.zeros(regular_mesh.num_vertices) # Fallback

# --- Compute Curvatures and Shape Index ---
print("DEBUG: Computing curvatures and shape index...")
try:
    regular_mesh.add_attribute("vertex_mean_curvature")
    H = regular_mesh.get_attribute("vertex_mean_curvature")
    regular_mesh.add_attribute("vertex_gaussian_curvature")
    K = regular_mesh.get_attribute("vertex_gaussian_curvature")

    elem = np.square(H) - K
    elem[elem < 0] = 1e-8 # Set to small positive if negative due to numerical issues
    k1 = H + np.sqrt(elem)
    k2 = H - np.sqrt(elem)

    # Avoid division by zero or very small numbers for shape index
    diff_k1_k2 = k1 - k2
    # Set si to 0 where k1 and k2 are too close, to avoid instability
    # A common default for si when k1=k2 is 0 or undefined, often handled as flat (si=0) or spherical (si=-1 or 1)
    # Here, if diff is near zero, si would explode.
    si = np.zeros_like(H)
    valid_si_mask = np.abs(diff_k1_k2) > 1e-6 # Threshold to consider k1 and k2 different
    si[valid_si_mask] = np.arctan((k1[valid_si_mask] + k2[valid_si_mask]) / diff_k1_k2[valid_si_mask]) * (2.0 / np.pi)
    print(f"DEBUG: Shape index computed. Min: {np.min(si):.2f}, Max: {np.max(si):.2f}")
except Exception as e:
    print(f"DEBUG: ERROR - Failed to compute curvatures or shape index: {e}. Shape index will be zero.")
    si = np.zeros(regular_mesh.num_vertices) # Fallback


# --- Save Final PLY File ---
output_ply_filename = "surface.ply" # Fixed output name as per daemon logic
output_ply_path = os.path.join(outdir, output_ply_filename)
print(f"DEBUG: Saving final surface to PLY file: {output_ply_path}")

try:
    save_ply(
        output_ply_path,
        regular_mesh.vertices,
        regular_mesh.faces,
        normals=vertex_normal,
        charges=vertex_charges,
        normalize_charges=True, # MaSIF typically normalizes charges
        hbond=vertex_hbond if use_hbond else np.zeros(regular_mesh.num_vertices),
        hphob=vertex_hphobicity if use_hphob else np.zeros(regular_mesh.num_vertices),
        si=si
    )
    print(f"--- Surface generation successful. Output saved to {output_ply_path} ---")

except Exception as e:
    print(f"DEBUG: FATAL ERROR - Failed to save PLY file {output_ply_path}: {e}")
    sys.exit(f"Failed to save PLY file: {e}")

os.system('rm -rf temp* io.mc workdir')

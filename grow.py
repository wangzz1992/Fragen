# Example for fragment-based generation:
# python grow.py --outdir example_frag --check_point ./ckpt/surfgen.pt --ply_file ./example/adrb1/adrb_pocket_8.0.ply --frag_path ./example/adrb1/2VT4_frag.sdf

import os
import argparse
from glob import glob
from easydict import EasyDict
from rdkit import Chem
import torch
from copy import deepcopy
import shutil
import numpy as np
from tqdm.auto import tqdm
import pickle
import os.path as osp
from plyfile import PlyData # Assuming plyfile is installed

# --- Import necessary utilities ---
# Adjust paths based on your project structure if these are not directly in utils
from utils.transforms import (
    RefineData, LigandCountNeighbors, FeaturizeLigandAtom, FeaturizeProteinAtom,
    LigandMaskZero, LigandMaskAll, AtomComposer, Geodesic_builder, Compose,
    ContrastiveSample # Assuming ContrastiveSample is needed for num_classes
)
from utils.misc import load_config, get_new_log_dir, seed_all # Assuming these exist
from utils.reconstruct import reconstruct_from_generated_with_edges, MolReconsError # Assuming these exist
from models.fragen import Fragen
from utils.chem import read_pkl, write_pkl # Assuming these exist
# Make sure these necessary functions are available, either defined here or imported
from utils.protein_ligand import parse_rdmol, parse_sdf_file # Assuming parse_sdf_file reads SDF
from utils.data import torchify_dict, ProteinLigandData # Assuming these exist
from utils.sample import get_init, get_next, logp_to_rank_prob, pdb_to_pocket_data
from utils.sample import STATUS_FINISHED, STATUS_RUNNING, STATUS_FAILED # Ensure STATUS_FAILED is defined or handled
from utils.surface import read_ply, parse_face, geodesic_matrix, dst2knnedge # Import functions to read PLY and parse faces
from torch_geometric.utils import to_undirected

from rdkit import RDLogger
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)
import warnings
warnings.filterwarnings("ignore", category=UserWarning) # Suppress common warnings

# --- Helper functions (ensure they are defined or imported correctly) ---

# read_ply is now imported from utils.surface
# read_sdf function (similar to delete.py)
def read_sdf(sdf_file):
    """Reads SDF file and returns a list of RDKit molecules."""
    try:
        # Use parse_sdf_file if it handles reading correctly, otherwise implement supplier logic
        # Assuming parse_sdf_file returns a dictionary, we need the molecule object
        # Let's use SDMolSupplier directly for clarity if parse_sdf_file isn't suitable
        supp = Chem.SDMolSupplier(sdf_file, sanitize=False, removeHs=False)
        mols_list = [mol for mol in supp if mol is not None]
        if not mols_list:
            raise ValueError(f"No valid molecules found in SDF file: {sdf_file}")
        return mols_list
    except Exception as e:
        print(f"Error reading SDF file {sdf_file}: {e}")
        raise

# surfdata_prepare function (similar to delete.py)
def surfdata_prepare(ply_file, frag_kept_sdf):
    """Prepares ProteinLigandData from PLY and SDF fragment."""
    try:
        # Use read_ply from utils.surface which returns a dict
        protein_dict = read_ply(ply_file)
        # Read the fragment molecule
        keep_frag_mol = read_sdf(frag_kept_sdf)[0] # Use the first molecule in the SDF

        # *** Add Sanitization Here ***
        try:
            Chem.SanitizeMol(keep_frag_mol)
            print(f"Successfully sanitized fragment molecule from {frag_kept_sdf}")
        except Exception as sanitize_error:
            print(f"Warning: Failed to sanitize fragment molecule from {frag_kept_sdf}: {sanitize_error}")
            # Consider raising the error if sanitization is critical for downstream steps
            # raise sanitize_error

        # Now call parse_rdmol with the (potentially) sanitized molecule
        ligand_dict = parse_rdmol(keep_frag_mol) # Ensure parse_rdmol is available

        data = ProteinLigandData.from_protein_ligand_dicts(
            protein_dict=torchify_dict(protein_dict), # Ensure torchify_dict is available
            ligand_dict=torchify_dict(ligand_dict)
        )
        # Add face data required by Geodesic_builder
        data.face = parse_face(ply_file) # Ensure parse_face is available
        if data.face is None:
             print(f"Warning: Could not parse faces from {ply_file}. Geodesic_builder might fail.")

        # --- Add Geodesic KNN Calculation ---
        if hasattr(data, 'protein_pos') and data.protein_pos.size(0) > 0 and hasattr(data, 'face') and data.face is not None:
            try:
                edge_index = torch.cat([data.face[:2], data.face[1:], data.face[::2]], dim=1)
                dlny_edge_index = to_undirected(edge_index, num_nodes=data.protein_pos.shape[0])
                gds_mat = geodesic_matrix(data.protein_pos, dlny_edge_index)
                # Ensure dst2knnedge returns tensors of the correct type and shape
                # Use a default knn value (e.g., 16) or get it from config if available
                gds_knn_edge_index, gds_knn_edge_dist = dst2knnedge(gds_mat, num_knn=16)
                data.gds_knn_edge_index = gds_knn_edge_index.long() # Ensure correct dtype
                data.gds_dist = gds_knn_edge_dist.float() # Ensure correct dtype
                print("Successfully calculated and added geodesic KNN graph data.")
            except Exception as gds_error:
                print(f"Error calculating geodesic KNN graph: {gds_error}")
                # Assign empty tensors or handle this case as appropriate
                data.gds_knn_edge_index = torch.empty((2, 0), dtype=torch.long)
                data.gds_dist = torch.empty((0,), dtype=torch.float)
        else:
            print("Warning: Cannot calculate geodesic KNN graph due to missing protein_pos or face data.")
            # Assign empty tensors or handle this case as appropriate
            data.gds_knn_edge_index = torch.empty((2, 0), dtype=torch.long)
            data.gds_dist = torch.empty((0,), dtype=torch.float)
        # --- End Geodesic KNN Calculation ---

        data.protein_filename = osp.basename(ply_file)
        data.ligand_filename = osp.basename(frag_kept_sdf) # Store fragment filename
        return data
    except Exception as e:
        # Print the original error for more details
        print(f"Error preparing data from {ply_file} and {frag_kept_sdf}: {e}")
        # Re-raise the exception to indicate failure
        raise

# transform_data function (as in original gen.py, slightly modified for safety)
def transform_data(data, transform):
    # Check if protein data exists before applying transforms that might need it
    protein_exists = hasattr(data, 'protein_pos') and data.protein_pos is not None and data.protein_pos.size(0) > 0
    compose_exists = hasattr(data, 'compose_pos') and data.compose_pos is not None

    if not protein_exists and not compose_exists:
         print("Warning: Neither protein_pos nor compose_pos found in data. Transforms might fail.")
    elif not protein_exists and compose_exists:
         # This might be okay if composer handles it, but issue a warning
         # print("Warning: protein_pos not found, relying on compose_pos. Ensure transforms handle this.")
         pass # Composer should handle this case

    if transform is not None:
        try:
            data = transform(data)
        except Exception as e:
            print(f"Error during data transformation: {e}")
            # Decide whether to raise the error or return None/original data
            raise # Re-raise the error to stop execution if transform fails critically
    return data


# --- Argument Parser ---
parser = argparse.ArgumentParser()
parser.add_argument(
    '--config', type=str, default='./configs/sample.yml', help='Path to sampling config file.'
)
parser.add_argument(
    '--outdir', type=str, default='outputs', help='Directory to save generated molecules.'
)
parser.add_argument(
    '--device', type=str, default='cuda', help='Device to use (cuda or cpu).'
)
parser.add_argument(
    '--check_point', type=str, default='./ckpt/surfgen.pt', help='Path to the pre-trained model checkpoint.'
)
# Made ply_file required as per the error message in the prompt
parser.add_argument(
    '--ply_file', type=str, required=True, help='Path to the protein surface PLY file.'
)
parser.add_argument(
    '--frag_path', type=str, default=None, help='Optional path to the fragment SDF file to start generation from.'
)
parser.add_argument(
    '--suboutdir', action='store',required=False,type=str,default=None,
    help='Optional subdirectory name within outdir. Defaults to PLY file name or fragment name.'
)
parser.add_argument(
    '--sdf_filename', type=str,default='gen_mols.sdf',
    help='Name for the output SDF file containing all generated molecules.'
)
parser.add_argument(
    '--save_split_sdf', action='store_true', help='Save each generated molecule to a separate SDF file.'
)

args = parser.parse_args()
config = load_config(args.config)
seed_all(config.sample.seed if hasattr(config, 'sample') and hasattr(config.sample, 'seed') else 42) # Seed for reproducibility

# --- Data Preparation ---
print("Preparing data...")
if args.frag_path:
    print(f"Loading fragment from: {args.frag_path}")
    try:
        data = surfdata_prepare(args.ply_file, args.frag_path)
        task_name = osp.basename(args.frag_path).replace('.sdf', '')
        is_fragment_based = True
    except Exception as e:
        print(f"Failed to prepare data with fragment: {e}")
        exit(1)
else:
    print(f"Loading surface for de novo generation from: {args.ply_file}")
    try:
        # pdb_to_pocket_data should also add face information if needed
        data = pdb_to_pocket_data(args.ply_file) # Assumes this reads PLY and prepares data with faces
        if not hasattr(data, 'face') or data.face is None:
             print(f"Warning: pdb_to_pocket_data did not add face info from {args.ply_file}. Geodesic_builder might fail.")
        task_name = osp.basename(args.ply_file).replace('.ply', '')
        is_fragment_based = False
    except Exception as e:
        print(f"Failed to prepare data for de novo generation: {e}")
        exit(1)

# --- Transforms ---
print("Setting up transforms...")
# Base transforms (common to both modes)
# ContrastiveSample might only be needed for training, check if num_elements is needed elsewhere
# If num_elements is fixed (e.g., 7), hardcode it. Otherwise, instantiate ContrastiveSample.
# Assuming num_elements is needed for the model definition:
contrastive_sampler = ContrastiveSample() # Use default values or load from config if needed
ligand_featurizer = FeaturizeLigandAtom()
protein_featurizer = FeaturizeProteinAtom()
# Geodesic_builder needs face information, ensured by surfdata_prepare and pdb_to_pocket_data
base_transform = Compose([
    RefineData(),
    LigandCountNeighbors(),
    Geodesic_builder(), # Ensure data.face exists
    ligand_featurizer,
    protein_featurizer
])

# Load checkpoint and determine composer KNN from training config
try:
    ckpt = torch.load(args.check_point, map_location=args.device)
    config_train = ckpt['config'] # Load training config from checkpoint
    # Ensure necessary keys exist in the loaded config
    if not hasattr(config_train, 'model') or not hasattr(config_train.model, 'encoder') or not hasattr(config_train.model.encoder, 'knn'):
         raise KeyError("Training config in checkpoint missing model.encoder.knn")
    composer_knn = config_train.model.encoder.knn # Get KNN value used during training
except FileNotFoundError:
    print(f"Error: Checkpoint file not found at {args.check_point}")
    exit(1)
except KeyError as e:
    print(f"Error: Missing key in checkpoint config: {e}")
    exit(1)
except Exception as e:
    print(f"Error loading checkpoint or config: {e}")
    exit(1)

# Conditional Masking Strategy & Composer
if is_fragment_based:
    print("Using fragment-based masking (LigandMaskZero).")
    mask = LigandMaskZero() # Keep fragment, mask nothing initially to add
    # Composer uses feature dimensions determined by featurizers
    composer = AtomComposer(protein_featurizer.feature_dim, ligand_featurizer.feature_dim, composer_knn)
else:
    print("Using de novo masking (LigandMaskAll).")
    mask = LigandMaskAll() # Mask all ligand atoms initially
    # Composer uses feature dimensions determined by featurizers
    composer = AtomComposer(protein_featurizer.feature_dim, ligand_featurizer.feature_dim, composer_knn)

masking_transform = Compose([
    mask,
    composer
])

# Apply transforms
try:
    data = transform_data(data, base_transform)
    # Apply masking and composer after base transforms
    data = transform_data(data, masking_transform)
    data = data.to(args.device)
    print("Data prepared and transformed.")
except Exception as e:
    print(f"Error applying transforms: {e}")
    exit(1)

# --- Model Loading ---
print("Loading model...")
try:
    model = Fragen(
        config_train.model, # Use model config from training checkpoint
        num_classes=contrastive_sampler.num_elements, # Ensure this matches training
        num_bond_types=3, # Ensure this matches training (usually 3: single, double, triple)
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim,
    ).to(args.device)
    model.load_state_dict(ckpt['model'],False)
    model.eval() # Set model to evaluation mode
    print(f'Model loaded. Num of parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.4f}M')
except Exception as e:
    print(f"Error initializing or loading model state dict: {e}")
    exit(1)

# --- Generation ---
# np.seterr(invalid='ignore') # Ignore invalid value warnings (e.g., log(0))
pool = EasyDict({
    'queue': [],
    'failed': [],
    'finished': [],
    'duplicate': [],
    'smiles': set(),
})

print('Starting generation...')
# Conditional Initialization
if is_fragment_based:
    print("Initializing generation from fragment.")
    # Start with the data containing the fragment
    # The initial 'data' object already contains the fragment and protein info
    # We need to ensure it has the necessary attributes expected by get_next
    # get_next likely expects a list, so wrap it
    data.status = STATUS_RUNNING # Set initial status
    # Add average_logp if get_next expects it (initialize reasonably)
    data.average_logp = torch.tensor([0.0] * 5) # Placeholder, adjust size if needed
    pool.queue = [data] # Start the queue with the fragment data
else:
    print("Initializing de novo generation.")
    # Sample initial atoms using get_init
    try:
        init_data_list = get_init(
            data.to(args.device),
            model=model,
            transform=composer, # Pass composer for potential internal use by get_init/get_next
            threshold=config.sample.threshold
        )
        pool.queue = init_data_list
    except Exception as e:
        print(f"Error during get_init: {e}")
        exit(1)

print(f'Initial queue size: {len(pool.queue)}')

global_step = 0
max_samples = config.sample.num_samples
max_steps = config.sample.max_steps

with torch.no_grad(): # Disable gradient calculations during generation
    pbar_samples = tqdm(total=max_samples, desc="Generated Samples")
    while len(pool.finished) < max_samples:
        global_step += 1
        if global_step > max_steps:
            print(f"\nReached max steps ({max_steps}). Stopping generation.")
            break
        if not pool.queue:
            print("\nQueue is empty, stopping generation.")
            break

        print(f"\nStep: {global_step}, Finished: {len(pool.finished)}/{max_samples}, Queue: {len(pool.queue)}")

        queue_tmp = []
        # Wrap pool.queue with tqdm for step progress
        for current_data in tqdm(pool.queue, desc=f"Step {global_step} Processing", leave=False):
            # Skip if already finished or failed (shouldn't happen with current logic, but good practice)
            if not hasattr(current_data, 'status') or current_data.status != STATUS_RUNNING:
                continue

            try:
                data_next_list = get_next(
                    current_data.to(args.device),
                    model=model,
                    transform=composer, # Pass composer for potential internal use
                    threshold=config.sample.threshold,
                    # Add other necessary args for get_next if any 
                )
            except Exception as e:
                print(f"\nError during get_next for a candidate: {e}")
                # Mark current data as failed or simply skip adding its children
                current_data.status = STATUS_FAILED # Assuming STATUS_FAILED exists
                pool.failed.append(current_data)
                continue # Skip processing results for this failed candidate

            for data_next in data_next_list:
                if data_next.status == STATUS_FINISHED:
                    try:
                        # Ensure reconstruction function is available
                        rdmol = reconstruct_from_generated_with_edges(data_next)
                        if rdmol is None:
                            # print('\nReconstruction failed: rdmol is None')
                            data_next.status = STATUS_FAILED
                            pool.failed.append(data_next)
                            continue

                        # Basic sanitization and SMILES generation
                        try:
                            Chem.SanitizeMol(rdmol) # Attempt sanitization
                        except Exception as sanitize_e:
                            # print(f'\nSanitization failed: {sanitize_e}')
                            # Decide if this counts as failure
                            data_next.status = STATUS_FAILED
                            pool.failed.append(data_next)
                            continue

                        smiles = Chem.MolToSmiles(Chem.RemoveHs(rdmol)) # Generate SMILES without Hs

                        if not smiles:
                             # print('\nSMILES generation failed.')
                             data_next.status = STATUS_FAILED
                             pool.failed.append(data_next)
                             continue

                        data_next.rdmol = rdmol
                        data_next.smiles = smiles

                        if smiles in pool.smiles:
                            # print(f'Duplicate molecule: {smiles}')
                            pool.duplicate.append(data_next)
                        elif '.' in smiles: # Check for multiple disconnected fragments
                            # print(f'Failed molecule (disconnected): {smiles}')
                            data_next.status = STATUS_FAILED
                            pool.failed.append(data_next)
                        else:   # Pass checks
                            # print(f'Success: {smiles}')
                            pool.finished.append(data_next)
                            pool.smiles.add(smiles)
                            pbar_samples.update(1) # Update progress bar for successful samples
                            # Optional: print progress periodically
                            # if len(pool.finished) % 10 == 0:
                            #     print(f"\nGenerated {len(pool.finished)} molecules...")

                    except MolReconsError as e:
                        # print(f'\nReconstruction error: {e}')
                        data_next.status = STATUS_FAILED
                        pool.failed.append(data_next)
                    except Exception as e:
                        print(f'\nError processing finished molecule: {e}')
                        data_next.status = STATUS_FAILED
                        pool.failed.append(data_next)

                elif data_next.status == STATUS_RUNNING:
                    queue_tmp.append(data_next)
                # Handle other statuses if necessary (e.g., failed status from get_next)
                elif hasattr(data_next, 'status') and data_next.status == STATUS_FAILED:
                     pool.failed.append(data_next)
                # else: # Unknown status or status missing
                #      print(f"\nWarning: data_next has unexpected status: {getattr(data_next, 'status', 'None')}")
                #      pool.failed.append(data_next)


        # Rank and select next candidates from the temporary queue
        if not queue_tmp:
            # print("\nTemporary queue is empty after processing step.")
            # Decide whether to continue (maybe more steps needed) or break
            if not pool.queue: # If main queue is also empty, definitely break
                 print("\nBoth main and temporary queues are empty. Stopping.")
                 break
            else:
                 continue # Main queue might still have items if beam search had >1 item

        try:
            # Ensure average_logp exists and has enough elements for ranking logic
            # The original code uses logp[2:], adapt if needed based on what get_next stores
            logps_for_ranking = []
            for p in queue_tmp:
                if hasattr(p, 'average_logp') and len(p.average_logp) > 2:
                    logp_val = p.average_logp[2:]
                    # Check if it's a tensor before calling .cpu()
                    if isinstance(logp_val, torch.Tensor):
                        logps_for_ranking.append(logp_val.cpu().numpy())
                    elif isinstance(logp_val, np.ndarray):
                        logps_for_ranking.append(logp_val) # Already a numpy array
                    else:
                        # Handle unexpected types if necessary, or skip
                        print(f"\nWarning: Unexpected type for average_logp: {type(logp_val)}")

            if not logps_for_ranking or len(logps_for_ranking) != len(queue_tmp):
                 # print("\nWarning: Not all candidates have valid logp for ranking. Selecting randomly.")
                 prob = np.ones(len(queue_tmp)) / len(queue_tmp)
            else:
                 # Ensure logp_to_rank_prob handles the list of tensors/arrays correctly
                 prob = logp_to_rank_prob(logps_for_ranking)
                 # Ensure probabilities sum to 1
                 prob = prob / np.sum(prob)


            n_tmp = len(queue_tmp)
            beam_size = min(config.sample.beam_size, n_tmp)
            if beam_size <= 0:
                 print("\nWarning: Beam size is zero or negative. Stopping.")
                 break

            # Ensure probabilities are valid for np.random.choice
            if np.any(np.isnan(prob)) or np.any(prob < 0):
                print("\nWarning: Invalid probabilities encountered during ranking. Selecting randomly.")
                prob = np.ones(n_tmp) / n_tmp

            next_indices = np.random.choice(np.arange(n_tmp), p=prob, size=beam_size, replace=False)
            pool.queue = [queue_tmp[idx] for idx in next_indices]
        except Exception as e:
            print(f"\nError during ranking/selection: {e}. Stopping generation.")
            break # Stop if ranking fails

    pbar_samples.close() # Close the progress bar for samples
print(f"\nGeneration finished. Total generated: {len(pool.finished)}, Failed: {len(pool.failed)}, Duplicates: {len(pool.duplicate)}")

# --- Save Results ---
if pool.finished:
    # Determine output directory name based on input
    sub_dir_name = args.suboutdir if args.suboutdir else task_name
    task_dir = osp.join(args.outdir, sub_dir_name)
    os.makedirs(task_dir, exist_ok=True)
    print(f"Saving results to: {task_dir}")

    # Save all molecules to a single SDF file
    sdf_path = osp.join(task_dir, args.sdf_filename)
    try:
        writer = Chem.SDWriter(sdf_path)
        mol_count = 0
        for data_mol in pool['finished']:
            if hasattr(data_mol, 'rdmol') and data_mol.rdmol is not None:
                # Add properties if needed, e.g., SMILES
                mol_name = data_mol.smiles if hasattr(data_mol, 'smiles') else f"GeneratedMol_{mol_count}"
                data_mol.rdmol.SetProp("_Name", mol_name)
                # Add log probabilities if available and desired
                if hasattr(data_mol, 'average_logp'):
                     try:
                          # Convert tensor to string representation
                          logp_str = str(data_mol.average_logp.cpu().numpy().round(4))
                          data_mol.rdmol.SetProp("LogP_Avg", logp_str)
                     except Exception: pass # Ignore errors setting property
                writer.write(data_mol.rdmol)
                mol_count += 1
        writer.close()
        print(f"Saved {mol_count} molecules to {sdf_path}")
    except Exception as e:
        print(f"Error writing combined SDF file {sdf_path}: {e}")


    # Optionally save molecules to individual SDF files
    if args.save_split_sdf:
        sdf_split_dir = osp.join(task_dir, 'SDF_split')
        os.makedirs(sdf_split_dir, exist_ok=True)
        mol_count_split = 0
        for i, data_mol in enumerate(pool['finished']):
             if hasattr(data_mol, 'rdmol') and data_mol.rdmol is not None:
                split_filename = osp.join(sdf_split_dir, f'mol_{i}.sdf')
                try:
                    writer_split = Chem.SDWriter(split_filename)
                    mol_name = data_mol.smiles if hasattr(data_mol, 'smiles') else f"GeneratedMol_{i}"
                    data_mol.rdmol.SetProp("_Name", mol_name)
                    # Add log probabilities if available
                    if hasattr(data_mol, 'average_logp'):
                         try:
                              logp_str = str(data_mol.average_logp.cpu().numpy().round(4))
                              data_mol.rdmol.SetProp("LogP_Avg", logp_str)
                         except Exception: pass
                    writer_split.write(data_mol.rdmol)
                    writer_split.close()
                    mol_count_split += 1
                except Exception as e:
                    print(f"Error writing individual SDF file {split_filename}: {e}")
        if mol_count_split > 0:
            print(f"Saved {mol_count_split} individual molecules to {sdf_split_dir}")

    # Copy input PLY file
    try:
        shutil.copy(args.ply_file, task_dir)
    except Exception as e:
        print(f"Could not copy PLY file: {e}")
    # Copy input fragment SDF file if it was used
    if args.frag_path:
        try:
            shutil.copy(args.frag_path, task_dir)
        except Exception as e:
            print(f"Could not copy fragment SDF file: {e}")
else:
    print("No molecules were successfully generated.")

print('Generation process complete.')

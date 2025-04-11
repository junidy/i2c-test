from system import System
import glob
import os
import json
import torch
import torchaudio
import pandas as pd
import numpy as np
import warnings
from postprocess import gainAdjustment
import argparse
import traceback

warnings.filterwarnings("ignore")

# --- Standardized Paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_TRACKS_SUBDIR = "input_tracks"
OUTPUT_FILENAME = "output.json"
DEFAULT_MODEL_FILENAME = "dmc_2.ckpt"

def main(): # Wrap main logic in a function
    # --- Argument Parser ---
    parser = argparse.ArgumentParser(description="Mixing script for music tracks.")
    parser.add_argument("--ckpt_path", type=str, default=os.path.join(SCRIPT_DIR, DEFAULT_MODEL_FILENAME), help="Path to the model checkpoint.")
    # Remove default for track_dir to make it explicit where state_manager placed files
    parser.add_argument("--track_dir", type=str, required=True, help="Path to the prepared track directory (containing only unmuted tracks).")
    parser.add_argument("--output_dir", type=str, default=SCRIPT_DIR, help="Directory to save the output file.")

    args = parser.parse_args()

    ckpt_path = args.ckpt_path
    track_dir = args.track_dir # Use the provided directory directly
    output_dir = args.output_dir

    output_json_path = os.path.join(output_dir, OUTPUT_FILENAME)
    print(f"Inference script running...")
    print(f"Using checkpoint: {ckpt_path}")
    print(f"Reading tracks from: {track_dir}") # Should contain only unmuted tracks
    print(f"Writing output to: {output_json_path}")

    ext = "wav"
    max_samples = 262144


    # --- Load Model ---
    print(f"DEBUG: Explicitly checking existence of: {ckpt_path}") # <-- ADD DEBUG
    if not os.path.exists(ckpt_path):
        print(f"ERROR: os.path.exists() check FAILED for {ckpt_path}") # <-- ADD DEBUG
        # Optionally add more checks like os.access(ckpt_path, os.R_OK) for read permission
        exit(1) # Exit if explicit check fails
    else:
        print(f"DEBUG: os.path.exists() check SUCCEEDED for {ckpt_path}") # <-- ADD DEBUG

    try:
        system = System.load_from_checkpoint(ckpt_path)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        system.to(device)
        system.eval()
        print(f"Model loaded successfully onto {device}. Expects MONO input.") # Adjusted assumption
    except FileNotFoundError:
        print(f"ERROR: Checkpoint file not found at {ckpt_path}")
        exit(1)
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        exit(1)

    # --- Load Input Tracks (Now dynamic based on files present) ---
    # Use glob to find all .wav files in the specified track_dir
    present_track_filepaths = sorted(glob.glob(os.path.join(track_dir, f"*.{ext}")))

    if not present_track_filepaths:
        print("ERROR: No input .wav files found in the specified track directory.")
        # Output an empty JSON result? Or just exit? Let's output empty.
        result = {"metadata": {"error": "No input tracks found"}, "tracks": {}}
        try:
            with open(output_json_path, 'w') as f: json.dump(result, f)
        except Exception as e: print(f"ERROR: Failed to write empty output JSON: {e}")
        exit(1) # Exit after writing empty result

    print(f"Found {len(present_track_filepaths)} tracks to process:")
    tracks_to_process = []
    # Keep track of the *original* channel number based on the filename
    original_channel_indices = []

    for track_filepath in present_track_filepaths:
        basename = os.path.basename(track_filepath)
        filename_no_ext = os.path.splitext(basename)[0]
        try:
            # Extract original channel number from filename ("1.wav", "2.wav", etc.)
            original_channel_num = int(filename_no_ext)
            if not (1 <= original_channel_num <= 8):
                print(f"Warning: Skipping file with unexpected name format: {basename}")
                continue

            print(f"  - Loading Track (Original Channel {original_channel_num}) from {basename}")
            x, sr = torchaudio.load(track_filepath)

            # --- Sample Rate Check ---
            if sr != system.hparams.sample_rate:
                print(f"    Warning: Track {original_channel_num} SR ({sr}Hz) differs from model SR ({system.hparams.sample_rate}Hz). Resampling.")
                resampler = torchaudio.transforms.Resample(sr, system.hparams.sample_rate)
                x = resampler(x)

            # --- Channel Count Check (Assuming Model Expects MONO) ---
            if x.shape[0] != 1:
                print(f"    ERROR: Track {original_channel_num} has {x.shape[0]} channels, model expects MONO.")
                print(f"    Skipping Track {original_channel_num} due to channel mismatch.")
                continue # Skip this track

            # --- Normalization & Processing ---
            x = x[:, : max_samples] # Trim/pad samples
            peak = x.abs().max()
            if peak > 1e-8: x /= peak.clamp(min=1e-8) # Peak normalize
            else: print(f"    Track {original_channel_num} appears silent, skipping normalization.")
            x *= 10 ** (-12 / 20.0) # Set peak to -12 dBFS

            tracks_to_process.append(x)
            original_channel_indices.append(str(original_channel_num)) # Store original index as string key

        except ValueError:
            print(f"Warning: Skipping file with non-numeric name: {basename}")
            continue
        except Exception as e:
            print(f"ERROR loading track {original_channel_num} ({track_filepath}): {e}")
            print(f"Skipping Track {original_channel_num}.")
            continue # Skip this track

    # Proceed only if we have valid tracks left
    if not tracks_to_process:
        print("ERROR: No tracks were successfully loaded or valid for processing.")
        result = {"metadata": {"error": "No valid tracks processed"}, "tracks": {}}
        try:
            with open(output_json_path, 'w') as f: json.dump(result, f)
        except Exception as e: print(f"ERROR: Failed to write error output JSON: {e}")
        exit(1)

 # --- Prepare Tensor for Model ---
    try:
        tracks_tensor = torch.stack(tracks_to_process, dim=0)
        tracks_tensor = tracks_tensor.unsqueeze(0) # Add batch dimension
        print(f"Tracks tensor prepared with shape: {tracks_tensor.shape} for {len(original_channel_indices)} tracks.")
    except Exception as e:
        print(f"ERROR: Failed to stack track tensors: {e}")
        traceback.print_exc() # Print traceback
        exit(1)

    tracks_tensor = tracks_tensor.to(device)

    # --- Run Inference ---
    print("Running model inference...")
    try:
        mixer = system.model.mixer
        num_expected_params = mixer.num_params

        with torch.no_grad():
             # --- Keep the direct capture for info, but error handling is broader now ---
            print("DEBUG: Calling system(tracks_tensor)...")
            return_value = system(tracks_tensor)
            print(f"DEBUG: Type of system() return value: {type(return_value)}")
            if isinstance(return_value, (list, tuple)):
                print(f"DEBUG: Length of system() return value: {len(return_value)}")
            else:
                print(f"DEBUG: system() return value is not a tuple/list.")
            # --- This unpacking might still fail, but the outer except block will catch it ---
            _, params_tensor = return_value

    except Exception as e: # Catch ANY exception during inference call/unpacking
        print(f"ERROR: Model inference or unpacking failed: {e}")
        traceback.print_exc() # <-- THIS WILL SHOW THE EXACT LINE
        exit(1)

    print("Inference complete. Processing parameters...")

    # --- Process Parameters ---
    try:
        params_np = params_tensor.squeeze().cpu().numpy() # Remove batch dim

        # Ensure shape matches the number of tracks PROCESSSED and expected params
        if params_np.ndim != 2 or params_np.shape[0] != len(original_channel_indices) or params_np.shape[1] != num_expected_params:
            print(f"ERROR: Unexpected parameter tensor shape after inference: {params_np.shape}.")
            print(f"Expected ({len(original_channel_indices)}, {num_expected_params})")
            exit(1)

        # Create a DataFrame using ORIGINAL CHANNEL INDICES as index
        df = pd.DataFrame(index=original_channel_indices, columns=mixer.param_names)

        # Fill the DataFrame - Iterate using the index of the PROCESSED tracks
        for i, track_idx_str in enumerate(original_channel_indices):
            track_params_np = params_np[i, :] # Get params corresponding to the i-th processed track

            # --- Denormalization logic (same as before) ---
            min_gain_dB = getattr(mixer, 'min_gain_dB', -60.0)
            max_gain_dB = getattr(mixer, 'max_gain_dB', 6.0)
            normalized_gain = float(track_params_np[0])
            gain_dB = min_gain_dB + normalized_gain * (max_gain_dB - min_gain_dB)
            df.loc[track_idx_str, "Gain dB"] = gain_dB

            eq_params_tensor = torch.tensor(track_params_np[1 : 1 + mixer.eq.num_params]).unsqueeze(0)
            eq_param_dict = mixer.eq.denormalize_param_dict(mixer.eq.extract_param_dict(eq_params_tensor))
            for param_name in mixer.eq.param_ranges.keys():
                df.loc[track_idx_str, param_name] = float(eq_param_dict[param_name][0])

            comp_params_tensor = torch.tensor(track_params_np[1 + mixer.eq.num_params : -1]).unsqueeze(0)
            comp_param_dict = mixer.comp.denormalize_param_dict(mixer.comp.extract_param_dict(comp_params_tensor))
            for param_name in mixer.comp.param_ranges.keys():
                df.loc[track_idx_str, param_name] = float(comp_param_dict[param_name][0])

            df.loc[track_idx_str, "Pan"] = float(track_params_np[-1])
            # --- End Denormalization ---

        print("Applying post-processing gain adjustment...")
        new_df = gainAdjustment(df) # Apply post-processing

        # Convert to dictionary using original track indices ("1", "2", etc.) as keys
        mixing_params_dict = new_df.to_dict(orient='index')

        # Prepare final JSON result
        result = {
            "metadata": {
                "model": getattr(system.hparams, 'automix_model', 'unknown'),
                "sample_rate": system.hparams.sample_rate,
                "checkpoint": os.path.basename(ckpt_path),
                "num_tracks_processed": len(original_channel_indices), # Add info on how many were actually processed
                "processed_channel_indices": original_channel_indices # List the channels included
            },
            "tracks": mixing_params_dict # Contains results ONLY for processed tracks, keyed by original channel num
        }

    except Exception as e:
        print(f"ERROR: Failed processing parameters: {e}")
        traceback.print_exc() # Print traceback
        exit(1)

    # --- Save Output JSON ---
    print(f"Saving results to {output_json_path}...")
    try:
        with open(output_json_path, 'w') as f: json.dump(result, f, indent=2)
        print("Results saved successfully.")
    except Exception as e:
        print(f"ERROR: Failed to save output JSON: {e}")
        traceback.print_exc() # Print traceback
        exit(1)

    print("Inference script finished.")
    # Implicit exit(0) on success

# --- Run the main function with top-level error catching ---
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"FATAL ERROR in script execution: {e}")
        traceback.print_exc() # Catch any unexpected error during setup/argparse etc.
        exit(1)
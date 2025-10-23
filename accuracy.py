import os
import argparse
import gc
from pathlib import Path
import json
from datetime import datetime # <-- Import datetime

# --- 1. SET UP ENVIRONMENT FOR LOCAL DATASETS ---
parser = argparse.ArgumentParser(description="Compare original and quantized model accuracy using local or Hub datasets.")
parser.add_argument(
    "--checkpoint_path",
    type=str,
    required=True,
    help="Path to the directory of the ORIGINAL, unquantized model."
)
parser.add_argument(
    "--local_datasets_path",
    type=str,
    default=None,
    help="Optional. Path to a local folder containing Hugging Face datasets. If provided, downloads will be skipped."
)
args = parser.parse_args()

if args.local_datasets_path:
    if os.path.isdir(args.local_datasets_path):
        print("="*60)
        print(f"Found local datasets path: {args.local_datasets_path}")
        print("Setting HF_DATASETS_CACHE to this directory.")
        os.environ["HF_DATASETS_CACHE"] = os.path.abspath(args.local_datasets_path)
        print("="*60)
    else:
        print(f"Error: Provided local datasets path does not exist: {args.local_datasets_path}")
        exit(1)

# --- Now, import the rest of the libraries ---
import torch
from tqdm import tqdm
import traceback # Import traceback for detailed errors

# --- LM-EVALUATION-HARNESS IMPORTS (Corrected) ---
try:
    import lm_eval
    from lm_eval.evaluator import simple_evaluate
    from lm_eval.models.huggingface import HFLM
    from lm_eval.tasks import get_task_dict, TaskManager
except ImportError as e:
    print("="*50)
    print("!!! A DETAILED IMPORT ERROR OCCURRED !!!")
    print("This is likely a dependency problem (like numpy/pandas).")
    print("\nHere is the full error message:")
    print(e)
    print("\nFull traceback:")
    traceback.print_exc()
    print("="*50)
    exit(1)

# --- IMPORTS FROM YOUR PROJECT FILE (static_ptq.py) ---
try:
    # This assumes 'accuracy.py' is in the same directory as 'static_ptq.py'
    from static_ptq import load_model_and_tokenizer, ConfigNamespace, MOR_MODEL_CLS
    print("Successfully imported functions from static_ptq.py")
except ImportError:
    print("ERROR: Could not import from 'static_ptq.py'. Ensure it's in the same directory.")
    exit(1)


def load_original_model_from_file(path, root_dir, device):
    """Loads the original unquantized model and moves it to the specified device."""
    print(f"--- Loading original model from: {path} to device: {device.upper()} ---")
    model, tokenizer = load_model_and_tokenizer(path, device, root_dir)
    return model, tokenizer

def load_quantized_model_from_pt(quantized_pt_path):
    """
    Loads the complete quantized model from its .pt file (always on CPU).
    Sets weights_only=False to allow loading custom classes.
    """
    print(f"--- Loading quantized .pt model from: {quantized_pt_path} to device: CPU ---")
    if not os.path.exists(quantized_pt_path):
        raise FileNotFoundError(f"Quantized model file not found at: {quantized_pt_path}")

    # Add weights_only=False for PyTorch 2.6+ compatibility
    model_quantized = torch.load(quantized_pt_path, map_location="cpu", weights_only=False)

    model_quantized.eval()
    print("Successfully loaded full quantized model.")
    return model_quantized


def evaluate_on_harness(model, tokenizer, tasks_list, device):
    """Wraps an in-memory model and runs evaluation on the specified device."""
    print(f"\n--- Starting Harness Evaluation on device: {device.upper()} for tasks: {tasks_list} ---")

    lm_eval_model = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        device=device
    )

    # Create the TaskManager, pointing it to local datasets if specified
    local_path_str = os.environ.get("HF_DATASETS_CACHE")
    task_manager = TaskManager(include_path=local_path_str)

    # Pass the list of task names and the manager directly to simple_evaluate
    results = simple_evaluate(
        model=lm_eval_model,
        tasks=tasks_list,
        task_manager=task_manager,
        num_fewshot=0,
        limit=5, # Using limit=5 for quick testing
        log_samples=True, # <-- Make sure samples are generated
        bootstrap_iters=2,
    )

    print(f"\n--- Harness Evaluation Complete on {device.upper()} ---")
    print("Full results:")
    print(json.dumps(results['results'], indent=2))
    return results # Return the full dictionary including samples


def main_harness_compare():
    """Main function to orchestrate the loading, evaluation, and comparison,
       and save detailed samples into a timestamped directory."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device auto-detected: {device.upper()}")

    ORIGINAL_PATH = args.checkpoint_path
    QUANTIZED_PATH = f"{ORIGINAL_PATH}_quantized/quantized_model.pt"
    PROJECT_ROOT = Path(__file__).resolve().parent

    # --- Directory to save detailed outputs (with timestamp) ---
    # Get current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") # Format: YYYYMMDD_HHMMSS

    # Define the base output directory
    BASE_OUTPUT_DIR = "./eval_outputs"

    # Create the full path with the timestamp
    TIMESTAMPED_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, timestamp)

    # Create the timestamped directory
    os.makedirs(TIMESTAMPED_OUTPUT_DIR, exist_ok=True)
    print(f"Detailed evaluation samples will be saved in: {TIMESTAMPED_OUTPUT_DIR}")
    # ---

    # TASKS_TO_RUN = ['ai2_arc','openbookqa']
    TASKS_TO_RUN = ['openbookqa']

    print("="*50)
    print(f"Starting comparison for tasks: {TASKS_TO_RUN}")
    print(f"Original:  {ORIGINAL_PATH}")
    print(f"Quantized: {QUANTIZED_PATH}")
    print("="*50)

    # --- Evaluate Original Model ---
    original_results = None
    tokenizer = None
    try:
        original_model, tokenizer = load_original_model_from_file(ORIGINAL_PATH, PROJECT_ROOT, device)
        # evaluate_on_harness now returns the full results dictionary
        original_results = evaluate_on_harness(original_model, tokenizer, TASKS_TO_RUN, device)

        # --- SAVE ORIGINAL SAMPLES (to timestamped directory) ---
        if original_results and 'samples' in original_results:
            # Use TIMESTAMPED_OUTPUT_DIR here
            orig_samples_path = os.path.join(TIMESTAMPED_OUTPUT_DIR, "original_samples.json")
            print(f"\nSaving original model samples to: {orig_samples_path}...")
            try:
                with open(orig_samples_path, "w", encoding='utf-8') as f:
                    # Use ensure_ascii=False for broader character support
                    json.dump(original_results['samples'], f, indent=2, ensure_ascii=False)
                print("Original samples saved successfully.")
            except Exception as e:
                print(f"Error saving original samples: {e}")
        # ---

        del original_model
        gc.collect()
        if device == "cuda": torch.cuda.empty_cache()
        print("Original model unloaded from memory.")
    except Exception as e:
        print(f"\n--- CRITICAL ERROR evaluating original model: {e} ---")
        traceback.print_exc()
        return

    # --- Evaluate Quantized Model ---
    quantized_results = None
    if tokenizer is None:
        print("Error: Tokenizer was not loaded. Cannot proceed.")
        return

    try:
        quantized_model = load_quantized_model_from_pt(QUANTIZED_PATH)
        quantized_results = evaluate_on_harness(quantized_model, tokenizer, TASKS_TO_RUN, "cpu")

        # --- SAVE QUANTIZED SAMPLES (to timestamped directory) ---
        if quantized_results and 'samples' in quantized_results:
            # Use TIMESTAMPED_OUTPUT_DIR here
            quant_samples_path = os.path.join(TIMESTAMPED_OUTPUT_DIR, "quantized_samples.json")
            print(f"\nSaving quantized model samples to: {quant_samples_path}...")
            try:
                with open(quant_samples_path, "w", encoding='utf-8') as f:
                    json.dump(quantized_results['samples'], f, indent=2, ensure_ascii=False)
                print("Quantized samples saved successfully.")
            except Exception as e:
                print(f"Error saving quantized samples: {e}")
        # ---

        del quantized_model
        gc.collect()
        print("Quantized model unloaded from memory.")
    except Exception as e:
        print(f"\n--- CRITICAL ERROR evaluating quantized model: {e} ---")
        traceback.print_exc()
        return

    # --- Print Final Comparison Table (Simplified) ---
    if original_results is None or quantized_results is None:
        print("\nComparison failed, one model did not produce results.")
        return

    print("\n" + "="*70)
    print(" " * 20 + "HARNESS ACCURACY COMPARISON")
    print("="*70)
    print(f"{'Benchmark':<18} | {'Metric':<15} | {'Original':<10} | {'Quantized':<10} | {'Change':<10}") # Adjusted Metric width
    print("-" * 70)

    # Directly iterate over the result keys from the original model's evaluation
    processed_tasks = 0
    for task_key in original_results['results']:
        # Check if the same task key exists in the quantized results
        if task_key not in quantized_results['results']:
            print(f"Task '{task_key}' found in original results but not in quantized. Skipping.")
            continue

        try:
            # --- METRIC FINDING LOGIC ---
            metric_name_found = None
            orig_acc = None
            quant_acc = None

            # Prioritize 'acc_norm,none'
            if 'acc_norm,none' in original_results['results'][task_key]:
                metric_name_found = 'acc_norm,none'
            # Fallback to 'acc,none'
            elif 'acc,none' in original_results['results'][task_key]:
                metric_name_found = 'acc,none'

            if metric_name_found:
                orig_acc = original_results['results'][task_key][metric_name_found]
                # Ensure the metric also exists in quantized results for this task key
                if metric_name_found in quantized_results['results'][task_key]:
                     quant_acc = quantized_results['results'][task_key][metric_name_found]
                else:
                     print(f"Metric '{metric_name_found}' found in original but not quantized results for '{task_key}'. Skipping comparison.")
                     continue # Skip this task if metric mismatch
            else:
                print(f"Could not find 'acc_norm,none' or 'acc,none' metric for task '{task_key}'. Skipping.")
                continue # Skip this task if no comparable metric found

            # Check if quantized_acc is valid before calculating change
            if quant_acc is None:
                continue

            # Handle potential "N/A" strings
            if isinstance(orig_acc, str) or isinstance(quant_acc, str):
                 change = "N/A"
                 print(f"{task_key:<18} | {metric_name_found:<15} | {str(orig_acc):<10} | {str(quant_acc):<10} | {change:<10}")
            else:
                 change = quant_acc - orig_acc
                 print(f"{task_key:<18} | {metric_name_found:<15} | {orig_acc:<10.4f} | {quant_acc:<10.4f} | {change:<+10.4f}")

            processed_tasks += 1
            # --- END OF METRIC FINDING ---

        except KeyError as e:
            print(f"KeyError accessing results for task '{task_key}': {e}. Skipping.")
        except Exception as e:
             print(f"An unexpected error occurred processing task '{task_key}': {e}. Skipping.")

    if processed_tasks == 0:
        print("No tasks were successfully compared. Check the results dictionaries and metric names.")

    print("="*70)
    print("✅ Comparison complete.")


if __name__ == "__main__":
    main_harness_compare()
    
#  python .\accuracy.py  --checkpoint_path checkpoints\250720_pretrain_smollm-360m_kv-share_rec3_middle_cycle_random_lr3e-3_mor_expert_linear_alpha_0.1_sigmoid_aux_loss_0.001 --local_datasets_path hf_datasets
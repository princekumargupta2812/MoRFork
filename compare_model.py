import time
import torch
import numpy as np
import argparse
import os
import gc  # Garbage collector
from pathlib import Path
import traceback

# --- NEW: Import psutil for memory checking ---
try:
    import psutil
except ImportError:
    print("="*50)
    print("ERROR: psutil is not installed. Please run: pip install psutil")
    print("="*50)
    exit(1)

# --- IMPORTS FROM YOUR PROJECT FILES ---
try:
    from static_ptq import load_model_and_tokenizer, predict_next_word, ConfigNamespace, MOR_MODEL_CLS
    print("Successfully imported functions from static_ptq.py")
except ImportError as e:
    print("="*50); print("ERROR: Could not import from 'static_ptq.py'."); print(f"Details: {e}"); traceback.print_exc(); print("="*50); exit(1)

try:
    from accuracy import load_quantized_model_from_pt
    print("Successfully imported function from accuracy.py")
except ImportError as e:
    print("="*50); print("ERROR: Could not import from 'accuracy.py'."); print(f"Details: {e}"); traceback.print_exc(); print("="*50); exit(1)

# --- CONFIGURATION ---
INPUT_TEXT = "Once upon a time"
NUM_RUNS = 20 # Number of times to run inference for averaging

# --- NEW: Helper function to print memory ---
def print_memory_usage(label: str):
    """Gets the current process's Resident Set Size (RSS) memory."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    rss_mb = mem_info.rss / (1024**2)
    print(f"[{label}] Current process RAM: {rss_mb:.2f} MB")
    return rss_mb

def main():
    # --- ARGUMENT PARSING ---
    parser = argparse.ArgumentParser(description="Compare inference speed and memory usage of original and quantized models.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the ORIGINAL model checkpoint.")
    args = parser.parse_args()

    # --- DEFINE PATHS ---
    ORIGINAL_PATH = args.checkpoint_path
    QUANTIZED_PATH = f"{ORIGINAL_PATH}_quantized/quantized_model.pt"
    PROJECT_ROOT = Path(__file__).resolve().parent

    if not os.path.isdir(ORIGINAL_PATH): print(f"Error: Original checkpoint path not found: {ORIGINAL_PATH}"); exit(1)
    if not os.path.exists(QUANTIZED_PATH): print(f"Error: Quantized model file not found: {QUANTIZED_PATH}"); exit(1)

    print(f"Comparing models:"); print(f"  Original:  {ORIGINAL_PATH}"); print(f"  Quantized: {QUANTIZED_PATH}")
    print(f"Input text: \"{INPUT_TEXT}\""); print(f"Timing {NUM_RUNS} inference runs for each model...")

    # --- MEASURE BASELINE MEMORY ---
    gc.collect() # Clean up before starting
    base_mem = print_memory_usage("Baseline")

    # --- LOAD ORIGINAL MODEL (CPU) & MEASURE MEMORY ---
    original_model = None
    tokenizer = None
    mem_after_original = base_mem
    try:
        print("\nLoading original model to CPU...")
        original_model, tokenizer = load_model_and_tokenizer(ORIGINAL_PATH, 'cpu', PROJECT_ROOT)
        mem_after_original = print_memory_usage("After Original Load")
        original_footprint = mem_after_original - base_mem
        print(f"Original Model RAM Footprint: {original_footprint:.2f} MB")
    except Exception as e:
        print(f"Failed to load original model: {e}"); traceback.print_exc(); return

    # --- LOAD QUANTIZED MODEL (CPU) & MEASURE MEMORY ---
    quantized_model = None
    mem_after_quantized = 0
    mem_after_cleanup = 0
    try:
        # --- Clean up original model first ---
        if original_model is not None:
            del original_model; original_model = None
        gc.collect()
        mem_after_cleanup = print_memory_usage("After Cleanup")
        # --- Load quantized model ---
        print("\nLoading quantized model...")
        quantized_model = load_quantized_model_from_pt(QUANTIZED_PATH) # Loads to CPU
        mem_after_quantized = print_memory_usage("After Quantized Load")
        quantized_footprint = mem_after_quantized - mem_after_cleanup
        print(f"Quantized Model RAM Footprint: {quantized_footprint:.2f} MB")
        # Ensure tokenizer is available
        if tokenizer is None:
             print("Re-loading tokenizer..."); tokenizer = AutoTokenizer.from_pretrained(ORIGINAL_PATH)
             if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"Failed to load quantized model: {e}"); traceback.print_exc(); return

    # --- RELOAD ORIGINAL MODEL FOR TIMING (if cleaned up) ---
    # We need both models in memory for the timing part
    if original_model is None:
        try:
            print("\nRe-Loading original model for timing...")
            original_model, _ = load_model_and_tokenizer(ORIGINAL_PATH, 'cpu', PROJECT_ROOT)
        except Exception as e:
            print(f"Failed to re-load original model for timing: {e}"); traceback.print_exc(); return


    # --- WARMUP ---
    print("\nWarming up inference...")
    try:
        _ = predict_next_word(original_model, tokenizer, INPUT_TEXT, 'cpu')
        _ = predict_next_word(quantized_model, tokenizer, INPUT_TEXT, 'cpu')
        print("Warmup complete.")
    except Exception as e:
        print(f"Error during warmup: {e}"); traceback.print_exc(); return

    # --- TIME ORIGINAL MODEL ---
    print(f"\n--- Timing Original Model ({NUM_RUNS} runs) ---")
    original_times = []
    try:
        for i in range(NUM_RUNS):
            start_time = time.perf_counter()
            _ = predict_next_word(original_model, tokenizer, INPUT_TEXT, 'cpu')
            end_time = time.perf_counter()
            original_times.append(end_time - start_time)
            # Optional: print(f" Run {i+1}/{NUM_RUNS}: {original_times[-1]:.6f} sec")
    except Exception as e:
        print(f"Error during original model timing: {e}"); traceback.print_exc(); return

    avg_original_time = np.mean(original_times) if original_times else 0
    print(f"Average Original Inference Time: {avg_original_time:.6f} seconds")

    # --- TIME QUANTIZED MODEL ---
    print(f"\n--- Timing Quantized Model ({NUM_RUNS} runs) ---")
    quantized_times = []
    try:
        for i in range(NUM_RUNS):
            start_time = time.perf_counter()
            _ = predict_next_word(quantized_model, tokenizer, INPUT_TEXT, 'cpu')
            end_time = time.perf_counter()
            quantized_times.append(end_time - start_time)
            # Optional: print(f" Run {i+1}/{NUM_RUNS}: {quantized_times[-1]:.6f} sec")
    except Exception as e:
        print(f"Error during quantized model timing: {e}"); traceback.print_exc(); return

    avg_quantized_time = np.mean(quantized_times) if quantized_times else 0
    print(f"Average Quantized Inference Time: {avg_quantized_time:.6f} seconds")

    # --- FINAL COMPARISON ---
    print("\n" + "="*50)
    print("            PERFORMANCE COMPARISON")
    print("="*50)

    # --- Memory Comparison ---
    print("## 🧠 Memory Usage (RAM Footprint)")
    original_footprint = mem_after_original - base_mem
    quantized_footprint = mem_after_quantized - mem_after_cleanup
    print(f"  Original Model:   {original_footprint:.2f} MB")
    print(f"  Quantized Model:  {quantized_footprint:.2f} MB")
    if original_footprint > 0 and quantized_footprint > 0:
        reduction_percent = (1 - (quantized_footprint / original_footprint)) * 100
        print(f"  Reduction:        {reduction_percent:.2f}%")
    else:
        print("  Could not calculate memory reduction.")
    print("-" * 50)

    # --- Speed Comparison ---
    print("## ⏱️ Inference Speed (CPU)")
    print(f"  Original Model (Avg):   {avg_original_time:.6f} seconds")
    print(f"  Quantized Model (Avg):  {avg_quantized_time:.6f} seconds")
    if avg_original_time > 0 and avg_quantized_time > 0:
        if avg_quantized_time < avg_original_time:
            speedup = avg_original_time / avg_quantized_time
            print(f"  Speedup (Quantized vs Original): {speedup:.2f}x faster")
        else:
            slowdown = avg_quantized_time / avg_original_time
            print(f"  Slowdown (Quantized vs Original): {slowdown:.2f}x slower")
    else:
        print("  Could not calculate speed comparison.")
    print("="*50)

if __name__ == "__main__":
    main()
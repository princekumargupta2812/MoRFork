# qunatization static code 
import argparse
import torch
import os
import yaml
from pathlib import Path
from transformers import AutoConfig, AutoTokenizer
import time # For potential performance comparison, though not explicitly asked

# Re-use ConfigNamespace and imports from the previous script
class ConfigNamespace(argparse.Namespace):
    """A subclass of argparse.Namespace that also supports the .get() method like a dict."""
    def get(self, key, default=None):
        return getattr(self, key, default)

try:
    from model.util import MOR_MODEL_CLS
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Please ensure you are running this script from the 'mixture_of_recursions' directory ")
    print("or that 'mixture_of_recursions' is in your PYTHONPATH.")
    exit(1)

def load_model_and_tokenizer(checkpoint_path: str, device: str, project_root: Path):
    """
    Loads the MoR model and tokenizer, applying MoR transformations from the YAML config.
    This is largely the same as the previous script.
    """
    print(f"Loading model and tokenizer from: {checkpoint_path}")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer loaded.")

    config = AutoConfig.from_pretrained(checkpoint_path)
    print(f"Model configuration loaded. Architecture: {config.architectures[0]}")

    model_arch = config.model_type
    if model_arch not in MOR_MODEL_CLS:
        if model_arch == "llama" and "smollm" in MOR_MODEL_CLS:
            print(f"Model architecture is 'llama', mapping to 'smollm' for MOR_MODEL_CLS.")
            model_arch = "smollm"
        else:
            raise ValueError(
                f"Model architecture '{model_arch}' not found in MOR_MODEL_CLS. "
                f"Available: {list(MOR_MODEL_CLS.keys())}"
            )
    
    model_cls = MOR_MODEL_CLS[model_arch]
    print(f"Using custom model class: {model_cls.__name__}")

    torch_dtype = config.torch_dtype if hasattr(config, "torch_dtype") else torch.bfloat16
    if isinstance(torch_dtype, str):
        torch_dtype = getattr(torch, torch_dtype)

    model = model_cls._from_config(config, torch_dtype=torch_dtype)
    print("Model instantiated from config.")

    # --- MoR Transformation Logic ---
    model_config = ConfigNamespace()
    yaml_config_path = None
    yaml_data = None
    checkpoint_name = os.path.basename(checkpoint_path)
    potential_yaml_path = os.path.join(project_root, "conf", "pretrain", f"{checkpoint_name}.yaml")

    if os.path.exists(potential_yaml_path):
        yaml_config_path = potential_yaml_path
        print(f"Found corresponding YAML config at: {yaml_config_path}")
        with open(yaml_config_path, 'r') as f:
            yaml_data = yaml.safe_load(f)
    else:
        print(f"Warning: Could not find a YAML config at {potential_yaml_path}. MoR transformations may not be applied correctly.")
        # Basic fallback if no YAML
        model_config.mor = ConfigNamespace()
        model_config.mor.enable = False # Assume no MoR if no config

    if yaml_data and yaml_data.get('mor', {}).get('enable', False):
        model_config.mor = ConfigNamespace()
        model_config.mor.enable = yaml_data['mor'].get('enable', True)
        model_config.mor.type = yaml_data['mor'].get('type')
        model_config.mor.rand_router = yaml_data['mor'].get('rand_router', False)
        model_config.mor.router_type = yaml_data['mor'].get('router_type', 'linear')
        model_config.mor.temp = yaml_data['mor'].get('temp', 1.0)
        
        if model_config.mor.type == "expert":
            model_config.mor.expert = ConfigNamespace()
            model_config.mor.expert.cap_warmup_step = yaml_data['mor']['expert'].get('cap_warmup_step', 0)
            model_config.mor.expert.router_func = yaml_data['mor']['expert'].get('router_func', 'sigmoid')
            model_config.mor.expert.alpha = yaml_data['mor']['expert'].get('alpha', 0.1)
            model_config.mor.expert.sampling = yaml_data['mor']['expert'].get('sampling', 'aux_loss')
            model_config.mor.capacity = yaml_data['mor'].get('capacity') 
            if model_config.mor.capacity is None:
                num_rec = yaml_data.get('recursive', {}).get('num_recursion', 3)
                model_config.mor.capacity = ",".join(["1.0"] * num_rec)
                print(f"Using default capacity: {model_config.mor.capacity}")
        elif model_config.mor.type == "token":
            model_config.mor.token = ConfigNamespace()
            model_config.mor.token.bal_warmup_step = yaml_data['mor']['token'].get('bal_warmup_step', 0)
            model_config.mor.router_func = yaml_data['mor']['token'].get('router_func', 'sigmoid')

        model_config.num_warmup_steps = yaml_data.get('num_warmup_steps', 0)
        model_config.gradient_accumulation_steps = yaml_data.get('gradient_accumulation_steps', 1)
        model_config.precision = yaml_data.get('precision', 'bf16')
        model_config.recursive = ConfigNamespace()
        model_config.recursive.enable = yaml_data.get('recursive', {}).get('enable', True)
        model_config.recursive.sharing = yaml_data['recursive'].get('sharing')
        model_config.recursive.num_recursion = yaml_data['recursive'].get('num_recursion')
        model_config.recursive.initialization = yaml_data['recursive'].get('initialization', 'random')
        model_config.model_name_or_path = yaml_data.get('model_name_or_path', checkpoint_path)

        if model_config.mor.enable:
            try:
                if model_config.mor.type == "expert":
                    print(f"Applying MoR expert transformation...")
                    model.transform_layer_to_mor_expert(model_config)
                elif model_config.mor.type == "token":
                    print(f"Applying MoR token transformation...")
                    model.transform_layer_to_mor_token(model_config)
                print(f"MoR transformation (type: {model_config.mor.type}) applied.")
            except Exception as e:
                print(f"Warning: Could not apply MoR transformations: {e}.")
    
    # --- Load Model State ---
    model_path = os.path.join(checkpoint_path, "pytorch_model.bin")
    if not os.path.exists(model_path):
        model_path = os.path.join(checkpoint_path, "model.safetensors")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights not found.")
        
    print(f"Loading model state from: {model_path}")
    if model_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(model_path)
    else:
        state_dict = torch.load(model_path, map_location="cpu")
    
    model.load_state_dict(state_dict)
    print("Model state loaded.")

    if hasattr(config, 'kv_sharing') and config.kv_sharing is not None:
        kv_config = config.kv_sharing.to_dict() if hasattr(config.kv_sharing, 'to_dict') else dict(config.kv_sharing)
        if "update_cache" not in kv_config: kv_config["update_cache"] = False
        final_kv_cfg = ConfigNamespace()
        final_kv_cfg.kv_sharing = ConfigNamespace(**kv_config)
        model_config.kv_sharing = final_kv_cfg.kv_sharing
        print("Setting KV sharing config...")
        model.set_kv_sharing_config(model_config)

    model.to(device)
    model.eval()
    print(f"Model moved to {device} and set to evaluation mode.")
    return model, tokenizer

def quantize_model(model: torch.nn.Module, tokenizer, device: str, checkpoint_path: str) -> torch.nn.Module:
    """
    Quantizes the model using static post-training quantization.
    This version fixes the wrapping logic to correctly handle top-level
    modules (like lm_head) in addition to nested modules.
    """
    print("Starting static model quantization...")
    model.eval()
    
    original_device = next(model.parameters()).device
    if original_device.type == 'cuda':
        model.to('cpu')

    # --- 1. Configure and Wrap Linear Layers ---
    
    modules_to_wrap = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # Add all linear layers, including lm_head
            modules_to_wrap.append((name, module))
        
        if isinstance(module, torch.nn.Embedding):
            print(f"Setting weight_only qconfig for: {name}")
            module.qconfig = torch.quantization.float_qparams_weight_only_qconfig

    # We must iterate in reverse to avoid parent/child naming conflicts
    for name, module in reversed(modules_to_wrap):
        try:
            # Get the qconfig
            qconfig = torch.quantization.get_default_qconfig('fbgemm')
            
            # Set qconfig on the nn.Linear module
            module.qconfig = qconfig
            
            # Create the wrapped module
            wrapped_module = torch.nn.Sequential(
                torch.quantization.QuantStub(),
                module,
                torch.quantization.DeQuantStub()
            )
            
            # Set the qconfig on the parent Sequential wrapper
            wrapped_module.qconfig = qconfig
            
            # --- *** THE CRITICAL FIX *** ---
            # Correctly find the parent module and child name
            if '.' in name:
                # Nested module (e.g., "model.layers.0.q_proj")
                parent_name, child_name = name.rsplit('.', 1)
                parent_module = model.get_submodule(parent_name)
            else:
                # Top-level module (e.g., "lm_head")
                parent_module = model
                child_name = name
            
            # Replace the original module with the wrapped one
            setattr(parent_module, child_name, wrapped_module)
            print(f"Successfully wrapped {name} with Quant/DeQuant stubs.")
        
        except Exception as e:
            # This is important for debugging
            print(f"CRITICAL: Failed to wrap {name}: {e}. This layer might not be quantized.")

    # --- 2. Prepare ---
    model_prepared = torch.quantization.prepare(model, inplace=True)
    print("Model prepared for quantization (observers inserted).")

    # --- 3. Calibrate ---
    print("Calibrating model with sample data...")
    calibration_sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a fascinating field of study.",
        "Once upon a time, in a land far, far away.",
        "To be, or not to be, that is the question.",
        "The capital of France is Paris."
    ]
    
    with torch.no_grad():
        for sentence in calibration_sentences:
            inputs = tokenizer(sentence, return_tensors="pt")
            inputs = {k: v.to('cpu') for k, v in inputs.items()}
            _ = model_prepared(**inputs)
    print("Calibration complete.")

    # --- 4. Convert ---
    # This will now correctly fuse all wrapped layers
    model_quantized = torch.quantization.convert(model_prepared)
    
    print("Static model quantization complete.")
    # print("The quantized model is+++++++++++++++++++++++++++++++", model_quantized)
    
    # # --- 5. Save ---
    # quantized_model_path = checkpoint_path + "_quantized"
    # os.makedirs(quantized_model_path, exist_ok=True)
    
    # torch.save(model_quantized.state_dict(), os.path.join(quantized_model_path, "pytorch_model.bin"))
    # model_quantized.config.save_pretrained(quantized_model_path)
    
    # print(f"Quantized model saved to: {quantized_model_path}")

    quantized_model_path = checkpoint_path + "_quantized"
    os.makedirs(quantized_model_path, exist_ok=True)
    full_model_path = os.path.join(quantized_model_path, "quantized_model.pt")
    torch.save(model_quantized, full_model_path)

    print(f"Full quantized model saved to: {full_model_path}")
    
    return model_quantized

def predict_next_word(model: torch.nn.Module, tokenizer, text: str, device: str) -> str:
    """
    Predicts the next word for the given input text using the provided model.
    Suppresses all special tokens.
    """
    # Check if model is quantized and adjust device accordingly
    model_device = next(model.parameters()).device
    if model_device.type == 'cpu':
        # For quantized models, ensure inputs are on CPU
        inputs = tokenizer(text, return_tensors="pt").to('cpu')
    else:
        inputs = tokenizer(text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[0, -1, :]

    if hasattr(tokenizer, "all_special_ids"):
        all_special_token_ids = set(tokenizer.all_special_ids)
        for special_token_id in all_special_token_ids:
            if 0 <= special_token_id < logits.shape[0]:
                logits[special_token_id] = -float("inf")
    
    predicted_token_id = torch.argmax(logits).unsqueeze(0)
    predicted_word = tokenizer.decode(predicted_token_id, skip_special_tokens=True)
    return predicted_word.strip()

def main():
    parser = argparse.ArgumentParser(description="Compare unquantized and quantized MoR model inference.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the model checkpoint directory.")
    parser.add_argument("--text", type=str, required=True, help="Input text for next-word prediction.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run inference on.")
    args = parser.parse_args()

    if not os.path.isdir(args.checkpoint_path):
        print(f"Error: Checkpoint path '{args.checkpoint_path}' is not a valid directory.")
        exit(1)

    project_root = Path(__file__).resolve().parent
    if str(project_root) not in os.sys.path:
        os.sys.path.append(str(project_root))

    # 1. Load the original (unquantized) model
    print("="*50)
    print("Loading original model...")
    original_model, tokenizer = load_model_and_tokenizer(args.checkpoint_path, args.device, project_root)
    # print("The orignal model is+++++++++++++++++++++++++++++++",original_model)
    
    # 2. Create a quantized version of the model
    print("="*50)
    # It's good practice to quantize a copy of the model if you need to keep the original
    # For this script, we can quantize directly as we don't need the original after this point.
    # However, to be safe and allow comparison, let's create a copy.
    # Note: A deep copy might be tricky with state dicts and device placements.
    # A simpler way is to reload, or quantize and then compare.
    # For this script, we'll load once, quantize, and then the original_model variable is no longer the original.
    # Let's adjust: load, then clone for quantization.
    
    # To properly compare, we need two separate model instances.
    # Let's reload the model for the quantized version to ensure they are independent.
    print("Loading model again for quantization to ensure independent instances...")
    model_to_quantize, _ = load_model_and_tokenizer(args.checkpoint_path, args.device, project_root)
    quantized_model = quantize_model(model_to_quantize, tokenizer, args.device, args.checkpoint_path)
    
    print("="*50)
    print(f"Input text: \"{args.text}\"")
    
    # 3. Perform inference with the original model
    print("\n--- Running inference with original model ---")
    original_prediction = predict_next_word(original_model, tokenizer, args.text, args.device)
    print(f"Original model prediction: \"{original_prediction}\"")

    # 4. Perform inference with the quantized model
    print("\n--- Running inference with quantized model ---")
    quantized_prediction = predict_next_word(quantized_model, tokenizer, args.text, args.device)
    print(f"Quantized model prediction: \"{quantized_prediction}\"")

    # 5. Compare the results
    print("\n--- Comparison ---")
    if original_prediction == quantized_prediction:
        print("Result: The original and quantized models predicted the same next word.")
    else:
        print("Result: The original and quantized models predicted different next words.")
        print(f"  Original: \"{original_prediction}\"")
        print(f"  Quantized: \"{quantized_prediction}\"")
    
    print("="*50)

if __name__ == "__main__":
    main()
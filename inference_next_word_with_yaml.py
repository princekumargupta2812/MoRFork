# inference next_word_with_yaml.py

import argparse
import torch
import os
import yaml # Added for parsing YAML config files
from pathlib import Path
from transformers import AutoConfig, AutoTokenizer

class ConfigNamespace(argparse.Namespace):
    """A subclass of argparse.Namespace that also supports the .get() method like a dict."""
    def get(self, key, default=None):
        return getattr(self, key, default)

# Ensure the project's modules can be imported
# This assumes the script is run from the 'mixture_of_recursions' directory
# or that the 'mixture_of_recursions' directory is in the PYTHONPATH.
try:
    from model.util import MOR_MODEL_CLS
    # from util.tokenizer import load_tokenizer_from_config # Not strictly needed if AutoTokenizer works
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Please ensure you are running this script from the 'mixture_of_recursions' directory ")
    print("or that 'mixture_of_recursions' is in your PYTHONPATH.")
    exit(1)

def load_model_and_tokenizer(checkpoint_path: str, device: str, project_root: Path):
    """
    Loads the MoR model and tokenizer from the given checkpoint path.
    project_root is needed to locate the YAML configuration files.
    """
    print(f"Loading model and tokenizer from: {checkpoint_path}")

    # 1. Load Tokenizer
    # The checkpoint path contains all tokenizer files.
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer loaded.")

    # 2. Load Model Configuration
    config = AutoConfig.from_pretrained(checkpoint_path)
    print(f"Model configuration loaded. Architecture: {config.architectures[0]}")

    # 3. Determine Model Architecture for MOR_MODEL_CLS
    # The config.model_type is usually 'llama', 'gpt2', etc.
    # For MoR models based on Llama (like SmolLM), the config.model_type might be 'llama'
    # but MOR_MODEL_CLS uses 'smollm' or 'smollm2' as keys.
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
    
    # 4. Get Custom Model Class
    model_cls = MOR_MODEL_CLS[model_arch]
    print(f"Using custom model class: {model_cls.__name__}")

    # 5. Instantiate Model from Config
    # We need to pass torch_dtype for correct loading.
    torch_dtype = config.torch_dtype if hasattr(config, "torch_dtype") else torch.bfloat16
    if isinstance(torch_dtype, str):
        torch_dtype = getattr(torch, torch_dtype)

    model = model_cls._from_config(config, torch_dtype=torch_dtype)
    print("Model instantiated from config.")

    # 6. Apply MoR Transformations (if applicable)
    # The MoR transformations require a cfg object with specific parameters.
    # We will load these from the corresponding YAML configuration file.
    print("Attempting to apply MoR transformations...")
    model_config = ConfigNamespace()
    yaml_config_path = None
    yaml_data = None

    # Construct the path to the YAML config file
    # Assumes checkpoint_path is like 'checkpoints/checkpoint_name'
    # and YAMLs are in 'conf/pretrain/checkpoint_name.yaml'
    checkpoint_name = os.path.basename(checkpoint_path)
    potential_yaml_path = os.path.join(project_root, "conf", "pretrain", f"{checkpoint_name}.yaml")

    if os.path.exists(potential_yaml_path):
        yaml_config_path = potential_yaml_path
        print(f"Found corresponding YAML config at---------------------: {yaml_config_path}")
        with open(yaml_config_path, 'r') as f:
            yaml_data = yaml.safe_load(f)
    else:
        print(f"Warning: Could not find a YAML config at {potential_yaml_path}. Using heuristic defaults.++++++++++++++++++++++++++++")
        # Fallback to heuristic inference if YAML not found
        if "mor_expert" in checkpoint_path.lower():
            model_config.mor = ConfigNamespace()
            model_config.mor.enable = True
            model_config.mor.type = "expert"
        elif "mor_token" in checkpoint_path.lower():
            model_config.mor = ConfigNamespace()
            model_config.mor.enable = True
            model_config.mor.type = "token"
        else:
            model_config.mor = ConfigNamespace()
            model_config.mor.enable = False
    
    if yaml_data and yaml_data.get('mor', {}).get('enable', False):
        # Populate model_config from YAML data
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
            # Capacity might be null in YAML, transform_layer_to_mor_expert might have a default.
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

        # Populate recursive config
        model_config.recursive = ConfigNamespace()
        model_config.recursive.enable = yaml_data.get('recursive', {}).get('enable', True)
        model_config.recursive.sharing = yaml_data['recursive'].get('sharing')
        model_config.recursive.num_recursion = yaml_data['recursive'].get('num_recursion')
        model_config.recursive.initialization = yaml_data['recursive'].get('initialization', 'random')
        
        # Ensure model_name_or_path is loaded from the YAML data
        model_config.model_name_or_path = yaml_data.get('model_name_or_path', checkpoint_path)
        print("This is the required path for the model++++++++++++++++++++++++", model_config.model_name_or_path)

    if not model_config.mor.enable: 
        print("MoR transformations disabled as per YAML or heuristic check.")
    else:
        try:
            if model_config.mor.type == "expert":
                print(f"Applying MoR expert transformation with config from YAML...")
                model.transform_layer_to_mor_expert(model_config)
            elif model_config.mor.type == "token":
                print(f"Applying MoR token transformation with config from YAML...")
                model.transform_layer_to_mor_token(model_config)
            print(f"MoR transformation (type: {model_config.mor.type}) applied.")
        except Exception as e:
            print(f"Warning: Could not apply MoR transformations: {e}. The model might not behave as a pure MoR model.")
            print("This might be due to missing or incorrect parameters in the configuration.")
    
    # 7. Load Model State
    model_path = os.path.join(checkpoint_path, "pytorch_model.bin")
    if not os.path.exists(model_path):
        model_path = os.path.join(checkpoint_path, "model.safetensors")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights not found at {os.path.join(checkpoint_path, 'pytorch_model.bin')} or {model_path}")
        
    print(f"Loading model state from: {model_path}")
    if model_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(model_path)
    else:
        state_dict = torch.load(model_path, map_location="cpu")
    
    model.load_state_dict(state_dict)
    print("Model state loaded.")

    # 8. Set KV sharing config on the model
    if hasattr(config, 'kv_sharing') and config.kv_sharing is not None:
        kv_config = config.kv_sharing.to_dict() if hasattr(config.kv_sharing, 'to_dict') else dict(config.kv_sharing)
        if "update_cache" not in kv_config:
            kv_config["update_cache"] = False
        
        final_kv_cfg = ConfigNamespace()
        final_kv_cfg.kv_sharing = ConfigNamespace(**kv_config)
        model_config.kv_sharing = final_kv_cfg.kv_sharing

        print("Setting KV sharing config on the model...")
        model.set_kv_sharing_config(model_config)
        print("KV sharing config set.")
    else:
        print("Warning: config.kv_sharing not found. Recursive KV caching may not work as expected.")

    model.to(device)
    model.eval()
    print(f"Model moved to {device} and set to evaluation mode.")
    return model, tokenizer

def predict_next_word(model, tokenizer, text: str, device: str):
    """
    Predicts the next word for the given input text.
    Suppresses all special tokens (like EOS, chat tokens) to encourage generating a real word.
    """
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits[0, -1, :]

    if hasattr(tokenizer, "add_special_tokens") and hasattr(tokenizer, "special_tokens_map"):
        all_special_token_ids = set(tokenizer.all_special_ids)
        for special_token_id in all_special_token_ids:
            if 0 <= special_token_id < logits.shape[0]:
                logits[special_token_id] = -float("inf")
    
    predicted_token_id = torch.argmax(logits).unsqueeze(0)
    predicted_word = tokenizer.decode(predicted_token_id, skip_special_tokens=True)

    return predicted_word.strip()

def main():
    parser = argparse.ArgumentParser(description="Next-word prediction inference script using YAML config.")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the model checkpoint directory (e.g., checkpoints/your_model_name/).",
    )
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Input text for next-word prediction.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the inference on ('cuda' or 'cpu').",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.checkpoint_path):
        print(f"Error: Checkpoint path '{args.checkpoint_path}' is not a valid directory.")
        exit(1)

    project_root = Path(__file__).resolve().parent
    if str(project_root) not in os.sys.path:
        os.sys.path.append(str(project_root))

    model, tokenizer = load_model_and_tokenizer(args.checkpoint_path, args.device, project_root)
    
    input_text = args.text
    # if args.text == "The capital of France is":
    #     input_text = "What is the capital of France? The answer is"
    #     print(f"Input text modified for better factual completion: \"{input_text}\"")
    # else:
    print(f"\nInput text: \"{input_text}\"")
        
    next_word = predict_next_word(model, tokenizer, input_text, args.device)
    print(f"Predicted next word: \"{next_word}\"")

if __name__ == "__main__":
    main()
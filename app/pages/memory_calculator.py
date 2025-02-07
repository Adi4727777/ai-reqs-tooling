import streamlit as st
import requests
from PIL import Image
from utils.memory_utils import llm_memory_GPU_distribution, calculate_memory

#Flag if using endpoint config
backend_endpoint = False 

# Load Logos
amd_logo = Image.open('./digital_assets/amd.png')
nvidia_logo = Image.open('./digital_assets/nvidia.png')

# Model Data Dictionary
model_dictionaries = {
    "Llama 3 70B Instruct": {"parameters": 70, "hidden_size": 8192, "layer_count": 80, "attention_heads": 64},
    "Llama 3 8B Instruct": {"parameters": 8, "hidden_size": 4096, "layer_count": 32, "attention_heads": 32},
    "Qwen 72B Instruct": {"parameters": 72, "hidden_size": 8192, "layer_count": 80, "attention_heads": 64},
    "Mixtral 8x7B": {"parameters": 47, "hidden_size": 4096, "layer_count": 32, "attention_heads": 32},
    "DeepSeek-R1": {"parameters": 671, "hidden_size": 7168, "layer_count": 61, "attention_heads": 128},
}

Models = list(model_dictionaries.keys()) + ["Custom"]

st.set_page_config(page_title="LLM Memory Calculator", layout="wide")
st.title("📊 LLM Memory Requirement Calculator")

with st.sidebar:
    st.header("⚙️ Model Configuration")
    prefill_model = st.selectbox("Prefill Model Selection", options=Models)
    
    # Dynamically set default values
    defaults = model_dictionaries.get(prefill_model, {})
    
    parameters = st.number_input("Parameters (B)", min_value=1, value=defaults.get("parameters", 1))
    batch_size = st.number_input("Batch Size", min_value=1, value=1)
    precision = st.selectbox("Precision", ["FP16", "8BIT", "4BIT", "FP32"])
    sequence_length = st.number_input("Sequence Length", min_value=1, value=2048)
    hidden_size = st.number_input("Hidden Size", min_value=1, value=defaults.get("hidden_size", 1))
    layer_count = st.number_input("Layer Count", min_value=1, value=defaults.get("layer_count", 1))
    attention_heads = st.number_input("Attention Heads", min_value=1, value=defaults.get("attention_heads", 1))
    tensor_parallelism = st.number_input("Tensor Parallelism", min_value=1, value=1)
    optimizer = st.selectbox("Optimizer", ["AdamW", "Adam", "bitsandbytes_8bit", "SGD-like"])
    percent_trainable_parameters = st.slider("% Trainable Parameters", min_value=1, max_value=100, value=100)
    

    if st.button("🔍 Calculate Memory Usage"):

        if backend_endpoint:
            URL = 'http://localhost:8000/calculate_memory'
            DATA = {
                'parameters': parameters, 'batch_size': batch_size, 'precision': precision,
                'sequence_length': sequence_length, 'hidden_size': hidden_size,
                'layer_count': layer_count, 'attention_heads': attention_heads,
                'tensor_parallelism': tensor_parallelism, 'optimizer': optimizer,
                'percent_trainable_parameters': percent_trainable_parameters
            }
            response = requests.post(url=URL, json=DATA)
            st.session_state["response"] = response.json()
        else:
            st.session_state["response"] = calculate_memory(parameters, batch_size, precision, sequence_length, hidden_size, layer_count, attention_heads,
            tensor_parallelism, optimizer, percent_trainable_parameters)

        st.success("✅ Calculation Complete!")

# Display Results
if "response" in st.session_state:

    try:
        response_data = st.session_state["response"]["Calculation"]
    except:
        response_data = st.session_state["response"]
    
    inference_tab, training_tab = st.tabs(["🚀 Inference", "🎯 Training"])
    
    with inference_tab:
        st.metric(label="**Total Inference Memory (GB)**", value=f"{response_data['standard_inference_total_memory_gb']:.2f}")
        st.write(f"- **Model Weights**: {response_data['model_weights_memory']:.2f} GB")
        st.write(f"- **KV Cache**: {response_data['kv_cache_memory']:.2f} GB")
        st.write(f"- **Activation Memory**: {response_data['activations_memory']:.2f} GB")
    
    with training_tab:
        st.metric(label="**Total Training Memory (GB)**", value=f"{response_data['standard_training_total_memory_gb']:.2f}")
        st.write(f"- **Model Weights**: {response_data['model_weights_memory']:.2f} GB")
        st.write(f"- **KV Cache**: {response_data['kv_cache_memory']:.2f} GB")
        st.write(f"- **Activation Memory**: {response_data['activations_memory']:.2f} GB")
        st.write(f"- **Optimizer Memory**: {response_data['optimizer_memory']:.2f} GB")
        st.write(f"- **Gradient Memory**: {response_data['gradient_memory']:.2f} GB")
    
    st.subheader("🖥️ GPU Requirements")
    col1, col2, col3, col4 = st.columns(4)
    
    gpu_configs = [
        ("MI300X (192GB)", amd_logo, 192),
        ("H100 (80GB)", nvidia_logo, 80),
        ("MI325X (256GB)", amd_logo, 256),
        ("H200 (141GB)", nvidia_logo, 141)
    ]
    
    for col, (gpu_name, logo, memory) in zip([col1, col2, col3, col4], gpu_configs):
        with col:
            st.image(logo, width=80)
            st.subheader(gpu_name)
    
            # Get inference/training memory distributions
            inference_gpu = llm_memory_GPU_distribution(response_data['standard_inference_total_memory_gb'], memory, 8, attention_heads, tensor_parallelism)
            training_gpu = llm_memory_GPU_distribution(response_data['standard_training_total_memory_gb'], memory, 8, attention_heads, tensor_parallelism)
    
            # Debugging: Ensure the function returns expected data
            st.write("Debug - Inference GPU:", inference_gpu)
            st.write("Debug - Training GPU:", training_gpu)
    
            # Inference Check
            if inference_gpu.get("attention_heads_divisible", False):
                if inference_gpu.get("fits_on_one_gpu", False):
                    st.success("✅ Fits on Single GPU for Inference")
                else:
                    st.warning(f"⚠️ Requires {inference_gpu.get('num_gpus_needed', '?')} GPUs for Inference")
            else:
                st.error("❌ Attention Heads Not Evenly Divisible - Check Tensor Parallelism")
    
            # Training Check
            if training_gpu.get("attention_heads_divisible", False):
                if training_gpu.get("fits_on_one_gpu", False):
                    st.success("✅ Fits on Single GPU for Training")
                else:
                    st.warning(f"⚠️ Requires {training_gpu.get('num_gpus_needed', '?')} GPUs for Training")
            else:
                st.error("❌ Attention Heads Not Evenly Divisible - Check Tensor Parallelism")
    
    

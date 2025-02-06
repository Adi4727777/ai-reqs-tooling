import streamlit as st
import requests
from PIL import Image
from utils.memory_utils import llm_memory_GPU_distribution

RESPONSE = None

model_dictionaries = {"Llama 3 70B Instruct": {"parameters":70, "hidden_size": 8192, "layer_count": 80, "attention_heads":64},
    "Llama 3 8B Instruct": {"parameters":8, "hidden_size": 4096, "layer_count": 32, "attention_heads":32},
    "Qwen 72B Instruct": {"parameters":72, "hidden_size": 8192, "layer_count": 80, "attention_heads":64},
    "Mixtral 8x7B": {"parameters":47, "hidden_size": 4096, "layer_count": 32, "attention_heads":32},
    "DeepSeek-R1": {"parameters":671, "hidden_size": 7168, "layer_count": 61, "attention_heads":128},
    }

Models = ["Llama 3 70B Instruct","Llama 3 8B Instruct","Qwen 72B Instruct","Mixtral 8x7B","DeepSeek-R1","Custom"]

amd_logo = Image.open('./digital_assets/amd.png')
nvidia_logo = Image.open('./digital_assets/nvidia.png')

with st.sidebar:
    prefill_model = st.selectbox(label='Prefill Model Selection', options=Models)
    parameters = st.number_input(label='Parameters', min_value = 1, value = model_dictionaries.get(prefill_model).get("parameters"), key='par-input')
    batch_size = st.number_input(label='Batch Size', min_value = 1, key='bs-input')
    precision = st.selectbox(label='Precision', options=["FP16", "8BIT", "4BIT", "FP32"], key='prec-input')
    sequence_length = st.number_input(label='Sequence Length', min_value = 1, value = 2048, key='seq-input')
    hidden_size = st.number_input(label='Hidden Size', min_value = 1, value = model_dictionaries.get(prefill_model).get("hidden_size"), key='hs-input')
    layer_count = st.number_input(label='Layer Count', min_value = 1, value = model_dictionaries.get(prefill_model).get("layer_count"), key='lc-input')
    attention_heads = st.number_input(label='Attention Heads', min_value = 1, value = model_dictionaries.get(prefill_model).get("attention_heads"), key='ah-input')
    tensor_parallelism = st.number_input(label='Tensor Parallism', min_value = 1, key='tp-input')
    optimizer = st.selectbox(label='Optimizer', options=["AdamW","Adam","bitsandbytes_8bit","SGD-like"], key='opt-input')
    percent_trainable_parameters = st.slider(label='Percent Trainable Parameters', min_value = 1, max_value=100, value= 100, key='trp-input')

    if st.button("Calculate"):
        URL = 'http://localhost:8000/calculate_memory'
        DATA = {'parameters': parameters, 'batch_size':batch_size, 'precision':precision, 'sequence_length':sequence_length, 
                  'hidden_size': hidden_size, 'layer_count': layer_count, 'attention_heads':attention_heads,
                  'tensor_parallelism':tensor_parallelism, 'optimizer':optimizer, 
                  'percent_trainable_parameters':percent_trainable_parameters}
        RESPONSE = requests.post(url = URL, json = DATA)
        st.success('Calculation Complete!')


st.title("LLM Memory Requirement Calculator")
inference, training = st.tabs(["Inference", "Training"])

if RESPONSE:
    standard_inference_total_memory_gb = RESPONSE.json().get('Calculation').get('standard_inference_total_memory_gb')
    inference.write(f"**Total Inference Memory for Model: {prefill_model}**: {standard_inference_total_memory_gb:.2f}")
    model_weights_memory = RESPONSE.json().get('Calculation').get('model_weights_memory')
    inference.write(f"- **Model Weights**: {model_weights_memory:.2f}")
    kv_cache_memory = RESPONSE.json().get('Calculation').get('kv_cache_memory')
    inference.write(f"- **KV Cache**: {kv_cache_memory:.2f}")
    activations_memory = RESPONSE.json().get('Calculation').get('activations_memory')
    inference.write(f"- **Activation Memory**: {activations_memory:.2f}")
    
    standard_training_total_memory_gb = RESPONSE.json().get('Calculation').get('standard_training_total_memory_gb')
    training.write(f"**Total Training Memory for Model: {prefill_model}**: {standard_training_total_memory_gb:.2f}")
    training.write(f"- **Model Weights**: {model_weights_memory:.2f}")
    training.write(f"- **KV Cache**: {kv_cache_memory:.2f}")
    training.write(f"- **Activation Memory**: {activations_memory:.2f}")
    optimizer_memory = RESPONSE.json().get('Calculation').get('optimizer_memory')
    training.write(f"- **Optimizer Memory**: {optimizer_memory:.2f}")
    gradient_memory = RESPONSE.json().get('Calculation').get('gradient_memory')
    training.write(f"- **Gradient Memory**: {gradient_memory:.2f}")


    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.header("MI300X (192GB)")
        st.image(amd_logo)
        mi300x_inference_gpu = llm_memory_GPU_distribution(standard_inference_total_memory_gb, 192, 8, attention_heads, tensor_parallelism)
        mi300x_training_gpu = llm_memory_GPU_distribution(standard_training_total_memory_gb, 192, 8, attention_heads, tensor_parallelism)

        if mi300x_inference_gpu.get('attention_heads_divisible'):
            if mi300x_inference_gpu.get("fits_on_one_gpu"):
                st.success("Model Fits on Single GPU for Inference")
            else:
                st.warning("Model Requires more than one GPU for Inference")
                st.write(f"- **GPUs Required for Inference**: {mi300x_inference_gpu.get('num_gpus_needed')}")
                if mi300x_inference_gpu.get('num_gpus_needed') > 8:
                    st.write(f"- **Nodes Required for Inference**: {mi300x_inference_gpu.get('num_nodes_needed')}")
        else:
            st.error("Attention Heads Not Evenly Divisible - Tensor Parallelism Config Error.")

        
        if mi300x_training_gpu.get('attention_heads_divisible'):
            if mi300x_training_gpu.get("fits_on_one_gpu"):
                st.success("Model Fits on Single GPU for Training")
            else:
                st.warning("Model Requires more than one GPU for Training")
                st.write(f"- **GPUs Required for Training**: {mi300x_training_gpu.get('num_gpus_needed')}")
                if mi300x_training_gpu.get('num_gpus_needed') > 8:
                    st.write(f"- **Nodes Required for Training**: {mi300x_training_gpu.get('num_nodes_needed')}")
        else:
            st.error("Attention Heads Not Evenly Divisible - Tensor Parallelism Config Error.")
            
        
    with col2:
        st.header("H100 (80GB)")
        st.image(nvidia_logo)
        h100_inference_gpu = llm_memory_GPU_distribution(standard_inference_total_memory_gb, 80, 8, attention_heads, tensor_parallelism)
        h100_training_gpu = llm_memory_GPU_distribution(standard_training_total_memory_gb, 80, 8, attention_heads, tensor_parallelism)
        
        if h100_inference_gpu.get('attention_heads_divisible'):
        
            if h100_inference_gpu.get("fits_on_one_gpu"):
                st.success("Model Fits on Single GPU for Inference")
            else:
                st.warning("Model Requires more than one GPU for Inference")
                st.write(f"- **GPUs Required for Inference**: {h100_inference_gpu.get('num_gpus_needed')}")
                if h100_inference_gpu.get('num_gpus_needed') > 8:
                    st.write(f"- **Nodes Required for Inference**: {h100_inference_gpu.get('num_nodes_needed')}")
        else:
            st.error("Attention Heads Not Evenly Divisible - Tensor Parallelism Config Error.")

        
        if h100_training_gpu.get('attention_heads_divisible'):
            if h100_training_gpu.get("fits_on_one_gpu"):
                st.success("Model Fits on Single GPU for Training")
            else:
                st.warning("Model Requires more than one GPU for Training")
                st.write(f"- **GPUs Required for Training**: {h100_training_gpu.get('num_gpus_needed')}")
                if h100_training_gpu.get('num_gpus_needed') > 8:
                    st.write(f"- **Nodes Required for Training**: {h100_training_gpu.get('num_nodes_needed')}")
        else:
            st.error("Attention Heads Not Evenly Divisible - Tensor Parallelism Config Error.")


    with col3:
        st.header("MI325X (256GB)")
        st.image(amd_logo)
        mi300x_inference_gpu = llm_memory_GPU_distribution(standard_inference_total_memory_gb, 256, 8, attention_heads, tensor_parallelism)
        mi300x_training_gpu = llm_memory_GPU_distribution(standard_training_total_memory_gb, 256, 8, attention_heads, tensor_parallelism)

        if mi300x_inference_gpu.get('attention_heads_divisible'):
            if mi300x_inference_gpu.get("fits_on_one_gpu"):
                st.success("Model Fits on Single GPU for Inference")
            else:
                st.warning("Model Requires more than one GPU for Inference")
                st.write(f"- **GPUs Required for Inference**: {mi300x_inference_gpu.get('num_gpus_needed')}")
                if mi300x_inference_gpu.get('num_gpus_needed') > 8:
                    st.write(f"- **Nodes Required for Inference**: {mi300x_inference_gpu.get('num_nodes_needed')}")
        else:
            st.error("Attention Heads Not Evenly Divisible - Tensor Parallelism Config Error.")

        
        if mi300x_training_gpu.get('attention_heads_divisible'):
            if mi300x_training_gpu.get("fits_on_one_gpu"):
                st.success("Model Fits on Single GPU for Training")
            else:
                st.warning("Model Requires more than one GPU for Training")
                st.write(f"- **GPUs Required for Training**: {mi300x_training_gpu.get('num_gpus_needed')}")
                if mi300x_inference_gpu.get('num_gpus_needed') > 8:
                    st.write(f"- **Nodes Required for Training**: {mi300x_training_gpu.get('num_nodes_needed')}")
        else:
            st.error("Attention Heads Not Evenly Divisible - Tensor Parallelism Config Error.")
    
    with col4:
        st.header("H200 (141GB)")
        st.image(nvidia_logo)
        mi300x_inference_gpu = llm_memory_GPU_distribution(standard_inference_total_memory_gb, 141, 8, attention_heads, tensor_parallelism)
        mi300x_training_gpu = llm_memory_GPU_distribution(standard_training_total_memory_gb, 141, 8, attention_heads, tensor_parallelism)

        if mi300x_inference_gpu.get('attention_heads_divisible'):
            if mi300x_inference_gpu.get("fits_on_one_gpu"):
                st.success("Model Fits on Single GPU for Inference")
            else:
                st.warning("Model Requires more than one GPU for Inference")
                st.write(f"- **GPUs Required for Inference**: {mi300x_inference_gpu.get('num_gpus_needed')}")
                if mi300x_inference_gpu.get('num_gpus_needed') > 8:
                    st.write(f"- **Nodes Required for Inference**: {mi300x_inference_gpu.get('num_nodes_needed')}")
        else:
            st.error("Attention Heads Not Evenly Divisible - Tensor Parallelism Config Error.")

        
        if mi300x_training_gpu.get('attention_heads_divisible'):
            if mi300x_training_gpu.get("fits_on_one_gpu"):
                st.success("Model Fits on Single GPU for Training")
            else:
                st.warning("Model Requires more than one GPU for Training")
                st.write(f"- **GPUs Required for Training**: {mi300x_training_gpu.get('num_gpus_needed')}")
                if mi300x_inference_gpu.get('num_gpus_needed') > 8:
                    st.write(f"- **Nodes Required for Training**: {mi300x_training_gpu.get('num_nodes_needed')}")
        else:
            st.error("Attention Heads Not Evenly Divisible - Tensor Parallelism Config Error.")

    

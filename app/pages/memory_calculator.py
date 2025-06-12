
import streamlit as st
import requests
from PIL import Image
from utils.memory_utils import (
    llm_memory_GPU_distribution,
    calculate_memory,
    recommend_parallelism_strategy,
)


def show_strategy_if_instinct(
    gpu_name: str,
    memory: int,
    inference_data: dict,
    training_data: dict,
    layer_count: int,
    parameters: int,
) -> None:
    """Display parallelism recommendations only for Instinct GPUs."""

    if not gpu_name.startswith("MI"):
        return

    inference_strategy = recommend_parallelism_strategy(
        inference_data["standard_inference_total_memory_gb"],
        inference_data["model_weights_memory"],
        layer_count,
        parameters,
        memory,
    )
    training_strategy = recommend_parallelism_strategy(
        training_data["standard_training_total_memory_gb"],
        training_data["model_weights_memory"],
        layer_count,
        parameters,
        memory,
    )

    st.markdown(
        f"""
        <div style='border:1px solid #ccc;padding:4px;margin-top:4px;border-radius:4px;'>
        <span title='{inference_strategy['reason']}'>{inference_strategy['icon']} <strong>Inference: {inference_strategy['strategy']}</strong></span><br>
        <span title='{training_strategy['reason']}'>{training_strategy['icon']} <strong>Training: {training_strategy['strategy']}</strong></span>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Flag if using endpoint config
backend_endpoint = False

# Load Logos
amd_logo = Image.open("./app/digital_assets/amd.png")
nvidia_logo = Image.open("./app/digital_assets/nvidia.png")

# Model Data Dictionary
model_dictionaries = {
    "Llama 3 70B Instruct": {
        "parameters": 70,
        "hidden_size": 8192,
        "layer_count": 80,
        "attention_heads": 64,
    },
    "Llama 3 8B Instruct": {
        "parameters": 8,
        "hidden_size": 4096,
        "layer_count": 32,
        "attention_heads": 32,
    },
    "Qwen 72B Instruct": {
        "parameters": 72,
        "hidden_size": 8192,
        "layer_count": 80,
        "attention_heads": 64,
    },
    "DeepSeek-R1": {
        "parameters": 671,
        "hidden_size": 7168,
        "layer_count": 61,
        "attention_heads": 128,
    },
    "Llama 3.1 405B": {
        "parameters": 405,
        "hidden_size": 16384,
        "layer_count": 126,
        "attention_heads": 128,
    },
}


Models = list(model_dictionaries.keys()) + ["Custom"]

st.set_page_config(page_title="LLM Memory Calculator", layout="wide")
st.title("📊 LLM Memory Requirement Calculator")
st.write(
    "All total memory requirements include and additional 5% memory buffer to account for minor calculation errors and to reserve compute for background tasks"
)

with st.sidebar:
    st.header("⚙️ Model Configuration")
    prefill_model = st.selectbox("Prefill Model Selection", options=Models)

    # Dynamically set default values
    defaults = model_dictionaries.get(prefill_model, {})

    parameters = st.number_input(
        "Parameters (B)", min_value=1, value=defaults.get("parameters", 1)
    )
    batch_size = st.number_input("Batch Size", min_value=1, value=1)
    precision = st.selectbox("Precision", ["FP16", "8BIT", "4BIT", "FP32"])
    sequence_length = st.number_input(
        "Sequence Length",
        min_value=1,
        value=2048,
        help="Considerations must be made for the context length support intended for training and inference. 2048 is just a placeholder based on Llama 3 limits.",
    )
    hidden_size = st.number_input(
        "Hidden Size", min_value=1, value=defaults.get("hidden_size", 1)
    )
    layer_count = st.number_input(
        "Layer Count", min_value=1, value=defaults.get("layer_count", 1)
    )
    attention_heads = st.number_input(
        "Attention Heads", min_value=1, value=defaults.get("attention_heads", 1)
    )
    tensor_parallelism = st.number_input("Tensor Parallelism", min_value=1, value=1)
    optimizer = st.selectbox(
        "Optimizer", ["AdamW", "Adam", "bitsandbytes_8bit", "SGD-like"]
    )
    percent_trainable_parameters = st.slider(
        "% Trainable Parameters", min_value=1, max_value=100, value=100
    )
    gradient_checkpointing = st.checkbox("Enable Gradient Checkpointing", value=True)

    if st.button("🔍 Calculate Memory Usage"):
        try:
            training_data = calculate_memory(
                parameters,
                batch_size,
                precision,
                sequence_length,
                hidden_size,
                layer_count,
                attention_heads,
                tensor_parallelism,
                optimizer,
                percent_trainable_parameters,
                mode="training",
                gradient_checkpointing=gradient_checkpointing,
            )
            inference_data = calculate_memory(
                parameters,
                batch_size,
                precision,
                sequence_length,
                hidden_size,
                layer_count,
                attention_heads,
                tensor_parallelism,
                optimizer,
                percent_trainable_parameters,
                mode="inference",
                gradient_checkpointing=gradient_checkpointing,
            )

            # Store the results for later use in the session state
            st.session_state["training"] = training_data
            st.session_state["inference"] = inference_data
            st.success("✅ Calculation Complete!")
        except Exception:
            st.warning("Calculation Failed!")

# Display Results
if "training" in st.session_state and "inference" in st.session_state:

    training_data = st.session_state["training"]
    inference_data = st.session_state["inference"]

    inference_tab, training_tab = st.tabs(["🚀 Inference", "🎯 Training"])

    with inference_tab:
        st.metric(
            label="**Total Inference Memory (GB)**",
            value=f"{inference_data['standard_inference_total_memory_gb']:.2f}",
        )
        st.write(
            f"- **Model Weights**: {inference_data['model_weights_memory']:.2f} GB"
        )
        st.write(f"- **KV Cache**: {inference_data['kv_cache_memory']:.2f} GB")
        st.write(
            f"- **Activation Memory**: {inference_data['activations_memory']:.2f} GB"
        )

    with training_tab:
        st.metric(
            label="**Total Training Memory (GB)**",
            value=f"{training_data['standard_training_total_memory_gb']:.2f}",
        )
        st.write(
            f"- **Model Weights**: {training_data['model_weights_memory']:.2f} GB"
        )
        st.write(f"- **KV Cache**: {training_data['kv_cache_memory']:.2f} GB")
        st.write(
            f"- **Activation Memory**: {training_data['activations_memory']:.2f} GB"
        )
        st.write(f"- **Optimizer Memory**: {training_data['optimizer_memory']:.2f} GB")
        st.write(f"- **Gradient Memory**: {training_data['gradient_memory']:.2f} GB")

    st.subheader("🖥️ GPU Requirements")
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    gpu_configs = [
        ("MI300X (192GB)", amd_logo, 192),
        ("H100 (80GB)", nvidia_logo, 80),
        ("MI325X (256GB)", amd_logo, 256),
        ("H200 (141GB)", nvidia_logo, 141),
        ("MI355X (288GB)", amd_logo, 288),
        ("B200 (192GB)", nvidia_logo, 192),
    ]

    for col, (gpu_name, logo, memory) in zip(
        [col1, col2, col3, col4, col5, col6], gpu_configs
    ):
        with col:
            st.image(logo, width=80)
            st.subheader(gpu_name)

            # Get inference/training memory distributions
            inference_gpu = llm_memory_GPU_distribution(
                inference_data["standard_inference_total_memory_gb"],
                memory,
                8,
                attention_heads,
                tensor_parallelism,
            )
            training_gpu = llm_memory_GPU_distribution(
                training_data["standard_training_total_memory_gb"],
                memory,
                8,
                attention_heads,
                tensor_parallelism,
            )


            # Debugging: Ensure the function returns expected data
            # st.write("Debug - Inference GPU:", inference_gpu)
            # st.write("Debug - Training GPU:", training_gpu)

            # Inference Check
            if inference_gpu.get("attention_heads_divisible", False):
                if inference_gpu.get("fits_on_one_gpu", False):
                    st.success("✅ Fits on Single GPU for Inference")
                else:
                    st.warning(
                        f"⚠️ Requires {inference_gpu.get('num_gpus_needed', '?')} GPUs for Inference"
                    )
            else:
                st.error(
                    "❌ Attention Heads Not Evenly Divisible - Check Tensor Parallelism"
                )


            # Training Check
            if training_gpu.get("attention_heads_divisible", False):
                if training_gpu.get("fits_on_one_gpu", False):
                    st.success("✅ Fits on Single GPU for Training")
                else:
                    st.warning(
                        f"⚠️ Requires {training_gpu.get('num_gpus_needed', '?')} GPUs for Training"
                    )
           else:
                st.error(
                    "❌ Attention Heads Not Evenly Divisible - Check Tensor Parallelism"
                )

            # Only show strategy box for Instinct GPUs
            if gpu_name.startswith("MI"):
                show_strategy_if_instinct(
                    gpu_name,
                    memory,
                    inference_data,
                    training_data,
                    layer_count,
                    parameters,
                )

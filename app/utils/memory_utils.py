# Memory Calculation Utilities
from typing import Dict


def llm_memory_GPU_distribution(
    model_memory_gb, gpu_memory_gb, gpus_per_node, attention_heads, tensor_parallelism
):
    """
    Determines whether an LLM can fit on a GPU, how many GPUs and nodes are needed,
    and checks if attention heads are evenly divisible by tensor parallelism.

    Parameters:
    - model_memory_gb (float): The memory required to run the LLM (in GB).
    - gpu_memory_gb (float): The memory capacity of a single GPU (in GB).
    - gpus_per_node (int): Number of GPUs available in a single node.
    - attention_heads (int): The number of attention heads in the model.
    - tensor_parallelism (int): The tensor parallelism value.

    Returns:
    - dict: A dictionary containing whether the model fits in one GPU, how many GPUs are needed,
            how many nodes are needed, and whether attention heads are evenly divisible.
    """

    # Check if attention heads are evenly divisible by tensor parallelism
    if attention_heads % tensor_parallelism != 0:
        return {
            "error": "Attention heads must be evenly divisible by the tensor parallelism value."
        }

    # Determine if model fits on a single GPU
    fits_on_one_gpu = model_memory_gb <= gpu_memory_gb

    # Determine the number of GPUs required
    num_gpus_needed = -(-model_memory_gb // gpu_memory_gb)  # Equivalent to math.ceil()

    # Determine the number of nodes required
    num_nodes_needed = -(-num_gpus_needed // gpus_per_node)  # Equivalent to math.ceil()

    return {
        "fits_on_one_gpu": fits_on_one_gpu,
        "num_gpus_needed": int(num_gpus_needed),
        "num_nodes_needed": int(num_nodes_needed),
        "attention_heads_divisible": True,
    }


def calculate_memory(
    parameters: float,
    batch_size: int,
    precision: str,
    sequence_length: int,
    hidden_size: int,
    layer_count: int,
    attention_heads: int,
    tensor_parallelism: int,
    optimizer: str,
    percent_trainable_parameters: float,
    mode: str = "training",
    gradient_checkpointing: bool = True,
) -> Dict[str, float]:
    """Calculate memory requirements for a given model configuration."""

    precision_dict = {"FP32": 4, "FP16": 2, "8BIT": 1, "4BIT": 0.5}
    optimizer_dict = {"AdamW": 12, "Adam": 8, "bitsandbytes_8bit": 6, "SGD-like": 8}
    parameters_billions = parameters * 1e9
    bytes_to_gb_factor = 1024**3

    # Model weights
    model_weights_memory = (
        parameters_billions * precision_dict[precision]
    ) / bytes_to_gb_factor

    # KV cache
    kv_cache_memory = (
        2
        * batch_size
        * sequence_length
        * layer_count
        * hidden_size
        * precision_dict[precision]
    ) / bytes_to_gb_factor

    # Attention overhead term
    attention_overhead = (5 * sequence_length * attention_heads) / hidden_size

    # Activation memory
    if mode == "inference":
        activations_memory = (
            batch_size
            * sequence_length
            * hidden_size
            * (34 + attention_overhead)
            * precision_dict[precision]
        ) / bytes_to_gb_factor
    else:
        if gradient_checkpointing:
            activations_memory = (
                batch_size
                * sequence_length
                * hidden_size
                * 24
                * precision_dict[precision]
            ) / bytes_to_gb_factor
        else:
            activations_memory = (
                batch_size
                * sequence_length
                * hidden_size
                * (34 + attention_overhead)
                * precision_dict[precision]
                * layer_count
            ) / bytes_to_gb_factor

    # Optimizer memory
    optimizer_memory = (
        optimizer_dict[optimizer] * parameters_billions
    ) / bytes_to_gb_factor

    # Gradient memory (always FP32)
    gradient_memory = (
        parameters_billions * precision_dict["FP32"]
    ) / bytes_to_gb_factor

    if mode == "inference":
        total_memory = (
            model_weights_memory + kv_cache_memory + activations_memory
        ) * 1.05
    else:
        total_memory = (
            model_weights_memory
            + kv_cache_memory
            + activations_memory
            + ((optimizer_memory + gradient_memory) * (percent_trainable_parameters / 100))
        ) * 1.05

    return {
        "model_weights_memory": model_weights_memory,
        "kv_cache_memory": kv_cache_memory,
        "activations_memory": activations_memory,
        "optimizer_memory": optimizer_memory,
        "gradient_memory": gradient_memory,
        "standard_inference_total_memory_gb": total_memory if mode == "inference" else None,
        "standard_training_total_memory_gb": total_memory if mode == "training" else None,
    }

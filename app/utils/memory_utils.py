# Memory Calculation Utilities
from typing import Dict

def llm_memory_GPU_distribution(model_memory_gb, gpu_memory_gb, gpus_per_node, attention_heads, tensor_parallelism):
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
        "attention_heads_divisible": True
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
) -> Dict[str, float]:
    """
    Calculates memory consumption for model training and inference.

    Parameters:
        parameters (float): Number of model parameters in billions.
        batch_size (int): Batch size used for training/inference.
        precision (str): Model precision (FP32, FP16, 8BIT, 4BIT).
        sequence_length (int): Length of input sequences.
        hidden_size (int): Hidden layer size.
        layer_count (int): Number of layers in the model.
        attention_heads (int): Number of attention heads.
        tensor_parallelism (int): Degree of tensor parallelism.
        optimizer (str): Optimizer type (AdamW, Adam, bitsandbytes_8bit, SGD-like).
        percent_trainable_parameters (float): Percentage of parameters that are trainable.

    Returns:
        Dict[str, float]: Dictionary containing various memory consumption values in GB.
    """
    # Memory requirements per precision type (Bytes per parameter)
    precision_dict = {"FP32": 4, "FP16": 2, "8BIT": 1, "4BIT": 0.5}
    optimizer_dict = {"AdamW": 12, "Adam": 8, "bitsandbytes_8bit": 6, "SGD-like": 8}
    parameters_billions = parameters * 1e9  # Convert billions to actual parameter count
    bytes_to_gb_factor = 1024 ** 3  # Factor to convert Bytes to GB

    # Model weight memory
    model_weights_memory = (parameters_billions * precision_dict[precision]) / bytes_to_gb_factor
    
    # KV Cache memory
    kv_cache_memory = (
        2 * batch_size * sequence_length * layer_count * hidden_size * precision_dict[precision]
    ) / bytes_to_gb_factor
    
    # Activations memory
    activations_memory = (
        batch_size * sequence_length * hidden_size * 
        (34 + (5 * sequence_length * attention_heads) / hidden_size) * precision_dict[precision]
    ) / bytes_to_gb_factor
    
    # Optimizer memory
    optimizer_memory = (optimizer_dict[optimizer] * parameters_billions) / bytes_to_gb_factor
    
    # Gradient memory (always stored in FP32 precision)
    gradient_memory = (parameters_billions * precision_dict["FP32"]) / bytes_to_gb_factor
    
    # Total training memory
    standard_training_total_memory_gb = (
        model_weights_memory + kv_cache_memory + activations_memory + 
        ((optimizer_memory + gradient_memory) * (percent_trainable_parameters / 100))
    )
    
    # Total inference memory
    standard_inference_total_memory_gb = model_weights_memory + kv_cache_memory + activations_memory

    return {
        "model_weights_memory": model_weights_memory,
        "kv_cache_memory": kv_cache_memory,
        "activations_memory": activations_memory,
        "optimizer_memory": optimizer_memory,
        "gradient_memory": gradient_memory,
        "standard_inference_total_memory_gb": standard_inference_total_memory_gb,
        "standard_training_total_memory_gb": standard_training_total_memory_gb,
    }

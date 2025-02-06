# Memory Calculation Utilities

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
from fastapi import FastAPI, Form
import uvicorn
from data_model import Memory
import requests

app = FastAPI()


@app.get("/ping")
async def ping():
    """Ping server to determine status

    Returns
    -------
    API response
        response from server on health status
    """
    return {"message": "Server is Running"}


@app.post("/calculate_memory")
async def calculate_memory(payload: Memory):
    # Example memory calculation logic
    precision_dict = {"FP32": 4, "FP16": 2, "8BIT": 1, "4BIT": 0.5}
    optimizer_dict = {"AdamW": 12, "Adam": 8, "bitsandbytes_8bit": 6, "SGD-like": 8}
    parameters_billions = payload.parameters * 1000000000
    bytes_to_gb_factor = 1024**3  # factor to convert Bytes to GBs for Human Readability

    # parsing inputs
    batch_size = payload.batch_size
    precision = payload.precision
    precision = payload.precision
    sequence_length = payload.sequence_length
    hidden_size = payload.hidden_size
    layer_count = payload.layer_count
    attention_heads = payload.attention_heads
    tensor_parallelism = payload.tensor_parallelism
    optimizer = payload.optimizer
    percent_trainable_parameters = payload.percent_trainable_parameters
    gradient_checkpointing = payload.gradient_checkpointing

    model_weights_memory = (
        parameters_billions * precision_dict[precision]
    ) / bytes_to_gb_factor
    kv_cache_memory = (
        2
        * batch_size
        * sequence_length
        * layer_count
        * hidden_size
        * precision_dict[precision]
    ) / bytes_to_gb_factor
    if gradient_checkpointing:
        activations_memory = (
            batch_size * sequence_length * hidden_size * 24 * precision_dict[precision]
        ) / bytes_to_gb_factor
    else:
        activations_memory = (
            batch_size
            * sequence_length
            * hidden_size
            * (34 + (5 * sequence_length * attention_heads) / hidden_size)
            * precision_dict[precision]
            * layer_count
        ) / bytes_to_gb_factor
    # activations_memory = (batch_size * sequence_length * hidden_size * layer_count * (
    #     10 + (24 / tensor_parallelism) + (5 * ((attention_heads * sequence_length) / (hidden_size * tensor_parallelism)))
    # )) / bytes_to_gb_factor
    optimizer_memory = (
        optimizer_dict[optimizer] * parameters_billions
    ) / bytes_to_gb_factor
    gradient_memory = (
        parameters_billions * precision_dict["FP32"]
    ) / bytes_to_gb_factor

    standard_training_total_memory_gb = (
        model_weights_memory
        + kv_cache_memory
        + activations_memory
        + ((optimizer_memory + gradient_memory) * (percent_trainable_parameters / 100))
    ) * 1.05
    standard_inference_total_memory_gb = (
        model_weights_memory + kv_cache_memory + activations_memory
    ) * 1.05

    RESULTS = {
        "model_weights_memory": model_weights_memory,
        "kv_cache_memory": kv_cache_memory,
        "activations_memory": activations_memory,
        "optimizer_memory": optimizer_memory,
        "gradient_memory": gradient_memory,
        "standard_inference_total_memory_gb": standard_inference_total_memory_gb,
        "standard_training_total_memory_gb": standard_training_total_memory_gb,
    }

    return {"msg": "Memory Calculation Complete", "Calculation": RESULTS}


if __name__ == "__main__":
    uvicorn.run("serve:app", host="0.0.0.0", port=8000, log_level="info")

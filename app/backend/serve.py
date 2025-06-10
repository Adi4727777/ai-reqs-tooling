from fastapi import FastAPI, Form
import uvicorn
from data_model import Memory
import requests
from utils.memory_utils import calculate_memory as calculate_memory_util

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
    result = calculate_memory_util(
        payload.parameters,
        payload.batch_size,
        payload.precision,
        payload.sequence_length,
        payload.hidden_size,
        payload.layer_count,
        payload.attention_heads,
        payload.tensor_parallelism,
        payload.optimizer,
        payload.percent_trainable_parameters,
        mode=payload.mode,
        gradient_checkpointing=payload.gradient_checkpointing,
    )

    return {"msg": "Memory Calculation Complete", "Calculation": result}


if __name__ == "__main__":
    uvicorn.run("serve:app", host="0.0.0.0", port=8000, log_level="info")

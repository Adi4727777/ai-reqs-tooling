from pydantic import BaseModel


class Memory(BaseModel):
    parameters: int = None
    batch_size: int = None
    precision: str = None
    sequence_length: int = None
    hidden_size: int = None
    layer_count: int = None
    attention_heads: int = None
    tensor_parallelism: int = None
    optimizer: str = None
    percent_trainable_parameters: int = None
    gradient_checkpointing: bool = True

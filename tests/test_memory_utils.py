import sys
sys.path.append('app')
import pytest

from utils import memory_utils as mu


def test_llm_memory_gpu_distribution_fits():
    res = mu.llm_memory_GPU_distribution(10, 20, 8, 8, 4)
    assert res == {
        "fits_on_one_gpu": True,
        "num_gpus_needed": 1,
        "num_nodes_needed": 1,
        "attention_heads_divisible": True,
    }


def test_llm_memory_gpu_distribution_not_divisible():
    res = mu.llm_memory_GPU_distribution(10, 5, 8, 8, 3)
    assert res == {
        "error": "Attention heads must be evenly divisible by the tensor parallelism value."
    }


def test_calculate_memory_training_values():
    res = mu.calculate_memory(
        parameters=1,
        batch_size=1,
        precision="FP16",
        sequence_length=1,
        hidden_size=1,
        layer_count=1,
        attention_heads=1,
        tensor_parallelism=1,
        optimizer="AdamW",
        percent_trainable_parameters=100,
        mode="training",
        gradient_checkpointing=True,
    )
    assert pytest.approx(res["model_weights_memory"], rel=1e-6) == 1.862645149230957
    assert pytest.approx(res["kv_cache_memory"], rel=1e-6) == 3.725290298461914e-09
    assert pytest.approx(res["standard_training_total_memory_gb"], rel=1e-6) == 17.60199671108276


def test_recommend_parallelism_strategy_paths():
    assert mu.recommend_parallelism_strategy(10, 1, 10, 10, 20)["strategy"] == "Data Parallelism"
    assert mu.recommend_parallelism_strategy(200, 50, 100, 120, 80)["strategy"] == "Hybrid Parallelism (TP+PP)"
    assert mu.recommend_parallelism_strategy(200, 100, 50, 50, 80)["strategy"] == "Tensor Parallelism"
    assert mu.recommend_parallelism_strategy(200, 50, 70, 50, 80)["strategy"] == "Pipeline Parallelism"
    assert mu.recommend_parallelism_strategy(200, 50, 50, 50, 80)["strategy"] == "Tensor Parallelism"

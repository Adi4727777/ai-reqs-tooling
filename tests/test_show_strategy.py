import sys
import ast
sys.path.append('app')
from utils import memory_utils

source = open('app/pages/memory_calculator.py').read()
module = ast.parse(source)
func_src = None
for node in module.body:
    if isinstance(node, ast.FunctionDef) and node.name == 'show_strategy_if_instinct':
        func_src = ast.get_source_segment(source, node)
        break

globals_dict = {
    'recommend_parallelism_strategy': memory_utils.recommend_parallelism_strategy,
    'st': None,
}
exec(func_src, globals_dict)
show_strategy_if_instinct = globals_dict['show_strategy_if_instinct']

class DummySt:
    def __init__(self):
        self.called = False
        self.last = None
    def markdown(self, text, unsafe_allow_html=False):
        self.called = True
        self.last = text


def test_show_strategy_runs_on_instinct(monkeypatch):
    st = DummySt()
    monkeypatch.setitem(globals_dict, 'st', st)
    monkeypatch.setitem(globals_dict, 'recommend_parallelism_strategy', lambda *a, **k: {'strategy': 'Test', 'icon': 'I', 'reason': 'R'})
    show_strategy_if_instinct(
        'MI300X',
        100,
        {'standard_inference_total_memory_gb': 1, 'model_weights_memory': 1},
        {'standard_training_total_memory_gb': 1, 'model_weights_memory': 1},
        1,
        1,
    )
    assert st.called
    assert 'Inference: Test' in st.last and 'Training: Test' in st.last


def test_show_strategy_skips_non_instinct(monkeypatch):
    st = DummySt()
    monkeypatch.setitem(globals_dict, 'st', st)
    monkeypatch.setitem(globals_dict, 'recommend_parallelism_strategy', lambda *a, **k: {'strategy': 'Test', 'icon': 'I', 'reason': 'R'})
    show_strategy_if_instinct(
        'H100',
        100,
        {'standard_inference_total_memory_gb': 1, 'model_weights_memory': 1},
        {'standard_training_total_memory_gb': 1, 'model_weights_memory': 1},
        1,
        1,
    )
    assert not st.called

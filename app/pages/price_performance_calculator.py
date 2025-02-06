import streamlit as st
import pandas as pd

st.title("Price Performance Calculator")

# Create five columns for inputs, adding model names
col0, col1, col2, col3, col4 = st.columns(5)

with col0:
    st.write("Model Names (Optional)")
    model_a_name = st.text_input("Model A Name", placeholder="Model A")
    model_b_name = st.text_input("Model B Name", placeholder="Model B")
    model_c_name = st.text_input("Model C Name", placeholder="Model C")
    model_d_name = st.text_input("Model D Name", placeholder="Model D")
    model_e_name = st.text_input("Model E Name", placeholder="Model E")

with col1:
    st.write("Throughput (tokens/sec)")
    model_a_throughput = st.number_input(label='Model A - Throughput', key='ma-input', min_value=0.0, step=1.0)
    model_b_throughput = st.number_input(label='Model B - Throughput', key='mb-input', min_value=0.0, step=1.0)
    model_c_throughput = st.number_input(label='Model C - Throughput', key='mc-input', min_value=0.0, step=1.0)
    model_d_throughput = st.number_input(label='Model D - Throughput', key='md-input', min_value=0.0, step=1.0)
    model_e_throughput = st.number_input(label='Model E - Throughput', key='me-input', min_value=0.0, step=1.0)

with col2:
    st.write("Model Serving Pricing")
    model_a_serving_pricing = st.number_input(label='Model A - Serving Pricing', key='ma-pricing-input', min_value=0.0, step=0.000001, format="%.6f")
    model_b_serving_pricing = st.number_input(label='Model B - Serving Pricing', key='mb-pricing-input', min_value=0.0, step=0.000001, format="%.6f")
    model_c_serving_pricing = st.number_input(label='Model C - Serving Pricing', key='mc-pricing-input', min_value=0.0, step=0.000001, format="%.6f")
    model_d_serving_pricing = st.number_input(label='Model D - Serving Pricing', key='md-pricing-input', min_value=0.0, step=0.000001, format="%.6f")
    model_e_serving_pricing = st.number_input(label='Model E - Serving Pricing', key='me-pricing-input', min_value=0.0, step=0.000001, format="%.6f")

with col3:
    st.write("Hardware Costs")
    a100_hourly_costs = st.number_input(label='A100 Hourly Costs', key='a100 cost', min_value=0.0, step=0.1)
    h100_hourly_costs = st.number_input(label='H100 Hourly Costs', key='h100 cost', min_value=0.0, step=0.1)
    h200_hourly_costs = st.number_input(label='H200 Hourly Costs', key='h200 cost', min_value=0.0, step=0.1)
    MI300X_hourly_costs = st.number_input(label='MI300X Hourly Costs', key='mi300x cost', min_value=0.0, step=0.1)
    MI325X_hourly_costs = st.number_input(label='MI325X Hourly Costs', key='mi325x cost', min_value=0.0, step=0.1)

with col4:
    st.write("Test Configuration")
    prompts = st.number_input(label='Number of Prompts', key='prompts', min_value=1, step=1)
    input_sequence_length = st.number_input(label='Input Sequence Length', key='isl', min_value=0, step=1)
    output_sequence_length = st.number_input(label='Output Sequence Length', key='osl', min_value=0, step=1)

st.write("----")

# Function to compute cost metrics for a given model.
def calculate_costs(throughput, hardware_cost, input_seq, output_seq):
    if throughput <= 0:
        return None, None, None  # Avoid division by zero; you could also show a warning.
    cost_per_token = hardware_cost / (throughput * 3600)
    cost_for_1M_tokens = cost_per_token * 1_000_000
    cost_per_prompt = hardware_cost * ((input_seq + output_seq) / throughput) / 3600
    return cost_for_1M_tokens, cost_per_token, cost_per_prompt

# Calculate metrics for each model.
a_1M, a_token, a_prompt = calculate_costs(model_a_throughput, a100_hourly_costs, input_sequence_length, output_sequence_length)
b_1M, b_token, b_prompt = calculate_costs(model_b_throughput, h100_hourly_costs, input_sequence_length, output_sequence_length)
c_1M, c_token, c_prompt = calculate_costs(model_c_throughput, h200_hourly_costs, input_sequence_length, output_sequence_length)
d_1M, d_token, d_prompt = calculate_costs(model_d_throughput, MI300X_hourly_costs, input_sequence_length, output_sequence_length)
e_1M, e_token, e_prompt = calculate_costs(model_e_throughput, MI325X_hourly_costs, input_sequence_length, output_sequence_length)

# Use custom names if provided, otherwise use default names
model_names = [
    model_a_name if model_a_name else "Model A",
    model_b_name if model_b_name else "Model B",
    model_c_name if model_c_name else "Model C",
    model_d_name if model_d_name else "Model D",
    model_e_name if model_e_name else "Model E"
]

# Create a summary table.
data = {
    "Model": model_names,
    "Cost for 1M output tokens ($)": [a_1M, b_1M, c_1M, d_1M, e_1M],
    "Cost per token ($)": [a_token, b_token, c_token, d_token, e_token],
    "Cost per prompt ($)": [a_prompt, b_prompt, c_prompt, d_prompt, e_prompt],
    "Serving Price ($ per token)": [model_a_serving_pricing, model_b_serving_pricing, model_c_serving_pricing, model_d_serving_pricing, model_e_serving_pricing]
}

df = pd.DataFrame(data)

# Apply styling
def highlight_min(s):
    """Highlight the minimum value in green and maximum in red."""
    if s.name in ["Cost per token ($)", "Cost for 1M output tokens ($)", "Cost per prompt ($)"]:
        min_val = s.min()
        max_val = s.max()
        return ['background-color: #8FBC8F' if v == min_val else ('background-color: #FF6F61' if v == max_val else '') for v in s]
    return [''] * len(s)

styled_df = df.style.format(
    {
        "Cost for 1M output tokens ($)": "{:,.2f}",
        "Cost per token ($)": "{:.8f}",
        "Cost per prompt ($)": "{:,.6f}",
        "Serving Price ($ per token)": "{:.6f}"
    }
).apply(highlight_min, subset=["Cost for 1M output tokens ($)", "Cost per token ($)", "Cost per prompt ($)"])

st.subheader("Cost Analysis")
st.dataframe(styled_df)

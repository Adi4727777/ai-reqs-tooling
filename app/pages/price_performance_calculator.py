import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

st.title("Price Performance Calculator")

# Create five columns for inputs, adding model names
col0, col1 = st.columns(2)

with col0:
    st.write("Hardware Costs - per 8x Node System ☁️")
    H100_hourly_costs = st.number_input(label='H100 Hourly Costs', key='h100 cost', min_value=0.0, step=0.1, value=80.0)
    H200_hourly_costs = st.number_input(label='H200 Hourly Costs', key='h200 cost', min_value=0.0, step=0.1, value=80.0)
    MI300x_hourly_costs = st.number_input(label='MI300x Hourly Costs', key='MI300x cost', min_value=0.0, step=0.1, value=48.0)
    MI325x_hourly_costs = st.number_input(label='MI325x Hourly Costs', key='MI325x cost', min_value=0.0, step=0.1)

with col1:
    st.write("Test Configuration 🧪")
    prompts = st.number_input(label='Number of Prompts', key='prompts', min_value=1, step=1, value=8)
    input_sequence_length = st.number_input(label='Input Sequence Length', key='isl', min_value=0, step=1, value=128)
    output_sequence_length = st.number_input(label='Output Sequence Length', key='osl', min_value=0, step=1, value=128)

st.write("----")
st.subheader("Model Configurations🛠️")
mcol0, mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(6)

with mcol0:
    model_a_name = st.text_input("Model A Name", placeholder="Model A")
    model_a_throughput_MI300X = st.number_input(label='MI300X Output Token Throughput (tk/s)', key='ma11-input', min_value=0.0, step=1.0, value=1.0)
    model_a_throughput_MI325X = st.number_input(label='MI325X Output Token Throughput (tk/s)', key='ma12-input', min_value=0.0, step=1.0, value=1.0)
    model_a_throughput_H100 = st.number_input(label='H100 Output Token Throughput (tk/s)', key='ma13-input', min_value=0.0, step=1.0, value=1.0)
    model_a_throughput_H200 = st.number_input(label='H200 Output Token Throughput (tk/s)', key='ma14-input', min_value=0.0, step=1.0, value=1.0)
    model_a_serving_pricing = st.number_input(label='Model A - Serving Pricing', key='ma-pricing-input', min_value=0.0, step=0.000001, format="%.6f")

with mcol1:
    st.markdown("<div style='border-left: 2px solid black; height: 100vh'></div>", unsafe_allow_html=True)

with mcol2:
    model_b_name = st.text_input("Model B Name", placeholder="Model B")
    model_b_throughput_MI300X = st.number_input(label='MI300X Output Token Throughput (tk/s)', key='ma21-input', min_value=0.0, step=1.0, value=1.0)
    model_b_throughput_MI325X = st.number_input(label='MI325X Output Token Throughput (tk/s)', key='ma22-input', min_value=0.0, step=1.0, value=1.0)
    model_b_throughput_H100 = st.number_input(label='H100 Output Token Throughput (tk/s)', key='ma23-input', min_value=0.0, step=1.0, value=1.0)
    model_b_throughput_H200 = st.number_input(label='H200 Output Token Throughput (tk/s)', key='ma24-input', min_value=0.0, step=1.0, value=1.0)
    model_b_serving_pricing = st.number_input(label='Model B - Serving Pricing', key='mb-pricing-input', min_value=0.0, step=0.000001, format="%.6f")
    st.markdown("<div style='border-left: 2px solid black; height: 100vh'></div>", unsafe_allow_html=True)

with mcol3:
    model_c_name = st.text_input("Model C Name", placeholder="Model C")
    model_c_throughput_MI300X = st.number_input(label='MI300X Output Token Throughput (tk/s)', key='ma31-input', min_value=0.0, step=1.0, value=1.0)
    model_c_throughput_MI325X = st.number_input(label='MI325X Output Token Throughput (tk/s)', key='ma32-input', min_value=0.0, step=1.0, value=1.0)
    model_c_throughput_H100 = st.number_input(label='H100 Output Token Throughput (tk/s)', key='ma33-input', min_value=0.0, step=1.0, value=1.0)
    model_c_throughput_H200 = st.number_input(label='H200 Output Token Throughput (tk/s)', key='ma34-input', min_value=0.0, step=1.0, value=1.0)
    model_c_serving_pricing = st.number_input(label='Model C - Serving Pricing', key='mc-pricing-input', min_value=0.0, step=0.000001, format="%.6f")

with mcol4:
    model_d_name = st.text_input("Model D Name", placeholder="Model D")
    model_d_throughput_MI300X = st.number_input(label='MI300X Output Token Throughput (tk/s)', key='ma41-input', min_value=0.0, step=1.0, value=1.0)
    model_d_throughput_MI325X = st.number_input(label='MI325X Output Token Throughput (tk/s)', key='ma42-input', min_value=0.0, step=1.0, value=1.0)
    model_d_throughput_H100 = st.number_input(label='H100 Output Token Throughput (tk/s)', key='ma43-input', min_value=0.0, step=1.0, value=1.0)
    model_d_throughput_H200 = st.number_input(label='H200 Output Token Throughput (tk/s)', key='ma44-input', min_value=0.0, step=1.0, value=1.0)
    model_d_serving_pricing = st.number_input(label='Model D - Serving Pricing', key='md-pricing-input', min_value=0.0, step=0.000001, format="%.6f")

with mcol5:
    model_e_name = st.text_input("Model E Name", placeholder="Model E")
    model_e_throughput_MI300X = st.number_input(label='MI300X Output Token Throughput (tk/s)', key='ma51-input', min_value=0.0, step=1.0, value=1.0)
    model_e_throughput_MI325X = st.number_input(label='MI325X Output Token Throughput (tk/s)', key='ma52-input', min_value=0.0, step=1.0, value=1.0)
    model_e_throughput_H100 = st.number_input(label='H100 Output Token Throughput (tk/s)', key='ma5-input', min_value=0.0, step=1.0, value=1.0)
    model_e_throughput_H200 = st.number_input(label='H200 Output Token Throughput (tk/s)', key='ma54-input', min_value=0.0, step=1.0, value=1.0)
    model_e_serving_pricing = st.number_input(label='Model E - Serving Pricing', key='me-pricing-input', min_value=0.0, step=0.000001, format="%.6f")

st.write("----")

# Function to compute cost metrics for a given model.
def calculate_costs(throughput, hardware_cost, input_seq, output_seq):
    if throughput <= 0:
        return None, None, None  # Avoid division by zero; you could also show a warning.
    cost_per_token = hardware_cost / (throughput * 3600)
    cost_for_1M_tokens = cost_per_token * 1_000_000
    cost_per_prompt = hardware_cost * ((input_seq + output_seq) / throughput) / 3600
    return cost_for_1M_tokens, cost_per_token, cost_per_prompt

# Calculate metrics for each model and platform

#MI300x
a_1M_MI300x, a_token_MI300x, a_prompt_MI300x = calculate_costs(model_a_throughput_MI300X, MI300x_hourly_costs, input_sequence_length, output_sequence_length)
b_1M_MI300x, b_token_MI300x, b_prompt_MI300x = calculate_costs(model_b_throughput_MI300X, MI300x_hourly_costs, input_sequence_length, output_sequence_length)
c_1M_MI300x, c_token_MI300x, c_prompt_MI300x = calculate_costs(model_c_throughput_MI300X, MI300x_hourly_costs, input_sequence_length, output_sequence_length)
d_1M_MI300x, d_token_MI300x, d_prompt_MI300x = calculate_costs(model_d_throughput_MI300X, MI300x_hourly_costs, input_sequence_length, output_sequence_length)
e_1M_MI300x, e_token_MI300x, e_prompt_MI300x = calculate_costs(model_e_throughput_MI300X, MI300x_hourly_costs, input_sequence_length, output_sequence_length)

#MI325x
a_1M_MI325x, a_token_MI325x, a_prompt_MI325x = calculate_costs(model_a_throughput_MI325X, MI325x_hourly_costs, input_sequence_length, output_sequence_length)
b_1M_MI325x, b_token_MI325x, b_prompt_MI325x = calculate_costs(model_b_throughput_MI325X, MI325x_hourly_costs, input_sequence_length, output_sequence_length)
c_1M_MI325x, c_token_MI325x, c_prompt_MI325x = calculate_costs(model_c_throughput_MI325X, MI325x_hourly_costs, input_sequence_length, output_sequence_length)
d_1M_MI325x, d_token_MI325x, d_prompt_MI325x = calculate_costs(model_d_throughput_MI325X, MI325x_hourly_costs, input_sequence_length, output_sequence_length)
e_1M_MI325x, e_token_MI325x, e_prompt_MI325x = calculate_costs(model_e_throughput_MI325X, MI325x_hourly_costs, input_sequence_length, output_sequence_length)

#H100
a_1M_H100, a_token_H100, a_prompt_H100 = calculate_costs(model_a_throughput_H100, H100_hourly_costs, input_sequence_length, output_sequence_length)
b_1M_H100, b_token_H100, b_prompt_H100 = calculate_costs(model_b_throughput_H100, H100_hourly_costs, input_sequence_length, output_sequence_length)
c_1M_H100, c_token_H100, c_prompt_H100 = calculate_costs(model_c_throughput_H100, H100_hourly_costs, input_sequence_length, output_sequence_length)
d_1M_H100, d_token_H100, d_prompt_H100 = calculate_costs(model_d_throughput_H100, H100_hourly_costs, input_sequence_length, output_sequence_length)
e_1M_H100, e_token_H100, e_prompt_H100 = calculate_costs(model_e_throughput_H100, H200_hourly_costs, input_sequence_length, output_sequence_length)

#H200
a_1M_H200, a_token_H200, a_prompt_H200 = calculate_costs(model_a_throughput_H200, H200_hourly_costs, input_sequence_length, output_sequence_length)
b_1M_H200, b_token_H200, b_prompt_H200 = calculate_costs(model_b_throughput_H200, H200_hourly_costs, input_sequence_length, output_sequence_length)
c_1M_H200, c_token_H200, c_prompt_H200 = calculate_costs(model_c_throughput_H200, H200_hourly_costs, input_sequence_length, output_sequence_length)
d_1M_H200, d_token_H200, d_prompt_H200 = calculate_costs(model_d_throughput_H200, H200_hourly_costs, input_sequence_length, output_sequence_length)
e_1M_H200, e_token_H200, e_prompt_H200 = calculate_costs(model_e_throughput_H200, H200_hourly_costs, input_sequence_length, output_sequence_length)


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
    "Serving Price ($ per token)": [model_a_serving_pricing, model_b_serving_pricing, model_c_serving_pricing, model_d_serving_pricing, model_e_serving_pricing],
    "MI300x Cost for 1M output tokens ($)": [a_1M_MI300x, b_1M_MI300x, c_1M_MI300x, d_1M_MI300x, e_1M_MI300x],
    "H100 Cost for 1M output tokens ($)": [a_1M_H100, b_1M_H100, c_1M_H100, d_1M_H100, e_1M_H100],
    "MI300x Cost per token ($)": [a_token_MI300x, b_token_MI300x, c_token_MI300x, d_token_MI300x, e_token_MI300x],
    "H100 Cost per token ($)": [a_token_H100, b_token_H100, c_token_H100, d_token_H100, e_token_H100],
    "MI300x Cost per prompt ($)": [a_prompt_MI300x, b_prompt_MI300x, c_prompt_MI300x, d_prompt_MI300x, e_prompt_MI300x],
    "H100 Cost per prompt ($)": [a_prompt_H100, b_prompt_H100, c_prompt_H100, d_prompt_H100, e_prompt_H100],
    "MI325x Cost for 1M output tokens ($)": [a_1M_MI325x, b_1M_MI325x, c_1M_MI325x, d_1M_MI325x, e_1M_MI325x],
    "H200 Cost for 1M output tokens ($)": [a_1M_H200, b_1M_H200, c_1M_H200, d_1M_H200, e_1M_H200],
    "MI325x Cost per token ($)": [a_token_MI325x, b_token_MI325x, c_token_MI325x, d_token_MI325x, e_token_MI325x],
    "H200 Cost per token ($)": [a_token_H200, b_token_H200, c_token_H200, d_token_H200, e_token_H200],
    "MI325x Cost per prompt ($)": [a_prompt_MI325x, b_prompt_MI325x, c_prompt_MI325x, d_prompt_MI325x, e_prompt_MI325x],
    "H200 Cost per prompt ($)": [a_prompt_H200, b_prompt_H200, c_prompt_H200, d_prompt_H200, e_prompt_H200],
    
}

df = pd.DataFrame(data)

styled_df = df.style.format(
    {
        "Cost for 1M output tokens ($)": "{:,.2f}",
        "Cost per token ($)": "{:.8f}",
        "Cost per prompt ($)": "{:,.6f}",
        "Serving Price ($ per token)": "{:.6f}"
    }
)

st.subheader("Inference Price/Performance Analysis💰")
st.dataframe(styled_df)


st.write("----")

# Existing code... (keep everything you have already implemented)

st.subheader("Visualization of Inference Price/Performance Metrics 📊")

# Allow user to select metrics to visualize
selected_metrics = []
metrics = {
    "Serving Price ($ per token)": "Serving Price ($ per token)",
    "MI300x Cost for 1M output tokens ($)": "MI300x Cost for 1M output tokens ($)",
    "H100 Cost for 1M output tokens ($)": "H100 Cost for 1M output tokens ($)",
    "MI300x Cost per token ($)": "MI300x Cost per token ($)",
    "H100 Cost per token ($)": "H100 Cost per token ($)",
    "MI300x Cost per prompt ($)": "MI300x Cost per prompt ($)",
    "H100 Cost per prompt ($)": "H100 Cost per prompt ($)",
    "MI325x Cost for 1M output tokens ($)": "MI325x Cost for 1M output tokens ($)",
    "H200 Cost for 1M output tokens ($)": "H200 Cost for 1M output tokens ($)",
    "MI325x Cost per token ($)": "MI325x Cost per token ($)",
    "H200 Cost per token ($)": "H200 Cost per token ($)",
    "MI325x Cost per prompt ($)": "MI325x Cost per prompt ($)",
    "H200 Cost per prompt ($)": "H200 Cost per prompt ($)"
}

col1, col2 = st.columns(2)

with col1:
    for key in list(metrics.keys())[:len(metrics)//2]:
        if st.checkbox(key, value=False):
            selected_metrics.append(metrics[key])

with col2:
    for key in list(metrics.keys())[len(metrics)//2:]:
        if st.checkbox(key, value=False):
            selected_metrics.append(metrics[key])

# Generate interactive bar chart if any metric is selected
if selected_metrics:
    st.write("### Selected Metrics Visualization")
    
    df_selected = df.set_index("Model")[selected_metrics]  # Filter only selected columns
    df_melted = df_selected.reset_index().melt(id_vars=["Model"], var_name="Metric", value_name="Cost")
    
    fig = px.bar(df_melted, x="Model", y="Cost", color="Metric", 
                 barmode="group", text_auto=True,
                 title="InferencePrice/Performance Analysis",
                 labels={"Cost": "Cost ($)", "Model": "Models"},
                 hover_name="Metric")
    
    fig.update_layout(
        template="plotly_dark",
        title_font_size=18,
        xaxis_title_font_size=14,
        yaxis_title_font_size=14,
        legend_title_font_size=14,
        hoverlabel_font_size=12,
        bargap=0.2,
        barmode="group",
        plot_bgcolor="#1E1E1E",
        paper_bgcolor="#1E1E1E",
        font_color="white"
    )
    
    st.plotly_chart(fig)
else:
    st.write("Select metrics to generate a chart.")

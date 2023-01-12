import streamlit as st
import json
import requests
import pandas as pd
from st_aggrid import AgGrid, ColumnsAutoSizeMode, GridOptionsBuilder
import numpy as np

st.set_page_config(layout="wide")

if True:
    hide_streamlit_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                </style>
                """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Remove whitespace from the top of the page and sidebar
st.markdown("""
        <style>
            .css-18e3th9 {
                padding-top: 0rem;
                padding-bottom: 10rem;
                padding-left: 5rem;
                padding-right: 5rem;
            }
            .css-1kyxreq.etr89bj2 {
                margin-top: -50px;
            }
        </style>
        """, unsafe_allow_html=True)

st.title("Fracture Detection")

# Take user inputs
with st.sidebar:
    st.image("img/fpfn_logo_700_crop.png")

    st.header("Matching")
    with st.expander("Matching parameters"):
        min_iou = st.slider("min_iou", 0.0, 1.0, 0.2)
        threshold = st.slider("threshold", 0.0, 1.0, 0.5)

    st.header("Filters")
    with st.expander("Study filters"):
        st.write("Placeholder")

    with st.expander("Image filters"):
        # List all available fields and let user select which ones to filter for

        #  @st.experimental_memo(show_spinner=False)
        def post_describe_metadata(url):
            return requests.post(url=url)

        describe_metadata_result = post_describe_metadata("http://127.0.0.1:8000/describe_metadata")
        describe_metadata_data = describe_metadata_result.json()
        metadata_boolean_expression = ""
        for key, values in describe_metadata_data["str"].items():
            selected_options = st.multiselect(f"{key}", values)
            if len(selected_options) > 0:
                if len(metadata_boolean_expression) > 0:
                    metadata_boolean_expression += " and "
                metadata_boolean_expression += f"image.metadata['{key}'] in {selected_options}"
        for key, values in describe_metadata_data["float"].items():
            col1, col2 = st.columns(2)
            with col1:
                selected_min_value = st.number_input(f"Min {key}", values["min_value"], values["max_value"], values["min_value"])
            with col2:
                selected_max_value = st.number_input(f"Max {key}", values["min_value"], values["max_value"], values["max_value"])
            if selected_min_value > values["min_value"]:
                if len(metadata_boolean_expression) > 0:
                    metadata_boolean_expression += " and "
                metadata_boolean_expression += f"image.metadata['{key}'] >= {selected_min_value}"
            if selected_max_value < values["max_value"]:
                if len(metadata_boolean_expression) > 0:
                    metadata_boolean_expression += " and "
                metadata_boolean_expression += f"image.metadata['{key}'] <= {selected_max_value}"
        if metadata_boolean_expression == "":
            metadata_boolean_expression = "True"

    with st.expander("GT box filters"):
        st.write("Placeholder")

    st.header("Dimensions")
    available_str_dimensions = list(describe_metadata_data["str"].keys())
    available_float_dimensions = list(describe_metadata_data["float"].keys())
    available_dimensions = available_str_dimensions + available_float_dimensions
    selected_dimension_1 = st.selectbox("Dimension 1", [None] + available_dimensions)
    selected_dimension_1_type = "str" if selected_dimension_1 in available_str_dimensions else "float"
    selected_dimension_2 = st.selectbox("Dimension 2", [None] + available_dimensions)
    selected_dimension_2_type = "str" if selected_dimension_2 in available_str_dimensions else "float"

    st.header("Other")
    with st.expander("Metrics"):
        available_metrics = ["tp_rate", "fps_per_image", "n_tp", "n_fn", "n_fp", "n_images", "n_positive_images", "n_negative_images"]
        selected_metrics = {}
        for metric in available_metrics:
            selected_metrics[metric] = st.checkbox(metric, True)

    with st.expander("Float field buckets"):
        for key in describe_metadata_data["float"].keys():
            min_value = describe_metadata_data["float"][key]["min_value"]
            max_value = describe_metadata_data["float"][key]["max_value"]
            default_edges = ", ".join([str(number) for number in np.round(np.arange(min_value, max_value + 0.01, (max_value - min_value) / 10), 2)])
            selected_edges = st.text_input(f"{key} edges", default_edges)
            describe_metadata_data["float"][key]["selected_edges"] = selected_edges

# Convert inputs to json format
inputs = {
    "metadata_boolean_expression": metadata_boolean_expression,
    "min_iou": min_iou,
    "threshold": threshold
}

if selected_dimension_1:
    inputs["dimension_1_name"] = selected_dimension_1
    inputs["dimension_1_type"] = selected_dimension_1_type
    if selected_dimension_1_type == "str":
        inputs["dimension_1_values"] = list(describe_metadata_data["str"][selected_dimension_1])
    else:
        inputs["dimension_1_values"] = [float(value) for value in describe_metadata_data["float"][selected_dimension_1]["selected_edges"].split(', ')]
if selected_dimension_2 and (selected_dimension_1 != selected_dimension_2):
    inputs["dimension_2_name"] = selected_dimension_2
    inputs["dimension_2_type"] = selected_dimension_2_type
    if selected_dimension_2_type == "str":
        inputs["dimension_2_values"] = list(describe_metadata_data["str"][selected_dimension_2])
    else:
        inputs["dimension_2_values"] = [float(value) for value in describe_metadata_data["float"][selected_dimension_2]["selected_edges"].split(', ')]
data = json.dumps(inputs)

# Make a request to the FastAPI

#  @st.experimental_memo(show_spinner=False)
def post_report(url, data):
    return requests.post(url=url, data=data)

report_result = post_report("http://127.0.0.1:8000/report", data)
report_data = report_result.json()
report_dataframe = pd.DataFrame(report_data)
for metric in selected_metrics:
    if selected_metrics[metric] is False:
        report_dataframe.drop(metric, axis=1, inplace=True)

# Print report
filter_text = f"**Study filter:** None" + "  \n" + \
    f"**Image filter:** {'None' if metadata_boolean_expression == 'True' else metadata_boolean_expression}" + "  \n" \
    f"**GT box filter:** None"
st.write(filter_text)

options_builder = GridOptionsBuilder.from_dataframe(report_dataframe)
if selected_metrics["tp_rate"]:
    options_builder.configure_column("tp_rate", type=["numericColumn", "numberColumnFilter", "customNumericFormat"], precision=1)
if selected_metrics["fps_per_image"]:
    options_builder.configure_column("fps_per_image", type=["numericColumn", "numberColumnFilter", "customNumericFormat"], precision=3)
# options_builder.configure_grid_options(masterDetail=True, detailRowAutoHeight=True)
grid_options = options_builder.build()
AgGrid(report_dataframe, gridOptions=grid_options, columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS)



# st.header(f"Dataset name: {report_data['dataset_name']}")
# st.write(f"**Images:** {report_data['n_images_filtered']} out of {report_data['n_images']}")
# st.write(f"**TP-rate:** {report_data['tp_rate'] * 100:.1f}%")
# st.write(f"**FPs per image:** {report_data['fps_per_image']:.3f}")
# st.write(f"**Number of TPs:** {report_data['n_tp']}")
# st.write(f"**Number of FNs:** {report_data['n_fn']}")
# st.write(f"**Number of FPs:** {report_data['n_fp']}")
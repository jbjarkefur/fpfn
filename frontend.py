import streamlit as st
import json
import requests
import pandas as pd
from st_aggrid import AgGrid, ColumnsAutoSizeMode, GridOptionsBuilder
import numpy as np
import plotly.express as px
from skimage import io

st.set_page_config(layout="wide")

def ceil_with_decimals(a, decimals=0):
    return np.true_divide(np.ceil(a * 10**decimals), 10**decimals)

def floor_with_decimals(a, decimals=0):
    return np.true_divide(np.floor(a * 10**decimals), 10**decimals)

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
            .css-zt5igj.e16nr0p33 {
                margin-top: 0px;
            }
            .css-1kyxreq.etr89bj2 {
                margin-top: -50px;
            }
        </style>
        """, unsafe_allow_html=True)

# Take user inputs
with st.sidebar:
    st.image("img/fpfn_logo_700_crop.png")

    st.header("Matching")
    with st.expander("Matching parameters"):
        min_iou = st.slider("min_iou", 0.0, 1.0, 0.2)
        threshold = st.slider("threshold", 0.0, 1.0, 0.5)
    
    def filtering(level: str, metadata_description: dict) -> str:
        metadata_boolean_expression = ""
        for key, values in metadata_description[level]["str"].items():
            selected_options = st.multiselect(f"{key}", values)
            if len(selected_options) > 0:
                if len(metadata_boolean_expression) > 0:
                    metadata_boolean_expression += " and "
                metadata_boolean_expression += f"{level}.metadata['{key}'] in {selected_options}"
        for key, values in metadata_description[level]["int"].items():
            col1, col2 = st.columns(2)
            min_value = values["min_value"]
            max_value = values["max_value"]
            with col1:
                selected_min_value = st.number_input(f"Min {key}", min_value, max_value, min_value, step=1)
            with col2:
                selected_max_value = st.number_input(f"Max {key}", min_value, max_value, max_value, step=1)
            if selected_min_value > min_value:
                if len(metadata_boolean_expression) > 0:
                    metadata_boolean_expression += " and "
                metadata_boolean_expression += f"{level}.metadata['{key}'] >= {selected_min_value}"
            if selected_max_value < max_value:
                if len(metadata_boolean_expression) > 0:
                    metadata_boolean_expression += " and "
                metadata_boolean_expression += f"{level}.metadata['{key}'] <= {selected_max_value}"
        for key, values in metadata_description[level]["float"].items():
            col1, col2 = st.columns(2)
            min_value = floor_with_decimals(values["min_value"], 2)
            max_value = ceil_with_decimals(values["max_value"], 2)
            with col1:
                selected_min_value = st.number_input(f"Min {key}", min_value, max_value, min_value, step=0.01)
            with col2:
                selected_max_value = st.number_input(f"Max {key}", min_value, max_value, max_value, step=0.01)
            if selected_min_value > min_value:
                if len(metadata_boolean_expression) > 0:
                    metadata_boolean_expression += " and "
                metadata_boolean_expression += f"{level}.metadata['{key}'] >= {selected_min_value}"
            if selected_max_value < max_value:
                if len(metadata_boolean_expression) > 0:
                    metadata_boolean_expression += " and "
                metadata_boolean_expression += f"{level}.metadata['{key}'] <= {selected_max_value}"
        if metadata_boolean_expression == "":
            metadata_boolean_expression = "True"
        return metadata_boolean_expression

    #  @st.experimental_memo(show_spinner=False)
    def post_describe_metadata(url):
        return requests.post(url=url)

    describe_metadata_result = post_describe_metadata("http://localhost:8000/describe_metadata")
    describe_metadata_data = describe_metadata_result.json()

    st.header("Filters")
    with st.expander("Study filters"):
        study_boolean_expression = filtering("study", describe_metadata_data)

    with st.expander("Image filters"):
        image_boolean_expression = filtering("image", describe_metadata_data)

    with st.expander("GT box filters"):
        st.write("Placeholder")

    st.header("Dimensions")
    available_study_str_dimensions = ["Study " + dimension for dimension in describe_metadata_data["study"]["str"].keys()]
    available_study_int_dimensions = ["Study " + dimension + "_bucket" for dimension in describe_metadata_data["study"]["int"].keys()]
    available_study_float_dimensions = ["Study " + dimension + "_bucket" for dimension in describe_metadata_data["study"]["float"].keys()]
    available_image_str_dimensions = ["Image " + dimension for dimension in describe_metadata_data["image"]["str"].keys()]
    available_image_int_dimensions = ["Image " + dimension + "_bucket" for dimension in describe_metadata_data["image"]["int"].keys()]
    available_image_float_dimensions = ["Image " + dimension + "_bucket" for dimension in describe_metadata_data["image"]["float"].keys()]

    available_dimensions = available_study_str_dimensions + available_study_int_dimensions + available_study_float_dimensions + available_image_str_dimensions + available_image_int_dimensions + available_image_float_dimensions
    selected_dimension_1 = st.selectbox("Dimension 1", [None] + available_dimensions)
    selected_dimension_2 = st.selectbox("Dimension 2", [None] + available_dimensions)

    available_study_dimensions = available_study_str_dimensions + available_study_int_dimensions + available_study_float_dimensions
    selected_dimension_1_level = "study" if selected_dimension_1 in available_study_dimensions else "image"
    selected_dimension_2_level = "study" if selected_dimension_2 in available_study_dimensions else "image"

    def get_dimension_type_from_name(selected_dimension):
        available_str_dimensions = available_study_str_dimensions + available_image_str_dimensions
        available_int_dimensions = available_study_int_dimensions + available_image_int_dimensions

        if selected_dimension in available_str_dimensions:
            dimension_type = "str"
        elif selected_dimension in available_int_dimensions:
            dimension_type = "int"
        else:
            dimension_type = "float"

        return dimension_type

    selected_dimension_1_type = get_dimension_type_from_name(selected_dimension_1)
    selected_dimension_2_type = get_dimension_type_from_name(selected_dimension_2)

    if selected_dimension_1 is not None:
        selected_dimension_1_name = selected_dimension_1.replace("Study ", "").replace("Image ", "").replace("_bucket", "")
    if selected_dimension_2 is not None:
        selected_dimension_2_name = selected_dimension_2.replace("Study ", "").replace("Image ", "").replace("_bucket", "")

    st.header("Visualization")
    col1, col2, col3 = st.columns(3)
    with col1:
        min_image_tp = st.number_input("Min number of TP per image", 0, 9, 0)
    with col2:
        min_image_fn = st.number_input("Min number of FN per image", 0, 9, 0)
    with col3:
        min_image_fp = st.number_input("Min number of FP per image", 0, 9, 0)
    always_show_images_for_first_row = st.checkbox("Always show images for first row", False)

    st.header("Other")
    with st.expander("Metrics"):
        available_metrics = ["tp_rate", "fps_per_image", "n_tp", "n_fn", "n_fp", "n_images", "n_positive_images", "n_negative_images"]
        selected_metrics = {}
        for metric in available_metrics:
            selected_metrics[metric] = st.checkbox(metric, True)

    with st.expander("Field buckets"):

        def _get_default_edges_str_for_int_field(field_description: dict) -> str:
            min_value = field_description["min_value"]
            max_value = field_description["max_value"]
            default_edges = np.linspace(min_value, max_value, 11)
            default_edges_round = [int(default_edge) for default_edge in default_edges]
            default_edges_str = ", ".join([str(default_edge) for default_edge in default_edges_round])
            return default_edges_str

        def _get_default_edges_str_for_float_field(field_description: dict) -> str:
            min_value = field_description["min_value"]
            max_value = field_description["max_value"]
            default_edges = np.linspace(min_value, max_value, 11)
            default_edges_round = [floor_with_decimals(default_edge, 2) for default_edge in default_edges[:-1]] + [ceil_with_decimals(default_edges[-1], 2)]
            default_edges_str = ", ".join([str(default_edge) for default_edge in default_edges_round])
            return default_edges_str

        st.subheader("Study")
        for field_name, field_description in describe_metadata_data["study"]["int"].items():
            default_edges = _get_default_edges_str_for_int_field(field_description)
            selected_edges = st.text_input(f"{field_name} edges (int values)", default_edges)
            describe_metadata_data["study"]["int"][field_name]["selected_edges"] = selected_edges
        for field_name, field_description in describe_metadata_data["study"]["float"].items():
            default_edges = _get_default_edges_str_for_float_field(field_description)
            selected_edges = st.text_input(f"{field_name} edges (float values)", default_edges)
            describe_metadata_data["study"]["float"][field_name]["selected_edges"] = selected_edges

        st.subheader("Image")
        for field_name, field_description in describe_metadata_data["image"]["int"].items():
            default_edges = _get_default_edges_str_for_int_field(field_description)
            selected_edges = st.text_input(f"{field_name} edges (int values)", default_edges)
            describe_metadata_data["image"]["int"][field_name]["selected_edges"] = selected_edges
        for field_name, field_description in describe_metadata_data["image"]["float"].items():
            default_edges = _get_default_edges_str_for_float_field(field_description)
            selected_edges = st.text_input(f"{field_name} edges (float values)", default_edges)
            describe_metadata_data["image"]["float"][field_name]["selected_edges"] = selected_edges

# Convert inputs to json format
inputs = {
    "study_boolean_expression": study_boolean_expression,
    "image_boolean_expression": image_boolean_expression,
    "min_iou": min_iou,
    "threshold": threshold
}

if selected_dimension_1:
    inputs["dimension_1_name"] = selected_dimension_1_name
    inputs["dimension_1_level"] = selected_dimension_1_level
    inputs["dimension_1_type"] = selected_dimension_1_type
    if selected_dimension_1_type == "str":
        inputs["dimension_1_values"] = list(describe_metadata_data[selected_dimension_1_level]["str"][selected_dimension_1_name])
    elif selected_dimension_1_type == "int":
        inputs["dimension_1_values"] = [int(float(value)) for value in describe_metadata_data[selected_dimension_1_level]["int"][selected_dimension_1_name]["selected_edges"].split(', ')]
    else:
        inputs["dimension_1_values"] = [float(value) for value in describe_metadata_data[selected_dimension_1_level]["float"][selected_dimension_1_name]["selected_edges"].split(', ')]
if selected_dimension_2 and (selected_dimension_1 != selected_dimension_2):
    inputs["dimension_2_name"] = selected_dimension_2_name
    inputs["dimension_2_level"] = selected_dimension_2_level
    inputs["dimension_2_type"] = selected_dimension_2_type
    if selected_dimension_2_type == "str":
        inputs["dimension_2_values"] = list(describe_metadata_data[selected_dimension_2_level]["str"][selected_dimension_2_name])
    elif selected_dimension_2_type == "int":
        inputs["dimension_2_values"] = [int(float(value)) for value in describe_metadata_data[selected_dimension_2_level]["int"][selected_dimension_2_name]["selected_edges"].split(', ')]
    else:
        inputs["dimension_2_values"] = [float(value) for value in describe_metadata_data[selected_dimension_2_level]["float"][selected_dimension_2_name]["selected_edges"].split(', ')]
data = json.dumps(inputs)

filter_text = f"**Study filter:** {'None' if study_boolean_expression == 'True' else study_boolean_expression}" + ", " \
    f"**Image filter:** {'None' if image_boolean_expression == 'True' else image_boolean_expression}" + ", " \
    f"**GT box filter:** None"
st.write(filter_text)

#  @st.experimental_memo(show_spinner=False)
def post_report(url, data):
    return requests.post(url=url, data=data)

report_result = post_report("http://localhost:8000/report", data)
report_data = report_result.json()
report_dataframe = pd.DataFrame(report_data)
if len(report_dataframe) > 0:
    for metric in selected_metrics:
        if selected_metrics[metric] is False:
            report_dataframe.drop(metric, axis=1, inplace=True)

    # Print report
    options_builder = GridOptionsBuilder.from_dataframe(report_dataframe)
    if selected_metrics["tp_rate"]:
        options_builder.configure_column("tp_rate", type=["numericColumn", "numberColumnFilter", "customNumericFormat"], precision=1)
    if selected_metrics["fps_per_image"]:
        options_builder.configure_column("fps_per_image", type=["numericColumn", "numberColumnFilter", "customNumericFormat"], precision=3)
    # options_builder.configure_grid_options(masterDetail=True, detailRowAutoHeight=True)
    options_builder.configure_grid_options(minHeight=0, domLayout="autoHeight")
    options_builder.configure_selection(use_checkbox=False)
    if always_show_images_for_first_row:
        options_builder.configure_selection(pre_selected_rows=[0])
    grid_options = options_builder.build()
    aggrid_result = AgGrid(report_dataframe, gridOptions=grid_options, columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS)

    if len(aggrid_result.selected_rows) > 0:

        selected_row_input = {
            "min_image_tp": min_image_tp,
            "min_image_fn": min_image_fn,
            "min_image_fp": min_image_fp,
        }

        if selected_dimension_1 is not None:
            selected_dimension_1_value = aggrid_result.selected_rows[0][selected_dimension_1]
            selected_row_input["dimension_1_name"] = selected_dimension_1_name
            selected_row_input["dimension_1_level"] = selected_dimension_1_level
            selected_row_input["dimension_1_type"] = selected_dimension_1_type
            selected_row_input["dimension_1_value"] = selected_dimension_1_value
        if selected_dimension_2 is not None:
            selected_dimension_2_value = aggrid_result.selected_rows[0][selected_dimension_2]
            selected_row_input["dimension_2_name"] = selected_dimension_2_name
            selected_row_input["dimension_2_level"] = selected_dimension_2_level
            selected_row_input["dimension_2_type"] = selected_dimension_2_type
            selected_row_input["dimension_2_value"] = selected_dimension_2_value

        selected_study_result = post_report("http://localhost:8000/select_study", json.dumps(selected_row_input))
        selected_study = selected_study_result.json()
        images = selected_study

        st.write(f"Images with >= {min_image_tp} TP, >= {min_image_fn} FN and >= {min_image_fp} FP:")

        for row_index in range(2):
            columns = st.columns(4)
            for column_index, column in enumerate(columns):
                image_index = column_index + 4*row_index
                if image_index < len(images):
                    image = images[column_index + 4*row_index]
                    with column:
                        im = io.imread(image["filename"])
                        if len(im.shape) < 3:
                            im = np.repeat(im[:, :, np.newaxis], 3, axis=2)

                        fig = px.imshow(im)
                        fig.update_layout(
                            margin=dict(l=0, r=0, t=0, b=0),
                            hovermode=False,
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),  # , fixedrange=False
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        )

                        gt_line = line=dict(color="green", width=2)
                        pred_line = line=dict(color="orange", width=2)
                        ground_truth_shapes = [{"type": "rect", "x0": gt["x1"], "y0": gt["y1"], "x1": gt["x2"], "y1": gt["y2"], "line": gt_line} for gt in image["ground_truths"]]
                        prediction_shapes = [{"type": "rect", "x0": gt["x1"], "y0": gt["y1"], "x1": gt["x2"], "y1": gt["y2"], "line": pred_line} for gt in image["predictions"]]
                        shapes = ground_truth_shapes + prediction_shapes
                        if len(shapes) > 0:
                            fig.update_layout(shapes=shapes)

                        config = dict({'scrollZoom': True, 'displayModeBar': False})
                        st.plotly_chart(fig, config=config, use_container_width=True)
    else:
        st.write("Click on any row in the table to look at the corresponding images")
else:
    st.write("No matching data, change filters or dimensions")

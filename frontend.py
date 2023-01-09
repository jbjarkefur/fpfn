import streamlit as st
import json
import requests

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

st.title("FPFN.ai")

# Take user inputs
with st.sidebar:
    st.header("Matching")
    min_iou = st.slider("min_iou", 0.0, 1.0, 0.2)
    threshold = st.slider("threshold", 0.0, 1.0, 0.5)

    st.header("Filtering")
    st.subheader("Image")
    # List all available fields and let user select which ones to filter for
    describe_metadata_result = requests.post(url="http://127.0.0.1:8000/describe_metadata")
    describe_metadata_data = describe_metadata_result.json()
    metadata_boolean_expression = ""
    for key, values in describe_metadata_data.items():
        print(values)
        selected_options = st.multiselect(f"{key}", values)
        if len(selected_options) > 0:
            if len(metadata_boolean_expression) > 0:
                metadata_boolean_expression += " and "
            metadata_boolean_expression += f"image.metadata['{key}'] in {selected_options}"
    if metadata_boolean_expression == "":
        metadata_boolean_expression = "True"
    # st.write(f"Image metadata boolean expression: {metadata_boolean_expression}")
    # metadata_boolean_expression = st.text_input("Image metadata boolean expression", "True")

    st.header("Presenting")

# Convert inputs to json format
inputs = {
    "metadata_boolean_expression": metadata_boolean_expression,
    "min_iou": min_iou,
    "threshold": threshold    
}
data = json.dumps(inputs)

# Make a request to the FastAPI
report_result = requests.post(url="http://127.0.0.1:8000/report", data=data)
report_data = report_result.json()

# Print report
st.header(f"Dataset name: {report_data['dataset_name']}")
st.write(f"**Images:** {report_data['n_images_filtered']} out of {report_data['n_images']}")
st.write(f"**TP-rate:** {report_data['tp_rate'] * 100:.1f}%")
st.write(f"**FPs per image:** {report_data['fps_per_image']:.3f}")
st.write(f"**Number of TPs:** {report_data['n_tp']}")
st.write(f"**Number of FNs:** {report_data['n_fn']}")
st.write(f"**Number of FPs:** {report_data['n_fp']}")
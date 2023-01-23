from fastapi import FastAPI
from pydantic import BaseModel

from generate_test_data import generate_test_dataset
from categorizer import match_and_classify, filter_predictions_using_threshold
from describe_metadata import describe
from report import report, filter_dataset, filter_studies_and_images_based_on_level, get_metrics


class Create_report_user_input(BaseModel):
    dimension_1_name: str | None = None
    dimension_1_level: str | None  = None
    dimension_1_type: str | None  = None
    dimension_1_values: list  = [None]
    dimension_2_name: str | None  = None
    dimension_2_level: str | None  = None
    dimension_2_type: str | None  = None
    dimension_2_values: list = [None]
    study_boolean_expression: str = "True"
    image_boolean_expression: str = "True"
    min_iou: float
    threshold: float


class Select_images_user_input(BaseModel):
    dimension_1_name: str | None = None
    dimension_1_level: str | None  = None
    dimension_1_type: str | None  = None
    dimension_1_value: str | None  = None
    dimension_2_name: str | None  = None
    dimension_2_level: str | None  = None
    dimension_2_type: str | None  = None
    dimension_2_value: str | None = None
    min_image_tp: int = 0
    min_image_fn: int = 0
    min_image_fp: int = 0


class Get_images_user_input(BaseModel):
    first_image_idx: int = 0
    last_image_idx: int = 7


app = FastAPI()
dataset = generate_test_dataset(5000)
description = describe(dataset)
metrics = get_metrics()

dataset_thresholded = None
dataset_matched_and_classified = None
dataset_filtered = None

last_create_report_input = None
performance_report = None

last_select_images_input = None
selected_images = None
n_selected_studies = None


@app.post("/describe_metadata")
def describe_metadata():
    return description


@app.post("/describe_metrics")
def describe_metrics():
    return metrics


@app.post("/create_report")
def create_report(input:Create_report_user_input):
    global last_create_report_input
    global performance_report

    global dataset
    global dataset_thresholded
    global dataset_matched_and_classified
    global dataset_filtered

    global selected_images

    recalculate_thresholding = True
    recalculate_matching_and_classification = True
    recalculate_filtering = True
    recalculate_reporting = True

    if last_create_report_input is None or performance_report is None:
        print("Create report: Startup")
    elif input.threshold != last_create_report_input.threshold:
        pass
    elif input.min_iou != last_create_report_input.min_iou:
        recalculate_thresholding = False
    elif input.study_boolean_expression != last_create_report_input.study_boolean_expression or \
        input.image_boolean_expression != last_create_report_input.image_boolean_expression:
        recalculate_thresholding = False
        recalculate_matching_and_classification = False
    elif input.dimension_1_name != last_create_report_input.dimension_1_name or input.dimension_2_name != last_create_report_input.dimension_2_name or \
        input.dimension_1_level != last_create_report_input.dimension_1_level or input.dimension_2_level != last_create_report_input.dimension_2_level or \
        input.dimension_1_type != last_create_report_input.dimension_1_type or input.dimension_2_type != last_create_report_input.dimension_2_type or \
        input.dimension_1_values != last_create_report_input.dimension_1_values or input.dimension_2_values != last_create_report_input.dimension_2_values:
        recalculate_thresholding = False
        recalculate_matching_and_classification = False
        recalculate_filtering = False
    else:
        recalculate_thresholding = False
        recalculate_matching_and_classification = False
        recalculate_filtering = False
        recalculate_reporting = False

    last_create_report_input = input

    if recalculate_thresholding:
        print("Create report: Recalculate thresholding")
        dataset_thresholded = filter_predictions_using_threshold(dataset, input.threshold)
    
    if recalculate_matching_and_classification:
        print("Create report: Recalculate matching and classification")
        dataset_matched_and_classified = match_and_classify(dataset_thresholded, input.min_iou)
    
    if recalculate_filtering:
        print("Create report: Recalculate filtering")
        dataset_filtered = filter_dataset(dataset_matched_and_classified, input.study_boolean_expression, input.image_boolean_expression)

    if recalculate_reporting:
        print("Create report: Recalculate reporting")
        performance_report = report(
            dataset_filtered,
            input.dimension_1_name, input.dimension_1_level, input.dimension_1_type, input.dimension_1_values,
            input.dimension_2_name, input.dimension_2_level, input.dimension_2_type, input.dimension_2_values,
        )
        selected_images = None # Reset selection since new report is calculated and user needs to select again
    else:
        print("Create report: Recalculating nothing, reusing last report")

    return performance_report


# Must be called after "/create_report" has been called
@app.post("/select_images")
def select_images(input:Select_images_user_input):
    global last_select_images_input
    global selected_images
    global n_selected_studies

    if input.dimension_1_name is not None:
        if input.dimension_1_type in ["int", "float"]:
            input.dimension_1_name += "_bucket"
    if input.dimension_2_name is not None:
        if input.dimension_2_type in ["int", "float"]:
            input.dimension_2_name += "_bucket"

    if selected_images is None or last_select_images_input is None:
        print("Select images: Startup")
        recalculate_selection = True
    else:
        if input.dimension_1_name != last_select_images_input.dimension_1_name or input.dimension_2_name != last_select_images_input.dimension_2_name or \
            input.dimension_1_level != last_select_images_input.dimension_1_level or input.dimension_2_level != last_select_images_input.dimension_2_level or \
            input.dimension_1_value != last_select_images_input.dimension_1_value or input.dimension_2_value != last_select_images_input.dimension_2_value or \
            input.min_image_tp != last_select_images_input.min_image_tp or \
            input.min_image_fn != last_select_images_input.min_image_fn or \
            input.min_image_fp != last_select_images_input.min_image_fp:
            recalculate_selection = True
        else:
            recalculate_selection = False

    last_select_images_input = input

    if recalculate_selection:
        print("Select images: Recalculate selection")
        selected_studies = filter_studies_and_images_based_on_level(
            dataset_filtered.studies,
            input.dimension_1_name, input.dimension_1_level, input.dimension_1_value,
            input.dimension_2_name, input.dimension_2_level, input.dimension_2_value,
            input.min_image_tp, input.min_image_fn, input.min_image_fp
        )
        n_selected_studies = len(selected_studies)
        selected_images = []
        for study in selected_studies:
            for image in study.images:
                selected_images.append(image)
    else:
        print("Select images: Recalculating nothing, reusing last selection")

    return {"n_selected_studies": n_selected_studies, "n_selected_images": len(selected_images)}


# Must be called after "/select_images" has been called
@app.post("/get_images")
def get_images(input:Get_images_user_input):
    return selected_images[input.first_image_idx:input.last_image_idx + 1]

from fastapi import FastAPI
from pydantic import BaseModel

from generate_test_data import generate_test_dataset
from categorizer import match_and_classify, filter_predictions_using_threshold
from describe_metadata import describe
from report import report, filter_dataset, filter_studies_and_images_based_on_level, get_metrics


class Select_data_user_input(BaseModel):
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


app = FastAPI()
dataset = generate_test_dataset(5000)
description = describe(dataset)
metrics = get_metrics()

dataset_thresholded = None
dataset_matched_and_classified = None
dataset_filtered = None

last_input = None
last_result = None


@app.post("/describe_metadata")
def describe_metadata():
    return description


@app.post("/describe_metrics")
def describe_metrics():
    return metrics


@app.post("/report")
def create_report(input:Create_report_user_input):
    global last_input
    global last_result
    global dataset
    global dataset_thresholded
    global dataset_matched_and_classified
    global dataset_filtered

    recalculate_thresholding = True
    recalculate_matching_and_classification = True
    recalculate_filtering = True
    recalculate_reporting = True

    if last_input is None or last_result is None:
        print("Status: Startup")
    else:
        if input.threshold != last_input.threshold:
            pass
        elif input.min_iou != last_input.min_iou:
            recalculate_thresholding = False
        elif input.study_boolean_expression != last_input.study_boolean_expression or \
            input.image_boolean_expression != last_input.image_boolean_expression:
            recalculate_thresholding = False
            recalculate_matching_and_classification = False
        elif input.dimension_1_name != last_input.dimension_1_name or input.dimension_2_name != last_input.dimension_2_name or \
            input.dimension_1_level != last_input.dimension_1_level or input.dimension_2_level != last_input.dimension_2_level or \
            input.dimension_1_type != last_input.dimension_1_type or input.dimension_2_type != last_input.dimension_2_type or \
            input.dimension_1_values != last_input.dimension_1_values or input.dimension_2_values != last_input.dimension_2_values:
            recalculate_thresholding = False
            recalculate_matching_and_classification = False
            recalculate_filtering = False
        else:
            recalculate_thresholding = False
            recalculate_matching_and_classification = False
            recalculate_filtering = False
            recalculate_reporting = False

    if recalculate_thresholding:
        print("Status: Recalculate thresholding")
        dataset_filtered = filter_predictions_using_threshold(dataset, input.threshold)
    
    if recalculate_matching_and_classification:
        print("Status: Recalculate matching and classification")
        dataset_matched_and_classified = match_and_classify(dataset_filtered, input.min_iou)
    
    if recalculate_filtering:
        print("Status: Recalculate filtering")
        dataset_filtered = filter_dataset(dataset_matched_and_classified, input.study_boolean_expression, input.image_boolean_expression)

    if recalculate_reporting:
        print("Status: Recalculate reporting")
        result = report(
            dataset_filtered,
            input.dimension_1_name, input.dimension_1_level, input.dimension_1_type, input.dimension_1_values,
            input.dimension_2_name, input.dimension_2_level, input.dimension_2_type, input.dimension_2_values,
        )
        last_result = result
    else:
        print("Status: Recalculating nothing, reusing last report")
        result = last_result

    last_input = input

    return result


# Must be called after "/report" has been called
@app.post("/select_images")
def select_images(input:Select_data_user_input):

    if input.dimension_1_name is not None:
        if input.dimension_1_type in ["int", "float"]:
            input.dimension_1_name += "_bucket"
    if input.dimension_2_name is not None:
        if input.dimension_2_type in ["int", "float"]:
            input.dimension_2_name += "_bucket"

    selected_studies = filter_studies_and_images_based_on_level(dataset_filtered.studies,
        input.dimension_1_name, input.dimension_1_level, input.dimension_1_value,
        input.dimension_2_name, input.dimension_2_level, input.dimension_2_value,
        input.min_image_tp, input.min_image_fn, input.min_image_fp
    )

    # Get the first 8 images and return them
    images = []
    found_enough_images = False
    for study in selected_studies:
        for image in study.images:
            if len(images) < 8:
                images.append(image)
            else:
                found_enough_images = True
                break
        if found_enough_images:
            break

    return images
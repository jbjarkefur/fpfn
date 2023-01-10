import json
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from generate_test_data import generate_test_dataset
from categorizer import match_and_classify, filter_predictions_using_threshold
from report import report_dimensions

class User_input(BaseModel):
    dimension: dict
    metadata_boolean_expression: str
    min_iou: float
    threshold: float

app = FastAPI()
dataset = generate_test_dataset(10000)
dataset_filtered = None
dataset_matched_and_classified = None
last_input = None


@app.post("/describe_metadata")
def describe_metadata():
    description = {}
    for image in dataset.images:
        for key, value in image.metadata.items():
            if key in description:
                if value in description[key]:
                    description[key][value] += 1
                else:
                    description[key][value] = 0
            else:
                description[key] = {}

    return description


@app.post("/report")
def create_report(input:User_input):
    global last_input
    global dataset
    global dataset_filtered
    global dataset_matched_and_classified
    if last_input is None:
        print("Status: Recalculate all 1")
        dataset_filtered = filter_predictions_using_threshold(dataset, input.threshold)
        dataset_matched_and_classified = match_and_classify(dataset_filtered, input.min_iou)
        result = report_dimensions(dataset_matched_and_classified, input.dimension, input.metadata_boolean_expression)
    else:
        if input.threshold != last_input.threshold:
            print("Status: Recalculate all 2")
            dataset_filtered = filter_predictions_using_threshold(dataset, input.threshold)
            dataset_matched_and_classified = match_and_classify(dataset_filtered, input.min_iou)
            result = report_dimensions(dataset_matched_and_classified, input.dimension, input.metadata_boolean_expression)
        elif input.min_iou != last_input.min_iou:
            print("Status: Recalculate match and report")
            dataset_matched_and_classified = match_and_classify(dataset_filtered, input.min_iou)
            result = report_dimensions(dataset_matched_and_classified, input.dimension, input.metadata_boolean_expression)
        else:
            print("Status: Recalculate report")
            result = report_dimensions(dataset_matched_and_classified, input.dimension, input.metadata_boolean_expression)

    last_input = input

    return result

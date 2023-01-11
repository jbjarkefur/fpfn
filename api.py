import json
from typing import List
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from generate_test_data import generate_test_dataset
from categorizer import match_and_classify, filter_predictions_using_threshold
from report import report

class User_input(BaseModel):
    dimension_1_name: str = None
    dimension_1_values: List[str] = []
    dimension_2_name: str = None
    dimension_2_values: List[str] = []
    metadata_boolean_expression: str
    min_iou: float
    threshold: float

app = FastAPI()
dataset = generate_test_dataset(25000)
dataset_filtered = None
dataset_matched_and_classified = None
last_input = None
last_result = None


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
    global last_result
    global dataset
    global dataset_filtered
    global dataset_matched_and_classified


    recalculate_filtering = True
    recalculate_matching_and_classification = True
    recalculate_report = True

    if last_input is None or last_result is None:
        print("Status: Recalculate all - startup")
    else:
        if input.threshold != last_input.threshold:
            print("Status: Recalculate all")
        elif input.min_iou != last_input.min_iou:
            print("Status: Recalculate matching, classification and report")
            recalculate_filtering = False
        elif input.metadata_boolean_expression != last_input.metadata_boolean_expression or input.dimension_1_name != last_input.dimension_1_name or input.dimension_2_name != last_input.dimension_2_name:
            print("Status: Recalculate report")
            recalculate_filtering = False
            recalculate_matching_and_classification = False
        else:
            print("Status: Recalculate nothing")
            recalculate_filtering = False
            recalculate_matching_and_classification = False
            recalculate_report = False
            result = last_result

    if recalculate_filtering:
        dataset_filtered = filter_predictions_using_threshold(dataset, input.threshold)
    
    if recalculate_matching_and_classification:
        dataset_matched_and_classified = match_and_classify(dataset_filtered, input.min_iou)

    if recalculate_report:
        result = report(
            dataset_matched_and_classified,
            input.dimension_1_name, input.dimension_1_values,
            input.dimension_2_name, input.dimension_2_values,
            input.metadata_boolean_expression)
        last_result = result

    last_input = input

    return result

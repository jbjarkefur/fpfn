from fastapi import FastAPI
from pydantic import BaseModel

from generate_test_data import generate_test_dataset
from categorizer import match_and_classify, filter_predictions_using_threshold
from report import report

class User_input(BaseModel):
    metadata_boolean_expression: str
    min_iou: float
    threshold: float

app = FastAPI()
dataset = generate_test_dataset(500)

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
    dataset_filtered = filter_predictions_using_threshold(dataset, input.threshold)
    dataset_matched_and_classified = match_and_classify(dataset_filtered, input.min_iou)
    result = report(dataset_matched_and_classified, True, input.metadata_boolean_expression)

    return result

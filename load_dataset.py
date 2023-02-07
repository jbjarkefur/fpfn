import json
from pathlib import Path
from typing import List, Tuple

from data import Dataset, GroundTruth, Image, Prediction, Study, StudyDataset


def read_json(json_file: Path):
    with open(json_file) as file:
        return json.load(file)


def _get_metadata(coco_image: dict, metadata_field_names: List[str]) -> dict:
    selected_metadata = {}
    for metadata_field_name in metadata_field_names:
        if metadata_field_name in coco_image:
            selected_metadata[metadata_field_name] = coco_image[metadata_field_name]
        else:
            raise ValueError(f"Metadata field name '{metadata_field_name} is not available in the COCO image.")

    return selected_metadata


def _get_bounding_box(coco_bbox: dict) -> Tuple[float, float, float, float]:
    x1 = coco_bbox[0]
    y1 = coco_bbox[1]
    x2 = x1 + coco_bbox[2]
    y2 = y1 + coco_bbox[3]

    return x1, y1, x2, y2


def _get_ground_truths(coco_annotations: list) -> dict:
    ground_truths = {}
    for coco_annotation in coco_annotations:
        x1, y1, x2, y2 = _get_bounding_box(coco_annotation["bbox"])
        image_id = coco_annotation["image_id"]
        if image_id not in ground_truths:
            ground_truths[image_id] = [GroundTruth(x1, y1, x2, y2)]
        else:
            ground_truths[image_id].append(GroundTruth(x1, y1, x2, y2))

    return ground_truths


def _get_predictions(coco_predictions: list) -> dict:
    predictions = {}
    for coco_prediction in coco_predictions:
        x1, y1, x2, y2 = _get_bounding_box(coco_prediction["bbox"])
        score = coco_prediction["score"]
        image_id = coco_prediction["image_id"]
        prediction = Prediction(x1, y1, x2, y2, score=score)
        if image_id not in predictions:
            predictions[image_id] = [prediction]
        else:
            predictions[image_id].append(prediction)

    return predictions


def load_dataset_from_coco(
        ground_truth_coco_file: Path,
        prediction_coco_file: Path = None,
        dataset_name: str = "Loaded COCO Dataset",
        image_metadata_field_names: List[str] = []) -> Dataset:

    # Load CSV files
    coco_ground_truth = read_json(ground_truth_coco_file)
    coco_predictions = [] if prediction_coco_file is None else read_json(prediction_coco_file)

    # Load ground truths
    ground_truths = _get_ground_truths(coco_ground_truth["annotations"])

    # Load predictions
    predictions = _get_predictions(coco_predictions)

    # Load images
    images = []
    for coco_image in coco_ground_truth["images"]:
        image_id = coco_image["id"]
        study_id = None

        images.append(Image(image_id, study_id,
                      coco_image["width"], coco_image["height"],
                      coco_image["file_name"],
                      ground_truths=[] if image_id not in ground_truths else ground_truths[image_id],
                      predictions=[] if image_id not in predictions else predictions[image_id],
                      metadata=_get_metadata(coco_image, image_metadata_field_names)))

    return Dataset(images=images, name=dataset_name)


def load_studydataset_from_coco(
        ground_truth_coco_file: Path,
        study_id_field_name: str,
        prediction_coco_file: Path = None,
        dataset_name: str = "Loaded COCO StudyDataset",
        study_metadata_field_names: List[str] = [],
        image_metadata_field_names: List[str] = []) -> StudyDataset:

    # Load CSV files
    coco_ground_truth = read_json(ground_truth_coco_file)
    coco_predictions = [] if prediction_coco_file is None else read_json(prediction_coco_file)

    # Load ground truths
    ground_truths = _get_ground_truths(coco_ground_truth["annotations"])

    # Load predictions
    predictions = _get_predictions(coco_predictions)

    # Load images and studies
    studies = {}  # Key = study_id_field_name
    for coco_image in coco_ground_truth["images"]:
        image_id = coco_image["id"]
        study_id = coco_image[study_id_field_name]

        image = Image(image_id, study_id,
                      coco_image["width"], coco_image["height"],
                      coco_image["file_name"],
                      ground_truths=[] if image_id not in ground_truths else ground_truths[image_id],
                      predictions=[] if image_id not in predictions else predictions[image_id],
                      metadata=_get_metadata(coco_image, image_metadata_field_names))

        # Create a new study if not already added, otherwise add image to available study
        if study_id not in studies:
            # Populate the study metadata
            study_metadata = {}
            for study_metadata_field_name in study_metadata_field_names:
                study_metadata[study_metadata_field_name] = coco_image[study_metadata_field_name]

            # Create a new study
            study = Study(study_id, images=[image], metadata=study_metadata)
            studies[study_id] = study
        else:
            studies[study_id].images.append(image)

    return StudyDataset(studies=[studies[study_id] for study_id in studies], name=dataset_name)


if __name__ == "__main__":
    # int IDs
    study_dataset = load_studydataset_from_coco(
        ground_truth_coco_file=Path("example_coco_files/gt.json"),
        prediction_coco_file=Path("example_coco_files/prediction.json"),
        dataset_name="Example StudyDataset with int IDs",
        study_id_field_name="my_custom_study_id",
        study_metadata_field_names=["patient_age", "patient_gender"],
        image_metadata_field_names=["image_sharpness"])

    print("\n----- StudyDataset int IDs -----")
    print(study_dataset)

    # str IDs
    study_dataset = load_studydataset_from_coco(
        ground_truth_coco_file=Path("example_coco_files/gt_str_id.json"),
        prediction_coco_file=Path("example_coco_files/prediction_str_id.json"),
        dataset_name="Example StudyDataset with str IDs",
        study_id_field_name="my_custom_study_id",
        study_metadata_field_names=["patient_age", "patient_gender"],
        image_metadata_field_names=["image_sharpness"])

    print("\n----- StudyDataset str IDs -----")
    print(study_dataset)

    if False:
        image_dataset = load_dataset_from_coco(
            ground_truth_coco_file=Path("example_coco_gt.json"),
            dataset_name="Example Dataset",
            image_metadata_field_names=["image_sharpness"])

        print("\n----- Dataset -----")
        print(image_dataset)

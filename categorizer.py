from copy import deepcopy
from typing import List

import numpy as np

from data import GroundTruth, Image, Prediction, StudyDataset, Study
from generate_test_data import generate_test_dataset
from report import filter_dataset, report


def filter_predictions_using_threshold(dataset: StudyDataset, threshold: float) -> StudyDataset:
    dataset_filtered = deepcopy(dataset)

    for study in dataset_filtered.studies:
        for image in study.images:
            image.predictions = [
                prediction for prediction in image.predictions if prediction.score >= threshold]

    return dataset_filtered


def _calculate_ious(ground_truths: List[GroundTruth], predictions: List[Prediction]):
    ious = np.zeros((len(ground_truths), len(predictions)))
    for ground_truth_idx, ground_truth in enumerate(ground_truths):
        for prediction_idx, prediction in enumerate(predictions):
            iou = _calculate_iou(ground_truth, prediction)
            if iou > 0:
                ious[ground_truth_idx, prediction_idx] = iou

    return ious


def _calculate_iou(ground_truth: GroundTruth, prediction: Prediction) -> float:
    # Calculate the x1, y1, x2, and y2 coordinates of the intersection
    # of the ground truth bounding box and the predicted bounding box.
    x1 = max(ground_truth.x1, prediction.x1)
    y1 = max(ground_truth.y1, prediction.y1)
    x2 = min(ground_truth.x2, prediction.x2)
    y2 = min(ground_truth.y2, prediction.y2)

    # If there is no overlap, return 0
    if x1 >= x2 or y1 >= y2:
        return 0

    intersection_area = (x2 - x1) * (y2 - y1)
    gt_bbox_area = ground_truth.area()
    pred_bbox_area = prediction.area()
    union_area = gt_bbox_area + pred_bbox_area - intersection_area

    return intersection_area / union_area


def match_and_classify(dataset: StudyDataset, min_iou: float = 0.0) -> StudyDataset:
    dataset_matched_and_classified = deepcopy(dataset)

    for study in dataset_matched_and_classified.studies:
        for image in study.images:
            image.n_tp, image.n_fn, image.n_fp = 0, 0, 0

            ground_truths = image.ground_truths
            predictions = image.predictions

            # Find matches by calculating an IoU matrix
            # Rows are ground truths, columns are predictions
            ious = _calculate_ious(ground_truths, predictions)

            if len(ground_truths) > 0 and len(predictions) > 0:
                # Go through each GroundTruth and check if it is FN or TP, higher IoU matches are checked first
                ground_truths_max_iou = np.amax(ious, axis=1)
                ground_truth_idxes_sorted_iou = np.argsort(ground_truths_max_iou)

                for ground_truth_idx in reversed(ground_truth_idxes_sorted_iou):
                    max_iou = ground_truths_max_iou[ground_truth_idx]
                    if max_iou > min_iou:
                        image.n_tp += 1
                        ground_truths[ground_truth_idx].tp = True
                        # Remove all other matches with this GroundTruth to prevent that other Predictions can match it
                        highest_iou_prediction_idx = np.argmax(
                            ious[ground_truth_idx, :])
                        ious[ground_truth_idx, :] = 0
                        ious[ground_truth_idx, highest_iou_prediction_idx] = max_iou
                    else:
                        image.n_fn += 1
                        ground_truths[ground_truth_idx].tp = False
            else:
                # Label all ground truths as FN
                for ground_truth in ground_truths:
                    image.n_fn += 1
                    ground_truth.tp = False

            # Go through each Prediction and check if they are FPs
            for prediction_idx, prediction in enumerate(predictions):
                if not np.any(ious[:, prediction_idx] > 0, axis=0):
                    image.n_fp += 1
                    prediction.fp = True
                else:
                    prediction.fp = False

    return dataset_matched_and_classified


if __name__ == "__main__":
    # 2 predictions on 1 ground truth -> 1 TP, 0 FN, 1 FP
    ground_truths = [GroundTruth(50, 60, 150, 160)]
    predictions = [Prediction(140, 150, 300, 310, 0.5), Prediction(10, 20, 60, 70, 0.5)]
    dataset = StudyDataset(studies=[Study(id=0, images=[Image(id=0, study_id=0, width=900, height=900, filename="test.jpg", ground_truths=ground_truths, predictions=predictions)])])
    dataset_matched_and_classified = match_and_classify(dataset)
    print(report(dataset_matched_and_classified))

    # 1 prediction on 2 ground truth -> 2 TP, 0 FN, 0 FP
    ground_truths = [GroundTruth(50, 60, 150, 160), GroundTruth(60, 70, 160, 170)]
    predictions = [Prediction(140, 150, 300, 310, 0.5)]
    dataset = StudyDataset(studies=[Study(id=0, images=[Image(id=0, study_id=0, width=950, height=950, filename="test.jpg", ground_truths=ground_truths, predictions=predictions)])])
    dataset_matched_and_classified = match_and_classify(dataset)
    print(report(dataset_matched_and_classified))

    # 1 prediction on 2 ground truth, 1 prediction not overlapping anything, 1 ground truth not detected -> 2 TP, 1 FN, 1 FP
    ground_truths = [GroundTruth(50, 60, 150, 160), GroundTruth(60, 70, 160, 170), GroundTruth(10, 20, 100, 110)]
    predictions = [Prediction(140, 150, 300, 310, 0.5), Prediction(161, 171, 300, 310, 0.5)]
    dataset = StudyDataset(studies=[Study(id=0, images=[Image(id=0, study_id=0, width=1000, height=1000, filename="test.jpg", ground_truths=ground_truths, predictions=predictions)])])
    dataset_matched_and_classified = match_and_classify(dataset)
    print(report(dataset_matched_and_classified))

    # Larger amount of data
    dataset = generate_test_dataset(n_studies=100)
    dataset_filtered = filter_predictions_using_threshold(dataset, threshold=0.2)
    dataset_matched_and_classified = match_and_classify(dataset_filtered, min_iou=0.2)
    dataset_filtered = filter_dataset(dataset_matched_and_classified,
                                      study_boolean_expression='study.metadata["age"] is not None and study.metadata["age"] >= 50',
                                      image_boolean_expression='image.metadata["bodypart"] in ["Knee", "Hand"] and image.metadata["view"] == "PA/AP"')
    print(report(dataset_filtered))

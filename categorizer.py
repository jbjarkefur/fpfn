from dataclasses import dataclass
import numpy as np


@dataclass
class BoundingBox:
  x1: float
  y1: float
  x2: float
  y2: float

  def area(self) -> float:
    return (self.x2 - self.x1) * (self.y2 - self.y1)

@dataclass
class GroundTruth(BoundingBox):
  tp: bool = None
  fn: bool = None

@dataclass
class Prediction(BoundingBox):
  fp: bool = None
  score: float = 0


def _calculate_ious(ground_truths: list[GroundTruth], predictions: list[Prediction]):
  ious = np.zeros((len(ground_truths), len(predictions)))
  for ground_truth_idx, ground_truth in enumerate(ground_truths):
    for prediction_idx, prediction in enumerate(predictions):
      iou = _calculate_iou(ground_truth, prediction)
      if iou > 0:
        ious[ground_truth_idx, prediction_idx] = iou
  return ious


def _calculate_iou(ground_truth: GroundTruth, prediction: Prediction):
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


def _categorize(ground_truths: list[GroundTruth], predictions: list[Prediction]):

  # Find matches by calculating an IoU matrix
  # Rows are ground truths, columns are predictions
  ious = _calculate_ious(ground_truths, predictions)
  
  # Go through each GroundTruth and check if it is FN or TP, higher IoU matches are checked first
  n_TP, n_FN, n_FP = 0, 0, 0
  ground_truths_max_iou = np.amax(ious, axis=1)
  ground_truth_idxes_sorted_iou = np.argsort(ground_truths_max_iou)
  for ground_truth_idx in reversed(ground_truth_idxes_sorted_iou):
    max_iou = ground_truths_max_iou[ground_truth_idx]
    if max_iou > 0:
      n_TP += 1
      ground_truths[ground_truth_idx].tp = True
      ground_truths[ground_truth_idx].fn = False
      # Remove all other matches with this GroundTruth to prevent that other Predictions can match it
      highest_iou_prediction_idx = np.argmax(ious[ground_truth_idx, :])
      ious[ground_truth_idx, :] = 0
      ious[ground_truth_idx, highest_iou_prediction_idx] = max_iou
    else:
      n_FN += 1
      ground_truths[ground_truth_idx].tp = False
      ground_truths[ground_truth_idx].fn = True
  
  # Go through each Prediction and check if they are FPs
  for prediction_idx, prediction in enumerate(predictions):
    if not np.any(ious[:, prediction_idx] > 0, axis=0):
      n_FP += 1
      prediction.fp = True
    else:
      prediction.fp = False

  return n_TP, n_FN, n_FP


if __name__ == "__main__":
  # 2 predictions on 1 ground truth -> 1 TP, 0 FN, 1 FP
  gt_1 = GroundTruth(50, 60, 150, 160)
  pred_1 = Prediction(140, 150, 300, 310)
  pred_2 = Prediction(10, 20, 60, 70)
  TP, FN, FP = _categorize([gt_1], [pred_1, pred_2])
  print(f"TP: {TP}, FN: {FN}, FP: {FP}")

  # 1 prediction on 2 ground truth -> 2 TP, 0 FN, 0 FP
  gt_1 = GroundTruth(50, 60, 150, 160)
  gt_2 = GroundTruth(60, 70, 160, 170)
  pred_1 = Prediction(140, 150, 300, 310)
  TP, FN, FP = _categorize([gt_1, gt_2], [pred_1, pred_2])
  print(gt_1, gt_2, pred_1, pred_2)
  print(f"TP: {TP}, FN: {FN}, FP: {FP}")

  # 1 prediction on 2 ground truth, 1 prediction not overlapping anything, 1 ground truth not detected -> 2 TP, 1 FN, 1 FP
  gt_1 = GroundTruth(50, 60, 150, 160)
  gt_2 = GroundTruth(60, 70, 160, 170)
  gt_3 = GroundTruth(10, 20, 100, 110)
  pred_1 = Prediction(140, 150, 300, 310)
  pred_2 = Prediction(161, 171, 300, 310)
  TP, FN, FP = _categorize([gt_1, gt_2, gt_3], [pred_1, pred_2])
  print(gt_1, gt_2, gt_3, pred_1, pred_2)
  print(f"TP: {TP}, FN: {FN}, FP: {FP}")
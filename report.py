from typing import List
from data import Dataset, Image
from copy import deepcopy
import pandas as pd
from multiprocessing import Pool


def _images_overview(images: List[Image]):
    n_ground_truths = 0
    n_predictions = 0
    for image in images:
        n_ground_truths += len(image.ground_truths)
        n_predictions += len(image.predictions)
    print(f"{len(images)} images, {n_ground_truths} ground truths and {n_predictions} predictions.")


def report(dataset, print_to_terminal=True, metadata_boolean_expression: str = "True"):

    n_tp, n_fn, n_fp, n_images_filtered = 0, 0, 0, 0
    for image in dataset.images:
        # TODO: Implement safer approach than using "eval"
        if eval(metadata_boolean_expression):
            n_images_filtered += 1
            n_tp += image.n_tp
            n_fn += image.n_fn
            n_fp += image.n_fp

    tp_rate = n_tp / (n_tp + n_fn)
    fps_per_image = n_fp / n_images_filtered

    result = {}
    result["dataset_name"] = dataset.name
    result["n_images"] = len(dataset)
    result["n_images_filtered"] = n_images_filtered
    result["n_tp"] = n_tp
    result["n_fn"] = n_fn
    result["n_fp"] = n_fp
    result["tp_rate"] = tp_rate
    result["fps_per_image"] = fps_per_image

    if print_to_terminal:
        print(result)

    return result


def _report_dimension_value(arguments):
    images, metadata_boolean_expression, dimension_name, dimension_value = arguments
    n_images, n_positive_images, n_negative_images, n_tp, n_fn, n_fp = 0, 0, 0, 0, 0, 0
    for image in images:
        # TODO: Implement safer approach than using "eval"
        if eval(metadata_boolean_expression + f" and image.metadata['{dimension_name}'] == '{dimension_value}'"):
            n_images += 1
            if len(image.ground_truths) > 0:
                n_positive_images += 1
            else:
                n_negative_images += 1
            n_tp += image.n_tp
            n_fn += image.n_fn
            n_fp += image.n_fp
    if n_images > 0:
        if (n_tp + n_fn) > 0:
            tp_rate = 100 * (n_tp / (n_tp + n_fn))
            fps_per_image = n_fp / n_images
        else:
            tp_rate = -1
            fps_per_image = -1
        report = pd.DataFrame({
            dimension_name: [dimension_value],
            "tp_rate": [tp_rate],
            "fps_per_image": [fps_per_image],
            "n_images": [float(n_images)],
            "n_positive_images": [float(n_positive_images)],
            "n_negative_images": [float(n_negative_images)]
        })
    else:
        report = None
    
    return report


def report_dimensions(dataset, dimension: dict, metadata_boolean_expression: str = "True"):
    dimension_name = list(dimension.keys())[0]
    dimension_values = dimension[dimension_name]
    with Pool() as pool:
        reports = pool.map(_report_dimension_value, [(dataset.images, metadata_boolean_expression, dimension_name, dimension_value) for dimension_value in dimension_values])

    reports = [report for report in reports if report is not None]
    report = pd.concat(reports, ignore_index=True)

    return report

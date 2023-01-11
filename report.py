from typing import List
from data import Dataset, Image
from copy import deepcopy
import pandas as pd
from multiprocessing import Pool

def _report(images, dimension_1_name, dimension_1_value, dimension_2_name, dimension_2_value):
    n_images, n_positive_images, n_negative_images, n_tp, n_fn, n_fp = 0, 0, 0, 0, 0, 0
    for image in images:
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
            "tp_rate": [tp_rate],
            "fps_per_image": [fps_per_image],
            "n_tp" : [float(n_tp)],
            "n_fn" : [float(n_fn)],
            "n_fp" : [float(n_fp)],
            "n_images": [float(n_images)],
            "n_positive_images": [float(n_positive_images)],
            "n_negative_images": [float(n_negative_images)]
        })
        if dimension_2_name is not None:
            report.insert(loc=0, column=dimension_2_name, value=[dimension_2_value])
        if dimension_1_name is not None:
            report.insert(loc=0, column=dimension_1_name, value=[dimension_1_value])
    else:
        report = None
    
    return report


def _report_row(arguments):
    images, dimension_1_name, dimension_1_value, dimension_2_name, dimension_2_value = arguments

    # TODO: Implement safer approach than using "eval"
    if dimension_1_name is None and dimension_2_name is None:
        images_filtered = images
    elif dimension_2_name is None:
        images_filtered = [image for image in images if eval(f"image.metadata['{dimension_1_name}'] == '{dimension_1_value}'")]
    elif dimension_1_name is None:
        images_filtered = [image for image in images if eval(f"image.metadata['{dimension_2_name}'] == '{dimension_2_value}'")]
    else:
        images_filtered = [image for image in images if eval(f"image.metadata['{dimension_1_name}'] == '{dimension_1_value}' and image.metadata['{dimension_2_name}'] == '{dimension_2_value}'")]

    report = _report(images_filtered, dimension_1_name, dimension_1_value, dimension_2_name, dimension_2_value)

    return report


def report(dataset: Dataset, dimension_1_name: str, dimension_1_values: List[str], dimension_2_name: str, dimension_2_values: List[str], metadata_boolean_expression: str):

    images = [image for image in dataset.images if eval(metadata_boolean_expression)]

    if dimension_1_name or dimension_2_name:
        pool_arguments = []
        if dimension_1_name and dimension_2_name:
            for dimension_1_value in dimension_1_values:
                for dimension_2_value in dimension_2_values:
                    pool_arguments.append((images, dimension_1_name, dimension_1_value, dimension_2_name, dimension_2_value))
        elif dimension_1_name:
            for dimension_1_value in dimension_1_values:
                pool_arguments.append((images, dimension_1_name, dimension_1_value, None, None))
        else:
            for dimension_2_value in dimension_2_values:
                pool_arguments.append((images, None, None, dimension_2_name, dimension_2_value))

        with Pool() as pool:
            reports = pool.map(_report_row, pool_arguments)
        reports = [report for report in reports if report is not None]  # Use empty dataframes instead (still with columns)
        report = pd.concat(reports, ignore_index=True)
    else:
        report = _report(images, dimension_1_name, dimension_1_values, dimension_2_name, dimension_2_values)

    return report

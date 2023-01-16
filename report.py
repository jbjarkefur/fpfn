from typing import List
from data import StudyDataset, Study, Image
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
        if (dimension_1_name is not None) and (dimension_1_name != dimension_2_name):
            report.insert(loc=0, column=dimension_1_name, value=[dimension_1_value])
    else:
        report = None
    
    return report


def _report_row(arguments):
    images, dimension_1_name, dimension_1_value, dimension_1_expression, dimension_2_name, dimension_2_value, dimension_2_expression = arguments

    # TODO: Implement safer approach than using "eval"
    if dimension_1_name is None and dimension_2_name is None:
        images_filtered = images
    elif dimension_2_name is None:
        images_filtered = [image for image in images if eval(dimension_1_expression)]
    elif dimension_1_name is None:
        images_filtered = [image for image in images if eval(dimension_2_expression)]
    else:
        images_filtered = [image for image in images if eval(dimension_1_expression + " and " + dimension_2_expression)]

    report = _report(images_filtered, dimension_1_name, dimension_1_value, dimension_2_name, dimension_2_value)

    return report


def _get_bucket_names(edges) -> List[str]:
    bucket_names = []
    for lower_edge, upper_edge in zip(edges[:-2], edges[1:-1]):
        bucket_names.append(f"[{lower_edge}, {upper_edge})")
    bucket_names.append(f"[{edges[-2]}, {edges[-1]}]")
    return bucket_names


def _set_float_bucket(value, edges) -> str:
    bucket = None
    for lower_edge, upper_edge in zip(edges[:-2], edges[1:-1]):
        if value >= lower_edge and value < upper_edge:
            bucket = f"[{lower_edge}, {upper_edge})"
            break
    if bucket == None:
        if value <= edges[-1]:
            bucket = f"[{edges[-2]}, {edges[-1]}]"
        else:
            bucket = "Other"
    return bucket


def report(dataset: StudyDataset, study_boolean_expression: str = "True", image_boolean_expression: str = "True", dimension_1_name: str = None, dimension_1_type: str = None, dimension_1_values: List = None, dimension_2_name: str = None, dimension_2_type: str = None, dimension_2_values: List = None):

    # TODO: Replace with double list comprehension
    images = []
    for study in dataset.studies:
        if eval(study_boolean_expression):
            for image in study.images:
                if eval(image_boolean_expression):
                    images.append(image)

    # Handle numeric dimensions
    if dimension_1_type == "float":
        for image in images:
            image.metadata[f"{dimension_1_name}_bucket"] = _set_float_bucket(image.metadata[dimension_1_name], dimension_1_values)
        dimension_1_name += "_bucket"
        dimension_1_values = _get_bucket_names(dimension_1_values)
        
    if dimension_2_type == "float":
        for image in images:
            image.metadata[f"{dimension_2_name}_bucket"] = _set_float_bucket(image.metadata[dimension_2_name], dimension_2_values)
        dimension_2_name += "_bucket"
        dimension_2_values = _get_bucket_names(dimension_2_values)

    if dimension_1_name or dimension_2_name:
        pool_arguments = []
        if dimension_1_name and dimension_2_name:
            for dimension_1_value in dimension_1_values:
                dimension_1_expression = f"image.metadata['{dimension_1_name}'] == '{dimension_1_value}'"
                for dimension_2_value in dimension_2_values:
                    dimension_2_expression = f"image.metadata['{dimension_2_name}'] == '{dimension_2_value}'"
                    pool_arguments.append((images, dimension_1_name, dimension_1_value, dimension_1_expression, dimension_2_name, dimension_2_value, dimension_2_expression))
        elif dimension_1_name:
            for dimension_1_value in dimension_1_values:
                dimension_1_expression = f"image.metadata['{dimension_1_name}'] == '{dimension_1_value}'"
                pool_arguments.append((images, dimension_1_name, dimension_1_value, dimension_1_expression, None, None, None))
        else:
            for dimension_2_value in dimension_2_values:
                dimension_2_expression = f"image.metadata['{dimension_2_name}'] == '{dimension_2_value}'"
                pool_arguments.append((images, None, None, None, dimension_2_name, dimension_2_value, dimension_2_expression))

        with Pool() as pool:
            reports = pool.map(_report_row, pool_arguments)
        reports = [report for report in reports if report is not None]  # Use empty dataframes instead (still with columns)
        report = pd.concat(reports, ignore_index=True)
    else:
        report = _report(images, dimension_1_name, dimension_1_values, dimension_2_name, dimension_2_values)

    return report

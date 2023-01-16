from typing import List
from data import StudyDataset, Study
import pandas as pd
from multiprocessing import Pool
import time


def _report(studies: List[Study],
    dimension_1_name: str = None, dimension_1_level: str = None, dimension_1_value: str = None,
    dimension_2_name: str = None, dimension_2_level: str = None, dimension_2_value: str = None):

    n_images, n_pos_images = 0, 0
    n_studies, n_pos_studies = 0, 0
    n_tp, n_fn, n_fp = 0, 0, 0
    for study in studies:
        n_studies += 1
        study_is_positive = False
        for image in study.images:
            n_images += 1
            if len(image.ground_truths) > 0:
                study_is_positive = True
                n_pos_images += 1
            n_tp += image.n_tp
            n_fn += image.n_fn
            n_fp += image.n_fp
        if study_is_positive:
            n_pos_studies += 1

    n_neg_studies = n_studies - n_pos_studies
    n_neg_images = n_images - n_pos_images

    if n_images > 0:
        # Calculate KPIs
        if (n_tp + n_fn) > 0:
            tp_rate = 100 * (n_tp / (n_tp + n_fn))
            fps_per_image = n_fp / n_images
        else:
            tp_rate = -1
            fps_per_image = -1

        # Write a report
        report = pd.DataFrame({
            "tp_rate": [tp_rate],
            "fps_per_image": [fps_per_image],
            "n_tp" : [float(n_tp)],
            "n_fn" : [float(n_fn)],
            "n_fp" : [float(n_fp)],
            "n_images": [float(n_images)],
            "n_pos_images": [float(n_pos_images)],
            "n_neg_images": [float(n_neg_images)],
            "n_studies": [float(n_studies)],
            "n_pos_studies": [float(n_pos_studies)],
            "n_neg_studies": [float(n_neg_studies)]
        })
        if dimension_2_name is not None:
            report.insert(loc=0, column=dimension_2_level.capitalize() + " " + dimension_2_name, value=[dimension_2_value])
        if (dimension_1_name is not None) and (dimension_1_name != dimension_2_name):
            report.insert(loc=0, column=dimension_1_level.capitalize() + " " + dimension_1_name, value=[dimension_1_value])
    else:
        report = None
    
    return report


def _report_row(arguments):
    studies, dimension_1_name, dimension_1_level, dimension_1_value, dimension_2_name, dimension_2_level, dimension_2_value = arguments

    if dimension_1_name is None and dimension_2_name is None:
        pass
    elif dimension_2_name is None:
        studies = _filter_studies_and_images_based_on_level(studies, dimension_1_name, dimension_1_level, dimension_1_value)
    elif dimension_1_name is None:
        studies = _filter_studies_and_images_based_on_level(studies, dimension_2_name, dimension_2_level, dimension_2_value)
    else:
        studies = _filter_studies_and_images_based_on_level(studies,
            dimension_1_name, dimension_1_level, dimension_1_value,
            dimension_2_name, dimension_2_level, dimension_2_value
        )

    report = _report(studies,
        dimension_1_name, dimension_1_level, dimension_1_value,
        dimension_2_name, dimension_2_level, dimension_2_value)

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


def _add_buckets_for_numeric_dimension(studies, dimension_name, dimension_level, dimension_type, dimension_values):
    if dimension_type == "float":
        for study in studies:
            if dimension_level == "study":
                study.metadata[f"{dimension_name}_bucket"] = _set_float_bucket(study.metadata[dimension_name], dimension_values)
            else:
                for image in study.images:
                    image.metadata[f"{dimension_name}_bucket"] = _set_float_bucket(image.metadata[dimension_name], dimension_values)

        dimension_name += "_bucket"
        dimension_values = _get_bucket_names(dimension_values)

    return dimension_name, dimension_values


def _filter_studies_and_images_based_on_level(studies: List[Study],
    dimension_1_name: str, dimension_1_level: str, dimension_1_value: str,
    dimension_2_name: str = None, dimension_2_level: str = None, dimension_2_value: str = None):

    if dimension_1_level == "study":
        study_boolean_expression = f"study.metadata['{dimension_1_name}'] == '{dimension_1_value}'"
        image_boolean_expression = "True"
    elif dimension_1_level == "image":
        study_boolean_expression = "True"
        image_boolean_expression = f"image.metadata['{dimension_1_name}'] == '{dimension_1_value}'"
    else:
        raise ValueError(f"Unsupported dimension level {dimension_1_level}")

    if dimension_2_name is not None:
        if dimension_2_level == "study":
            study_boolean_expression += f" and study.metadata['{dimension_2_name}'] == '{dimension_2_value}'"
        elif dimension_2_level == "image":
            image_boolean_expression += f" and image.metadata['{dimension_2_name}'] == '{dimension_2_value}'"
        else:
            raise ValueError(f"Unsupported dimension level {dimension_2_level}")

    return _filter_studies_and_images(studies, study_boolean_expression, image_boolean_expression)


def _filter_studies_and_images(studies: List[Study], study_boolean_expression: str, image_boolean_expression: str) -> List[Study]:

    # TODO: Implement safer approach than using "eval"
    studies_filtered = []
    for study in studies:
        if eval(study_boolean_expression):
            study_filtered = Study(metadata=study.metadata, classification=study.classification)
            studies_filtered.append(study_filtered)
            for image in study.images:
                if eval(image_boolean_expression):
                    study_filtered.images.append(image)

    # List comprehension does not seem to be faster
    # studies_filtered = [Study(images=[image for image in study.images if eval(image_boolean_expression)], metadata=study.metadata, classification=study.classification) for study in studies if eval(study_boolean_expression)]

    return studies_filtered


def report(dataset: StudyDataset,
    study_boolean_expression: str = "True", image_boolean_expression: str = "True",
    dimension_1_name: str = None, dimension_1_level: str = None, dimension_1_type: str = None, dimension_1_values: List = [None],
    dimension_2_name: str = None, dimension_2_level: str = None, dimension_2_type: str = None, dimension_2_values: List = [None]):

    # Filter studies and images based on boolean filters
    studies = _filter_studies_and_images(dataset.studies, study_boolean_expression, image_boolean_expression)

    if dimension_1_name is None and dimension_2_name is None:
        report = _report(studies)
    else:
        # Handle numeric dimensions
        dimension_1_name, dimension_1_values = _add_buckets_for_numeric_dimension(studies, dimension_1_name, dimension_1_level, dimension_1_type, dimension_1_values)
        dimension_2_name, dimension_2_values = _add_buckets_for_numeric_dimension(studies, dimension_2_name, dimension_2_level, dimension_2_type, dimension_2_values)

        pool_arguments = []
        for dimension_1_value in dimension_1_values:
            for dimension_2_value in dimension_2_values:
                pool_arguments.append(
                    (studies,
                     dimension_1_name, dimension_1_level, dimension_1_value,
                     dimension_2_name, dimension_2_level, dimension_2_value))

        with Pool() as pool:
            reports = pool.map(_report_row, pool_arguments)
        reports = [report for report in reports if report is not None]
        report = pd.concat(reports, ignore_index=True)

    return report

from typing import List
from data import Dataset, Image
from copy import deepcopy


def _images_overview(images: List[Image]):
    n_ground_truths = 0
    n_predictions = 0
    for image in images:
        n_ground_truths += len(image.ground_truths)
        n_predictions += len(image.predictions)
    print(f"{len(images)} images, {n_ground_truths} ground truths and {n_predictions} predictions.")


def report(dataset, print_to_terminal=True, metadata_boolean_expression: str = "True"):

    # TODO: Implement safer approach than using "eval"
    # images_filtered = [image for image in dataset.images if eval(metadata_boolean_expression)]

    # for key, values in metadata_filters.items():
    #     if isinstance(values, list):
    #         images_filtered = list(
    #             filter(lambda image: image.metadata[key] in values, images_filtered))
    #     else:
    #         images_filtered = list(
    #             filter(lambda image: image.metadata[key] == values, images_filtered))

    n_tp, n_fn, n_fp, n_images_filtered = 0, 0, 0, 0
    for image in dataset.images:
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

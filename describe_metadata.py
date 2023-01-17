from data import StudyDataset


def _describe_metadata(description: dict, metadata: dict):

    for key, value in metadata.items():
        if isinstance(value, str):
            if key in description["str"]:
                if value in description["str"][key]:
                    description["str"][key][value] += 1
                else:
                    description["str"][key][value] = 0
            else:
                description["str"][key] = {}
                description["str"][key][value] = 1
        elif isinstance(value, int):
            if key in description["int"]:
                if value < description["int"][key]["min_value"]:
                    description["int"][key]["min_value"] = value
                elif value > description["int"][key]["max_value"]:
                    description["int"][key]["max_value"] = value
            else:
                description["int"][key] = {}
                description["int"][key]["min_value"] = value
                description["int"][key]["max_value"] = value
        elif isinstance(value, float):
            if key in description["float"]:
                if value < description["float"][key]["min_value"]:
                    description["float"][key]["min_value"] = value
                elif value > description["float"][key]["max_value"]:
                    description["float"][key]["max_value"] = value
            else:
                description["float"][key] = {}
                description["float"][key]["min_value"] = value
                description["float"][key]["max_value"] = value


def _describe_study_metadata(dataset: StudyDataset) -> dict:
    description = {
        "str": {},
        "int": {},
        "float": {}
    }
    for study in dataset.studies:
        _describe_metadata(description, study.metadata)

    return description


def _describe_image_metadata(dataset: StudyDataset) -> dict:
    description = {
        "str": {},
        "int": {},
        "float": {}
    }
    for study in dataset.studies:
        for image in study.images:
            _describe_metadata(description, image.metadata)

    return description


def describe(dataset: StudyDataset) -> dict:
    return {
        "study": _describe_study_metadata(dataset),
        "image": _describe_image_metadata(dataset)
    }

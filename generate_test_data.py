from typing import List
from data import StudyDataset, Study, Image, BoundingBox, GroundTruth, Prediction
import random


def _generate_random_bounding_box(image_width, image_height) -> BoundingBox:
    width = random.randint(20, 200)
    height = random.randint(20, 200)
    x1 = random.randint(0, image_width - width)
    y1 = random.randint(0, image_height - height)
    x2 = x1 + width
    y2 = y1 + height
    return BoundingBox(x1, y1, x2, y2)


def _generate_random_ground_truth(image_width, image_height) -> GroundTruth:
    return GroundTruth.from_parent(_generate_random_bounding_box(image_width, image_height))


def _generate_random_prediction(image_width, image_height) -> Prediction:
    bounding_box = _generate_random_bounding_box(image_width, image_height)
    score = random.uniform(0, 1)
    return Prediction.from_parent(bounding_box, score)


def _generate_random_matching_prediction(image_width, image_height, ground_truth) -> Prediction:
    width = random.uniform(0.95, 1.05) * ground_truth.width()
    height = random.uniform(0.95, 1.05) * ground_truth.height()
    x1 = min(ground_truth.x1 + random.uniform(-0.05, 0.05)
             * width, image_width - width)
    y1 = min(ground_truth.y1 + random.uniform(-0.05, 0.05)
             * height, image_height - height)
    x2 = x1 + width
    y2 = y1 + height
    score = random.uniform(0, 1)
    return Prediction(x1, y1, x2, y2, score=score)


def generate_test_dataset(n_studies: int = 100) -> StudyDataset:
    random.seed(42)

    image_descriptions = {
        "test_data_images/arm_1.jpg": {"width": 1574, "height": 1920},
        "test_data_images/foot_1.jpg": {"width": 1082, "height": 1920},
        "test_data_images/hand_1.jpg": {"width": 1597, "height": 1920},
        "test_data_images/hand_2.jpg": {"width": 1064, "height": 1280},
        "test_data_images/knee_1.jpg": {"width": 1440, "height": 1920},
        "test_data_images/knee_2.jpg": {"width": 1440, "height": 1920},
        "test_data_images/wrist_1.jpg": {"width": 1080, "height": 1920}
    }

    studies = []
    current_image_id = 0
    for study_id in range(n_studies):
        # Create a new study
        study = Study(id=study_id)

        # Set some random study metadata
        study.metadata["country"] = random.choices(
            ["US", "France", "Germany", "UK", "Sweden", "Denmark", "Spain"], [0.25, 0.2, 0.1, 0.2, 0.1, 0.1, 0.05])[0]
        study.metadata["gender"] = random.choices(
            ["Male", "Female"], [0.7, 0.3])[0]
        study.metadata["age"] = random.randint(0, 100)

        # Add 2-5 images to the study
        n_images = random.randint(2, 5)
        for _ in range(n_images):

            # Create a new image
            image_id = current_image_id
            current_image_id += 1

            filename = random.choices(list(image_descriptions.keys()))[0]

            image_width = image_descriptions[filename]["width"]
            image_height = image_descriptions[filename]["height"]
            image = Image(image_id, study_id, image_width, image_height)

            image.filename = filename

            # Set some random image metadata
            image.metadata["bodypart"] = random.choices(
                ["Knee", "Hand", "Foot", "Shoulder", "Elbow", "Hip", "Forearm"], [0.2, 0.1, 0.2, 0.2, 0.15, 0.1, 0.05])[0]
            image.metadata["view"] = random.choices(
                ["Lateral", "PA/AP", "Oblique"], [0.2, 0.4, 0.2])[0]
            image.metadata["machine"] = random.choices(
                ["GE", "Fujifilm", "Siemens"], [0.1, 0.1, 0.8])[0]
            image.metadata["sharpness"] = random.uniform(0, 10)

            # Add some random FPs
            n_fps = random.choices(
                [0, 1, 2, 3, 4], [0.7, 0.15, 0.1, 0.025, 0.025])[0]
            for _ in range(n_fps):
                fp = _generate_random_prediction(image_width, image_height)
                image.predictions.append(fp)

            # Add ground truth and some matching predictions
            if random.uniform(0, 1) > 0.3:
                # This image should have ground truth
                n_ground_truths = random.randint(1, 5)
                for _ in range(n_ground_truths):
                    ground_truth = _generate_random_ground_truth(
                        image_width, image_height)
                    image.ground_truths.append(ground_truth)

                    if random.uniform(0, 1) > 0.2:
                        prediction = _generate_random_matching_prediction(
                            image_width, image_height, ground_truth)
                        image.predictions.append(prediction)

            study.images.append(image)
        studies.append(study)

    return StudyDataset(studies=studies, name="Test dataset")


if __name__ == "__main__":
    # Generate 100 images where around half have ground truth (1-5) and where around 80% of the ground truths have a prediction
    studies = generate_test_dataset()
    a = 1

from dataclasses import dataclass, field
from enum import Enum
from typing import List


@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    def width(self) -> float:
        return self.x2 - self.x1

    def height(self) -> float:
        return self.y2 - self.y1

    def area(self) -> float:
        return self.width() * self.height()


@dataclass
class GroundTruth(BoundingBox):
    tp: bool = None

    @classmethod
    def from_parent(cls, bounding_box):
        return cls(x1=bounding_box.x1, y1=bounding_box.y1, x2=bounding_box.x2, y2=bounding_box.y2)

    def __repr__(self):
        return f"\nGround truth x1: {self.x1}, y1: {self.y1}, x2: {self.x2}, y2: {self.y2}, width: {self.width()}, height: {self.height()}, area: {self.area()}"


@dataclass
class Prediction(BoundingBox):
    score: float
    fp: bool = None

    @classmethod
    def from_parent(cls, bounding_box, score):
        return cls(x1=bounding_box.x1, y1=bounding_box.y1, x2=bounding_box.x2, y2=bounding_box.y2, score=score)

    def __repr__(self):
        return f"\nPrediction x1: {self.x1}, y1: {self.y1}, x2: {self.x2}, y2: {self.y2}, width: {self.width()}, height: {self.height()}, area: {self.area()}, score: {self.score}"


class Classification(Enum):
    TP = 1
    FN = 2
    FP = 3
    TN = 4


@dataclass
class Image():
    id: int | str
    study_id: int | str | None
    width: int
    height: int
    filename: str
    ground_truths: List[GroundTruth] = field(default_factory=list)
    predictions: List[Prediction] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    n_tp: int = None
    n_fn: int = None
    n_fp: int = None
    classification: Classification = None

    def area(self) -> float:
        return self.width * self.height

    def __repr__(self):
        image_text = f"\n\nImage id: {self.id}, Study id: {self.study_id}\nWidth: {self.width}, Height: {self.height}"
        metadata_text = ""
        for key, value in self.metadata.items():
            metadata_text += f"\n{key}: {value}"
        ground_truths_text = ""
        for ground_truth in self.ground_truths:
            ground_truths_text += ground_truth.__repr__()
        predictions_text = ""
        for prediction in self.predictions:
            predictions_text += prediction.__repr__()

        return image_text + metadata_text + ground_truths_text + predictions_text


@dataclass
class Study():
    id: int | str
    images: List[Image] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    classification: Classification = None

    def __len__(self):
        return len(self.images)

    def __repr__(self):
        study_text = f"\n\nStudy id: {self.id}\nNumber of images: {len(self.images)}"
        metadata_text = ""
        for key, value in self.metadata.items():
            metadata_text += f"\n{key}: {value}"
        images_text = ""
        for image in self.images:
            images_text += image.__repr__()

        return study_text + metadata_text + images_text


@dataclass
class Dataset():
    images: List[Image] = field(default_factory=list)
    name: str = "A Dataset"

    def __len__(self):
        return len(self.images)

    def __repr__(self):
        dataset_text = f"Dataset name: '{self.name}'.\nNumber of images: {len(self.images)}"
        images_text = ""
        for image in self.images:
            images_text += image.__repr__()

        return dataset_text + images_text


@dataclass
class StudyDataset():
    studies: List[Study] = field(default_factory=list)
    name: str = "A StudyDataset"

    def __len__(self):
        return len(self.studies)

    def __repr__(self):
        dataset_text = f"StudyDataset name: '{self.name}'.\nNumber of studies: {len(self.studies)}"
        studies_text = ""
        for study in self.studies:
            studies_text += study.__repr__()

        return dataset_text + studies_text

from typing import List
from dataclasses import dataclass, field
from enum import Enum


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


@dataclass
class Prediction(BoundingBox):
    fp: bool = None
    score: float = 0

    @classmethod
    def from_parent(cls, bounding_box, score):
        return cls(x1=bounding_box.x1, y1=bounding_box.y1, x2=bounding_box.x2, y2=bounding_box.y2, score=score)


class Classification(Enum):
    TP = 1
    FN = 2
    FP = 3
    TN = 4


@dataclass
class Image():
    width: int
    height: int
    filename: str = None
    ground_truths: List[GroundTruth] = field(default_factory=list)
    predictions: List[Prediction] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    n_tp: int = None
    n_fn: int = None
    n_fp: int = None
    classification: Classification = None

    def area(self) -> float:
        return self.width * self.height


@dataclass
class Study():
    images: List[Image] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    classification: Classification = None

    def __len__(self):
        return len(self.images)


@dataclass
class StudyDataset():
    studies: List[Study] = field(default_factory=list)
    name: str = "Fracture Dataset"

    def __len__(self):
        return len(self.studies)

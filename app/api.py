import abc
from enum import Enum
from typing import Any

from PIL.Image import Image
from pydantic import BaseModel


class Nsfw(str, Enum):
    nsfw = "NSFW"
    normal = "normal"


class NsfwPrediction(BaseModel):
    label: Nsfw
    score: float


class IImageGenerator(abc.ABC):
    @abc.abstractmethod
    async def generate(self, prompt: str, width: int, height: int, num_inference_steps: int, guidance_scale=7, seed: int = -1):
        pass


class ITextNsfwClassifier(abc.ABC):
    @abc.abstractmethod
    async def classify(self, text: str) -> Any:
        pass


class IImageNsfwClassifier(abc.ABC):
    @abc.abstractmethod
    async def classify(self, image: Image) -> NsfwPrediction:
        pass

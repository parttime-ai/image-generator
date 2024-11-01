import abc
from typing import Any


class IImageGenerator(abc.ABC):
    @abc.abstractmethod
    async def generate(self, prompt: str, width: int, height: int, num_inference_steps: int, guidance_scale=7, seed: int = -1):
        pass


class ITextNsfwClassifier(abc.ABC):
    @abc.abstractmethod
    async def classify(self, text: str) -> Any:
        pass

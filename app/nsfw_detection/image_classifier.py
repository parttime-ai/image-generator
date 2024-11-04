import base64
from io import BytesIO

import PIL
import torch
from PIL.Image import Image
from azure.ai.contentsafety import ContentSafetyClient
from azure.ai.contentsafety.models import AnalyzeImageOptions, ImageData, ImageCategory
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError
from transformers import AutoModelForImageClassification, ViTImageProcessor

from app.api import NsfwPrediction, IImageNsfwClassifier, Nsfw
import logging

class FtVitNsfwClassifier(IImageNsfwClassifier):
    processor: ViTImageProcessor
    model_name = "Falconsai/nsfw_image_detection"

    def __init__(self):
        self.model = AutoModelForImageClassification.from_pretrained(self.model_name)
        self.processor = ViTImageProcessor.from_pretrained(self.model_name)

    async def classify(self, b64: str) -> NsfwPrediction:
        with torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt")
            outputs = self.model(**inputs)
            logits = outputs.logits

        predicted_label = logits.argmax(-1).item()
        label = self.model.config.id2label[predicted_label]

        # get the prediction label and score
        print({"label": label, "score": logits.softmax(-1).max().item()})
        return NsfwPrediction(
            label=label,
            score=logits.softmax(-1).max().item())


def image_to_base64(image: PIL.Image) -> bytes:
    buffered = BytesIO()
    image = image.convert("RGB")
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue())


class AzureImageNsfwContentClassifier(IImageNsfwClassifier):
    endpoint: str
    key: str
    client: ContentSafetyClient
    categories_to_check: list[ImageCategory] = [
        ImageCategory.HATE,
        ImageCategory.SELF_HARM,
        ImageCategory.SEXUAL,
        ImageCategory.VIOLENCE
    ]

    def __init__(self, endpoint: str, key: str):
        self.endpoint = endpoint
        self.key = key
        self.client = ContentSafetyClient(endpoint=self.endpoint, credential=AzureKeyCredential(self.key))

    async def classify(self, b64: bytes) -> NsfwPrediction:
        request = AnalyzeImageOptions(image=ImageData(content=b64))
        try:
            result = self.client.analyze_image(request)
        except HttpResponseError as e:
            logging.error("Analyze image failed.")
            if e.error:
                logging.error(f"Error code: {e.error.code}")
                logging.error(f"Error message: {e.error.message}")
                raise
            print(e)
            raise

        results = [next(item for item in result.categories_analysis if item.category == category)
                   for category in self.categories_to_check]

        if any(result.severity > 2 for result in results):
            return NsfwPrediction(label=Nsfw.nsfw, score=1.0)
        else:
            return NsfwPrediction(label=Nsfw.normal, score=1.0)

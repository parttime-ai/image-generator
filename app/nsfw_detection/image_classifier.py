import torch
from PIL.Image import Image
from transformers import AutoModelForImageClassification, ViTImageProcessor

from app.api import NsfwPrediction


class FtVitNsfwClassifier:
    processor: ViTImageProcessor
    model_name = "Falconsai/nsfw_image_detection"

    def __init__(self):
        self.model = AutoModelForImageClassification.from_pretrained(self.model_name)
        self.processor = ViTImageProcessor.from_pretrained(self.model_name)

    async def classify(self, image: Image) -> NsfwPrediction:
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

import logging
from typing import Any

import torch
from tokenizers import Tokenizer
from transformers import pipeline, Pipeline, AutoTokenizer, AutoModelForSequenceClassification

from app.api import ITextNsfwClassifier


class DistilRobertaNsfwClassifier(ITextNsfwClassifier):
    classifier: Pipeline

    def __init__(self):
        self.classifier = pipeline("sentiment-analysis", model="michellejieli/NSFW_text_classifier")

    async def classify(self, text: str) -> Any:
        output = self.classifier(text)[0]
        return output  # {"label": 'NSFW' | 'SFW', "score": float}


class RobertaNsfwClassifier(ITextNsfwClassifier):
    classifier: Pipeline

    def __init__(self):
        model_repo = "MichalMlodawski/nsfw-text-detection-large"

        self.tokenizer = AutoTokenizer.from_pretrained(model_repo)
        self.classifier = AutoModelForSequenceClassification.from_pretrained(model_repo)

    async def classify(self, text: str) -> Any:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

        with torch.no_grad():
            outputs = self.classifier(**inputs)

        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
        predicted_class = {"label": str(predicted_class), "score": float(logits.softmax(dim=1).max())}
        return predicted_class  # {"label": '0' | '1' | '2', "score": float}
    
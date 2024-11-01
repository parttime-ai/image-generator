import asyncio
import json
import logging
from typing import Any

import torch
from together import AsyncTogether
from transformers import pipeline, Pipeline, AutoTokenizer, AutoModelForSequenceClassification

from app.api import ITextNsfwClassifier
from app.nsfw_detection.prompt_templates import CONTENT_CHECK_PROMPT, AGGREGATOR_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


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


class MixtureOfAgentsClassifier(ITextNsfwClassifier):
    together_client: AsyncTogether
    aggregator_system_prompt: str
    reference_models: list[str]
    aggregator_model: str
    attempt = 0
    retries = 5

    def __init__(self, api_key: str):
        self.together_client = AsyncTogether(api_key=api_key)

        self.reference_models = [
            "Qwen/Qwen2-72B-Instruct",
            "Qwen/Qwen2.5-72B-Instruct-Turbo",
            "mistralai/Mixtral-8x22B-Instruct-v0.1",
            "databricks/dbrx-instruct",
        ]

        self.aggregator_model = "mistralai/Mixtral-8x22B-Instruct-v0.1"
        self.aggregator_model2 = "Qwen/Qwen2.5-72B-Instruct-Turbo"
        self.aggregator_system_prompt = AGGREGATOR_SYSTEM_PROMPT + "\n\n Responses from models:"

    async def classify(self, text: str) -> dict:
        # single_responses = await asyncio.gather(*[await self.chat(model, text) for model in self.reference_models])
        single_responses = []
        for model in self.reference_models:
            response = await self.chat(model, text)
            single_responses.append(response)
            # await asyncio.sleep(5)

        self.attempt = 0

        while self.attempt < self.retries:
            try:
                final_response = await self.together_client.chat.completions.create(
                    model=self.aggregator_model,
                    messages=[
                        {"role": "system", "content": self.aggregator_system_prompt},
                        {"role": "user", "content": ",".join(str(elem) for elem in single_responses)},
                    ],
                    stream=False
                )
                logger.info("Attempt: %s", self.attempt)
                logger.info("Model response: %s", final_response.choices[0].message.content)

                response_dict = self.__parse_response__(final_response.choices[0].message.content)
                return response_dict
            except Exception as e:
                logger.error(f"Attempt {self.attempt + 1} failed with error: {e}")
                self.attempt += 1
                if self.attempt < self.retries:
                    await asyncio.sleep(5)
                else:
                    raise e

    async def chat(self, model: str, user_prompt: str) -> str:
        """Run a single LLM call with a reference model"""
        user_prompt = CONTENT_CHECK_PROMPT.format(user_prompt=user_prompt)

        response = await self.together_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=0.7,
            max_tokens=512,
        )

        logger.info("Using model: %s", model)

        return response.choices[0].message.content

    @staticmethod
    def __parse_response__(response: str) -> dict:
        """Parse the response from the aggregator model"""
        try:
            response_dict = json.loads(response)
            return response_dict
        except json.JSONDecodeError:
            logger.info("Could not parse response: %s. Retry...", response)

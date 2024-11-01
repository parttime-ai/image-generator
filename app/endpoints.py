import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from app.config import AppConfiguration

from app.models.requests import ImageRequest, TextPrompt

router = APIRouter()
appSettings = AppConfiguration()


@router.post("/generate-image/together")
async def generate_image_together(request: Request, image_request: ImageRequest) -> str:
    model = request.app.state.together_ai
    image = await model.generate(
        prompt=image_request.prompt,
        width=image_request.width,
        height=image_request.height,
        num_inference_steps=image_request.num_inference_steps,
        guidance_scale=image_request.guidance_scale,
        seed=image_request.seed
    )

    if image is None:
        raise HTTPException(status_code=500, detail="Could not generate image")

    return image


@router.post("/generate-image/local-sd")
async def generate_image_local_sd(request: Request, image_request: ImageRequest) -> str:
    model = request.app.state.stable_diffusion
    image = await model.generate(
        prompt=image_request.prompt,
        width=image_request.width,
        height=image_request.height,
        num_inference_steps=image_request.num_inference_steps,
        guidance_scale=image_request.guidance_scale,
        seed=image_request.seed
    )

    if image is None:
        raise HTTPException(status_code=500, detail="Could not generate image")

    return image


@router.post("/nsfw-text-detection/distil-clf")
async def nsfw_text_detection_distil(request: Request, text_prompt: TextPrompt) -> Any:
    model = request.app.state.distil_clf
    prediction = await model.classify(text_prompt.prompt)

    return prediction


@router.post("/nsfw-text-detection/roberta-clf")
async def nsfw_text_detection_roberta(request: Request, text_prompt: TextPrompt) -> Any:
    model = request.app.state.roberta_clf
    prediction = await model.classify(text_prompt.prompt)

    return prediction
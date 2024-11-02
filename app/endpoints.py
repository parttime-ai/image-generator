import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from app.config import AppConfiguration

from app.models.requests import ImageRequest, TextPrompt
from app.models.response import ContentAssessment, OverallAssessment, ConfidenceLevel

router = APIRouter()
appSettings = AppConfiguration()
logger = logging.getLogger(__name__)


@router.post("/generate-image/together")
async def generate_image_together(request: Request, image_request: ImageRequest) -> str:
    model = request.app.state.together_ai

    text_content_assessor = request.app.state.moa_clf
    try:
        if image_request.nsfw_check:
            logger.info(f"Classifying text: {image_request.prompt}")
            prediction = await text_content_assessor.classify(image_request.prompt)
            assessment = ContentAssessment(**prediction)
            if (assessment.overall_assessment == OverallAssessment.inappropriate and
                    (assessment.confidence_level == ConfidenceLevel.medium or
                     assessment.confidence_level == ConfidenceLevel.high)):
                raise HTTPException(status_code=400, detail=f"NSFW content detected. Reason: {assessment.reason}")

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
    except Exception as e:
        logging.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Could not classify text")


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


@router.post("/nsfw-text-detection/moa-clf")
async def nsfw_text_detection_moa(request: Request, text_prompt: TextPrompt) -> ContentAssessment:
    model = request.app.state.moa_clf
    try:
        logger.info(f"Classifying text: {text_prompt.prompt}")
        prediction = await model.classify(text_prompt.prompt)
        assessment = ContentAssessment(**prediction)

        return assessment
    except Exception as e:
        logging.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Could not classify text")

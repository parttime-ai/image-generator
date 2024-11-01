from pydantic import BaseModel


class ImageRequest(BaseModel):
    prompt: str = "Dog with a party hat in the forest"
    width: int = 1024
    height: int = 1024
    num_inference_steps: int = 6
    guidance_scale: int | None = 7
    seed: int | None = 42


class TextPrompt(BaseModel):
    prompt: str

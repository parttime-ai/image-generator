import base64
from io import BytesIO

import torch
from PIL.Image import Image
from diffusers import StableDiffusionPipeline
from torch import Generator
from app.api import IImageGenerator


class StableDiffusion(IImageGenerator):
    pipe: StableDiffusionPipeline
    generator: Generator
    device: str

    def __init__(self, model: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"):
        self.device = self.__get_device__()
        if "stable-diffusion" not in model:
            raise ValueError("Invalid model")
        self.pipe = StableDiffusionPipeline.from_pretrained(model)
        self.pipe = self.pipe.to(self.device)
        self.generator = torch.Generator(device=self.device)

    async def generate(self,
                 prompt: str,
                 width: int = 512,
                 height: int = 512,
                 num_inference_steps: int | None = 10,
                 guidance_scale: int | None = 7,
                 seed: int | None = 42) -> str | None:

        self.generator.manual_seed(seed)
        image = self.pipe(
            prompt=prompt,
            height=height,
            width=width,
            negative_prompt=None,
            num_images_per_prompt=1,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=self.generator).images[0]

        image = self.__PIL_image_to_B64__(image)
        return image

    @staticmethod
    def __get_device__() -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def __PIL_image_to_B64__(image: Image) -> str:
        im_file = BytesIO()
        image.save(im_file, format="JPEG")
        im_bytes = im_file.getvalue()
        im_b64 = base64.b64encode(im_bytes).decode("utf-8")

        return im_b64

from together import AsyncTogether

from app.api import IImageGenerator


class TogetherAI(IImageGenerator):
    client: AsyncTogether

    def __init__(self, api_key: str, model: str = "black-forest-labs/FLUX.1-schnell"):
        self.client = AsyncTogether(api_key=api_key)
        self.model = model

    async def generate(self,
                       prompt: str,
                       width: int = 1024,
                       height: int = 1024,
                       num_inference_steps: int | None = 20,
                       guidance_scale: int | None = 7,
                       seed: int | None = None) -> str | None:
        image_response = await self.client.images.generate(
            prompt=prompt,
            model=self.model,
            steps=num_inference_steps,
            seed=seed,
            n=1,
            height=height,
            width=width,
            negative_prompt=None,
            response_format="b64_json")

        return image_response.data[0].b64_json

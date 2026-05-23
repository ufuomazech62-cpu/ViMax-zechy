# https://ai.google.dev/gemini-api/docs/image-generation

import logging
import os
import asyncio
from PIL import Image
from typing import List, Optional
from google import genai
from google.genai import types
from google.genai.errors import ClientError
from tenacity import retry, stop_after_attempt
from interfaces.image_output import ImageOutput
from utils.retry import after_func
from utils.rate_limiter import RateLimiter


class ImageGeneratorNanobananaGoogleAPI:
    def __init__(
        self,
        api_key: str = "",
        project: str = "",
        location: str = "",
        model: str = "gemini-2.5-flash-image",
        rate_limiter: Optional[RateLimiter] = None,
    ):
        self.model = model
        self.rate_limiter = rate_limiter

        # Prefer Vertex AI when project is provided; fall back to API key
        if project:
            client_kwargs = {
                "project": project,
                "location": location or "us-central1",
            }
            logging.info(
                "Image gen: using Vertex AI (project=%s, location=%s)",
                project, client_kwargs["location"],
            )
        else:
            env_project = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
            env_location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
            env_key = os.environ.get("GOOGLE_API_KEY", "") or api_key

            if env_project:
                client_kwargs = {
                    "project": env_project,
                    "location": env_location,
                }
                logging.info(
                    "Image gen: using Vertex AI from env (project=%s, location=%s)",
                    env_project, env_location,
                )
            elif env_key:
                client_kwargs = {"api_key": env_key}
                logging.info("Image gen: using Google AI Studio API key")
            else:
                raise ValueError(
                    "No Vertex AI project or Google API key configured. "
                    "Set GOOGLE_CLOUD_PROJECT + GOOGLE_CLOUD_LOCATION for Vertex AI, "
                    "or GOOGLE_API_KEY for AI Studio."
                )

        self.client = genai.Client(**client_kwargs)

    @retry(stop=stop_after_attempt(3), after=after_func)
    async def generate_single_image(
        self,
        prompt: str,
        reference_image_paths: List[str] = [],
        aspect_ratio: Optional[str] = "16:9",
        **kwargs,
    ) -> ImageOutput:

        """
            aspect_ratio: The aspect ratio of the image.
        """

        logging.info(f"Calling {self.model} to generate image...")

        # Apply rate limiting if configured
        if self.rate_limiter:
            await self.rate_limiter.acquire()

        reference_images = [Image.open(path) for path in reference_image_paths]

        # Retry logic for rate limit errors
        max_retries = 3
        retry_delay = 5

        for attempt in range(max_retries):
            try:
                response = await self.client.aio.models.generate_content(
                    model=self.model,
                    contents=reference_images + [prompt],
                    config=types.GenerateContentConfig(
                        response_modalities=["IMAGE"],
                        image_config=types.ImageConfig(
                            aspect_ratio=aspect_ratio,
                        ),
                    ),
                )
                break
            except ClientError as e:
                if e.status_code == 429 and attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    logging.warning(f"Rate limit hit (429), retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(wait_time)
                else:
                    raise

        image = None
        text = ""
        for part in response.candidates[0].content.parts:
            if part.text is not None:
                text += part.text
            elif part.inline_data is not None:
                image = part.as_image()

        if image is None:
            logging.error(f"No image generated. The response text is: {text}")
            raise ValueError("No image generated")

        return ImageOutput(fmt="pil", ext="png", data=image)

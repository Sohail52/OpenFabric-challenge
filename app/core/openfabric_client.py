import logging
from typing import Optional, Tuple
from .llm_processor import LLMProcessor  # Make sure llm_processor.py is in the same directory

logger = logging.getLogger(__name__)

class OpenfabricClient:
    """
    Client for handling image and 3D model generation using local models.
    """

    def __init__(self):
        """Initialize the client with local AI processors."""
        self.llm_processor = LLMProcessor()
        logger.info("Local AI models initialized successfully")

    def generate_image(self, prompt: str) -> Tuple[Optional[bytes], Optional[str]]:
        """
        Generate an image from text using local Stable Diffusion model.

        Args:
            prompt (str): The user's input prompt.

        Returns:
            Tuple[Optional[bytes], Optional[str]]: (image_data, error_message)
        """
        try:
            logger.info(f"Generating image for prompt: {prompt}")

            # Expand the prompt using the LLM
            expanded_prompt = self.llm_processor.expand_prompt(prompt)
            if expanded_prompt:
                logger.info(f"Expanded prompt: {expanded_prompt}")
            else:
                expanded_prompt = prompt  # fallback

            # Generate image from expanded prompt
            image_data, error = self.llm_processor.generate_image(expanded_prompt)
            if error:
                return None, error

            return image_data, None

        except Exception as e:
            error_msg = f"Failed to generate image: {str(e)}"
            logger.error(error_msg)
            return None, error_msg

    def generate_3d_model(self, image_data: bytes) -> Tuple[Optional[bytes], Optional[str]]:
        """
        Generate a 3D model from an image (placeholder for now).

        Args:
            image_data (bytes): Image bytes to convert to 3D model.

        Returns:
            Tuple[Optional[bytes], Optional[str]]: (model_data, error_message)
        """
        try:
            logger.info("Generating 3D model from image (test placeholder)")
            return self._generate_test_model(), None
        except Exception as e:
            error_msg = f"Failed to generate 3D model: {str(e)}"
            logger.error(error_msg)
            return None, error_msg

    def _generate_test_model(self) -> bytes:
        """
        Temporary method to simulate 3D model generation.

        Returns:
            bytes: Dummy 3D model data.
        """
        # This would normally be a real GLB model file
        return b"Test 3D Model Data"

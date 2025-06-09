from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import logging
from typing import Optional, Tuple
import os

logger = logging.getLogger(__name__)

class LLMProcessor:
    """
    Handles creative prompt expansion using a local LLM
    """
    
    def __init__(self, model_name: str = "facebook/opt-125m"):
        """Initialize the LLM processor with proper memory management"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing LLM on {self.device}")
        if self.device.type == "cuda":
            logger.info(f"CUDA Device: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        else:
            logger.info("Running on CPU - this will be significantly slower")
        
        # Create cache directories
        self.cache_dir = os.path.join(os.getcwd(), "model_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        try:
            # Try to load from local cache first
            logger.info("Attempting to load model from local cache...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=self.cache_dir,
                local_files_only=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=self.cache_dir,
                local_files_only=True,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                low_cpu_mem_usage=True
            ).to(self.device)  # Move model to device immediately after loading
            
        except Exception as e:
            logger.warning(f"Could not load model from local cache: {str(e)}")
            logger.info("Falling back to simple text expansion...")
            # Fallback to simple text expansion
            self.tokenizer = None
            self.model = None
            
        logger.info("LLM initialized successfully")
    
    def expand_prompt(self, prompt: str) -> Optional[str]:
        """
        Expand a user prompt into a more detailed, creative description
        
        Args:
            prompt (str): The original user prompt
            
        Returns:
            Optional[str]: The expanded prompt, or None if expansion fails
        """
        try:
            logger.info(f"Starting prompt expansion for: {prompt}")
            
            # If model failed to load, use simple expansion
            if self.model is None or self.tokenizer is None:
                logger.info("Using simple text expansion")
                # Simple template-based expansion
                template = f"""A detailed description of {prompt}:
                - The scene is set in a {prompt}
                - The lighting creates a dramatic atmosphere
                - Rich colors and textures fill the scene
                - The mood is immersive and engaging
                - Every detail is carefully crafted for visual impact"""
                return template
            
            # Create a more creative and detailed expansion template
            template = f"""You are a creative AI assistant that transforms simple ideas into vivid, detailed descriptions.
            Your task is to expand this basic concept into a rich, artistic description that captures every visual detail.
            
            Original concept: "{prompt}"
            
            Create a detailed description that includes:
            - Visual elements and composition
            - Lighting and atmosphere
            - Colors and textures
            - Mood and emotion
            - Artistic style and details
            
            Focus on making the description vivid and cinematic, perfect for image generation.
            Respond with ONLY the expanded description, no explanations or other text."""
            
            logger.info("Tokenizing input...")
            # Tokenize and generate
            inputs = self.tokenizer(template, return_tensors="pt").to(self.device)  # Move inputs to same device as model
            
            logger.info("Generating expanded prompt...")
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    max_length=512,  # Increased for more detailed descriptions
                    temperature=0.8,  # Slightly increased for more creativity
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.2  # Added to prevent repetitive phrases
                )
            
            logger.info("Decoding output...")
            # Decode and clean up the response
            expanded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract only the expanded description, removing the template
            expanded = expanded.split("Respond with ONLY the expanded description, no explanations or other text.")[-1].strip()
            
            logger.info(f"Successfully expanded prompt to: {expanded}")
            return expanded
            
        except Exception as e:
            logger.error(f"Failed to expand prompt: {str(e)}", exc_info=True)
            return None
            
    def __call__(self, prompt: str) -> Optional[str]:
        """Convenience method to call expand_prompt directly"""
        return self.expand_prompt(prompt)

    def generate_image(self, prompt: str) -> Tuple[Optional[bytes], Optional[str]]:
        """
        Generate an image from text using local Stable Diffusion model.
        
        Args:
            prompt (str): The text prompt to generate an image from
            
        Returns:
            Tuple[Optional[bytes], Optional[str]]: (image_data, error_message)
        """
        try:
            from diffusers import StableDiffusionPipeline
            import torch
            from huggingface_hub import scan_cache_dir
            import shutil
            
            logger.info(f"Current device: {self.device}")
            if self.device.type == "cuda":
                logger.info(f"CUDA Memory before loading: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            
            # Check if model is already downloaded
            cache_info = scan_cache_dir()
            model_id = "runwayml/stable-diffusion-v1-5"
            model_exists = any(repo.repo_id == model_id for repo in cache_info.repos)
            
            if model_exists:
                logger.info("Using existing Stable Diffusion model from cache")
            else:
                logger.info("Stable Diffusion model not found in cache. Downloading...")
            
            # Initialize the pipeline if not already done
            if not hasattr(self, 'image_pipeline'):
                logger.info("Loading Stable Diffusion pipeline...")
                self.image_pipeline = StableDiffusionPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                    local_files_only=model_exists  # Only use local files if model exists
                ).to(self.device)
                logger.info("Pipeline loaded successfully")
                if self.device.type == "cuda":
                    logger.info(f"CUDA Memory after loading: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            
            # Generate the image
            logger.info("Generating image...")
            image = self.image_pipeline(prompt).images[0]
            logger.info("Image generation completed")
            
            # Convert to bytes
            import io
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            
            return img_byte_arr, None
            
        except Exception as e:
            error_msg = f"Failed to generate image: {str(e)}"
            logger.error(error_msg)
            return None, error_msg
            
    def cleanup_models(self):
        """
        Clean up downloaded models to free up space.
        """
        try:
            from huggingface_hub import scan_cache_dir, delete_from_cache
            
            # Scan cache directory
            cache_info = scan_cache_dir()
            
            # Delete Stable Diffusion model
            model_id = "runwayml/stable-diffusion-v1-5"
            for repo in cache_info.repos:
                if repo.repo_id == model_id:
                    delete_from_cache(repo.repo_id)
                    logger.info(f"Deleted model: {model_id}")
                    break
            
            # Clear pipeline if it exists
            if hasattr(self, 'image_pipeline'):
                del self.image_pipeline
                logger.info("Cleared image pipeline from memory")
                
        except Exception as e:
            logger.error(f"Failed to cleanup models: {str(e)}")
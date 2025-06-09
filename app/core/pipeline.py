import logging
import time
from typing import Dict, Optional
import torch
from pathlib import Path
from datetime import datetime

from .llm_processor import LLMProcessor
from .memory import MemorySystem
from .openfabric_client import OpenfabricClient

logger = logging.getLogger(__name__)

class AICreativePipeline:
    """
    Main pipeline that orchestrates the entire generation process:
    1. LLM prompt expansion
    2. Text to Image generation
    3. Image to 3D model conversion
    4. Memory storage
    """
    
    def __init__(self):
        self.llm = LLMProcessor()
        self.memory = MemorySystem()
        self.openfabric = OpenfabricClient()
        
        # Ensure assets directory exists
        Path("generated_assets").mkdir(exist_ok=True)
        
    def process_prompt(
        self,
        prompt: str,
        progress_callback=None,
        use_memory: bool = True,
        enhance_prompt: bool = True
    ) -> Optional[Dict]:
        """
        Process a user prompt through the entire pipeline
        
        Args:
            prompt (str): The user's original prompt
            progress_callback (callable): Optional callback for progress updates
            use_memory (bool): Whether to use memory for enhancement
            enhance_prompt (bool): Whether to use LLM for prompt enhancement
            
        Returns:
            Dict: Generation results including paths and metadata
        """
        def update_progress(stage: str, progress: float, status: str = ""):
            if progress_callback:
                progress_callback(stage, progress, status)
            logger.info(f"Progress - {stage}: {progress:.0f}% - {status}")
        
        start_time = time.time()
        try:
            update_progress("Initialization", 0, "Starting generation pipeline")
            
            # Step 1: Memory Enhancement (5%)
            if use_memory:
                update_progress("Memory Search", 5, "Checking similar generations")
                similar = self.memory.find_similar_generations(prompt, k=1)
                if similar:
                    logger.info(f"Found similar generation: {similar[0]['original_prompt']}")
            
            # Step 2: LLM Enhancement (15%)
            expanded_prompt = None
            if enhance_prompt:
                update_progress("Prompt Enhancement", 15, "Expanding prompt with LLM")
                expanded_prompt = self.llm.expand_prompt(prompt)
                if not expanded_prompt:
                    logger.warning("LLM expansion failed, using original prompt")
                    expanded_prompt = prompt
            else:
                expanded_prompt = prompt
            
            # Step 3: Generate Image (50%)
            update_progress("Image Generation", 30, "Generating image from prompt")
            image_data, error = self.openfabric.generate_image(expanded_prompt)
            if error:
                update_progress("Error", 100, f"Image generation failed: {error}")
                return {
                    'error': error,
                    'stage': 'image_generation',
                    'original_prompt': prompt,
                    'expanded_prompt': expanded_prompt
                }
            
            # Save image (60%)
            update_progress("Saving", 60, "Saving generated image")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_path = f"generated_assets/image_{timestamp}.png"
            with open(image_path, 'wb') as f:
                f.write(image_data)
            
            # Step 4: Generate 3D Model (80%)
            update_progress("3D Generation", 70, "Converting image to 3D model")
            model_data, error = self.openfabric.generate_3d_model(image_data)
            if error:
                update_progress("Error", 100, f"3D model generation failed: {error}")
                return {
                    'error': error,
                    'stage': '3d_generation',
                    'original_prompt': prompt,
                    'expanded_prompt': expanded_prompt,
                    'image_path': image_path
                }
            
            # Save 3D model (90%)
            update_progress("Saving", 90, "Saving generated 3D model")
            model_path = f"generated_assets/model_{timestamp}.glb"
            with open(model_path, 'wb') as f:
                f.write(model_data)
            
            # Step 5: Store in Memory (95%)
            update_progress("Finalizing", 95, "Storing results in memory")
            generation_time = time.time() - start_time
            cuda_memory = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
            
            metadata = {
                'generation_time': generation_time,
                'cuda_memory_used': cuda_memory,
                'similar_found': bool(similar) if use_memory else None,
                'prompt_enhanced': enhance_prompt
            }
            
            self.memory.store_generation(
                "anonymous",  # TODO: Add user management
                prompt,
                expanded_prompt,
                image_path,
                model_path,
                metadata
            )
            
            update_progress("Complete", 100, "Generation completed successfully")
            
            return {
                'original_prompt': prompt,
                'expanded_prompt': expanded_prompt,
                'image_path': image_path,
                'model_path': model_path,
                'generation_time': f"{generation_time:.2f}s",
                'cuda_memory_used': cuda_memory
            }
            
        except Exception as e:
            error_msg = f"Pipeline processing failed: {str(e)}"
            logger.error(error_msg)
            update_progress("Error", 100, error_msg)
            return {
                'error': error_msg,
                'stage': 'pipeline',
                'original_prompt': prompt
            }
            
    def cleanup_old_assets(self, max_age_days: int = 7):
        """Clean up old generated assets to free space"""
        try:
            cutoff = datetime.now().timestamp() - (max_age_days * 24 * 60 * 60)
            
            for ext in ['png', 'glb', 'stl']:
                for file in Path("generated_assets").glob(f"*.{ext}"):
                    if file.stat().st_mtime < cutoff:
                        file.unlink()
                        logger.info(f"Cleaned up old asset: {file}")
                        
        except Exception as e:
            logger.error(f"Asset cleanup failed: {str(e)}") 
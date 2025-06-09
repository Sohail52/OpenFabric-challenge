import os
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict
import subprocess
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from diffusers import StableDiffusionPipeline
import io

logger = logging.getLogger(__name__)

class OneClickGenerator:
    """Simplified one-click generation pipeline"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.assets_dir = Path("generated_assets")
        self.assets_dir.mkdir(exist_ok=True)
        
        # Initialize models
        self.init_models()
        
    def init_models(self):
        """Initialize both LLM and diffusion models"""
        try:
            # LLM for prompt expansion
            self.llm_tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
            self.llm_model = AutoModelForCausalLM.from_pretrained("facebook/opt-1.3b").to(self.device)
            
            # Diffusion model for image generation
            self.diffusion_pipe = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
            ).to(self.device)
            
        except Exception as e:
            logger.error(f"Model initialization failed: {str(e)}")
            raise

    def expand_prompt(self, prompt: str) -> str:
        """Enhanced prompt expansion with fallback"""
        try:
            inputs = self.llm_tokenizer(
                f"Enhance this image generation prompt with vivid details: {prompt}",
                return_tensors="pt"
            ).to(self.device)
            
            outputs = self.llm_model.generate(
                inputs["input_ids"],
                max_length=200,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.llm_tokenizer.eos_token_id
            )
            
            expanded = self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
            return expanded if len(expanded) > len(prompt) + 20 else prompt
            
        except Exception as e:
            logger.warning(f"Prompt expansion failed, using original: {str(e)}")
            return prompt

    def generate(self, prompt: str, stub) -> Dict:
        """One-click generation pipeline"""
        start_time = time.time()
        
        try:
            # Step 1: Expand prompt
            enhanced_prompt = self.expand_prompt(prompt)
            
            # Step 2: Generate image
            image = self.diffusion_pipe(enhanced_prompt).images[0]
            
            # Save image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            image_path = str(self.assets_dir / f"image_{timestamp}.png")
            image.save(image_path)
            
            # Step 3: Generate 3D model
            model_path = str(self.assets_dir / f"model_{timestamp}.glb")
            success, message = self.generate_3d_model(stub, image_path, model_path)
            
            return {
                "status": "success" if success else "partial_success",
                "original_prompt": prompt,
                "enhanced_prompt": enhanced_prompt,
                "image_path": image_path,
                "model_path": model_path if success else None,
                "generation_time": f"{time.time() - start_time:.2f}s",
                "cuda_memory_used": f"{torch.cuda.memory_allocated() / 1024**2:.1f} MB" if torch.cuda.is_available() else "N/A",
                "message": message
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "generation_time": f"{time.time() - start_time:.2f}s"
            }

    def generate_3d_model(self, stub, image_path: str, output_path: str) -> Tuple[bool, str]:
        """Generate 3D model using Openfabric's app"""
        try:
            with open(image_path, 'rb') as f:
                image_data = f.read()
            
            result = stub.call(
                '69543f29-4d41-4afc-7f29-3d51591f11eb',  # Image-to-3D app ID
                {'image': image_data},
                'super-user'
            )
            
            model_data = result.get('result')
            if not model_data or not model_data.startswith(b'glTF'):
                return False, "Invalid GLB data received"
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'wb') as f:
                f.write(model_data)
                
            return True, "Success"
            
        except Exception as e:
            return False, f"3D generation failed: {str(e)}"

# Example usage
if __name__ == "__main__":
    from openfabric_pysdk.stub import Stub
    
    # Initialize generator
    generator = OneClickGenerator()
    stub = Stub(app_ids=['69543f29-4d41-4afc-7f29-3d51591f11eb'])
    
    # One-click generation
    result = generator.generate("a peaceful image surrounded by cherry blossom trees during spring", stub)
    print(result)
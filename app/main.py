import logging
from typing import Dict
import os
from datetime import datetime
import json
from pathlib import Path
import torch
import gc
from huggingface_hub import hf_hub_download

from ontology_dc8f06af066e4a7880a5938933236037.config import ConfigClass
from ontology_dc8f06af066e4a7880a5938933236037.input import InputClass
from ontology_dc8f06af066e4a7880a5938933236037.output import OutputClass
from openfabric_pysdk.context import AppModel, State, ExecutionResult
from openfabric_pysdk.stub import Stub
from openfabric_pysdk.schema import InputClass, OutputClass, ConfigClass
from openfabric_pysdk.config import configurations
from core.llm import LLMProcessor
from core.memory import MemorySystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Force CUDA usage
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Please ensure CUDA is properly installed.")

# Set CUDA device
torch.cuda.set_device(0)
logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
logger.info(f"CUDA Version: {torch.version.cuda}")

# Configurations for the app
configurations: Dict[str, ConfigClass] = dict()

# Initialize LLM and Memory System
llm_processor = None
memory_system = None

def init_components():
    """Initialize LLM and Memory System components"""
    global llm_processor, memory_system
    try:
        # Clear GPU memory
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("Cleared GPU memory cache")
        
        # Set CUDA device properties
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        
        # Create necessary directories
        os.makedirs("generated_assets", exist_ok=True)
        os.makedirs("model_cache", exist_ok=True)
        
        # Initialize components
        llm_processor = LLMProcessor()
        memory_system = MemorySystem()
        logger.info("Successfully initialized LLM and Memory System")
        
        # Log memory status
        logger.info(f"CUDA Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        logger.info(f"CUDA Memory Reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
        
    except Exception as e:
        logger.error(f"Error initializing components: {str(e)}")
        raise

def cleanup():
    """Cleanup resources and free memory"""
    global llm_processor, memory_system
    try:
        if llm_processor and hasattr(llm_processor, 'model'):
            del llm_processor.model
            torch.cuda.empty_cache()
            gc.collect()
        logger.info("Successfully cleaned up resources")
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")

############################################################
# Config callback function
############################################################
def config(configuration: Dict[str, ConfigClass], state: State) -> None:
    """
    Stores user-specific configuration data.

    Args:
        configuration (Dict[str, ConfigClass]): A mapping of user IDs to configuration objects.
        state (State): The current state of the application (not used in this implementation).
    """
    for uid, conf in configuration.items():
        logging.info(f"Saving new config for user with id:'{uid}'")
        configurations[uid] = conf

############################################################
# Execution callback function
############################################################
def execute(model: AppModel) -> None:
    """
    Main execution entry point for handling a model pass.

    Args:
        model (AppModel): The model object containing request and response structures.
    """
    global llm_processor, memory_system

    # Initialize components if not already done
    if llm_processor is None or memory_system is None:
        init_components()

    try:
        # Retrieve input
        request: InputClass = model.request
        prompt = request.prompt
        logger.info(f"Processing prompt: {prompt}")

        # Retrieve user config
        user_config: ConfigClass = configurations.get('super-user', None)
        logger.info(f"Using configuration: {user_config}")

        # Initialize the Stub with app IDs
        app_ids = user_config.app_ids if user_config else []
        stub = Stub(app_ids)

        # Step 1: Expand prompt using local LLM
        logger.info("Expanding prompt using local LLM...")
        expanded_prompt = llm_processor.expand_prompt(prompt)
        logger.info(f"Expanded prompt: {expanded_prompt}")

        # Step 2: Generate image using Text-to-Image app
        logger.info("Generating image...")
        text_to_image_result = stub.call(
            'f0997a01-d6d3-a5fe-53d8-561300318557',  # Text-to-Image app ID
            {'prompt': expanded_prompt},
            'super-user'
        )
        
        # Save the generated image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = f"generated_assets/image_{timestamp}.png"
        with open(image_path, 'wb') as f:
            f.write(text_to_image_result.get('result'))
        logger.info(f"Image saved to: {image_path}")

        # Step 3: Convert image to 3D model
        logger.info("Converting image to 3D model...")
        image_to_3d_result = stub.call(
            '69543f29-4d41-4afc-7f29-3d51591f11eb',  # Image-to-3D app ID
            {'image': image_path},
            'super-user'
        )

        # Save the 3D model
        model_path = f"generated_assets/model_{timestamp}.glb"
        with open(model_path, 'wb') as f:
            f.write(image_to_3d_result.get('result'))
        logger.info(f"3D model saved to: {model_path}")

        # Step 4: Store in memory system
        memory_system.save_generation(
            original_prompt=prompt,
            expanded_prompt=expanded_prompt,
            image_path=image_path,
            model_path=model_path,
            timestamp=timestamp
        )

        # Step 5: Search for similar generations
        similar_generations = memory_system.search_similar(prompt)
        
        # Prepare response
        response: OutputClass = model.response
        response.message = (
            f"Successfully generated 3D model from prompt: {prompt}\n"
            f"Image saved to: {image_path}\n"
            f"3D model saved to: {model_path}\n"
            f"Found {len(similar_generations)} similar generations"
        )
        logger.info("Processing completed successfully")

    except Exception as e:
        logger.error(f"Error in execution: {str(e)}")
        response: OutputClass = model.response
        response.message = f"Error processing request: {str(e)}"
    finally:
        # Cleanup resources
        cleanup()
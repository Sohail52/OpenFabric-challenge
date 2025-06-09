import os
import torch
import streamlit as st
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from diffusers import StableDiffusionPipeline
import logging
from dotenv import load_dotenv

# ======================
# SYSTEM CONFIGURATION
# ======================
# Auto-detect CUDA availability
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True if device == "cuda" else False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Create necessary directories
os.makedirs("model_cache", exist_ok=True)
os.makedirs("generated_assets", exist_ok=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'memory' not in st.session_state:
    st.session_state.memory = []
if 'generated_assets' not in st.session_state:
    st.session_state.generated_assets = []
if 'current_image' not in st.session_state:
    st.session_state.current_image = None

# ======================
# MODEL LOADING (GPU-OPTIMIZED)
# ======================

@st.cache_resource
def load_model():
    try:
        model_name = "facebook/opt-1.3b"  # Using a larger model since we have GPU
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto",
            offload_folder="model_cache",  # Specify offload folder
            local_files_only=False  # Allow downloading if needed
        )
        return model, tokenizer
    except Exception as e:
        logger.error(f"Failed to load language model: {str(e)}")
        st.error(f"Failed to load language model: {str(e)}")
        return None, None

@st.cache_resource
def load_stable_diffusion():
    try:
        model_id = "runwayml/stable-diffusion-v1-5"
        
        # First try downloading with safetensors
        try:
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                cache_dir="model_cache",
                local_files_only=False,
                safety_checker=None,
                use_safetensors=True
            )
        except Exception as e:
            logger.warning(f"Failed to load with safetensors, trying without: {str(e)}")
            # If safetensors fails, try without it
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                cache_dir="model_cache",
                local_files_only=False,
                safety_checker=None,
                use_safetensors=False
            )
        
        # Move to device
        if device == "cuda":
            pipe = pipe.to(device)
            pipe.enable_xformers_memory_efficient_attention()
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to(device)
            pipe.enable_attention_slicing()
        
        return pipe
    except Exception as e:
        logger.error(f"Failed to load Stable Diffusion: {str(e)}")
        st.error(f"Failed to load Stable Diffusion: {str(e)}")
        return None

# ======================
# CORE FUNCTIONS (GPU-OPTIMIZED)
# ======================

def generate_image(prompt, pipe):
    try:
        with torch.autocast(device):  # Automatic mixed precision for GPU
            image = pipe(
                prompt,
                num_inference_steps=50 if device == "cuda" else 25,
                guidance_scale=7.5
            ).images[0]
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = f"generated_assets/image_{timestamp}.png"
        os.makedirs("generated_assets", exist_ok=True)
        image.save(image_path)
        return image_path
    except Exception as e:
        st.error(f"Image generation failed: {str(e)}")
        return None

def generate_3d_model(image_path=None):
    """Generate a basic 3D model"""
    try:
        # Create a simple 3D model (cube)
        vertices = [
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1]
        ]
        
        faces = [
            [0,1,2], [2,3,0], [4,5,6], [6,7,4],
            [3,2,6], [6,7,3], [0,1,5], [5,4,0],
            [1,2,6], [6,5,1], [0,3,7], [7,4,0]
        ]
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        obj_path = f"generated_assets/model_{timestamp}.obj"
        os.makedirs("generated_assets", exist_ok=True)
        
        with open(obj_path, 'w') as f:
            for v in vertices:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")
            for face in faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
        
        return obj_path
    except Exception as e:
        st.error(f"3D model generation failed: {str(e)}")
        return None

# ======================
# STREAMLIT UI
# ======================

def main():
    st.title("AI Creative Pipeline (GPU Optimized)")
    st.write(f"Running on: {device.upper()}")
    st.write("Transform your ideas into images and 3D models!")

    # Initialize components
    try:
        model, tokenizer = load_model()
        pipe = load_stable_diffusion()
        if model is None or pipe is None:
            st.error("Failed to initialize one or more components. Please check the logs.")
            return
    except Exception as e:
        st.error(f"Error initializing components: {str(e)}")
        return

    # Chat interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if "assets" in message:
                for asset in message["assets"]:
                    if asset.endswith(".png"):
                        st.image(asset)
                    elif asset.endswith(".obj"):
                        st.write(f"3D Model: {asset}")

    # User input
    if prompt := st.chat_input("Describe what you want to create..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Process the prompt
        with st.chat_message("assistant"):
            with st.spinner("Processing your request..."):
                try:
                    # Generate image
                    image_path = generate_image(prompt, pipe)
                    if image_path:
                        st.image(image_path)
                        
                        # Generate 3D model
                        st.write("Generating 3D model...")
                        model_path = generate_3d_model(image_path)
                        
                        # Store results
                        assets = [image_path]
                        if model_path:
                            assets.append(model_path)
                            st.write(f"3D model generated: {model_path}")
                        
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"Generated image and 3D model from: {prompt}",
                            "assets": assets
                        })
                except Exception as e:
                    st.error(f"Error processing request: {str(e)}")

if __name__ == "__main__":
    main()
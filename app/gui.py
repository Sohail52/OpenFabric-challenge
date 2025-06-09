import streamlit as st
import requests
import json
from pathlib import Path
import base64
from PIL import Image
import io
import torch
import logging
from datetime import datetime
import os
import plotly.graph_objects as go
from stl import mesh
import numpy as np
from core.pipeline import AICreativePipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
API_URL = "http://localhost:8888/execution"
GENERATED_ASSETS_DIR = "generated_assets"

# Initialize session state
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = AICreativePipeline()
if 'progress' not in st.session_state:
    st.session_state.progress = (0, "", "")  # (progress, stage, status)

def verify_cuda():
    """Verify CUDA availability and display information"""
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        st.success("✅ CUDA is available!")
        st.info(f"GPU: {torch.cuda.get_device_name(0)}")
        st.info(f"CUDA Version: {torch.version.cuda}")
    else:
        st.warning("⚠️ CUDA is not available. Running on CPU.")
    return cuda_available

def load_image(image_path):
    """Load and display an image"""
    try:
        with open(image_path, "rb") as f:
            image_data = f.read()
        return Image.open(io.BytesIO(image_data))
    except Exception as e:
        logger.error(f"Error loading image: {str(e)}")
        st.error(f"Error loading image: {str(e)}")
        return None

def display_memory_info():
    """Display current CUDA memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**2
        reserved = torch.cuda.memory_reserved(0) / 1024**2
        st.metric("GPU Memory Usage", f"{allocated:.1f} MB / {reserved:.1f} MB")

def update_progress(stage: str, progress: float, status: str):
    """Update progress in session state and display progress bar"""
    st.session_state.progress = (progress, stage, status)

def main():
    st.set_page_config(
        page_title="AI Creative Pipeline",
        page_icon="🎨",
        layout="wide"
    )
    
    # GPU info if available
    if torch.cuda.is_available():
        st.sidebar.success(f"Using GPU: {torch.cuda.get_device_name(0)}")
        memory_used = torch.cuda.memory_allocated() / 1024**2
        st.sidebar.progress(min(memory_used / 8000, 1.0))  # Assuming 8GB GPU
        st.sidebar.text(f"Memory Used: {memory_used:.1f} MB")
    else:
        st.sidebar.warning("Running on CPU")
    
    # Main interface
    st.title("🚀 AI Creative Pipeline")
    st.write("Transform your ideas into 3D reality")
    
    # Input section
    prompt = st.text_area("Enter your creative prompt:", height=100)
    
    col1, col2 = st.columns(2)
    with col1:
        use_memory = st.checkbox("Use Memory Enhancement", value=True)
    with col2:
        enhance_prompt = st.checkbox("Use LLM Enhancement", value=True)
    
    # Preview expanded prompt if enhancement is enabled
    if enhance_prompt and prompt:
        with st.expander("Preview Enhanced Prompt", expanded=True):
            try:
                # Verify model is initialized
                if not hasattr(st.session_state.pipeline.llm, 'model'):
                    st.error("LLM model not properly initialized")
                    return
                    
                expanded = st.session_state.pipeline.llm.expand_prompt(prompt)
                if expanded:
                    st.write("Enhanced prompt:")
                    st.info(expanded)
                else:
                    st.warning("Could not generate enhanced prompt preview. Check the logs for details.")
            except Exception as e:
                st.error(f"Error during prompt expansion: {str(e)}")
                logger.exception("Error in prompt expansion preview")
    
    if st.button("Generate", type="primary"):
        if not prompt:
            st.error("Please enter a prompt first!")
            return
            
        try:
            # Create placeholder for progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def progress_callback(stage: str, progress: float, status: str):
                progress_bar.progress(int(progress))
                status_text.text(f"{stage}: {status}")
            
            # Process the prompt
            result = st.session_state.pipeline.process_prompt(
                prompt,
                progress_callback=progress_callback,
                use_memory=use_memory,
                enhance_prompt=enhance_prompt
            )
            
            if not result:
                st.error("Generation failed with an unknown error")
                return
                
            if 'error' in result:
                st.error(f"Generation failed at {result['stage']}: {result['error']}")
                
                # Show partial results if available
                if 'image_path' in result:
                    st.write("Partial results (image was generated):")
                    st.image(result['image_path'])
                return
            
            # Display results in tabs
            tab1, tab2, tab3 = st.tabs(["Generated Image", "3D Model", "Generation Details"])
            
            with tab1:
                st.image(result['image_path'])
                
            with tab2:
                if result['model_path'].endswith('.glb'):
                    st.write("3D model generated successfully!")
                    st.download_button(
                        "Download 3D Model",
                        open(result['model_path'], 'rb'),
                        file_name=os.path.basename(result['model_path'])
                    )
                else:
                    st.error("3D model format not supported for visualization")
                
            with tab3:
                st.json({
                    'original_prompt': prompt,
                    'enhanced_prompt': result['expanded_prompt'],
                    'generation_time': result['generation_time'],
                    'cuda_memory_used': f"{result['cuda_memory_used']:.1f} MB"
                })
                
        except Exception as e:
            logger.exception("Error during generation")
            st.error(f"An unexpected error occurred: {str(e)}")
        finally:
            # Clear progress display
            if 'progress_bar' in locals():
                progress_bar.empty()
            if 'status_text' in locals():
                status_text.empty()
    
    # Stats section
    st.header("📊 System Stats")
    stats = st.session_state.pipeline.memory.get_generation_stats()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Generations", stats['total_generations'])
    with col2:
        st.metric("Avg. CUDA Memory", f"{stats['average_cuda_memory_used_mb']:.1f} MB")
    with col3:
        st.metric("Most Used Device", stats['most_used_device'])
    
    # Memory Browser
    st.header("🗄 Generation History")
    
    # Search options
    search_type = st.radio(
        "Search by:",
        ["Recent", "Similar", "Date"],
        horizontal=True
    )
    
    if search_type == "Similar":
        search_prompt = st.text_input("Enter prompt to find similar generations:")
        if search_prompt:
            results = st.session_state.pipeline.memory.find_similar_generations(search_prompt)
            
    elif search_type == "Date":
        date_query = st.text_input("Enter date (e.g., 'last Thursday', 'yesterday'):")
        if date_query:
            results = st.session_state.pipeline.memory.find_by_date(date_query)
            
    else:  # Recent
        results = st.session_state.pipeline.memory.get_recent_generations(10)
        
    # Display results
    if 'results' in locals() and results:
        for idx, gen in enumerate(results):
            timestamp = gen.get('timestamp', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            with st.expander(f"Generation from {timestamp}"):
                st.write(f"Original prompt: {gen.get('original_prompt', 'No prompt available')}")
                if 'image_path' in gen and os.path.exists(gen['image_path']):
                    st.image(gen['image_path'], width=300)
                if 'model_path' in gen and os.path.exists(gen['model_path']):
                    st.download_button(
                        "Download 3D Model",
                        open(gen['model_path'], 'rb'),
                        file_name=os.path.basename(gen['model_path']),
                        key=f"download_btn_{idx}_{timestamp}"
                    )

if __name__ == "__main__":
    main() 
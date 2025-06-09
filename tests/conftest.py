import pytest
import os
import shutil
from datetime import datetime

@pytest.fixture(scope="session")
def test_dir():
    """Create a temporary test directory"""
    test_dir = os.path.join(os.getcwd(), "test_assets")
    os.makedirs(test_dir, exist_ok=True)
    yield test_dir
    # Cleanup after tests
    shutil.rmtree(test_dir)

@pytest.fixture(scope="session")
def sample_image_path(test_dir):
    """Create a sample image file for testing"""
    image_path = os.path.join(test_dir, "sample.png")
    # Create an empty file
    with open(image_path, "w") as f:
        f.write("")
    return image_path

@pytest.fixture(scope="session")
def sample_model_path(test_dir):
    """Create a sample 3D model file for testing"""
    model_path = os.path.join(test_dir, "sample.glb")
    # Create an empty file
    with open(model_path, "w") as f:
        f.write("")
    return model_path

@pytest.fixture(scope="session")
def sample_memory_data():
    """Create sample memory data for testing"""
    return {
        "prompt": "test prompt",
        "expanded_prompt": "expanded test prompt",
        "image_path": "test_image.png",
        "model_path": "test_model.glb",
        "timestamp": datetime.now().isoformat()
    } 
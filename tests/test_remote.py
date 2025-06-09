import pytest
from app.core.remote import RemoteAPI
import os

@pytest.fixture
def remote():
    """Create a remote API instance for testing"""
    return RemoteAPI()

def test_api_initialization(remote):
    """Test remote API initialization"""
    assert remote.api_key is not None
    assert isinstance(remote.api_key, str)
    assert len(remote.api_key) > 0

def test_text_to_image(remote):
    """Test text-to-image generation"""
    prompt = "a red car on a road"
    result = remote.text_to_image(prompt)
    assert isinstance(result, dict)
    assert "image_path" in result
    assert os.path.exists(result["image_path"])

def test_image_to_3d(remote):
    """Test image-to-3D conversion"""
    # First generate an image
    prompt = "a blue house"
    image_result = remote.text_to_image(prompt)
    
    # Then convert to 3D
    result = remote.image_to_3d(image_result["image_path"])
    assert isinstance(result, dict)
    assert "model_path" in result
    assert os.path.exists(result["model_path"])

def test_error_handling(remote):
    """Test error handling for invalid inputs"""
    with pytest.raises(Exception):
        remote.text_to_image("")
    
    with pytest.raises(Exception):
        remote.image_to_3d("nonexistent_image.png")

def test_file_cleanup(remote):
    """Test cleanup of generated files"""
    # Generate test files
    prompt = "test cleanup"
    image_result = remote.text_to_image(prompt)
    model_result = remote.image_to_3d(image_result["image_path"])
    
    # Verify files exist
    assert os.path.exists(image_result["image_path"])
    assert os.path.exists(model_result["model_path"])
    
    # Clean up
    os.remove(image_result["image_path"])
    os.remove(model_result["model_path"])
    
    # Verify files are removed
    assert not os.path.exists(image_result["image_path"])
    assert not os.path.exists(model_result["model_path"]) 
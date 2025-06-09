import pytest
import os
import json
from app.core.memory import MemorySystem

@pytest.fixture
def memory():
    """Create a memory system instance for testing"""
    return MemorySystem()

def test_memory_initialization(memory):
    """Test memory system initialization"""
    assert memory.memory_file is not None
    assert os.path.exists(memory.memory_file)

def test_save_and_load(memory):
    """Test saving and loading memory"""
    test_data = {
        "prompt": "test prompt",
        "expanded_prompt": "expanded test prompt",
        "image_path": "test_image.png",
        "model_path": "test_model.glb",
        "timestamp": "2024-01-01"
    }
    
    # Save data
    memory.save(test_data)
    
    # Load and verify
    loaded_data = memory.load()
    assert isinstance(loaded_data, list)
    assert len(loaded_data) > 0
    assert loaded_data[-1]["prompt"] == test_data["prompt"]

def test_memory_format(memory):
    """Test memory data format"""
    test_data = {
        "prompt": "test prompt",
        "expanded_prompt": "expanded test prompt",
        "image_path": "test_image.png",
        "model_path": "test_model.glb",
        "timestamp": "2024-01-01"
    }
    
    memory.save(test_data)
    loaded_data = memory.load()
    
    # Verify data structure
    latest_entry = loaded_data[-1]
    assert all(key in latest_entry for key in test_data.keys())
    assert isinstance(latest_entry["timestamp"], str)

def test_memory_persistence(memory):
    """Test memory persistence across instances"""
    test_data = {
        "prompt": "persistence test",
        "expanded_prompt": "expanded persistence test",
        "image_path": "persist_image.png",
        "model_path": "persist_model.glb",
        "timestamp": "2024-01-01"
    }
    
    # Save with first instance
    memory.save(test_data)
    
    # Create new instance and verify data
    new_memory = MemorySystem()
    loaded_data = new_memory.load()
    assert loaded_data[-1]["prompt"] == test_data["prompt"] 
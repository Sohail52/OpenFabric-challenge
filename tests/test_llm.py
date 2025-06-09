import pytest
from app.core.llm import LLMProcessor

def test_llm_initialization():
    """Test LLM processor initialization"""
    llm = LLMProcessor()
    assert llm.model is not None
    assert llm.tokenizer is not None
    assert llm.device is not None

def test_prompt_expansion():
    """Test prompt expansion functionality"""
    llm = LLMProcessor()
    prompt = "a red car"
    expanded = llm.expand_prompt(prompt)
    assert isinstance(expanded, str)
    assert len(expanded) > len(prompt)
    assert "car" in expanded.lower()

def test_cuda_availability():
    """Test CUDA availability check"""
    llm = LLMProcessor()
    assert isinstance(llm.device, str)
    assert llm.device in ["cuda", "cpu"]

def test_model_output():
    """Test model output format"""
    llm = LLMProcessor()
    prompt = "a blue house"
    expanded = llm.expand_prompt(prompt)
    assert isinstance(expanded, str)
    assert len(expanded) > 0
    assert not expanded.isspace() 
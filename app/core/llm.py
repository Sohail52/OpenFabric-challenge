import logging
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import os

class LLMProcessor:
    def __init__(self, model_name="distilgpt2"):
        """
        Initialize the LLM processor with the specified model.
        
        Args:
            model_name (str): Name of the model to use (default: distilgpt2)
        """
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name
        
        # Create necessary directories
        self.cache_dir = os.path.join(os.getcwd(), "model_cache")
        self.offload_folder = os.path.join(self.cache_dir, "offload")
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.offload_folder, exist_ok=True)
        
        # Setup device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"Using device: {self.device}")
        
        if torch.cuda.is_available():
            self.logger.info(f"CUDA Device: {torch.cuda.get_device_name(0)}")
            self.logger.info(f"CUDA Version: {torch.version.cuda}")
        
        # Load model and tokenizer
        self.model = self._load_model()
        self.tokenizer = self._load_tokenizer()

    def _load_model(self):
        """Load the model with proper configuration"""
        try:
            self.logger.info(f"Loading model {self.model_name}...")
            
            # Configure model loading
            model = GPT2LMHeadModel.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                local_files_only=False,
                resume_download=True,
                force_download=False,
                device_map="auto",  # Let the model decide the best device mapping
                offload_folder=self.offload_folder,  # Specify offload folder
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,  # Use FP16 for CUDA
                low_cpu_mem_usage=True
            )
            
            self.logger.info(f"Model loaded successfully on {self.device}")
            return model
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise
    
    def _load_tokenizer(self):
        """Load the tokenizer"""
        try:
            self.logger.info(f"Loading tokenizer for {self.model_name}...")
            return GPT2Tokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                local_files_only=False,
                resume_download=True,
                force_download=False
            )
        except Exception as e:
            self.logger.error(f"Error loading tokenizer: {str(e)}")
            raise

    def expand_prompt(self, prompt: str) -> str:
        """
        Expand the given prompt using the loaded model.
        
        Args:
            prompt (str): The input prompt to expand
            
        Returns:
            str: The expanded prompt
        """
        try:
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt")
            if self.device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Generate expanded prompt
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=100,
                    num_return_sequences=1,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True
                )
            
            # Decode and return
            expanded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return expanded
        except Exception as e:
            self.logger.error(f"Error expanding prompt: {str(e)}")
            raise 
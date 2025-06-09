import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

logger = logging.getLogger(__name__)

class LLMHandler:
    def __init__(self, model_name="facebook/opt-125m"):
        """
        Initialize the LLM handler with a local model.
        
        Args:
            model_name (str): Name of the model to load
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        """Load the model and tokenizer."""
        try:
            logger.info(f"Loading model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def expand_prompt(self, prompt: str) -> str:
        """
        Expand a user prompt into a detailed description.
        
        Args:
            prompt (str): The original user prompt
            
        Returns:
            str: Expanded prompt with additional details
        """
        try:
            system_prompt = (
                "You are a creative AI assistant. Expand the given prompt into a "
                "detailed, vivid description suitable for image generation. "
                "Include details about lighting, mood, style, and composition."
            )
            full_prompt = f"{system_prompt}\n\nUser: {prompt}\nAssistant:"
            
            inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)
            outputs = self.model.generate(
                **inputs,
                max_length=300,
                num_return_sequences=1,
                temperature=0.8,
                do_sample=True,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            expanded_prompt = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            expanded_prompt = expanded_prompt.replace(full_prompt, "").strip()
            
            logger.info(f"Expanded prompt: {expanded_prompt}")
            return expanded_prompt
            
        except Exception as e:
            logger.error(f"Error expanding prompt: {str(e)}")
            return prompt  # Return original prompt if expansion fails 
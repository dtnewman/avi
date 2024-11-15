import coremltools as ct
from transformers import pipeline
import numpy as np

class CoreMLLlamaInference:
    def __init__(self, model_path, tokenizer_path):
        """
        Initialize the CoreML Llama inference engine
        
        Args:
            model_path (str): Path to the .mlpackage CoreML model
            tokenizer_path (str): Path to the tokenizer files
        """
        # Load the CoreML model
        self.model = ct.models.MLModel(model_path)
        
        # Load the tokenizer
        self.tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)
        
        # Get model metadata
        self.max_tokens = self.model.input_description["input_ids"][0]
        
    def generate_text(self, prompt, max_new_tokens=128, temperature=0.7):
        """
        Generate text from a prompt using the CoreML Llama model
        
        Args:
            prompt (str): Input prompt
            max_new_tokens (int): Maximum number of tokens to generate
            temperature (float): Sampling temperature (0.0 to 1.0)
            
        Returns:
            str: Generated text
        """
        # Tokenize the input prompt
        inputs = self.tokenizer(prompt, return_tensors="np", padding=True)
        input_ids = inputs["input_ids"]
        
        # Initialize generation
        current_input_ids = input_ids
        generated_tokens = []
        
        # Generate tokens one at a time
        for _ in range(max_new_tokens):
            # Prepare inputs for CoreML model
            model_inputs = {
                "input_ids": current_input_ids.astype(np.int32),
                "temperature": np.array([temperature], dtype=np.float32)
            }
            
            # Run inference
            output = self.model.predict(model_inputs)
            next_token_logits = output["logits"][0, -1, :]
            
            # Sample next token
            next_token = self._sample_token(next_token_logits, temperature)
            
            # Break if end of sequence token is generated
            if next_token == self.tokenizer.eos_token_id:
                break
                
            generated_tokens.append(next_token)
            current_input_ids = np.concatenate([
                current_input_ids,
                np.array([[next_token]])
            ], axis=1)
            
            # Check if we've exceeded max context length
            if current_input_ids.shape[1] >= self.max_tokens:
                break
        
        # Decode generated tokens
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return generated_text
    
    def _sample_token(self, logits, temperature):
        """
        Sample next token from logits using temperature
        """
        if temperature == 0:
            return int(np.argmax(logits))
            
        probs = np.exp(logits / temperature)
        probs = probs / np.sum(probs)
        return int(np.random.choice(len(probs), p=probs))

# Example usage
def main():
    model_id = "meta-llama/Llama-3.2-1B-instruct"
    pipe = pipeline(
    "text-generation", 
        model=model_id, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )

if __name__ == "__main__":
    main()
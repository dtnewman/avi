import coremltools as ct
from transformers import AutoTokenizer
import numpy as np
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class CoreMLLlamaInference:
    def __init__(self, model_path, model_id):
        """Initialize the CoreML Llama inference engine"""
        self.model = ct.models.MLModel(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize state
        self.state = self.model.make_state()

    def generate_text(self, prompt, max_new_tokens=100, temperature=0.7, callback=None):
        """Generate text from a prompt"""
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="np", padding=True)
        input_ids = inputs["input_ids"].astype(np.int32)
        generated_sequence = input_ids.copy()
        
        # Initialize the generated text buffer
        current_text = ""
        
        for i in range(max_new_tokens):
            current_length = generated_sequence.shape[1]
            
            predictions = self.model.predict({
                'inputIds': generated_sequence,
                'causalMask': np.ones((1, 1, current_length, current_length), dtype=np.float32)
            }, state=self.state)
            
            next_token = self._sample_token(predictions['logits'][0][-1], temperature)
            
            # Check stopping conditions
            if next_token in [self.tokenizer.eos_token_id, self.tokenizer.pad_token_id]:
                break
            
            generated_sequence = np.concatenate([
                generated_sequence, 
                np.array([[next_token]], dtype=np.int32)
            ], axis=1)
            
            # Decode the new token and update the stream
            new_text = self.tokenizer.decode(generated_sequence[0], skip_special_tokens=True)
            if callback and new_text != current_text:
                callback(new_text[len(current_text):])
                current_text = new_text

        return current_text

    def _sample_token(self, logits, temperature):
        """Sample next token from logits using temperature"""
        logits = logits / temperature
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        return int(np.random.choice(len(probs), p=probs))

def main():
    model_id = "meta-llama/Llama-3.2-1B-instruct"
    model_path = './models/llama-3.2-1b-instruct.mlpackage'
    
    llama = CoreMLLlamaInference(model_path, model_id)
    
    prompt = "What is the capital of Japan?"
    
    # Define streaming callback
    def stream_callback(new_token):
        print(new_token, end='', flush=True)
    
    generated_text = llama.generate_text(prompt, callback=stream_callback)
    print("\n")  # Add newline at the end

if __name__ == "__main__":
    main()
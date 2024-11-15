import coremltools as ct
from transformers import AutoTokenizer
import numpy as np
import os
import time

from src.inference import CoreMLLlamaInference


os.environ["TOKENIZERS_PARALLELISM"] = "false"



def main():

    model_id = "meta-llama/Llama-3.2-1B-instruct"
    model_path = './models/llama-3.2-1b-instruct-quantized.mlpackage'
    
    llama = CoreMLLlamaInference(model_path, model_id)
    
    prompt = "What is the capital of Japan?"
    print(f"Prompt: {prompt}")
    
    # Define streaming callback
    def stream_callback(new_token):
        print(new_token, end='', flush=True)
    
    generated_text = llama.generate_text(prompt, callback=stream_callback)
    print("\n")  # Add newline at the end

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"\nGeneration time: {end_time - start_time:.2f} seconds")

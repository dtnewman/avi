import time
import torch
from transformers import pipeline


model_id = "meta-llama/Llama-3.2-1B-Instruct"
start_time = time.time()
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
end_time = time.time()
print(f"Time taken for pipeline initialization: {end_time - start_time:.2f} seconds")
messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "What is the capital of France?"},
]
start_time = time.time()
outputs = pipe(
    messages,
    max_new_tokens=256,
)
end_time = time.time()
print(outputs[0]["generated_text"][-1])
print(f"Time taken for text generation: {end_time - start_time:.2f} seconds")

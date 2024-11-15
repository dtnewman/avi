import coremltools as ct
from transformers import AutoTokenizer, LlamaForCausalLM
import numpy as np
from src.model import KvCacheStateLlamaForCausalLM
from src.converter import LlamaCoreMLConverter
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


model_id = "meta-llama/Llama-3.2-1B-instruct"
print("Loading model")
model = ct.models.MLModel('./models/llama-3.2-1b-instruct.mlpackage')
print("Loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

print("Setting prompt")
prompt = "How much wood can a woodchuck chuck?"
inputs = tokenizer(prompt, return_tensors="np", padding=True)
input_ids = inputs["input_ids"].astype(np.int32)
current_input_ids = input_ids
generated_tokens = []

print("Tokenizing prompt")
inputs = tokenizer(prompt, return_tensors="np", padding=True)
input_ids = inputs["input_ids"].astype(np.int32)
# print(inputs.keys())
# print(input_ids)
causal_mask = inputs["attention_mask"].astype(np.float32)
# print(causal_mask)

print("Initializing key and value caches")
# Reshape to the specified shapes in the model spec
key_cache = np.zeros((1, 32, 2048, 128), dtype=np.float32)
value_cache = np.zeros((1, 32, 2048, 128), dtype=np.float32)


# Reshape inputs as before
current_input_ids = input_ids
causal_mask = np.ones((1, 1, current_input_ids.shape[1], current_input_ids.shape[1]), dtype=np.float32)


# from coremltools.models.model import MLState
# state = MLState({
#     'key_cache': key_cache,
#     'value_cache': value_cache
# })

print("Making state")
state = model.make_state()
print(type(state))

print("Initializing generation parameters")
max_new_tokens = 100
stop_token_id = tokenizer.eos_token_id
generated_sequence = input_ids.copy()  # Start with the prompt tokens

print("Starting generation loop")
print("Prompt:", prompt)
for i in range(max_new_tokens):
    print(f"Generating token {i}")
    current_length = generated_sequence.shape[1]
    
    predictions = model.predict({
        'inputIds': generated_sequence,
        'causalMask': np.ones((1, 1, current_length, current_length), dtype=np.float32)
    }, state=state)
    
    logits = predictions['logits']
    # Apply temperature
    temperature = 0.7
    logits = logits[0][-1] / temperature
    # Apply softmax
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / np.sum(exp_logits)
    
    # Sample from the distribution
    next_token = int(np.random.choice(len(probs), p=probs))
    
    # Enhanced stopping conditions
    if next_token == stop_token_id or next_token == tokenizer.pad_token_id:
        break
    
    # Append the new token to the sequence
    generated_sequence = np.concatenate([generated_sequence, np.array([[next_token]], dtype=np.int32)], axis=1)

# Decode the full sequence
text = tokenizer.decode(generated_sequence[0], skip_special_tokens=True)
print(text)
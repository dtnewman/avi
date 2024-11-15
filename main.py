import coremltools as ct
from transformers import AutoTokenizer, LlamaForCausalLM
import numpy as np
from src.model import KvCacheStateLlamaForCausalLM
from src.converter import LlamaCoreMLConverter
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


model_id = "meta-llama/Llama-3.2-1B-instruct"
model = ct.models.MLModel('./models/llama-3.2-1b-instruct.mlpackage')
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

prompt = "How much would can a woodchuck chuck?"
inputs = tokenizer(prompt, return_tensors="np", padding=True)
input_ids = inputs["input_ids"].astype(np.int32)
current_input_ids = input_ids
generated_tokens = []

inputs = tokenizer(prompt, return_tensors="np", padding=True)
input_ids = inputs["input_ids"].astype(np.int32)
# print(inputs.keys())
# print(input_ids)
causal_mask = inputs["attention_mask"].astype(np.float32)
# print(causal_mask)

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


state = model.make_state()
print(type(state))

# Run inference with KV cache states

predictions = model.predict({
    'inputIds': current_input_ids.astype(np.int32),
    'causalMask': causal_mask
}, state=state)


logits = predictions['logits']
print(logits)

# take predictions and convert to tokens
next_token = int(np.argmax(logits[0][-1]))
generated_tokens.append(next_token)

# convert tokens to text
text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
print(text)
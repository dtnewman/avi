import coremltools as ct
from transformers import AutoTokenizer, LlamaForCausalLM
import numpy as np
from src.model import KvCacheStateLlamaForCausalLM
from src.converter import LlamaCoreMLConverter
import os

model_id = "meta-llama/Llama-3.2-1B-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

model = LlamaForCausalLM.from_pretrained(model_id)

model = KvCacheStateLlamaForCausalLM(
    model_id,
    batch_size=1,
    context_size=2048
)

converter = LlamaCoreMLConverter(
    model,
    batch_size=1,
    context_size=2048
)

mlmodel = converter.convert(quantize=True)
mlmodel.save("./models/llama-3.2-1b-instruct-quantized.mlpackage")

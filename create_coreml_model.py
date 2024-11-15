from transformers import LlamaForCausalLM
from src.model import KvCacheStateLlamaForCausalLM
from src.converter import LlamaCoreMLConverter

model_id = "meta-llama/Llama-3.2-1B-instruct"

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

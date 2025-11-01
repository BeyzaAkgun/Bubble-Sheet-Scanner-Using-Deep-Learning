#!pip install -U accelerate bitsandbytes einops safetensors peft transformers_stream_generator
#!pip install -q git+https://github.com/huggingface/peft.git
#!pip install --upgrade git+https://github.com/huggingface/transformers.git

from transformers import AutoProcessor, AutoModelForCausalLM
import torch
from PIL import Image
import requests

model_id = "Qwen/Qwen-VL-Chat"
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, device_map="auto", torch_dtype=torch.float16).eval()

image = Image.open("/content/1234567891019.png").convert("RGB")

prompt = (
    "<|im_start|>system\nYou are a helpful assistant that reads exam answer sheets.<|im_end|>\n"
    "<|im_start|>user\nThis is a scanned multiple-choice answer sheet. Please extract and report the selected answers for:\n"
    "<image>\n"
    "<|im_end|>\n"
    "<|im_start|>assistant\n"
)

inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)

generate_ids = model.generate(**inputs, max_new_tokens=512)
output = processor.batch_decode(generate_ids, skip_special_tokens=True)[0]

print(output)


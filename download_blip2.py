from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
print("Downloading BLIP-2 processor...")
Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
print("Processor done. Downloading model weights (~10GB)...")
Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
print("Done.")

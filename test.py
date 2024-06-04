from transformers import SiglipImageProcessor, SiglipVisionConfig
from PIL import Image
image=Image.open('/home/ubuntu/LLaVA-pp/LLaVA/images (2).jpeg')
processor_siglip=SiglipImageProcessor.from_pretrained("google/siglip-so400m-patch14-384")
image_sig=processor_siglip.preprocess(image, return_tensors='pt')['pixel_values'][0]
print(image_sig)
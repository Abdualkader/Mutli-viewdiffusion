---
license: apache-2.0
---
# ImageDream-diffusers Model Card
This is a port of https://huggingface.co/Peng-Wang/ImageDream into diffusers.

And get ported weights from https://huggingface.co/ashawkey/imagedream-ipmv-diffuser  
In ashawkey's work, UNet did not ported to diffusers.

This work has been fully ported to diffusers, including UNet.  
And separated the IP-adapter-plus from the unet.

## Diffusers
```python
import torch
from diffusers import DiffusionPipeline
from diffusers.utils import make_image_grid
from PIL import Image

pipe = DiffusionPipeline.from_pretrained(
    "kiigii/imagedream-ipmv-diffusers",
    torch_dtype=torch.float16,
    trust_remote_code=True,
)
pipe.load_ip_adapter()
pipe.to("cude")

prompt = ""  # no need to input prompt
image = Image.open(...)

mv_images = pipe(
    prompt=prompt,
    ip_adapter_image=image,
    guidance_scale=5,
    num_inference_steps=30,
    elevation=0,
    num_images_per_prompt=1
).images
mv_grid = make_image_grid(mv_images[:4], 2, 2)
mv_grid.save("mv_image.png")
```

## Citation
```
@article{wang2023imagedream,
  title={ImageDream: Image-Prompt Multi-view Diffusion for 3D Generation},
  author={Wang, Peng and Shi, Yichun},
  journal={arXiv preprint arXiv:2312.02201},
  year={2023}
}
```
## Misuse, Malicious Use, and Out-of-Scope Use
The model should not be used to intentionally create or disseminate images that create hostile or alienating environments for people. This includes generating images that people would foreseeably find disturbing, distressing, or offensive; or content that propagates historical or current stereotypes.
# sdwebuiapi
API client for AUTOMATIC1111/stable-diffusion-webui

Supports txt2img, img2img, extra-single-image, extra-batch-images API calls.

API support have to be enabled from webui. It's explained [here](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/API).

API calls are (almost) direct translation from http://127.0.0.1:7860/docs as of 2022/11/08.


# Usage

webuiapi_demo.ipynb contains example code with images.

## create API client
```
import webuiapi

# create API client
api = webuiapi.WebUIApi()

# you can set default sampler, steps.
api = webuiapi.WebUIApi(sampler='Euler a', steps=20)
```

## txt2img
```
result1 = api.txt2img(prompt="cute squirrel",
                    negative_prompt="ugly, out of frame",
                    seed=1003,
                    styles=["anime"],
                    cfg_scale=7,
#                      sampler_index='DDIM',
#                      steps=30,
                    )
# images contains the returned images (PIL images)
result1.images

# image is shorthand for images[0]
result1.image

# info contains text info about the api call
result1.info

# info contains paramteres of the api call
result1.parameters
```

## img2img
```
result2 = api.img2img(images=[result1.image], prompt="cute cat", seed=5555, cfg_scale=6.5, denoising_strength=0.6)
result2.image
```

## img2img inpainting
```
from PIL import Image, ImageDraw

mask = Image.new('RGB', result2.image.size, color = 'black')
# mask = result2.image.copy()
draw = ImageDraw.Draw(mask)
draw.ellipse((210,150,310,250), fill='white')
draw.ellipse((80,120,160,120+80), fill='white')

# mask

inpainting_result = api.img2img(images=[result2.image],
                                mask_image=mask,
                                inpainting_fill=1,
                                prompt="cute cat",
                                seed=104,
                                cfg_scale=5.0,
                                denoising_strength=0.7)
inpainting_result.image
```

## extra-single-image
```
result3 = api.extra_single_image(image=result2.image,
                                 upscaler_1="ESRGAN_4x",
                                 upscaling_resize=1.5)
print(result3.image.size)
result3.image
```

## extra-batch-images
```
result4 = api.extra_batch_images(images=[result1.image, result2.image],
                                 upscaler_1="ESRGAN_4x",
                                 upscaling_resize=1.5)
len(result4.images)
```

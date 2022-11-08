# sdwebuiapi
API client for AUTOMATIC1111/stable-diffusion-webui

Supports txt2img, img2img, extra-single-image, extra-batch-images API calls.

API support have to be enabled from webui. It's explained [here](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/API).

API calls are (almost) direct translation from http://127.0.0.1:7860/docs as of 2022/11/08.


# Usage

webuiapi_demo.ipynb contains example code with original images. Images are compressed as jpeg in this document.

## create API client
```
import webuiapi

# create API client
api = webuiapi.WebUIApi()

# set host, port
api = webuiapi.WebUIApi(host='127.0.0.1', port=7860)

# set default sampler, steps.
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

result1.image
```
![txt2img](https://user-images.githubusercontent.com/1288793/200459205-258d75bb-d2b6-4882-ad22-040bfcf95626.jpg)


## img2img
```
result2 = api.img2img(images=[result1.image], prompt="cute cat", seed=5555, cfg_scale=6.5, denoising_strength=0.6)
result2.image
```
![img2img](https://user-images.githubusercontent.com/1288793/200459294-ab1127e5-04e5-47ac-82b2-2bbd0648402a.jpg)

## img2img inpainting
```
from PIL import Image, ImageDraw

mask = Image.new('RGB', result2.image.size, color = 'black')
# mask = result2.image.copy()
draw = ImageDraw.Draw(mask)
draw.ellipse((210,150,310,250), fill='white')
draw.ellipse((80,120,160,120+80), fill='white')

mask
```
![mask](https://user-images.githubusercontent.com/1288793/200459372-7850c6b6-27c5-435a-93e2-8710948d316a.jpg)

```
inpainting_result = api.img2img(images=[result2.image],
                                mask_image=mask,
                                inpainting_fill=1,
                                prompt="cute cat",
                                seed=104,
                                cfg_scale=5.0,
                                denoising_strength=0.7)
inpainting_result.image
```
![img2img_inpainting](https://user-images.githubusercontent.com/1288793/200459398-9c1004be-1352-4427-bc00-442721a0e5a1.jpg)

## extra-single-image
```
result3 = api.extra_single_image(image=result2.image,
                                 upscaler_1=webuiapi.Upscaler.ESRGAN_4x,
                                 upscaling_resize=1.5)
print(result3.image.size)
result3.image
```
(768, 768)

![extra_single_image](https://user-images.githubusercontent.com/1288793/200459455-8579d740-3d8f-47f9-8557-cc177b3e99b7.jpg)

## extra-batch-images
```
result4 = api.extra_batch_images(images=[result1.image, inpainting_result.image],
                                 upscaler_1=webuiapi.Upscaler.ESRGAN_4x,
                                 upscaling_resize=1.5)
result4.images[0]
```
![extra_batch_images_1](https://user-images.githubusercontent.com/1288793/200459540-b0bd2931-93db-4d03-9cc1-a9f5e5c89745.jpg)
```
result4.images[1]
```
![extra_batch_images_2](https://user-images.githubusercontent.com/1288793/200459542-aa8547a0-f6db-436b-bec1-031a93a7b1d4.jpg)


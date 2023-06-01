# sdwebuiapi
API client for AUTOMATIC1111/stable-diffusion-webui

Supports txt2img, img2img, extra-single-image, extra-batch-images API calls.

API support have to be enabled from webui. Add --api when running webui.
It's explained [here](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/API).

You can use --api-auth user1:pass1,user2:pass2 option to enable authentication for api access.
(Since it's basic http authentication the password is transmitted in cleartext)

API calls are (almost) direct translation from http://127.0.0.1:7860/docs as of 2022/11/21.

# Install

```
pip install webuiapi
```

# Usage

webuiapi_demo.ipynb contains example code with original images. Images are compressed as jpeg in this document.

## create API client
```
import webuiapi

# create API client
api = webuiapi.WebUIApi()

# create API client with custom host, port
#api = webuiapi.WebUIApi(host='127.0.0.1', port=7860)

# create API client with custom host, port and https
#api = webuiapi.WebUIApi(host='webui.example.com', port=443, use_https=True)

# create API client with default sampler, steps.
#api = webuiapi.WebUIApi(sampler='Euler a', steps=20)

# optionally set username, password when --api-auth=username:password is set on webui.
# username, password are not protected and can be derived easily if the communication channel is not encrypted.
# you can also pass username, password to the WebUIApi constructor.
api.set_auth('username', 'password')
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
#                      enable_hr=True,
#                      hr_scale=2,
#                      hr_upscaler=webuiapi.HiResUpscaler.Latent,
#                      hr_second_pass_steps=20,
#                      hr_resize_x=1536,
#                      hr_resize_y=1024,
#                      denoising_strength=0.4,

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

### Async API support
txt2img, img2img, extra_single_image, extra_batch_images support async api call with use_async=True parameter. You need asyncio, aiohttp packages installed.
```
result = await api.txt2img(prompt="cute kitten",
                    seed=1001,
                    use_async=True
                    )
result.image
```

### Scripts support
Scripts from AUTOMATIC1111's Web UI are supported, but there aren't official models that define a script's interface.

To find out the list of arguments that are accepted by a particular script look up the associated python file from
AUTOMATIC1111's repo `scripts/[script_name].py`. Search for its `run(p, **args)` function and the arguments that come
after 'p' is the list of accepted arguments

#### Example for X/Y/Z Plot script:
```
(scripts/xyz_grid.py file from AUTOMATIC1111's repo)

    def run(self, p, x_type, x_values, y_type, y_values, z_type, z_values, draw_legend, include_lone_images, include_sub_grids, no_fixed_seeds, margin_size):
    ...
```
List of accepted arguments:
* _x_type_: Index of the axis for X axis. Indexes start from [0: Nothing]
* _x_values_: String of comma-separated values for the X axis 
* _y_type_: Index of the axis type for Y axis. As the X axis, indexes start from [0: Nothing]
* _y_values_: String of comma-separated values for the Y axis
* _z_type_: Index of the axis type for Z axis. As the X axis, indexes start from [0: Nothing]
* _z_values_: String of comma-separated values for the Z axis
* _draw_legend_: "True" or "False". IMPORTANT: It needs to be a string and not a Boolean value
* _include_lone_images_: "True" or "False". IMPORTANT: It needs to be a string and not a Boolean value
* _include_sub_grids_: "True" or "False". IMPORTANT: It needs to be a string and not a Boolean value
* _no_fixed_seeds_: "True" or "False". IMPORTANT: It needs to be a string and not a Boolean value
* margin_size: int value
```
# Available Axis options (Different for txt2img and img2img!)
XYZPlotAvailableTxt2ImgScripts = [
    "Nothing",
    "Seed",
    "Var. seed",
    "Var. strength",
    "Steps",
    "Hires steps",
    "CFG Scale",
    "Prompt S/R",
    "Prompt order",
    "Sampler",
    "Checkpoint name",
    "Sigma Churn",
    "Sigma min",
    "Sigma max",
    "Sigma noise",
    "Eta",
    "Clip skip",
    "Denoising",
    "Hires upscaler",
    "VAE",
    "Styles",
]

XYZPlotAvailableImg2ImgScripts = [
    "Nothing",
    "Seed",
    "Var. seed",
    "Var. strength",
    "Steps",
    "CFG Scale",
    "Image CFG Scale",
    "Prompt S/R",
    "Prompt order",
    "Sampler",
    "Checkpoint name",
    "Sigma Churn",
    "Sigma min",
    "Sigma max",
    "Sigma noise",
    "Eta",
    "Clip skip",
    "Denoising",
    "Cond. Image Mask Weight",
    "VAE",
    "Styles",
]

# Example call
XAxisType = "Steps"
XAxisValues = "20,30" 
YAxisType = "Sampler"
YAxisValues = "Euler a, LMS"
ZAxisType = "Nothing"
ZAxisValues = ""
drawLegend = "True"
includeLoneImages = "False"
includeSubGrids = "False"
noFixedSeeds = "False"
marginSize = 0


# x_type, x_values, y_type, y_values, z_type, z_values, draw_legend, include_lone_images, include_sub_grids, no_fixed_seeds, margin_size

result = api.txt2img(
                    prompt="cute girl with short brown hair in black t-shirt in animation style",
                    seed=1003,
                    script_name="X/Y/Z Plot",
                    script_args=[
                        XYZPlotAvailableTxt2ImgScripts.index(XAxisType),
                        XAxisValues,
                        XYZPlotAvailableTxt2ImgScripts.index(YAxisType),
                        YAxisValues,
                        XYZPlotAvailableTxt2ImgScripts.index(ZAxisType),
                        ZAxisValues,
                        drawLegend,
                        includeLoneImages,
                        includeSubGrids,
                        noFixedSeeds,
                        marginSize,                        ]
                    )

result.image
```
![txt2img_grid_xyz](https://user-images.githubusercontent.com/1288793/222345625-dc2e4090-6786-4a53-8619-700dc2f12412.jpg)


### Configuration APIs
```
# return map of current options
options = api.get_options()

# change sd model
options = {}
options['sd_model_checkpoint'] = 'model.ckpt [7460a6fa]'
api.set_options(options)

# when calling set_options, do not pass all options returned by get_options().
# it makes webui unusable (2022/11/21).

# get available sd models
api.get_sd_models()

# misc get apis
api.get_samplers()
api.get_cmd_flags()      
api.get_hypernetworks()
api.get_face_restorers()
api.get_realesrgan_models()
api.get_prompt_styles()
api.get_artist_categories() # deprecated ?
api.get_artists() # deprecated ?
api.get_progress()
api.get_embeddings()
api.get_cmd_flags()
api.get_scripts()
api.get_memory()

# misc apis
api.interrupt()
api.skip()
```

### Utility methods
```
# save current model name
old_model = api.util_get_current_model()

# get list of available models
models = api.util_get_model_names()

# set model (use exact name)
api.util_set_model(models[0])

# set model (find closest match)
api.util_set_model('robodiffusion')

# wait for job complete
api.util_wait_for_ready()

```

### LORA and alwayson_scripts example

```
r = api.txt2img(prompt='photo of a cute girl with green hair <lora:Moxin_10:0.6> shuimobysim __juice__',
                seed=1000,
                save_images=True,
                alwayson_scripts={"Simple wildcards":[]} # wildcards extension doesn't accept more parameters.
               )
r.image
```

### Extension support - Model-Keyword
```
# https://github.com/mix1009/model-keyword
mki = webuiapi.ModelKeywordInterface(api)
mki.get_keywords()
```
ModelKeywordResult(keywords=['nousr robot'], model='robo-diffusion-v1.ckpt', oldhash='41fef4bd', match_source='model-keyword.txt')


### Extension support - Instruct-Pix2Pix
```
# Instruct-Pix2Pix extension is now deprecated and is now part of webui.
# You can use normal img2img with image_cfg_scale when instruct-pix2pix model is loaded.
r = api.img2img(prompt='sunset', images=[pil_img], cfg_scale=7.5, image_cfg_scale=1.5)
r.image
```

### Extension support - ControlNet
```
# https://github.com/Mikubill/sd-webui-controlnet

api.controlnet_model_list()
```
<pre>
['control_v11e_sd15_ip2p [c4bb465c]',
 'control_v11e_sd15_shuffle [526bfdae]',
 'control_v11f1p_sd15_depth [cfd03158]',
 'control_v11p_sd15_canny [d14c016b]',
 'control_v11p_sd15_inpaint [ebff9138]',
 'control_v11p_sd15_lineart [43d4be0d]',
 'control_v11p_sd15_mlsd [aca30ff0]',
 'control_v11p_sd15_normalbae [316696f1]',
 'control_v11p_sd15_openpose [cab727d4]',
 'control_v11p_sd15_scribble [d4ba51ff]',
 'control_v11p_sd15_seg [e1f51eb9]',
 'control_v11p_sd15_softedge [a8575a2a]',
 'control_v11p_sd15s2_lineart_anime [3825e83e]',
 'control_v11u_sd15_tile [1f041471]']
 </pre>

```
api.controlnet_version()
api.controlnet_module_list()
```

```
# normal txt2img
r = api.txt2img(prompt="photo of a beautiful girl with blonde hair", height=512, seed=100)
img = r.image
img
```
![cn1](https://user-images.githubusercontent.com/1288793/222315754-43c6dc8c-2a62-4a31-b51a-f68523118e0d.png)

```
# txt2img with ControlNet (used 1.0 but also supports 1.1)
unit1 = webuiapi.ControlNetUnit(input_image=img, module='canny', model='control_canny-fp16 [e3fe7712]')

r = api.txt2img(prompt="photo of a beautiful girl", controlnet_units=[unit1])
r.image
```

![cn2](https://user-images.githubusercontent.com/1288793/222315791-c6c480eb-2987-4044-b673-5f2cb6135f87.png)


```
# img2img with multiple ControlNets (used 1.0 but also supports 1.1)
unit1 = webuiapi.ControlNetUnit(input_image=img, module='canny', model='control_canny-fp16 [e3fe7712]')
unit2 = webuiapi.ControlNetUnit(input_image=img, module='depth', model='control_depth-fp16 [400750f6]', weight=0.5)

r2 = api.img2img(prompt="girl",
            images=[img], 
            width=512,
            height=512,
            controlnet_units=[unit1, unit2],
            sampler_name="Euler a",
            cfg_scale=7,
           )
r2.image
```
![cn3](https://user-images.githubusercontent.com/1288793/222315816-1155b0c2-570d-4455-a68e-294fc7061b0a.png)

```
r2.images[1]
```
![cn4](https://user-images.githubusercontent.com/1288793/222315836-9a26afec-c407-426b-9a08-b2cef2a32ab1.png)

```
r2.images[2]
```
![cn5](https://user-images.githubusercontent.com/1288793/222315859-e6b6286e-854d-40c1-a516-5a08c827c49a.png)


```
r = api.controlnet_detect(images=[img], module='canny')
r.image
```

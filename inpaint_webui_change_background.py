import os
import numpy as np
import random
import cv2
from PIL import Image, ImageOps, ImageFilter, ImageChops
import torch
from diffusers import ControlNetModel
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_controlnet import MultiControlNetModel
from diffusers import WebUIStableDiffusionControlNetInpaintPipeline
from diffusers.schedulers import EulerAncestralDiscreteScheduler
import extensions.ESRGAN.test as esrgan
from diffusers.utils.promt_parser import get_promt_embedding, load_webui_textual_inversion


def get_fixed_seed(seed):
    if seed is None or seed == '' or seed == -1:
        return int(random.randrange(4294967294))

    return seed


def load_image(image_path):
    assert isinstance(image_path, str)
    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image)
    if image.mode == "RGBA":
        # returning an RGB mode image with no transparency
        img = np.array(image)[..., :3]
        image = Image.fromarray(img)
    return image.convert("RGB")


def mask_process(mask, inpainting_mask_invert=True, blur=4):
    mask = mask.convert("L")
    if inpainting_mask_invert:
        mask = ImageOps.invert(mask)
    if blur > 0:
        np_mask = np.array(mask)
        kernel_size = 2 * int(4 * blur + 0.5) + 1
        np_mask = cv2.GaussianBlur(np_mask, (kernel_size, 1), blur)
        np_mask = cv2.GaussianBlur(np_mask, (1, kernel_size), blur)
        mask = Image.fromarray(np_mask)
    return mask


def prompt_preprocess(pos_prompt, neg_prompt, pipe, device=torch.device("cuda")):
    pos_embeds = get_promt_embedding(pos_prompt, pipe.tokenizer, pipe.text_encoder, device)
    neg_embeds = get_promt_embedding(neg_prompt, pipe.tokenizer, pipe.text_encoder, device)
    # pad embedding
    len_pos, len_neg = pos_embeds.shape[1], neg_embeds.shape[1]
    if len_pos > len_neg:
        pad_embed = neg_embeds[:, -1].unsqueeze(1).repeat(1, len_pos - len_neg, 1)
        neg_embeds = torch.cat([neg_embeds, pad_embed], dim=1)
    elif len_neg > len_pos:
        pad_embed = pos_embeds[:, -1].unsqueeze(1).repeat(1, len_neg - len_pos, 1)
        pos_embeds = torch.cat([pos_embeds, pad_embed], dim=1)
    return pos_embeds, neg_embeds


### init pipeline ###
root_dir = "/xsl/wilson.xu/weights"
base_path = f"{root_dir}/CyberRealistic_V3.0-FP32"
canny_path = f"{root_dir}/control_v11p_sd15_canny"
face_restored_path = f"{root_dir}/codeformer-v0.1.0.pth"
device = torch.device("cuda")

canny_model = ControlNetModel.from_pretrained(canny_path, torch_dtype=torch.float16).to(device)
controlnet = MultiControlNetModel([canny_model])

pipe_control = WebUIStableDiffusionControlNetInpaintPipeline.from_pretrained(
    base_path, controlnet=controlnet, torch_dtype=torch.float16).to(device)
pipe_control.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe_control.scheduler.config)
# pipe_control.load_textual_inversion("embedding")
load_webui_textual_inversion("embedding", pipe_control)

prompt = "bokeh,outdoors background,clear background, beautiful light, best quality, masterpiece, ultra highres, very detailed skin, photorealistic, masterpiece, very high detailed,4k, 8k, 64k,photo detail"
negative_prompt = "floor reflection, tilt, people, ((messy hair:2)), tree, ((light at background)), human, ((no peoples background)), ((sun)),((backlight)), ((light above the head)), ng_deepnegative_v1_75t , (badhandv4), verybadimagenegative_v1.3, easynegative, EasyNegative, (worst quality:2), (low quality:2), (normal quality:2), lowres, ((monochrome)), ((grayscale)), 2 body, 2 pussy, 2 upper, 2 lower, ((2 head)), ((3 hands)), 3 feet, 3 ear, 4 ear, bad anatomy,((cross-eyed)), bad hands, bad feet, ((watermark:2)),((logo:2)), (moles:2), illustration ,3d, sepia ,painting ,cartoon, sketch, light above the head, stand on the table, high light on body"

# prompt_embedding
pos_prompt_embs, neg_prompt_embs = prompt_preprocess(prompt, negative_prompt, pipe_control, device)


input_size = (1024, 1024)
seed = get_fixed_seed(3365673641)
generator = torch.Generator("cuda").manual_seed(seed)
print(f'processing seed = {seed}')

img_file = "/xsl/wilson.xu/case/4.png"
mask_file = "/xsl/wilson.xu/case/3.png"
save_dir = f'./output'
os.makedirs(save_dir, exist_ok=True)

### processing input ###
init_image = load_image(img_file)
# TODO: scale any size input
upscale = 1 if init_image.size[1] < 1024 else 0
print(f"image_size = {init_image.size}, upscale = {upscale}")
if upscale:
    w, h = init_image.size
    init_image = Image.fromarray(esrgan.upscale(
        np.asarray(init_image))).resize((w * 2, h * 2))  # 放大四倍再resize回2倍
if init_image.size[0] < input_size[0] or init_image.size[1] < input_size[1]:
    bgImg: Image.Image = Image.new("RGB", input_size, (255, 255, 255))
    bgImg.paste(init_image, (round((input_size[0] - init_image.size[0]) / 2), round((input_size[1] - init_image.size[1]) / 2)))
    init_image = bgImg
else:
    init_image = init_image.resize(input_size)

ori_mask = load_image(mask_file)
mask_image = mask_process(ori_mask, inpainting_mask_invert=True)

# clothing canny
canny_image = cv2.Canny(np.array(ori_mask), 100, 200)[..., None]
canny_image = np.tile(canny_image, [1, 1, 3])
canny_image = Image.fromarray(canny_image)

final = np.concatenate([init_image, ori_mask, canny_image], axis=1)

### cloth mask for paste back ###
np_mask = np.array(mask_image, dtype=np.float32) / 255
np_mask = np.around(np_mask)[..., None]

### start pipeline ###
pil_img_list, has_nsfw_concept = pipe_control(
        image=init_image,
        mask_image=mask_image,
        control_image=[canny_image],
        height=input_size[0],
        width=input_size[1],
        strength=0.75,
        num_inference_steps=35,
        guidance_scale=9,  # CFGscale
        num_images_per_prompt=1,
        generator=generator,
        prompt_embeds=pos_prompt_embs,
        negative_prompt_embeds=neg_prompt_embs,
        return_dict=False,
        controlnet_conditioning_scale=1.0,)

### restore faces ###
for index in range(len(pil_img_list)):
    output_image = pil_img_list[index]

    ### cloth paste back ###
    output_image = np.array(output_image) * np_mask + np.array(init_image) * (1 - np_mask)
    final = np.concatenate((final, output_image), axis=1)

Image.fromarray(np.uint8(final)).save(f"{save_dir}/test_webui.png")

import os
import sys

sys.path.append(os.path.dirname(__file__))
print(os.path.dirname(__file__))

from share import *
from config import *

import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.hed import HEDdetector
from annotator.pidinet import PidiNetDetector
from annotator.util import nms
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler

class sketch_scribble:
    def __init__(self, model_name = 'control_v11p_sd15_scribble', preprocessor = None):
        self.preprocessor = preprocessor

        model = create_model(f'./ControlNet/models/{model_name}.yaml').cpu()
        model.load_state_dict(load_state_dict('./ControlNet/models/v1-5-pruned.ckpt', location='cuda'), strict=False)
        model.load_state_dict(load_state_dict(f'./ControlNet/models/{model_name}.pth', location='cuda'), strict=False)
        self.model = model.cuda()
        self.ddim_sampler = DDIMSampler(model)

    def process(
        self,
        input_image_path, 
        input_image,
        prompt, 
        det = "None", 
        a_prompt = 'best quality', 
        n_prompt = 'lowres, bad anatomy, bad hands, cropped, worst quality', 
        num_samples = 1, 
        image_resolution = 512, 
        detect_resolution =  512, 
        ddim_steps = 20, 
        guess_mode = False, 
        strength = 1.0, 
        scale = 9.0, 
        seed = 12345, 
        eta = 1.0):
        
        gen_path = []
        input_image = 255 - input_image # cv2.imread(input_image_path) The sketch images corresponding to 'input_image_path' might not be 224*224.

        preprocessor = self.preprocessor

        if 'HED' in det:
            if not isinstance(preprocessor, HEDdetector):
                preprocessor = HEDdetector()

        if 'PIDI' in det:
            if not isinstance(preprocessor, PidiNetDetector):
                preprocessor = PidiNetDetector()

        with torch.no_grad():
            input_image = HWC3(input_image)

            if det == 'None':
                detected_map = input_image.copy()
            else:
                detected_map = preprocessor(resize_image(input_image, detect_resolution))
                detected_map = HWC3(detected_map)

            img = resize_image(input_image, image_resolution)
            H, W, C = img.shape

            detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
            detected_map = nms(detected_map, 127, 3.0)
            detected_map = cv2.GaussianBlur(detected_map, (0, 0), 3.0)
            detected_map[detected_map > 4] = 255
            detected_map[detected_map < 255] = 0

            control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
            control = torch.stack([control for _ in range(num_samples)], dim=0)
            control = einops.rearrange(control, 'b h w c -> b c h w').clone()

            if seed == -1:
                seed = random.randint(0, 65535)
            seed_everything(seed)

            if config.save_memory:
                self.model.low_vram_shift(is_diffusing=False)

            cond = {"c_concat": [control], "c_crossattn": [self.model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
            un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [self.model.get_learned_conditioning([n_prompt] * num_samples)]}
            shape = (4, H // 8, W // 8)

            if config.save_memory:
                self.model.low_vram_shift(is_diffusing=True)

            self.model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)
            # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01

            samples, intermediates = self.ddim_sampler.sample(ddim_steps, num_samples,
                                                         shape, cond, verbose=False, eta=eta,
                                                         unconditional_guidance_scale=scale,
                                                         unconditional_conditioning=un_cond)

            if config.save_memory:
                self.model.low_vram_shift(is_diffusing=False)

            x_samples = self.model.decode_first_stage(samples)
            x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

            results = [x_samples[i] for i in range(num_samples)]
            for i in range(len(results)):
                result_rgb = cv2.resize(cv2.cvtColor(results[i], cv2.COLOR_BGR2RGB),(224,224))

                res_path = os.path.join(os.path.dirname(input_image_path), 'controlnet_res', input_image_path.split('/')[-1].replace('.png','').replace('.jpg','') + '_' + prompt.replace(' ', '+') + '_' + str(i) + '.png')
                if not os.path.exists(os.path.dirname(res_path)):
                    os.makedirs(os.path.dirname(res_path))

                cv2.imwrite(res_path, result_rgb)
                gen_path.append(res_path)

        return gen_path





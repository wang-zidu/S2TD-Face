import argparse
import cv2
import os
import sys

import torch
import numpy as np
from PIL import Image
import random

from recon.recon_sketch import face_model
from util.preprocess import get_data_path, resize_image, make_square, sketch_tensor
from util.io import visualize

from TDDFAv3.get_uv_texture import V3_get_uv_texture
from ControlNet.sketch_scribble import sketch_scribble
from clip_similarity.calculate_similarity import match_image_list


def main(args, albedo_setting):

    recon_model = face_model(args)
    gen_texture_from_controlnet = sketch_scribble()
    get_uv_texture_from_library = V3_get_uv_texture()

    for ind in range(len(albedo_setting)):
        this_setting = albedo_setting[ind]
        this_sketch = Image.open(this_setting['sketch_path']).convert('RGB')
        this_sketch = resize_image(make_square(this_sketch), (224, 224))
        this_sketch_tensor = sketch_tensor(this_sketch)
        recon_model.input_img = this_sketch_tensor.to(args.device)

        # Texture_Module Start
        # (1) Directly extract the texture from the image path.
        face_albedo_map_path_list = this_setting['direct_texture_path']
        face_albedo_map_in_uv_list = [True for i in range(len(face_albedo_map_path_list))]
        face_albedo_map_name_list = ['direct_' + str(i) for i in range(len(face_albedo_map_path_list))]

        # (2) Match textures in a preset texture library based on prompts using CLIP. (The original method in the paper)
        for this_prompt in this_setting['Prompt']:
            face_albedo_map_path_list += match_image_list(this_prompt, this_setting['clip_top_k'])
            face_albedo_map_in_uv_list += [True for i in range(this_setting['clip_top_k'])]
            face_albedo_map_name_list += ['library-' + this_prompt.replace(' ', '+') + '_top' + str(i) for i in range(this_setting['clip_top_k'])]
        
        # (3) Generate face images based on sketches and prompts using ControlNet, then extract textures. (New texture generation method)
        #     This method does not rely on the texture library, making it potentially more flexible.
        #     However, it may lead to issues with oversaturated albedo.
        for this_prompt in this_setting['Prompt']:
            random.seed(None)
            face_albedo_map_path_list += gen_texture_from_controlnet.process(
                input_image_path = this_setting['sketch_path'],
                input_image = np.asarray(this_sketch)[:,:,::-1], # The sketch images corresponding to this_setting['sketch_path'] might not be 224*224.
                prompt = this_prompt,
                num_samples = this_setting['controlnet_num'],
                seed = this_setting['controlnet_seed'] if 'controlnet_seed' in this_setting.keys() else random.randint(0, 65535),
            )
            face_albedo_map_in_uv_list += [False for i in range(this_setting['controlnet_num'])]
            face_albedo_map_name_list += ['controlnet-' + this_prompt.replace(' ', '+') + str(i) for i in range(this_setting['controlnet_num'])]

        face_albedo_map_list = []
        for i in range(len(face_albedo_map_path_list)):
            if face_albedo_map_in_uv_list[i]:
                temp = get_uv_texture_from_library.get(
                    im_path = face_albedo_map_path_list[i],
                    no_pca = False, # use pca-texture for texture blending
                )
                face_albedo_map_list.append(torch.from_numpy(cv2.resize(temp[:,:,::-1], (256, 256))).float().to(args.device).unsqueeze(0).permute(0,3,1,2) / 255.)
            else:
                temp = cv2.imread(face_albedo_map_path_list[i])
                face_albedo_map_list.append(torch.from_numpy(cv2.resize(temp, (224, 224))).float().to(args.device).unsqueeze(0).permute(0,3,1,2) / 255.)
        # Texture_Module End

        result_dict = recon_model.forward(
            beta_magnitude = args.beta_magnitude,
        )
        result_dict = recon_model.forward_texture(result_dict, face_albedo_map_list, face_albedo_map_in_uv_list, args.light_intensities)
        
        if not os.path.exists(os.path.join(args.savepath, this_setting['sketch_path'].split('/')[-1].replace('.png','').replace('.jpg',''))):
            os.makedirs(os.path.join(args.savepath, this_setting['sketch_path'].split('/')[-1].replace('.png','').replace('.jpg','')))
        my_visualize = visualize(result_dict, args)

        my_visualize.visualize_and_output(None, cv2.cvtColor(np.asarray(this_sketch), cv2.COLOR_RGB2BGR), \
            os.path.join(args.savepath, this_setting['sketch_path'].split('/')[-1].replace('.png','').replace('.jpg','')), \
            this_setting['sketch_path'].split('/')[-1].replace('.png','').replace('.jpg',''), \
            face_albedo_map_name_list)

        print(str(ind + 1) + '/' + str(len(albedo_setting)), this_setting['sketch_path'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='S2TD-Face')

    parser.add_argument('-s', '--savepath', default='examples/results', type=str,
                        help='path to the output directory, where results (obj, png files) will be stored.')
    parser.add_argument('--device', default='cuda', type=str)

    parser.add_argument('--beta_magnitude', default=0.12, type=float,
                        help='adjust the magnitude of the normal offset')
    parser.add_argument('--light_intensities', default=1.5, type=float)

    # save
    parser.add_argument('--ldm68', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='save and show 68 landmarks')
    parser.add_argument('--ldm106', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='save and show 106 landmarks')
    parser.add_argument('--ldm106_2d', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='save and show 106 landmarks, face profile is in 2d form')
    parser.add_argument('--ldm134', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='save and show 134 landmarks' )
    parser.add_argument('--seg', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='save and show segmentation in 2d without visible mask' )
    parser.add_argument('--seg_visible', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='save and show segmentation in 2d with visible mask' )

    albedo_setting = [
        {
            'sketch_path': 'examples/sketch1.png',
            'direct_texture_path': ['texture_library/498.png', 'texture_library/993.png'],
            'Prompt': ["Cartoon Boy", "Oil Painting"],
            'clip_top_k': 3,
            'controlnet_num': 2,
            'controlnet_seed': 23500, # (optional)
        },

        {
            'sketch_path': 'examples/sketch2.png',
            'direct_texture_path': ['texture_library/413.png', 'texture_library/453.png'],
            'Prompt': ["Beauty in Makeup", "Face Art"],
            'clip_top_k': 3,
            'controlnet_num': 2,
        },

        {
            'sketch_path': 'examples/sketch3.png',
            'direct_texture_path': ['texture_library/1099.png', 'texture_library/1093.png'],
            'Prompt': ["Old Man", "Moustache"],
            'clip_top_k': 3,
            'controlnet_num': 2,
        },

        {
            'sketch_path': 'examples/sketch4.jpg',
            'direct_texture_path': ['texture_library/392.png', 'texture_library/210.png'],
            'Prompt': ["Bearded Man", "Cartoon Bearded Man"],
            'clip_top_k': 3,
            'controlnet_num': 2,
        },

        {
            'sketch_path': 'examples/sketch5.png',
            'direct_texture_path': ['texture_library/413.png', 'texture_library/1018.png'],
            'Prompt': ["Beauty in Makeup", "Oil Painting of a Girl"],
            'clip_top_k': 3,
            'controlnet_num': 2,
            'controlnet_seed': 30300, # (optional)
        },

        {
            'sketch_path': 'examples/sketch6.png',
            'direct_texture_path': ['texture_library/831.png', 'texture_library/773.png'],
            'Prompt': ["Heavy Makeup", "Halloween"],
            'clip_top_k': 3,
            'controlnet_num': 2,
        },

        {
            'sketch_path': 'examples/sketch7.png',
            'direct_texture_path': ['texture_library/751.png', 'texture_library/673.png'],
            'Prompt': ["Freckles", "Cartoon Freckle Boy"],
            'clip_top_k': 3,
            'controlnet_num': 2,
            'controlnet_seed': 16632, # (optional)
        },

        {
            'sketch_path': 'examples/sketch8.png',
            'direct_texture_path': ['texture_library/815.png', 'texture_library/926.png'],
            'Prompt': ["Cartoon Black Woman", "Black Woman Makeup"],
            'clip_top_k': 3,
            'controlnet_num': 2,
        },

        {
            'sketch_path': 'examples/sketch9.png',
            'direct_texture_path': ['texture_library/386.png', 'texture_library/402.png'],
            'Prompt': ["Sculpture", "Male Painting Portrait"],
            'clip_top_k': 3,
            'controlnet_num': 2,
        },
    ]

    main(parser.parse_args(), albedo_setting)

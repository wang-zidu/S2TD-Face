import argparse
import cv2
import os
import sys
sys.path.append(os.path.dirname(__file__))

import torch
import numpy as np
from PIL import Image

from model.recon import face_model
from face_box import face_box

class V3_get_uv_texture:
    def __init__(self):
        parser = argparse.ArgumentParser(description='3DDFA-V3')

        parser.add_argument('-i', '--inputpath', default='examples/', type=str,
                        help='path to the test data, should be a image folder')
        parser.add_argument('-s', '--savepath', default='examples/results', type=str,
                            help='path to the output directory, where results (obj, png files) will be stored.')
        parser.add_argument('--device', default='cuda', type=str,
                            help='set device, cuda or cpu' )

        # process test images
        parser.add_argument('--iscrop', default=True, type=lambda x: x.lower() in ['true', '1'],
                            help='whether to crop input image, set false only when the test image are well cropped and resized into (224,224,3).' )
        parser.add_argument('--detector', default='retinaface', type=str,
                            help='face detector for cropping image, support for mtcnn and retinaface')

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
        parser.add_argument('--useTex', default=True, type=lambda x: x.lower() in ['true', '1'],
                            help='save obj use texture from BFM model')
        parser.add_argument('--extractTex', default=True, type=lambda x: x.lower() in ['true', '1'],
                            help='save obj use texture extracted from input image')

        # backbone
        parser.add_argument('--backbone', default='resnet50', type=str,
                            help='backbone for reconstruction, support for resnet50 and mbnetv3')

        parser.add_argument('--beta_magnitude', default=0.12, type=float,
                            help='adjust the magnitude of the normal offset')
        parser.add_argument('--light_intensities', default=1.5, type=float)

        self.args = parser.parse_args()
        self.recon_model = face_model(self.args)
        self.facebox_detector = face_box(self.args).detector


    def get(self, im_path, no_pca = False):

        im = Image.open(im_path).convert('RGB')
        trans_params, im_tensor = self.facebox_detector(im)
        self.recon_model.input_img = im_tensor.to(self.args.device)
        results = self.recon_model.forward_uv(no_pca = no_pca)
        return results['extractTex_uv']



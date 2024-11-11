# Download [S2TD-Face](https://arxiv.org/abs/2408.01218) Assets

Download [assets](https://huggingface.co/datasets/Zidu-Wang/S2TD-Face/tree/main/assets) to ```assets```.

Download [texture_library](https://huggingface.co/datasets/Zidu-Wang/S2TD-Face/tree/main/texture_library) to ```texture_library/```.

# Download [3DDFA_V3](https://arxiv.org/abs/2312.00311) Assets

Download [assets](https://huggingface.co/datasets/Zidu-Wang/3DDFA-V3/tree/main/assets) to ```TDDFAv3/assets/```.

# Download [ControlNet](https://github.com/lllyasviel/ControlNet-v1-1-nightly) Code
Download [ControlNet-v1-1-nightly](https://github.com/lllyasviel/ControlNet-v1-1-nightly) code to ```ControlNet/``` (Directly using `git clone` might work as well, but since there are many files in ControlNet, it may be more convenient to download and then copy them over).

# Download [ControlNet](https://github.com/lllyasviel/ControlNet-v1-1-nightly) Assets
Download [control_v11p_sd15_scribble.pth](https://huggingface.co/lllyasviel/ControlNet-v1-1/tree/main) to ```ControlNet/models/```
Download [v1-5-pruned.ckpt](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/blob/main/v1-5-pruned.ckpt) to ```ControlNet/models```.

# Final File Structure Should be:

```
assets/
├── bfm_uvs2.npy
├── faces.npy
├── sketch_recon.pth
├── uvcoords.npy
└── uvfaces.npy

ControlNet/
├── annotator/
├── ldm/
├── ......
├── share.py
└── models/
    ├── ......
    ├── control_v11p_sd15_scribble.pth
    └── v1-5-pruned.ckpt

texture_library/
├── ...png
├── ......
└── ...png

TDDFAv3/
└── assets/
    ├── face_model.npy
    ├── large_base_net.pth
    ├── net_recon.pth
    ├── retinaface_resnet50_2020-07-20_old_torch.pth
    └── similarity_Lm3D_all.mat
```

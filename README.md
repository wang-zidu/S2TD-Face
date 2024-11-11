# S2TD-Face: Reconstruct a Detailed 3D Face with Controllable Texture from a Single Sketch

By [Zidu Wang](https://wang-zidu.github.io/), [Xiangyu Zhu](https://xiangyuzhu-open.github.io/homepage/), [Jiang Yu](https://scholar.google.com/citations?view_op=list_works&hl=zh-CN&user=QYY-3I0AAAAJ), [Tianshuo Zhang](https://openreview.net/profile?id=~Tianshuo_Zhang2) and [Zhen Lei](http://www.cbsr.ia.ac.cn/users/zlei/).

![teaser](/teaser/1.jpg)

This repository is the official implementation of [S2TD-Face](https://arxiv.org/abs/2408.01218) in [ACM-MM 2024](https://2024.acmmm.org/). [S2TD-Face](https://arxiv.org/abs/2408.01218) can reconstruct controllable textured and detailed 3D
faces from sketches.

## Getting Started
### Environment
  ```bash
  # Clone the repo:
  git clone https://github.com/wang-zidu/S2TD-Face
  cd S2TD-Face

  conda create -n S2TDFACE python=3.9
  conda activate S2TDFACE

  # pytorch
  conda install pytorch=1.13.0 torchvision pytorch-cuda=11.6 -c pytorch -c nvidia
  conda install numpy=1.23.1

  # pytorch3d
  conda install -c iopath iopath
  conda install -c bottler nvidiacub
  conda install pytorch3d -c pytorch3d

  # requirements
  python -m pip install -r requirements.txt

  # nvdiffrast
  git clone https://github.com/NVlabs/nvdiffrast.git
  cd nvdiffrast
  python -m pip install .
  cd ..

  # CLIP
  python -m pip install ftfy regex tqdm
  python -m pip install git+https://github.com/openai/CLIP.git
  ```  

### Usage
1. Please refer to this [README](https://github.com/wang-zidu/S2TD-Face/blob/main/assets/) to prepare assets and pretrained models.

2. Encode the images in the texture library using [CLIP](https://github.com/openai/CLIP).

    ```
    python clip_similarity/calculate_similarity.py
    ```

3. Run demos. The user can use the default `albedo_setting` in `demo.py` (recommended) or adjust `albedo_setting` as needed.

    ```
    python demo.py --savepath examples/results --device cuda --beta_magnitude 0.12 --light_intensities 1.5 --ldm68 1 --ldm106 1 --ldm106_2d 1 --ldm134 1 --seg_visible 1 --seg 1
    ```

     - `--savepath`: path to the output directory, where results (obj, png files) will be stored.

     - `--beta_magnitude`: control the magnitude of the displacement map.

     - `--light_intensities`: control the light intensity.

     <br>With the 3D mesh annotations provided by [3DDFA_V3](https://arxiv.org/abs/2312.00311), we can generate 2D/3D landmarks and 2D facial segmentation results based on the 3D mesh:

     - `--ldm68`, `--ldm106`, `--ldm106_2d` and `--ldm134`: save and show landmarks.

      - `--seg_visible`: save and show segmentation in 2D with visible mask. When a part becomes invisible due to pose changes, the corresponding region will not be displayed. All segmentation results of the 8 parts will be shown in a single subplot. 

      - `--seg`: save and show segmentation in 2D. When a part becomes invisible due to pose changes, the corresponding segmented region will still be displayed (obtained from 3D estimation), and the segmentation information of the 8 parts will be separately shown in 8 subplots.

4. Results.
     - `image_name_res.png`: the visualization results.
     - `image_name_shape_base.obj`: 3D coarse geometry mesh in OBJ format.
     - `image_name_shape_detail_XXX.obj`: 3D detailed geometry mesh in OBJ format  with corresponding textures.
     - `image_name_uv_XXX.png`: UV texture.
     - `image_name_displacement_map.png` and `image_name_displacement_map.npy`: the detail displacement map in UV space.
     - `image_name_position_map.png` and `image_name_uv_sketch.png` are used to reconstruct detailed geometry.


    <br>![teaser](/teaser/sketch1_res.png)<br>
    <br>![teaser](/teaser/sketch5_res.png)<br>

## Texture Control Module

Compared to the original Texture Control Module in [S2TD-Face](https://github.com/wang-zidu/S2TD-Face), this GitHub repository introduces two main improvements:
1. We additionally support generating face images based on sketches and prompts using [ControlNet](https://github.com/lllyasviel/ControlNet-v1-1-nightly), then extracting textures. This method does not rely on the texture library, making it potentially more flexible. The UV texture in this part depends on the geometry estimation from the sketch (while the texture library's UV texture estimation uses [3DDFA_V3](https://arxiv.org/abs/2312.00311)).

2. We use [Poisson Blending](https://www.cs.jhu.edu/~misha/Fall07/Papers/Perez03.pdf) to merge invisible areas of the UV texture with PCA textures or flipped image textures. This approach results in smoother boundary transitions compared to the original median filtering method.

## Note

1. The [texture library](https://huggingface.co/datasets/Zidu-Wang/S2TD-Face/tree/main/texture_library) we provide is only an example, with data primarily sourced from [FFHQ](https://github.com/NVlabs/ffhq-dataset) and [Getty Images](https://www.gettyimages.hk/). If you wish to create a custom texture library, please ensure the diversity of selected images and, whenever possible, include frontal-face images to maintain the quality and accuracy of UV texture matching.

2. Although the Texture Control Module with [ControlNet](https://github.com/lllyasviel/ControlNet-v1-1-nightly) is more flexible, it may result in oversaturated albedo. We still recommend using the texture library method, which is simpler. Alternatively, you can choose to directly specify any facial image as the texture for the sketch. See the `albedo_setting` in `demo.py`.

3. [S2TD-Face](https://arxiv.org/abs/2408.01218) can also extract certain attributes from sketches, such as 2D/3D landmarks and segmentation. Although these features may not achieve perfect accuracy, they are rarely available in other projects.

4. For improved rendering results, consider using [Blender](https://www.blender.org/) or employing [Blender Python scripts](https://github.com/yuki-koyama/blender-cli-rendering).

## Citation
If you use our work in your research, please cite our publication:
```
@inproceedings{wang2024s2td,
  title={S2TD-Face: Reconstruct a Detailed 3D Face with Controllable Texture from a Single Sketch},
  author={Wang, Zidu and Zhu, Xiangyu and Yu, Jiang and Zhang, Tianshuo and Lei, Zhen},
  booktitle={Proceedings of the 32nd ACM International Conference on Multimedia},
  pages={6453--6462},
  year={2024}
}
```

## Summary & Discussion
- The results of S2TD-Face exhibit topological consistency and excel in matching the identity depicted in the input sketch.
- The geometry and texture provided by S2TD-Face can serve as regularization, initialization, and reference, potentially reducing issues like Janus problems, incorrect proportions, and oversaturated albedo commonly found in Score Distillation Sampling methods.
- The Texture Control Module in this repository may have broader applications.
  
## Contact
If you have any suggestions or requirements, please feel free to contact us at wangzidu0705@gmail.com. 

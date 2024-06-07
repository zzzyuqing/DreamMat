# DreamMat
DreamMat: High-quality PBR Material Generation with Geometry- and Light-aware Diffusion Models
## [Paper](https://arxiv.org/abs/2405.17176) | [Project page](https://zzzyuqing.github.io/dreammat.github.io/)

![](assets/teaser.png)

### Preparation for inference
1. Install packages in `requirements.txt`.
    We test our model on 3090/4090/V100/A6000 with 11.8 CUDA and 2.0.0 pytorch.
    ```
    git clone https://github.com/zzzyuqing/DreamMat.git
    cd DreamMat
    pip install -r requirements.txt
    ```
2. Install Blender 

    Download [blender-3.2.2-linux-x64.tar.xz](https://download.blender.org/release/Blender3.2/)
    
    Run:
    ```bash
    tar -xvf blender-3.2.2-linux-x64.tar.xz
    export PATH=$PATH:path_to_blender/blender-3.2.2-linux-x64
    ```


3. Download the pre-trained ControlNet checkpoints [here](https://pan.zju.edu.cn/share/78d6588ec65bcfa432ed22d262) or from [hugging face](https://huggingface.co/zzzyuqing/light-geo-controlnet), and put it to the `threestudio_dreammat/model/controlnet`
4. A docker env can be found at https://hub.docker.com/repository/docker/zzzyuqing/dreammat_image/general

### Inference

```
cd threestudio_dreammat
sh cmd/run_examples.sh
```

Upon initial execution, each model will undergo pre-rendering using Blender, with an approximate duration of 15 minutes on a 4090 GPU. During this period, there will be no output; thus, patience is requested. For subsequent runs, the `blender_generate` can be set to `false` to bypass this process.
### Geometry- and Light-aware ControlNet
You can also train your own geometry- and light-aware ControlNet. The methods for dataset generation and the training code are presented as follows.

![](assets/pipeline_controlnet.png)
#### Preparation for training
Make sure the environment map folder structure as
```bash
dataset
|-- <env_dir>
    |-- map1
        |-- map1.exr
    |-- map2
        |-- map2.exr
    |-- map3
        |-- map3.exr
    |-- map4
        |-- map4.exr
    |-- map5
        |-- map5.exr
```

Run the following code to generate pre-rendered data for training
```bash
cd controlnet_train
blender -b -P blender_script_geometry.py -- \
    --object_path ./dataset/model/046e3307c74746a58ec4bea5b33b7b97.glb \
    --output_dir ./dataset/training_data \
    --elevation 30 \
    --num_images 16


blender -b -P blender_script_light.py -- \
    --object_path ./dataset/model/046e3307c74746a58ec4bea5b33b7b97.glb \
    --env_dir ./dataset/envmap \
    --output_dir ./dataset/training_data \
    --elevation 30 \
    --num_images 16
```

The dataset folder structure will be as follows
```bash
dataset
|-- training_data
    |-- <uid_0>
        |-- color
            |-- 000_color_env1.png
            |-- ...
        |-- depth
            |-- 000.png
            |-- ...
        |-- light
            |-- 000_m0.0r0.0_env1.png
            |-- ...
        |-- normal
            |-- 000.png
            |-- ...
    |-- <uid_1>
    |-- ...
    
```

#### Training ControlNet

before training, make sure that the json file of prompts is in the format of 
```json
{
    "<uid_0>" : "<prompt_0>",
    "<uid_1>" : "<prompt_1>",
    "<uid_2>" : "<prompt_2>",
    ...
}
```


and the directory of training data is in the structure of
```bash
training_data
|-- <uid_0>
|-- <uid_1>
|-- <uid_2>
|-- ...

```
We provide several data as examples [here](https://pan.zju.edu.cn/share/b1724c30e0b5a3a9861a58570e).

run the training
```bash
cd controlnet_train
accelerate launch diffusers_train_controlnet.py --config config.json 
```


## Acknowledgement
We have intensively borrow codes from the following repositories. Many thanks to the authors for sharing their codes.
- [threestudio](https://github.com/threestudio-project/threestudio)
- [stable diffusion](https://github.com/CompVis/stable-diffusion)
- [CSD](https://github.com/CVMI-Lab/Classifier-Score-Distillation)
- [NeRO](https://github.com/liuyuan-pal/NeRO)
- [Fantasia3D](https://github.com/Gorilla-Lab-SCUT/Fantasia3D)
- [SyncDreamer](https://github.com/liuyuan-pal/SyncDreamer)
- [diffusers](https://github.com/huggingface/diffusers)
- [ControlNet](https://github.com/lllyasviel/ControlNet)

In addition to the 3D model from [Objaverse](https://objaverse.allenai.org/), we express our profound appreciation to the contributors of the following 3D models:
- [Bobcat machine](https://sketchfab.com/3d-models/bobcat-machine-7845344823cb4cdcb99963f561e5d866) by mohamed ouartassi.
- [Molino De Viento \_ Windmill](https://sketchfab.com/3d-models/molino-de-viento---windmill-2ea0a5296d4b49dbad71ce1975c0e3ff) by BC-X.
- [MedivalHouse | house for living | MedivalVilage](https://sketchfab.com/3d-models/medivalhousehouse-for-livingmedivalvilage-ba53607959b0476fb719043c406bc245) by JFred-chill.
- [Houseleek plant](https://sketchfab.com/3d-models/houseleek-plant-70679a304b324ca8941c214875acf6a9) by matousekfoto.
- [Jagernaut (Beyond Human)](https://sketchfab.com/3d-models/jagernaut-beyond-human-977e3a466dbc4c859071e342c6b6151e) by skartemka.
- [Grabfigur](https://sketchfab.com/3d-models/grabfigur-fbd44dd62766450abefaa0e43941633e) by noe-3d.at.
- [Teenage Mutant Ninja Turtles - Raphael](https://sketchfab.com/3d-models/teenage-mutant-ninja-turtles-raphael-191f64c3a6a44218a98a4d93f44229a9) by Hellbruch. 
- [Cat with jet pack](https://sketchfab.com/3d-models/cat-with-jet-pack-9afc8fd58c0d4f7d827f2007d6ac1e80) by Muru.
- [Transformers Universe: Autobot Showdown](https://sketchfab.com/3d-models/transformers-universe-autobot-showdown-7a3f2d273f354b29b31f247beb62d973) by Primus03.
- [PigMan](https://sketchfab.com/3d-models/pigman-f7597d3af7224f7e890710ac27d4d597) by Grigorii Ischenko.
- [Bulky Knight](https://sketchfab.com/3d-models/bulky-knight-002a90cbf12941b792f9685546a7502c) by Arthur Krut.
- [Sir Frog](https://sketchfab.com/3d-models/sir-frog-chrono-trigger-0af0c15e947143be8fab274841764bf1) by Adrian Carter.
- [Infantry Helmet](https://sketchfab.com/3d-models/infantry-helmet-ba3a571a8077417f80ae0e06150c91d2) by Masonsmith2020.
- [Sailing Ship Model](https://sketchfab.com/3d-models/sailing-ship-model-ac65e0168e8c423db9c9fdc71397c84e) by Andrea Spognetta (Spogna). 
- [Venice Mask](https://sketchfab.com/3d-models/venice-mask-4aace12762ee44cf97d934a6ced12e65) by DailyArt.
- [Bouddha Statue Photoscanned](https://sketchfab.com/3d-models/bouddha-statue-photoscanned-2d71e5b04f184ef89130eb26bc726add) by amcgi.
- [Bunny](https://sketchfab.com/3d-models/bunny-c362411a4a744b6bb18ce4ffcf4e7f43) by vivienne0716.
- [Baby Animals Statuettes](https://sketchfab.com/3d-models/baby-animals-statuettes-cadc2617612d47468e92360960583dc9) by Andrei Alexandrescu.
- [Durian The King of Fruits](https://sketchfab.com/3d-models/durian-the-king-of-fruits-62cc563e52514fa9b2e3dfdfc09e5377) by Laithai.
- [Wooden Shisa (Okinawan Guardian Lion)](https://www.artstation.com/artwork/LVvnk) by Vlad Erium.

We express our profound gratitude to [Ziyi Yang](https://github.com/ingra14m) for his insightful discussions during the project, and to Lei Yang for his comprehensive coordination and planning. This research work was supported by Information Technology Center and Tencent Lightspeed Studios.
Concurrently, we are also exploring more advanced 3D representation and inverse rendering technologies such as [Spec-Gaussian](https://github.com/ingra14m/Specular-Gaussians) and [SIRe-IR](https://github.com/ingra14m/SIRe-IR).

## Citation
If you find this repository useful in your project, please cite the following work. :)
```
@inproceedings{zhang2024dreammat,
  title={DreamMat: High-quality PBR Material Generation with Geometry- and Light-aware Diffusion Models},
  author={Zhang, Yuqing and Liu, Yuan and Xie, Zhiyu and Yang, Lei and Liu, Zhongyuan and Yang, Mengzhou and Zhang, Runze and Kou, Qilong and and Lin, Cheng and Wang, Wenping and Jin, Xiaogang},
  booktitle={SIGGRAPH},
  year={2024}
}
```

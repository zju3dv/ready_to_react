This repo is the official implementation of "Ready-to-React: Online Reaction Policy for Two-Character Interaction Generation".

<!-- [arxiv]() | [project page](https://zju3dv.github.io/ready_to_react/) -->


## News
[2025/02/20] We release the DuoBox dataset, training and evaluation code.

## Installation
```bash
conda create -n react python=3.9
conda activate react
# install pytorch
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.6 -c pytorch -c nvidia
# install pytroch3d
pip install pytorch3d-0.7.2-cp39-cp39-linux_x86_64.whl
# install other requirements
cat requirements.txt | sed -e '/^\s*-.*$/d' -e '/^\s*#.*$/d' -e '/^\s*$/d' | awk '{split($0, a, "#"); if (length(a) > 1) print a[1]; else print $0;}' | awk '{split($0, a, "@"); if (length(a) > 1) print a[2]; else print $0;}' | xargs -n 1 pip install
# install ready2react lib
pip install -e . --no-build-isolation --no-deps
```

NOTE:
`pytorch3d` [download link](https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu116_pyt1130/download.html).


## Data preparation
### DuoBox dataset
1. To download DuoBox, please fill in this [form](https://docs.google.com/forms/d/e/1FAIpQLSe8-G1PxykPoCR0wjH4HZKXkWz3_5UTa9x1if3L7bGPUXhmNA/viewform?usp=sharing). We provide skeleton extracted from OptiTrack and SMPL-X fitting results. In this project, we use the skeleton extracted from OptiTrack. The skeleton definition can be found in [link](https://docs.optitrack.com/motive/data-export/data-export-bvh). 

The data format of files under fbx_export_joints is:
```python
{
    f'{sbj_name}_position': # (T, J, 3), global position of joints
    f'{sbj_name}_rotation': # (T, J, 3, 3), [:, 1] are the root global orientations, the [:, 1:] are the other joints' local rotations relative to their parents
}
```

The data format of files under fbx_fit_smpl is:
```python
{
    f'{sbj_name}':
    {
        'transl':  # (T, 3)
        'global_orient': # (T, 3)
        'body_pose': # (T, 63)
        'betas': # (T, 10)
    }
}
```

2. Link to data/:
```bash
mkdir data
ln -s /path/to/DuoBox data/DuoBox
```

### Preprocess data

1. Dataset spliting and prepare for reactive motion model:
```bash
python reactmotion/scripts/main.py -t data_test -c configs/exps/vqvae_tokenizer.yaml dataloader_cfg.dataset_cfg.gen_motion_split=True
python reactmotion/scripts/main.py -t data_test -c configs/exps/vqvae_tokenizer.yaml dataloader_cfg.dataset_cfg.gen_preprocess=True
python reactmotion/scripts/main.py -t data_test -c configs/exps/reactive_model.yaml dataloader_cfg.dataset_cfg.gen_preprocess=True
```
Files will be saved in data/boxing/reactive.

2. Prepare for sparse control model:
```bash
python reactmotion/scripts/main.py -t data_test -c configs/exps/sparse_control.yaml dataloader_cfg.dataset_cfg.gen_motion_split=True
python reactmotion/scripts/main.py -t data_test -c configs/exps/sparse_control.yaml dataloader_cfg.dataset_cfg.gen_preprocess=True
```

### Pretrained models
Weights are shared in [link](https://drive.google.com/file/d/11H7-JoMobmnWFfx2TkF1TOKOBrxC4PQl/view?usp=sharing). Download and unzip under `data/`.

```bash
/data
|__ DuoBox
|__ trained_model
```

## Evaluation
Useful flags:
```bash
model_cfg.inference_func=reactive or twoagent # inference type
runner_cfg.evaluator_cfg.feature_compute=False # skip eval FID
runner_cfg.evaluator_cfg.apd_compute=True # eval APD
```

### Inference setting # 1: reactive motion generation
```bash
python reactmotion/scripts/main.py -t test -c configs/exps/reactive_model.yaml model_cfg.inference_func=reactive
```

### Inference setting # 2: two-character motion generation
```bash
python reactmotion/scripts/main.py -t test -c configs/exps/reactive_model.yaml model_cfg.inference_func=twoagent
```


### Inference setting # 3: long-term two-character motion generation
```bash
python reactmotion/scripts/main.py -t test -c configs/exps/reactive_model.yaml model_cfg.inference_func=twoagent model_cfg.use_gt_length=False model_cfg.num_new_frames=1800
```

### Inference setting # 4: sparse control
```bash
python reactmotion/scripts/main.py -t test -c configs/exps/sparse_control.yaml model_cfg.inference_func=reactive
```

## Training the model by yourself

To train the models, simply set `-t train` istead of `-t test`. Don't forget to change the `exp_name=yourexpname`.

### Step # 1: train the VQ-VAE motion tokenizer
```bash
python reactmotion/scripts/main.py -t train -c configs/exps/vqvae_tokenizer.yaml exp_name=yourexpname1
```

### Step # 2: train the reactive motion model
Set the config `motoken_cfg_file` to the config file generated in step # 1. It should be saved in `data/record/yourexpname1`.
```bash
python reactmotion/scripts/main.py -t train -c configs/exps/reactive_model.yaml exp_name=yourexpname2
```


## Acknowledgements
- [EasyVolCap](https://github.com/zju3dv/EasyVolcap) for their nice code base.
- [T2M-GPT](https://github.com/Mael-zys/T2M-GPT) for the VQ-VAE architecture.


## Citation
If you use our model or dataset, please cite our Ready-to-React paper. If you use easyvolcap codebase, please cite EasyVolcap as follows.
```
@inproceedings{cen2025_ready_to_react,
  title={Ready-to-React: Online Reaction Policy for Two-Character Interaction Generation},
  author={Cen, Zhi and Pi, Huaijin and Peng, Sida and Shuai, Qing and Shen, Yujun and Bao, Hujun and Zhou, Xiaowei and Hu, Ruizhen},
  booktitle={ICLR},
  year={2025}
}
@article{xu2023easyvolcap,
  title={EasyVolcap: Accelerating Neural Volumetric Video Research},
  author={Xu, Zhen and Xie, Tao and Peng, Sida and Lin, Haotong and Shuai, Qing and Yu, Zhiyuan and He, Guangzhao and Sun, Jiaming and Bao, Hujun and Zhou, Xiaowei},
  booktitle={SIGGRAPH Asia 2023 Technical Communications},
  year={2023}
}

```
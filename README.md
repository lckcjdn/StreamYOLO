# StreamYOLO for VisDrone MOT

这个仓库已经改成面向 `E:\VOD-dataset\VisDrone_MOT_TransVOD` 的 StreamYOLO one-future 训练版本，默认配置对齐环境 `E:\python\envs\TransVOD_py310_cu128` 的核心组合：

- Python `3.10.18`
- CUDA `12.8`
- PyTorch `2.9.1+cu128`
- TorchVision `0.24.1+cu128`
- TorchAudio `2.9.1+cu128`

## 这次改了什么

- 新增 VisDrone 11 类类别表：`exps/data/visdrone_class.py`
- 新增 VisDrone one-future 数据集：`exps/dataset/tal_flip_one_future_visdronedataset.py`
- 新增 VisDrone 评估器：`exps/evaluators/onex_stream_evaluator_visdrone.py`
- 新增 VisDrone 训练配置：
  - `cfgs/visdrone_s_s50_onex_dfp_tal_flip.py`
  - `cfgs/visdrone_m_s50_onex_dfp_tal_flip.py`
  - `cfgs/visdrone_l_s50_onex_dfp_tal_flip.py`
- 新增 WSL/Linux 训练脚本：`scripts/train_visdrone.sh`
- 新增 Conda 环境文件：`environment.yml`
- 新增依赖清单：`requirements.txt`
- 新增兼容别名文件：`requestment.txt`
- 在依赖里加入了 `conda-pack`，方便把训练环境整体打包迁移

## 数据集假设

当前实现直接复用你现有的 COCO 风格 JSON，不再依赖 Argoverse 的：

- `seq_dirs`
- `sid`
- `fid`
- `name`
- `Argoverse-1.1/tracking/...` 路径拼接

当前数据按照 `video_id + frame_id` 建立前一帧和目标帧关系：

- 支持帧：当前帧的前一帧，若是序列首帧则回退到当前帧
- 目标帧：当前帧的后一帧，若是序列末帧则回退到当前帧
- 评估时直接使用数据集返回的目标 `image_id`，不再写 Argoverse 专用跳过逻辑

你给的数据统计和这个实现是一致的：

- 训练集 56 个视频
- 24201 帧
- 1105516 条标注
- 11 个类别
- 每个视频内部尺寸固定

## 数据目录

默认根目录是：

```text
E:\VOD-dataset\VisDrone_MOT_TransVOD
```

目录结构应为：

```text
E:\VOD-dataset\VisDrone_MOT_TransVOD
├─annotations
│  ├─imagenet_vid_train.json
│  ├─imagenet_vid_val.json
│  ├─imagenet_vid_test.json
│  └─imagenet_vid_train_joint_30.json
└─Data
   └─VID
      ├─train
      ├─valid
      └─test
```

如果后续想换目录，不需要改代码，直接设置环境变量即可：

```powershell
$env:STREAMYOLO_VISDRONE_ROOT = 'E:\VOD-dataset\VisDrone_MOT_TransVOD'
```

## Conda 环境

推荐在仓库内创建独立环境：

```powershell
cd D:\SAR_Nir_dt\StreamYOLO
conda create -p .\.conda\streamyolo_visdrone_py310_cu128 python=3.10.18 pip=25.3 setuptools=80.9.0 wheel=0.45.1 -y
conda activate D:\SAR_Nir_dt\StreamYOLO\.conda\streamyolo_visdrone_py310_cu128
pip install -r requirements.txt
pip install --no-deps https://github.com/Megvii-BaseDetection/YOLOX/archive/refs/tags/0.3.0.zip
```

然后把仓库加入 `PYTHONPATH`：

```powershell
$env:PYTHONPATH = "D:\SAR_Nir_dt\StreamYOLO;$env:PYTHONPATH"
```

如果你更想用名字而不是前缀路径，也可以：

```powershell
conda create -n streamyolo_visdrone_py310_cu128 python=3.10.18 pip=25.3 setuptools=80.9.0 wheel=0.45.1 -y
conda activate streamyolo_visdrone_py310_cu128
pip install -r requirements.txt
pip install --no-deps https://github.com/Megvii-BaseDetection/YOLOX/archive/refs/tags/0.3.0.zip
```

## 依赖说明

`requirements.txt` 分成两部分：

- 和 `TransVOD_py310_cu128` 对齐的核心运行栈
- StreamYOLO 额外需要的工具库：`loguru`、`tabulate`、`tensorboard`、`thop`、`ninja`、`psutil`

`yolox` 建议单独安装：

```powershell
pip install --no-deps https://github.com/Megvii-BaseDetection/YOLOX/archive/refs/tags/0.3.0.zip
```

这样可以避开 `PyPI yolox==0.3.0` 在 Windows + Python 3.10 下继续拉起旧版 `onnx` 源码编译的问题。

其中还额外加入了：

- `conda-pack`

它用于把 Conda 环境打包成可迁移压缩包。

## 训练

直接训练 `m` 模型：

```powershell
python tools/train.py -f cfgs/visdrone_m_s50_onex_dfp_tal_flip.py -d 1 -b 8 -c <your_yolox_m_checkpoint.pth> --fp16
```

如果你手头是 `YOLOX-s` 或 `YOLOX-l` 的 COCO 预训练权重，请改用对应配置：

```powershell
python tools/train.py -f cfgs/visdrone_s_s50_onex_dfp_tal_flip.py -d 1 -b 8 -c .\pretrained\yolox_s.pth --fp16
python tools/train.py -f cfgs/visdrone_l_s50_onex_dfp_tal_flip.py -d 1 -b 4 -c .\pretrained\yolox_l.pth --fp16
```

常用说明：

- `-d`：GPU 数量
- `-b`：总 batch size
- `-c`：预训练权重路径
- `--fp16`：混合精度训练

输出目录默认写到：

```text
D:\SAR_Nir_dt\StreamYOLO\outputs\streamyolo_visdrone
```

## WSL/Linux 训练脚本

仓库内已经提供了一个 bash 脚本：

```bash
scripts/train_visdrone.sh
```

示例：

```bash
cd /mnt/d/SAR_Nir_dt/StreamYOLO
chmod +x scripts/train_visdrone.sh

MODEL_SIZE=s \
BATCH_SIZE=8 \
DEVICES=1 \
NUM_WORKERS=4 \
MAX_EPOCH=30 \
./scripts/train_visdrone.sh
```

可用环境变量包括：

- `MODEL_SIZE=s|m|l`
- `CKPT=/path/to/yolox_*.pth`
- `STREAMYOLO_VISDRONE_ROOT=/path/to/VisDrone_MOT_TransVOD`
- `BATCH_SIZE`
- `DEVICES`
- `NUM_WORKERS`
- `MAX_EPOCH`
- `EXPERIMENT_NAME`

## 评估

```powershell
python tools/eval.py -f cfgs/visdrone_m_s50_onex_dfp_tal_flip.py -c <your_ckpt.pth> -d 1 -b 8 --conf 0.01 --fp16
```

## 打包环境

环境安装完成后，可以直接打包：

```powershell
New-Item -ItemType Directory -Force -Path .\dist | Out-Null
conda pack -p .\.conda\streamyolo_visdrone_py310_cu128 -o .\dist\streamyolo_visdrone_py310_cu128.zip --format zip
```

如果你使用的是命名环境：

```powershell
conda pack -n streamyolo_visdrone_py310_cu128 -o .\dist\streamyolo_visdrone_py310_cu128.zip --format zip
```

## 说明

- 这个仓库仍然依赖外部 `yolox` 包，不是 vendor-in 仓库内部版本
- 由于当前会话网络受限，我没有办法在这里直接把 `yolox` 从外网安装完成
- 我已经把环境文件、依赖文件和数据集适配代码补齐了，后续只要在可联网或有内网镜像的环境里执行 `conda env create` 即可

## Citation

如果这份修改版仓库对你的工作有帮助，请同时引用原始 StreamYOLO 论文：

```bibtex
@inproceedings{streamyolo,
  title={Real-time Object Detection for Streaming Perception},
  author={Yang, Jinrong and Liu, Songtao and Li, Zeming and Li, Xiaoping and Sun, Jian},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={5385--5395},
  year={2022}
}
```

# 部署指南

拿到打包的环境和项目文件后，按以下步骤部署。


## 1. 恢复 Conda 环境
1.检查是否配好conda环境，打开cmd输入where conda ，若没报错，跳转到2.，若报错，则打开windows的设置-应用，搜索conda，确认没有安装conda后，下载网盘里的conda安装包，安装，然后跳转到1.1

1.1 win+S，搜索“编辑系统环境变量”，点击“环境变量”，在下方的“系统变量”里找到变量“Path”双击，点击右侧的新建，依次将miniconda根目录，以及根目录下的Script，Library\bin添加
    比如我的安装目录是E:\SOFT\miniconda，则添加这三条：E:\SOFT\miniconda，E:\SOFT\miniconda\Scripts，E:\SOFT\miniconda\Library\bin，然后打开cmd，输入conda init

2.打开cmd输入conda config --show envs_dirs，记住给出的几个路径，随便挑一个，将网盘里的环境dark-isp.zip解压进去，构成这样的结构：../envs/dark-isp/conda-meta...

3.将系统文件Dark-ISP解压到某个目录，比如D:/Dark-ISP/mmcv-2.1.0....，记住这个路径

4.回到之前的conda环境目录，../envs/dark-isp，打开以下这四个文件，将其中的内容改成右边的内容
  ..\envs\dark-isp\Lib\site-packages\mmcv.egg-link       → D:\Dark-ISP\mmcv-2.1.0 (第3步里的路径)
  ..\envs\dark-isp\Lib\site-packages\mmdet.egg-link      → D:\Dark-ISP\mmdetection_github (第3步里的路径)
  ..\envs\dark-isp\Lib\site-packages\mmengine.egg-link   → D:\Dark-ISP\mmengine-0.10.5 (第3步里的路径)
  ..\envs\dark-isp\Lib\site-packages\rawpy.egg-link      → D:\Dark-ISP\downloads\rawpy (第3步里的路径)


## 2. 配置数据集路径

数据集统一放在 `dataset/<DATA_NAME>/` 下：

```
dataset/
└── <DATA_NAME>/             # 如 RAWTest, NOD
    ├── RAW_dark_npz/        # npz 文件
    ├── RGB-dark-Annotations/# XML 标注
    └── trainval/
        ├── train.txt
        └── val.txt
```

然后运行：

```powershell
powershell -ExecutionPolicy Bypass -File setup_paths.ps1 <数据集完整路径>
```

示例：

```powershell
powershell -ExecutionPolicy Bypass -File setup_paths.ps1 D:\Dark-ISP\dataset\RAWTest
```

---


## 4. 验证

```bash
conda activate dark-isp
cd mmdetection_github
python check_env.py
python test_dataset.py
```

预期：`check_env.py` 打印 All imports OK + CUDA: True，`test_dataset.py` 打印 Dataset test PASSED。

---

## 5. 训练

```bash
conda activate dark-isp
cd mmdetection_github
python tools/train.py configs/LOD/VOCmetric/R_Net_denoise_50.py --work-dir work_dirs/my_train

$env:CUDA_LAUNCH_BLOCKING="1" ; python tools/train.py configs/LOD/VOCmetric/R_Net_denoise_50.py --work-dir work_dirs/my_train
python tools/train.py configs/LOD/VOCmetric/R_Net_DarkISP_18.py --work-dir work_dirs/2026_06_04_DarkISP_ResNet18 --resume checkpoints/2026_06_03_DarkISP_ResNet18/epoch_3.pth
从断点接着跑：--resume checkpoints/my_train/epoch_1.pth
用之前的权重：--cfg-options load_from=checkpoints/my_train/epoch_1.pth

```
## 6. 结果保存
1.权重：
  checkpoints/my_train/
  ├── epoch_1.pth          ← 每轮都存
  ├── epoch_2.pth
  ├── ...
  ├── epoch_15.pth
  ├── best_pascal_voc_mAP_epoch_8.pth   ← mAP 最高那个
  └── last_checkpoint                   ← 最后一次的，方便 --resume

2.日志和曲线：
  work_dirs/
  └── my_train/
      └── 20260512_191747/     ← 每次执行，若执行脚本的work_dirs设置的一样，就会保存在对应的时间戳里面

## 7. 结果查看
```bash
回到根目录
python plot_loss.py mmdetection_github/work_dirs/mytrain
```
自动找最新时间戳 → 读 scalars.json → 画出 loss + mAP，存为 training_curves.png。
所以原则上确保每次执行，都更新work_dirs的名字
---

## 8. 常见报错

### 8.1 `UnicodeDecodeError: 'gbk' codec can't decode ... cl.exe`

出现在 `mmengine/utils/dl_utils/collect_env.py:141`。中文 Windows 下 `cl.exe` 的输出不是本地
编码，mmengine 收集环境信息时解码失败直接把训练打断。

* 已在 `mmengine-0.10.5` 源码里加 fallback（`utf-8, errors='replace'`），正常情况不会再报。
* 若换了别的环境版本，可临时规避：`set PYTHONUTF8=1` 后再启动训练。

### 8.2 `FileNotFoundError: ...\RAW-dark-Annotations\2.xml`

说明跑的是**旧配置**（`R_Net_defaultISP_18/50.py`、`R_Net_demosaic_18/50.py`、`R_Net_enhance_*`、
`R_Net_LIS_*`、`R_Net_raw_adapter_*`、`SpRCNN_*`）。这批配置依赖**原版预渲染图片目录**
（`RGB_dark/`、`RAW_dark/`、`RAW-dark-Annotations/`），本地数据集只有：

```
dataset/RAWTest/
├── RAW_dark_npz/          # npz 原始数据
├── RGB-dark-Annotations/  # xml 标注
└── trainval/              # train.txt / val.txt
```

**本地可用的 ISP 对照配置**（都走 npz 管线，只有 `ISP_version` 不同）：

| 配置 | ISP | 说明 |
|---|---|---|
| `R_Net_DarkISP_18.py` | v2 | Dark-ISP 基线（61.90） |
| `R_Net_defaultISP_v0_18.py` | v0 | 传统 ISP 基线（D5 用这个） |
| `R_Net_demosaic_v1_18.py` | v1 | 仅去马赛克 |

> ⚠ 自动补全的坑：输入 `R_Net_defaultISP_<Tab>` 时，`R_Net_defaultISP_18.py`（旧，不可用）会排在
> `R_Net_defaultISP_v0_18.py`（新，可用）前面，务必打全名。

```bash
cd mmdetection_github
set PYTHONUTF8=1
python tools/train.py configs/LOD/VOCmetric/R_Net_defaultISP_v0_18.py --work-dir work_dirs/R_Net_defaultISP_v0_18
```


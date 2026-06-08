# Dark-ISP

***The implementation of paper "Dark-ISP: Enhancing RAW Image Processing for Low-Light Object Detection".***

## ⚙️ Dependencies and Installation:
Create the conda virtual environment with python version 3.8 and CUDA version 10.2
```bash
  conda create -n dark-isp python=3.8
  conda activate dark-isp
```
## 📦 Install PyTorch (CUDA)
```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```
## 📦 Install the requirement
```bash
  cd mmdetection_github
  pip install -r requirements.txt 
```
### ⚠️ There are several libraries that require editable installation. 
All of them must be the same as the given in this repository. If some libraries update automatically during the installation process, they need to be uninstalled and reinstalled.
#### 1.mmcv
```bash
cd ../mmcv-2.1.0
pip install -v -e .
```
#### 2.mmengine
``` bash
cd ../mmengine-0.10.5
pip install -v -e .
```
#### 3.mmdet
```bash
cd ../mmdetection_github
pip install -v -e .
```
#### 4.LibRaw and rawpy

> Windows 下编译安装步骤详见：[`downloads/LibRaw_Windows_Build_Guide.md`](downloads/LibRaw_Windows_Build_Guide.md)
## 🗄 Data Peparation
The LOD dataset and orignal RAW images are uploaded [here](https://pan.baidu.com/s/1Ek-q_9DaSLceLD4dhOYQAA?pwd=2025) (Extraction code：2025).

To accelerate training, it is necessary to preprocess RAW format files into npz files. The processed npz file was also uploaded to the cloud disk.

### Path Configuration
Run the setup script to configure dataset paths:
```powershell
powershell -ExecutionPolicy Bypass -File setup_paths.ps1 <your_dataset_dir>
```
Example:
```powershell
powershell -ExecutionPolicy Bypass -File setup_paths.ps1 E:\my_data\LOD
```
> This updates all config files, preprocessing scripts, and test scripts with your dataset path.

### Preprocessing (optional)
If you downloaded RAW files instead of preprocessed npz:
```python
python mmdetection_github/LOD_RAW_preprocess.py
```
## 🤖 Training and Evaluation
```python
python mmdetection_github/tools/train.py mmdetection_github/configs/LOD/VOCmetric/R_Net_denoise_50.py
```
## 📷 Test
Download the [checkpoint](https://pan.baidu.com/s/1xtvygvwL2FIoUkTJI7ogoA?pwd=2025) (Extraction code：2025).
```python
python mmdetection_github/tools/test.py mmdetection_github/configs/LOD/VOCmetric/R_Net_denoise_50.py mmdetection_github/checkpoints/Daek-ISP_LOD_70.4.pth
```

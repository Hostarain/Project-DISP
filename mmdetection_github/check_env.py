import mmcv
import mmengine
import mmdet
import rawpy
import torch

print('mmcv:', mmcv.__version__)
print('mmengine:', mmengine.__version__)
print('mmdet:', mmdet.__version__)
print('rawpy:', rawpy.__version__)
print('torch:', torch.__version__)
print('CUDA:', torch.cuda.is_available())
print('All imports OK')

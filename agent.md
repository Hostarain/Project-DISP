# Meta
- framework: MMDETECTION
- python_env: dark-isp

# Paths
- configs_dir: mmdetection_github/configs
- weights_dir: mmdetection_github/checkpoints
- data_dir: dataset
- output_dir: mmdetection_github/work_dirs

# Scripts
# 训练模型
```bash
python mmdetection_github/tools/train.py ${config_path} --work-dir ${work_dir}
```

# 检查环境
```bash
python mmdetection_github/check_env.py
```

# 测试数据集
```bash
python mmdetection_github/test_dataset.py
```

# 可视化推理结果
```bash
python infer_vis.py
```

# 可视化通用ISP结果
```bash
python vis_common_isp.py
```

# 绘制训练损失曲线
```bash
python plot_loss.py
```

# 预处理COCO合成数据集
```bash
python mmdetection_github/COCO_Syn_preprocess.py
```

# 预处理LOD RAW数据集
```bash
python mmdetection_github/LOD_RAW_preprocess.py
```

# 预处理NOD RAW数据集
```bash
python mmdetection_github/NOD_RAW_preprocess.py
```

# Parameters
| 参数名 | 状态 | 默认值 | 类型 | 描述 |
|--------|------|--------|------|------|
| config_path | 必填 | 无 | str | 训练配置文件路径 |
| work_dir | 可选 | ./work_dirs | str | 工作目录，用于保存日志和权重 |
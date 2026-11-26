# 项目记忆 — Dark-ISP（AGENTS.md）

> 最后更新：2026-10-28
> **本文件是项目级指令文件**：DSH 的 `dsh-agent-instructions` 包把 `AGENTS.md` 与 `CLAUDE.md` 一起作为候选，每次会话自动注入。
> Claude Code 不直接读本文件，由 `CLAUDE.md` 末尾的 `@AGENTS.md` 导入；原 `MEMORY.md` 已并入本文件（仅保留一个指针）。
> 配套文档：论证评估 `论文论证框架评估.md`、论文目录 `论文目录_定稿.md`。

---

## 0. 记忆维护约定

| 文件 | 记什么 | 不记什么 |
|---|---|---|
| `AGENTS.md`（本文件） | **状态**：当前阶段、实验事实（mAP/配置/可溯源路径）、已定决策、待定项、用户偏好 | 架构细节（在 `CLAUDE.md`）、论证审计（在评估文档） |
| `CLAUDE.md` | **稳定的工程说明**：环境、命令、架构、论文核心概念 | 任何"当前进度/最新结果" |
| `论文论证框架评估.md` | 论证审计、历史决议存档（§8 记录每轮的决定与发现的偏差） | 日常进度 |
| `论文目录_定稿.md` | 交付稿（给导师看的目录） | 内部待定项（写在末尾"编制说明"里） |

**何时写**：每轮对话结束前，若产生新事实（新 mAP、新结论、口径改动、新增待定项、发现的文档/代码偏差），立刻写回本文件。**过期结论直接改写，不要追加第二份**——本文件始终只描述"现在"。

**写法**：结论后面必须跟可溯源路径（`work_dirs/...`、`configs/...`、文件:行号）；拿不到溯源的结果要显式标注"未回传/不可复现"。

---

## 0.1 接手须知（给下一位智能体）

**这是什么项目**：论文复现 + 自研改进（Dark-ISP，低光 RAW 目标检测）。当前工作是**论证与实验补全**，不是工程开发——多数时间花在跑对照实验、算指标、改论文结构上。

**当前状态（2026-11-26）**：
- 论文目录第 2–4 章已定稿（`论文目录_定稿.md`），第 5 章待定。
- 第 2 章证据已齐（`Dark-ISP_res/`）；第 3–4 章证据在补。
- 结构算子 **GGNC 有正结果（65.27）**；第二个算子 **L_grad 三轮实验尚未显示增益**，正在用能量加权重试。

**三个必须先知道的坑**：
1. **受限进程导入 `mmdet` 会卡死**（yapf 构建语法表）。纯分析脚本用 numpy/cv2 自己实现；需要模型时用完全权限运行。
2. **受限进程无法在仓库目录内创建文件**。计算写 `%TEMP%`，交付物用一次完全权限的拷贝写出；受限与非受限模式下 `$env:TEMP` 指向不同目录。
3. **L_grad 曾两次 NaN 崩溃**：`sqrt(0)` 的反向为 inf，梯度裁剪把 `inf` 变成 `NaN`。修法见 §4。

**文档地图**：
| 想做什么 | 看哪个 |
|---|---|
| 了解项目状态与决策 | 本文件 |
| 写论文正文 | `方法论_两段跃进.md`（论证骨架）、`写作素材_细节与巧思.md`（可引用的细节与数字） |
| 知道还要跑什么 | `实验清单_第2-3章.md`、`实验清单_第3-4章.md` |
| 给导师看的目录 | `论文目录_定稿.md` |
| 查第 2 章数据 | `Dark-ISP_res/README.md` 及 `00_code`~`05_*` |
| 论证审计与历史决议 | `论文论证框架评估.md`（§8 记录每轮决定） |

**统一数值口径**：LOD / `dataset/RAWTest`、15 epoch、batch 2、VOC 11-point mAP、训练从 `mmdetection_github/` 启动。

---

## 1. 当前阶段

- 论文处于**论证设计**阶段，尚未开写正文。
- 目录：**第 2–4 章已定稿**（见 `论文目录_定稿.md`）；第 1 章绪论、**第 5 章待定**（与导师讨论后再排）。
- 论证链定稿为**单分支**（仅非线性分支）：GGNC v1/v2 + L_grad；线性分支的结构算子不再论证。
- **第 2–3 章实验清单已立**：`实验清单_第2-3章.md`（P1/P2 基建 + D1–D9 + R1 训练）。参考 ISP 定**方案 B：自建可分解实现**（六环节可开关），不用相机固件 ISP（黑盒、环节不可拆）。
- **第 2 章传统 ISP 侧实验已完成（2026-10-30，纯推理）**：D1–D4 + D3 像素级，结果已移入工作区 `Dark-ISP_res/`（`README.md` + `00_code`~`05_D4_invertibility`，可直接读写修改）。核心结论见 §3.2，**推翻了"量化是结构损耗主因"的原假设**。
- 已完成论证框架评估：发现 **3 个逻辑断点 + 2 处数据冲突**。
- **L_grad 三轮实验（2026-10-31 ~ 11-01）**：两次 `full` 掩码均在第 10 轮 NaN 崩溃（根因与修法见 §4）。诊断发现方向约束的**监督对象与权重都有问题**：弱梯度区被"近零梯度向量的夹角"这类噪声主导（中位 0.0008、p99 1.0、与 log 幅值相关 −0.82），且 λ=0.1 相对总损失仅占 0.08%~0.4%。已改为**能量加权**（`grad_mask='weighted'`）并重建配置 `R_Net_GGNC_v1_L_grad_energy_18.py`，待跑。详见 `写作素材_细节与巧思.md` 第 10/10b/10c 条。
- 用户明确：先论证，不急着开写。

---

## 2. 论文论证主线（待最终确认）

### 2.1 贯穿线

> 低光照检测的真正瓶颈不是“颜色/亮度”，而是“**结构信息的不可逆衰减**”。

### 2.2 四章链条

1. 传统 RGGB 链路：结构信息不可逆损耗。
2. 可微 Dark-ISP：恢复色调，但无显式结构约束。
3. 两个结构算子（GGNC + L_grad）：显式注入/约束结构。
4. 结构敏感检测器适配 + 端到端联合优化。

### 2.3 桥接命题

- **1 → 2**：损耗发生在 ISP 内部且不可逆，因此修复必须进入 ISP。
- **2 → 3**：残余缺陷可定位到非线性分支缺乏结构先验，因此需要结构算子。
- **3 → 4**：结构增益能否转化为检测增益，取决于检测器的监督与融合机制，因此需要检测器适配。
- **4 → 结论**：结构保真与任务监督必须联合优化，端到端是必要条件。

### 2.4 关键逻辑修正

- **不能**用“输入结构更强 → 必须换 YOLO”（non-sequitur）。
- 改为 **2 × 2 因子实验**，检验交互效应：`(D − C) > (B − A)`。
  - A：ResNet + 弱 ISP
  - B：ResNet + Dark-ISP/结构算子
  - C：YOLO 系 + 弱 ISP
  - D：YOLO 系 + Dark-ISP/结构算子
- 若交互效应不显著，第 5 章诚实地改为**工程适配章**（部署生态要求 YOLO 输出接口）。
- **不把 YOLO 放进大标题核心**；研究对象是“结构敏感检测器”，YOLO/RTMDet 作为实现载体。

### 2.5 标题草案

- 第 2 章　低光照 RGGB 成像链路的结构信息损耗机理
- 第 3 章　可微 Dark-ISP 的色调恢复与结构保持缺陷
- 第 4 章　结构引导的非线性校正算子与梯度一致性约束
- 第 5 章　结构敏感检测器的适配与端到端联合优化（待定）

---

## 3. 当前实验事实

| 实验 | 配置 | 最佳 mAP | 备注 |
|---|---|---|---|
| Dark-ISP baseline (R18) | `R_Net_DarkISP_18.py` | **61.90** | `work_dirs/2026_06_04_DarkISP_ResNet18/20260604_090135` |
| GGNC v1（feature-level） | `R_Net_GGNC_v1_18.py` | **65.27** | `work_dirs/R_Net_GGNC_18/20260610_173726`（历史命名） |
| GGNC v2（logit-level） | `R_Net_GGNC_v2_18.py` | **63.72** | `work_dirs/R_Net_GGNC_v2_18/20260627_093228`；热启动自 v2 的 epoch_5 |
| 线性分支 GGNC | 已删除配置 | 62.68 | +0.78，负结果；代码已删，不可复现 |
| YOLOX-s + Dark-ISP | `YOLOX_DarkISP_CSPDarknet_s.py` | **57.15** | `work_dirs/YOLOX/20260706_074505`；11.12M |
| L_grad | `R_Net_L_grad_18.py` | **无结果** | 配置存在，未完成训练 |
| GGNC v2 + L_grad | `R_Net_GGNC_v2_L_grad_18.py` | **无结果** | 配置存在，未完成训练 |
| **传统 ISP（v0）** | `R_Net_defaultISP_v0_18.py` | **63.41** | `work_dirs/2026_10_30_defaultISP_reNet18`；与论文 63.6 基本吻合（按当前口径只作内部参照，不进正文） |
| L_grad·full 第 1 次 | `R_Net_GGNC_v1_L_grad_18.py` | 62.16@9 后崩 | 第 10 轮 NaN；`work_dirs/2026_10_31_v1_Lgrad_full` |
| L_grad·full 第 2 次 | 同上 | 60.70@9 后崩 | 守卫只查 loss，未拦住梯度 NaN；`2026_10_31_v1_Lgrad_fix` |
| **L_grad·energy（待跑）** | `R_Net_GGNC_v1_L_grad_energy_18.py` | — | eps 进根号 + 梯度级守卫 + 能量加权，见 §4 |
| 论文 R-50 消融 | 原论文 | 66.6 / 68.7 / 70.4 | Linear / +Nonlinear / +Self-Boost |
| RTMDet-x + Dark-ISP | `DarkISP_RTMDet_x_H100.py` | **68.7** | 服务器 2×H100，100 epoch，COCO 预训练主干，**GGNC off**；work_dir 未回传，暂不可溯源（与论文 R-50 的 68.7 数值巧合，勿混淆） |
| RTMDet-s + Dark-ISP | `DarkISP_RTMDet_s_local.py` | **无结果** | 8.99M，Kaiming 从头训，50 epoch |

### 3.1 数据冲突

1. YOLOX-s 57.15 < Dark-ISP + ResNet18 61.90，不能直接宣称“换 YOLO 更强”。
2. **第 5 章的 YOLOX / RTMDet 配置关闭了 GGNC**：
   - `YOLOX_DarkISP_CSPDarknet_s.py` 未设置 `use_grad_guide`，`darkisp_cspdarknet.py` 默认 `False`。
   - `DarkISP_RTMDet_x_H100.py` / `DarkISP_RTMDet_s_local.py` 显式 `use_grad_guide=False`。
   - 连 68.7 这个最好的结果也是 GGNC off 得到的：第 4 章的结构算子没有带入第 5 章，论证链断裂。
3. **跨检测器比较的四个变量全不同**：参数量（20.19M / 11.12M / 8.99M / 94.91M）、初始化（ImageNet / COCO 检测主干 / Kaiming）、训练预算（15 / 15 / 50 / 100 epoch）、检测范式。故第 5 章以“同族内部差值”为主口径，不作绝对性能排名。

---

### 3.2 第 2 章传统 ISP 诊断结果（2026-10-30，纯推理，无训练）

数据：LOD/RAWTest 抽样 200 张（像素级量化分析 60 张）；参考 ISP 为自建六环节可分解实现，已与 `vis_isp_output/*_common.png`（原版 torch `default_ISP`）逐像素对齐，最大误差 0.0039（≈1/255）。结果目录：`Dark-ISP_res/`（位于工作区内）。

| 指标（暗区） | s1 黑电平 | +WB | +CCM | +降噪 | +gamma | +8bit |
|---|---|---|---|---|---|---|
| 方向一致性（vs 线性参考） | 0.9920 | 0.9304 | 0.7663 | 0.6335 | 0.6513 | 0.6498 |
| 梯度幅值 | 0.00545 | 0.00659 | 0.00791 | 0.00597 | 0.01393 | 0.01394 |

* **结构性损耗以方向失真为主**：方向一致性 0.992 → 0.650；单环节贡献为 WB −0.062、CCM **−0.164**、降噪 **−0.132**、gamma +0.017、量化 −0.0015。
* **幅值反而上升 2.56 倍**（gamma 拉伸暗区）→ 梯度幅值不能作为损耗指标（与第 3 章 3.3「幅值视角失灵」一致）。
* **留一法**：去掉降噪 → 暗区梯度回升 **+68.7%**（降噪抹掉约 41%）、方向 0.650→0.777；去掉 CCM → 方向 0.725；**去掉 8bit 量化 → 梯度仅 −0.08%**。
* **量化的真实量级**：像素级分析（1.03 亿暗区像素）显示 **11.53%** 的最弱暗区梯度低于自身亮度下的局部台阶，量化引起的梯度相对变化中位数仅 **2.28%**。
* **可逆性**：逆 gamma 后梯度幅值恢复 91%（0.00598 vs 0.00545），但方向一致性只有 0.632（未回到 0.992）→ **幅值可逆、方向不可逆**。

**对第 2 章写法的修订**：原计划把 8 比特量化当主因，实测不成立；结论应改为"损耗主体在降噪与色彩校正，量化只决定不可逆的边界"。2.2 的环节顺序与 2.3 的指标选择据此调整（指标以方向一致性为主、幅值为辅）。

---

## 4. 关键代码 / 配置事实

- ISP 三个版本（`dark_resnet.py`）：
  - `v0`：默认 ISP，detach。
  - `v1`：仅去马赛克。
  - `v2`：完整 Dark-ISP（线性 `Matrix_Predictor` + 非线性 `Color_Level_Process` + `mat_loss2`）。
- GGNC（`Color_Level_Process`）：
  - 梯度核 `grad_extract` 是**可学习卷积**（Sobel 初始化），不是“固定 Sobel”。
  - `grad_guide_mode='v1'`：Sobel → `grad_fuse`（6→32）→ 特征相加。
  - `grad_guide_mode='v2'`：Sobel → `grad_branch`（6→24，Tanh）→ `grad_scale` → 系数 logits。
- L_grad（`Dark_ResNet.grad_consistency_loss`）：
  - 对 `x_g` / `x_l` 做 Sobel，只惩罚梯度方向（`1 − cos`），不惩罚幅值。
  - **三档口径**（`grad_mask`）：
    - `'edge'`（默认，旧）：仅 `‖∇x_g‖ > mean+2σ` 的强梯度像素（约 3.7%）。实测该处方向偏差仅 0.0017，正则无对象，λ 再大也无效。
    - `'full'`：全图，仅 `min_mag=1e-4` 作数值下限。实测弱梯度区均值 0.0568，但**中位 0.0008、p99=1.0、与 log 幅值相关 −0.82**——主要由近零梯度向量间的无意义夹角贡献，训练信号被噪声主导。
    - `'weighted'`（推荐）：`L = Σ(1−cos)·‖∇x_g‖ / Σ‖∇x_g‖`。能量加权后真实失真为基线的 2.1 倍（0.0054→0.0114），信号落在中段（50–90% 分位，约一半梯度能量），无需人工阈值。
  - 分母必须写成 `sqrt(gx²+gy²+1e-12)`（**eps 在根号内**），否则平坦区反向为 inf → NaN。
  - 由 `use_grad_reg` / `lambda_grad` / `grad_mask` 控制。**λ=0.1 相对总损失只占 0.08%~0.4%，可能一直太弱**；若 `weighted` 版仍无差别，先升到 0.5 再下结论。
- `DarkISP_CSPDarknet` / `DarkISP_CSPNeXt`：
  - 仅支持 `use_grad_guide`，且不传 `grad_guide_mode`（只能跑 v1 模式）；**不支持 `use_grad_reg`（L_grad）**，第 5 章要用需先扩接口。
- `mat_loss2` **已加零行列式保护**（2026-10-31）：`|det| < 1e-8` 时跳过该项并返回 0，不再 `raise`；末尾另有 `isfinite` 兜底。`mat_loss`（旧接口）仍保留原 `raise` 行为。
- `DarkISP_RTMDet_*` 配置实为「RTMDet 骨架 + YOLOX 回归头」：有 TAL、有 `QualityFocalLoss`，**无 DFL**（`DistancePointBBoxCoder` + `GIoULoss` + `exp_on_reg`）；`cri_pix` 在 `SingleStageRAWDetector` 中 build 后从未调用，属死配置。
- 输入尺寸必须是 **32 的倍数**（PAFPN 严格 2× 上采样 + `torch.cat`）；当前配置用 `(672, 416)`。
- `YOLOXModeSwitchHook` 已修复 `RepeatDataset` 解包问题。
- 训练必须从 `mmdetection_github/` 目录启动，checkpoint 才会落在 `mmdetection_github/checkpoints/`。
- **NaN 崩溃与修复（2026-10-30/31）**：`R_Net_GGNC_v1_L_grad_18` 第 10 轮起全 NaN。根因是 `grad_consistency_loss` 里 `sqrt(gx²+gy²)` 在输出梯度为 0 处的反向是 1/0=inf，而全局梯度裁剪在 `total_norm=inf` 时 `scale=35/inf=0`，`inf×0=NaN`，一步毁掉全部参数（loss 全程正常、无尖峰）。修复：**eps 放进根号内** `sqrt(gx²+gy²+1e-12)` + `cos` 截断 + 有限性兜底（`dark_resnet.py`）。另加两道保险：`mat_loss2` 行列式保护（`|det|<1e-8` 跳过）、`NaNSkipHook`（`mmdet/engine/hooks/nan_skip_hook.py`，非有限损失跳过更新）。**`--resume epoch_9.pth` 会复活污染动量（mat_loss 有限但检测损失 NaN 即其特征），只能从零重跑**。
- **训练报错与修复（2026-10-30）**：
  - 报错一：`mmengine/utils/dl_utils/collect_env.py:141` `UnicodeDecodeError: 'gbk' codec can't decode`。中文 Windows 下 `cl.exe` 输出非本地编码，`collect_env()` 抛错直接中断训练。已在 `mmengine-0.10.5` 源码里改为 fallback 到 `utf-8 errors='replace'`（editable 安装，立即生效）；另可临时 `set PYTHONUTF8=1` 规避。
  - 报错二：`R_Net_defaultISP_18.py` / `R_Net_demosaic_18.py` / `R_Net_defaultISP_50.py` / `R_Net_demosaic_50.py` 依赖**预渲染图目录**（`RGB_dark/`、`RAW_dark/`、`RAW-dark-Annotations/`），本地数据集只有 `RAW_dark_npz/` + `RGB-dark-Annotations/`，必然 `FileNotFoundError`。**这批旧配置在本地不可用**。
  - 解决：新增两个薄配置继承主基线，只覆盖 ISP 版本 —— `R_Net_defaultISP_v0_18.py`（v0，D5 传统 ISP 基线）、`R_Net_demosaic_v1_18.py`（v1，仅去马赛克）。二者与 `R_Net_DarkISP_18.py` 共用 npz 管线，唯一差别是 `ISP_version`，是真正的同协议对照。
  - 训练实测：v0 配置在 2060 上约 0.31 s/iter（batch 2，2700 iter/epoch），15 epoch 约 3.5 小时；checkpoint 落在 `mmdetection_github/checkpoints/<work_dir 名>/`。
  - v0/v1 下 `params_predictor`、`nonlinear` 等模块仍会被构造但不参与计算（加载 ImageNet 权重时会报一堆 missing keys，属正常现象）。

- **环境坑（2026-10-30 实测）**：
  - `import mmdet` 在本机卡死（>4 分钟无返回，CPU 增长停滞）。已定位到导入链里的 **yapf**（`yapf_third_party/_ylib2to3/pgen2/grammar.py` 的语法表 dump，faulthandler 抓到的现场），CPU 空转。绕法：把 `site-packages/yapf` 临时改名（mmcv 对缺失 yapf 有 fallback，只失去 Config 的 pretty_text），或在非受限模式下运行。纯分析脚本**不要导入 mmdet/torch**，用 numpy/cv2 自己实现；需要对照原实现时，用仓库里已生成的产物（如 `vis_isp_output/*_common.png`）做像素级校验。
  - 受限进程（pwsh / python）**无法在仓库目录内创建文件或目录**（ACL 允许但沙箱 token 拒绝）；计算与中间产物写到 `%TEMP%`，交付物用一次完全权限的拷贝命令写出。注意：受限进程里 `$env:TEMP` 会被改写为会话子目录（如 `...\Temp\dsh-xxxx\`），完全权限进程里则是真实 `...\Temp`，跨模式传路径要用绝对路径。

---

## 5. 下一步（优先级）

**当前第一优先**：跑 `R_Net_GGNC_v1_L_grad_energy_18.py`（能量加权 + 双重 NaN 守卫，约 19h）。判据看前 6 轮相对 v1 同期（21.4/42.7/44.8/50.6/57.6/57.7）：持平或更好 → 跑满；落后 3 分以上 → λ 降到 0.05；几乎无差别 → λ 升到 0.5 再判。若最终仍不超 65.27，第 4 章按"GGNC 单算子 + L_grad 负结果"收尾（素材见 `写作素材_细节与巧思.md` 10/10b/10c）。

**第 2–3 章（成文前提）**：按 `实验清单_第2-3章.md` 执行。v0 基线已出（63.41），D5 的分桶漏检率与 D6 的结构-色度判别可用现有 ckpt 完成，**不再需要训练**。

**第 3–4 章（路线已定，细节待分章展开）**：`实验清单_第3-4章.md`，四次"发现缺陷→引出算子"决策（A 缺通路 / B 伪结构 / C 只能约束方向 / D 协同），E1–E10 + T1–T5。**E3（随机冻结梯度分支）优先级最高**：1 跑即可判断第 4 章核心贡献是"结构先验"还是"通路容量"。

### P0（不做则第 4 → 5 章断裂）

1. YOLO / RTMDet 打开 GGNC 的对照（同一检测器，`use_grad_guide=False` vs `True`）。
2. 结构指标：暗区梯度幅值、边缘密度、频域能量、暗区漏检率。
3. 公平比较协议：参数对齐，或改报增益转化率 `ΔmAP` + 2 × 2 交互实验。

### P1（第 4 章成立所需）

4. 跑完 `R_Net_L_grad_18.py` 与 `R_Net_GGNC_v2_L_grad_18.py`；另需新建 `v1 + L_grad`（v1 才是当前最优注入）。
5. 解决 v2（63.72）< v1（65.27）：重跑确认，或把 v1 作为主方法、v2 作为负向消融。
6. 端到端 vs detach ISP 对照（需给 `Dark_ResNet` 加 `detach_isp` 开关）。

### P2

7. RTMDet-s 跑通（本地 8.99M）。
8. 跨检测器泛化：GGNC 在 ResNet / YOLOX / RTMDet 上是否一致有效。

---

## 6. 用户偏好与工程约束

- **服务器无外网**：预训练权重必须离线打包（如 `checkpoints/cspnext_x_coco.pth`、`cspdarknet_s_coco.pth`）。
- **增量包**：以 `mmdetection_github` 为根目录，保持原目录结构，一键覆盖；不要额外脚本开训，用 `.py` 训练入口。
- **部署**：设备端脚本处理 YOLO 输出；导师要求 YOLO（Ultralytics 风格），曾嫌 “V3 太老”。
  - 应对策略：用精度与 mmdet 生态论证；RTMDet 作为 mmdet 中与 YOLOv8 同代的方案（CSPNeXt + PAFPN + TAL）。注意本项目 config **未启用 DFL**，不要对外表述成“完整 RTMDet”。
- **数据集**：LOD 训练集 1800 张，当前仍为 `.npz` 验证阶段；最终希望支持 `.raw`。
- **格式结论**：`.npz` 可用是因为预处理阶段已用 LibRaw 解包；裸 `.raw` 无标准元数据，需自定义 loader + 外部 ISP 先验。
- **论文写作**：先论证、后开写；关注逻辑链而非罗列工作。
- **实验请求方式**：只给要跑的东西（配置/命令清单），成本由用户自己权衡。

---

## 7. 文件指针

### 文档

- 详细论证评估：`论文论证框架评估.md`
- 论文目录（定稿）：`论文目录_定稿.md`
- 第 2–3 章实验清单：`实验清单_第2-3章.md`
- 第 3–4 章实验清单（路线级）：`实验清单_第3-4章.md`
- 方法论骨架（两段跃进）：`方法论_两段跃进.md`
- 写作素材（实现细节与巧思）：`写作素材_细节与巧思.md`
- 项目说明：`CLAUDE.md`
- 快速命令/路径：`agent.md`、`AGENT.md`
- 部署说明：`DEPLOY.md`
- 原论文文本：上级目录 `page_1.txt` ~ `page_11.txt`、`Dark-isp.pdf`

### 核心代码

- `mmdetection_github/mmdet/models/backbones/dark_resnet.py`
- `mmdetection_github/mmdet/models/backbones/Dark_modules/noise_predictor.py`
- `mmdetection_github/mmdet/models/backbones/Dark_modules/utils.py`
- `mmdetection_github/mmdet/models/backbones/darkisp_cspdarknet.py`
- `mmdetection_github/mmdet/models/backbones/darkisp_cspnext.py`
- `mmdetection_github/mmdet/engine/hooks/yolox_mode_switch_hook.py`

### 配置

- 基线/消融：`mmdetection_github/configs/LOD/VOCmetric/R_Net_*.py`
- 可用的 ISP 对照配置：`R_Net_DarkISP_18.py`（v2，61.90）、`R_Net_defaultISP_v0_18.py`（v0，D5）、`R_Net_demosaic_v1_18.py`（v1）
- YOLOX：`mmdetection_github/configs/LOD/VOCmetric/YOLOX_DarkISP_CSPDarknet_s.py`
- RTMDet：`mmdetection_github/configs/LOD/VOCmetric/DarkISP_RTMDet_s_local.py`、`DarkISP_RTMDet_x_H100.py`

### 打包

- `mmdetection_github/server_incremental/`
- `mmdetection_github/server_package/`
- `mmdetection_github/updates_20250709/`

---

## 8. 待确认事项

1. ~~“两个小算子”的最终指代~~ → **已定：GGNC + L_grad，且只走非线性单分支**。
2. 第 5 章主检测器：倾向 **RTMDet（CSPNeXt）**，YOLOX（11.12M，57.15）作为“绝对 mAP 比较失效”的反面例子；待与导师确认。
3. 第 5 章是否必须打开 GGNC / L_grad；是否先补 `use_grad_reg` 接口？
4. 是否接受“工程适配”作为第 5 章备选结论（若交互效应不显著）。
5. 第 5 章命题：**结构增益的转化率／可扩展性**，还是**结构算子的跨范式可移植性**？
6. v2 的“优劣 vs 判据”口径（等第 4 章 4.4 数据）。
7. L_grad 是否改为双阈值（强边缘保方向 + 平坦区压幅值）。
8. RTMDet-x 68.7 的服务器 `work_dir` 回传（`config.py` + `scalars.json` + log + best ckpt）。

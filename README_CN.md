# Joint Multi-View Embedding with Progressive Multi-Scale Alignment for Unaligned Infrared-Visible Image Fusion



[![Ubuntu 20.04.5](https://img.shields.io/badge/OS-Ubuntu%2020.04.5-yellow?style=flat-square)](https://ubuntu.com/)
![CUDA 12.0](https://img.shields.io/badge/CUDA-V12.0-FF6B6B?logo=nvidia&logoColor=white&style=flat-square-4c1)
![NVIDIA-SMI 550.107.02](https://img.shields.io/badge/NVIDIA--SMI-550.107.02-76B900?logo=nvidia&logoColor=white&style=flat-square-4c1)
![Python 3.9.18](https://img.shields.io/badge/Python-3.9.18-blue?logo=python&logoColor=white&style=flat-square-4c1)
![PyTorch 1.12.1](https://img.shields.io/badge/PyTorch-1.12.1-EE4C2C?logo=pytorch&logoColor=white&style=flat-square-4c1)
![Torchvision 0.13.1](https://img.shields.io/badge/Torchvision-0.13.1-ff69b4?logo=pytorch&logoColor=white&style=flat-square-4c1)
![多模态图像融合](https://img.shields.io/badge/Image-Fusion-4c1)
![图像配准](https://img.shields.io/badge/Image-Registration-ffcc99?style=flat-square-4c1)

<div style="background-color: #fff8c5; color: #000000; padding: 10px; border-left: 4px solid #f0ad4e; border-radius: 4px;">
  <strong>我们的论文正在投稿接受同行评审中，感谢您的关注与支持，论文将在接收后进一步完善README文档。</strong>
</div>

<br>

[README-Chinese](./README_CN.md) | [README-English](./README.md) 


## 1. 简介
本方法提出了一个基于特征级配准的端对端红外与可见光图像融合网络(ME-PMA)。该网络可以同时处理图像配准和融合任务，主要包含以下创新点:
- 未对齐场景下端对端的实现红外与可见光图像配准与融合任务
- 提出一种多视角嵌入与渐进式多尺度特征对齐策略，可从多个视角生成泛化的变形场，实现特征级的配准。
- 所提方法在多个数据集上表现出优异的融合性能，同时具有较低的模型复杂度与计算成本，且仅使用一组模型权重。

### 网络结构，如下图所示
![网络结构图](./Figures/PaperAcademic/figure1.png)

网络主要由以下部分组成:
- Feature Encoder: 特征编码器，包含SFE、UIB_Block和Restormer
- MSPA: 多尺度渐进式对齐模块，用于特征级配准
- Feature Decoder: 特征解码器，FFCM特征融合，FRRB用于重建输出最终融合结果

### 特征级配准模块MSPA，如下图所示

![模块结构图](./Figures/PaperAcademic/figure2.png)
- Restormer_Corr: 局部相关性全局特征提取模块
- UIB_CA: 通道注意力局部特征提取模块  
- Reg_flow: 多视角配准流预测模块

## 2. 环境配置

### 基础环境
```bash
git clone https://github.com/yidamyth/ME-PMA.git
cd ME-PMA

# 创建conda环境
conda create -n ME-PMA python=3.9.18
conda activate ME-PMA

# 安装PyTorch
pip install torch==1.12.1+cu113
pip install torchvision==0.13.1+cu113

# 安装依赖
pip install -r requirements.txt
```

## 3. 数据准备
```
.
└── ./DataSet/IVIF/
    ├── M3FD
        ├── test
            ├── ir
            ├── ir_move
            └── vis
    ├── MSRS
        ├── test
            ├── ir
            ├── ir_move
            └── vis
    └── RoadScene
        ├── RoadS_test
        │   ├── ir
        │   ├── ir_move
        │   └── vis
        └── RoadS_train
            ├── ir
            └── vis
```

## 4. 测试

端对端特征级配准与融合结果（输入未对齐场景下的图像对）
```bash
python test_phase2.py
# Save to: ./DataSet/IVIF/RoadScene/RoadS_test/Results/UnAligned/
```

直接融合结果，不采用配准模块（输入对齐场景下的图像对）
```bash
python test.py
# Save to: ./DataSet/IVIF/RoadScene/RoadS_test/Results/Aligned/
```

> 能够切换数据集来获得不同数据集的结果，default=test_path['RoadScene'] or ['M3FD'] or ['MSRS']；不同数据测试均是用的相同模型权重参数。

## 5. 训练

### 第一阶段训练 (融合网络)
```bash

# 1.查找python位置
which python
# ouput: /home/yida/anaconda3/envs/ME-PMA/bin/python

# 2.编辑conda路径
vim run.sh

# 3.切换到你anaconda3下面的conda路径
eval "$(/home/your_user_name_xxx/anaconda3/bin/conda shell.bash hook)"

# 4.保存vim

# 5.运行
sh ./run.sh

# 6.查看日志
tail -f ./Logs/nohup/2024-1119-1001_time.log

# 7.将程序在后台自动运行，可退出终端
# 模型保存路径：./Model/Parameters/24-1119-1001/

# 8.退出程序
control + z
```

### 第二阶段训练 (配准网络)
```bash
# 1.编辑conda路径
vim run_phase2.sh
eval "$(/home/your_user_name_xxx/anaconda3/bin/conda shell.bash hook)"

# 加载第一阶段模型路径
phase2_model_id='24-1119-1001'
phase2_ModelPath='./Model/Parameters/24-1119-1001/RegImageFusModel-best.pth'
# 保存vim

# 2.运行
sh ./run_phase2.sh

# 3.查看日志
tail -f ./Logs/nohup/2024-1119-1355_time.log

# 4.退出程序
control + z
```

## 实验结果

### 配准+融合对比可视化
![配准+融合](./Figures/PaperAcademic/figure3.png)

### 联合优化对比可视化
![联合优化](./Figures/PaperAcademic/figure4.png)

### 评估指标

### 融合评估指标
- $Q_{CE↓}$
- $Q_{MI↑}$
- $Q_{VIF↑}$
- $Q_{AB/F↑}$
- $Q_{CB↑}$
- $Q_{CV↓}$

读者能够获得我们详细的定量评估指标，使用示例如下:
```python
python ./Util/metrics_fus.py
```

### 配准评估指标
- $Q_{MI↑}$
- $Q_{MS-SSIM↑}$
- $Q_{NCC↑}$

读者能够获得我们详细的定量评估指标，使用示例如下:
```python
python ./Util/metrics_reg.py
```

> 为了方便读者使用，可直接进行指标测试，得到论文中的结果。但，具体细节我们将在论文接受后进一步补充；

> 值得注意的是配准评估指标是三个数据集总和的均值。

### 实验结果可视化


### 配准+融合优化结果
![配准+融合](./Figures/PaperAcademic/figure5.png)

### 📊 Table 1:  基于先配准再融合方法的详细定量比较结果
*(红色高亮为最优值，橙色高亮为次优值)*  


### 🚗 RoadScene Dataset
| Method      | $Q_{MI}$ ↑ | $Q_{CE}$ ↓ | $Q_{VIF}$ ↑ | $Q_{AB/F}$ ↑ | $Q_{CB}$ ↑ | $Q_{CV}$ ↓ |
| ----------- | ---------- | ---------- | ----------- | ------------ | ---------- | ---------- |
| Meta-Fusion | <span style="color:orange; font-weight:bold">3.1598</span> | <span style="color:orange; font-weight:bold">0.7210</span> | 0.5931 | 0.2711 | 0.3417 | 1304.8995 |
| TarDAL      | 2.5729 | 0.8588 | 0.5273 | 0.2098 | 0.3503 | 2309.3965 |
| U2Fusion    | 2.2014 | 1.0187 | 0.4798 | 0.2817 | 0.4266 | 1776.3013 |
| DATFuse     | 3.1237 | 0.8039 | 0.5918 | <span style="color:orange; font-weight:bold">0.3386</span> | 0.4283 | <span style="color:orange; font-weight:bold">864.8960</span> |
| LRRNet      | 2.6209 | 1.4728 | <span style="color:orange; font-weight:bold">0.6288</span> | 0.3230 | <span style="color:orange; font-weight:bold">0.4731</span> | 913.1348 |
| CoCoNet     | 2.3097 | 0.9942 | 0.6080 | 0.2634 | 0.4574 | 1635.9103 |
| DDBFusion   | 2.2559 | 1.1727 | 0.5400 | 0.2698 | 0.3908 | 1252.3511 |
| VDMUFusion  | 2.2052 | 0.8059 | 0.4935 | 0.2298 | 0.3802 | 1551.4407 |
| **Ours**    | <span style="color:red; font-weight:bold">4.3598</span> | <span style="color:red; font-weight:bold">0.4122</span> | <span style="color:red; font-weight:bold">0.7877</span> | <span style="color:red; font-weight:bold">0.3979</span> | <span style="color:red; font-weight:bold">0.4857</span> | <span style="color:red; font-weight:bold">636.7558</span> |


### 🏍️ M3FD Dataset
| Method      | $Q_{MI}$ ↑ | $Q_{CE}$ ↓ | $Q_{VIF}$ ↑ | $Q_{AB/F}$ ↑ | $Q_{CB}$ ↑ | $Q_{CV}$ ↓ |
| ----------- | ---------- | ---------- | ----------- | ------------ | ---------- | ---------- |
| Meta-Fusion | 3.0173 | 1.1911 | 0.5904 | 0.2818 | 0.4123 | 1326.7568 |
| TarDAL      | 2.9566 | 1.0932 | 0.5670 | 0.2068 | 0.3979 | 1532.9253 |
| U2Fusion    | 2.5369 | <span style="color:orange; font-weight:bold">1.0638</span> | 0.5965 | 0.4172 | <span style="color:red; font-weight:bold">0.4640</span> | 1115.3704 |
| DATFuse     | <span style="color:orange; font-weight:bold">3.3914</span> | 1.6030 | 0.6490 | 0.4032 | 0.4245 | 804.6190 |
| LRRNet      | 2.5597 | 1.5662 | 0.6672 | <span style="color:orange; font-weight:bold">0.4503</span> | <span style="color:orange; font-weight:bold">0.4472</span> | <span style="color:orange; font-weight:bold">782.3767</span> |
| CoCoNet     | 2.5524 | 1.1676 | <span style="color:orange; font-weight:bold">0.7961</span> | 0.3352 | 0.4115 | 1530.3093 |
| DDBFusion   | 2.4250 | 1.3975 | 0.5511 | 0.3461 | 0.4015 | 982.0145 |
| VDMUFusion  | 2.5092 | 1.4986 | 0.5172 | 0.2825 | 0.4114 | 984.2836 |
| **Ours**    | <span style="color:red; font-weight:bold">4.1585</span> | <span style="color:red; font-weight:bold">0.8221</span> | <span style="color:red; font-weight:bold">0.8083</span> | <span style="color:red; font-weight:bold">0.4641</span> | 0.4426 | <span style="color:red; font-weight:bold">766.2438</span> |


### 🌆 MSRS Dataset
| Method      | $Q_{MI}$ ↑ | $Q_{CE}$ ↓ | $Q_{VIF}$ ↑ | $Q_{AB/F}$ ↑ | $Q_{CB}$ ↑ | $Q_{CV}$ ↓ |
| ----------- | ---------- | ---------- | ----------- | ------------ | ---------- | ---------- |
| Meta-Fusion | 2.0020 | 2.1508 | 0.3831 | 0.2214 | 0.3855 | 818.3882 |
| TarDAL      | 2.0618 | 1.5758 | 0.4386 | 0.1357 | 0.4168 | 2474.7432 |
| U2Fusion    | 1.9934 | 1.1261 | 0.5081 | 0.3929 | 0.4770 | 969.1622 |
| DATFuse     | <span style="color:orange; font-weight:bold">3.5697</span> | 1.8426 | <span style="color:orange; font-weight:bold">0.7890</span> | <span style="color:orange; font-weight:bold">0.5438</span> | <span style="color:orange; font-weight:bold">0.5004</span> | <span style="color:orange; font-weight:bold">555.2225</span> |
| LRRNet      | 2.9645 | 2.5844 | 0.6438 | 0.4699 | 0.4360 | 694.7701 |
| CoCoNet     | 2.3879 | 2.5679 | 0.6974 | 0.3364 | 0.4616 | 1097.5978 |
| DDBFusion   | 2.1452 | <span style="color:orange; font-weight:bold">1.0623</span> | 0.5559 | 0.3118 | 0.4754 | 901.0095 |
| VDMUFusion  | 2.4895 | 1.1072 | 0.6095 | 0.2891 | 0.4799 | 854.6607 |
| **Ours**    | <span style="color:red; font-weight:bold">4.2775</span> | <span style="color:red; font-weight:bold">0.5673</span> | <span style="color:red; font-weight:bold">0.8594</span> | <span style="color:red; font-weight:bold">0.5742</span> | <span style="color:red; font-weight:bold">0.5226</span> | <span style="color:red; font-weight:bold">466.9209</span> |


📌 *注：每一列中，红色加粗数值表示最佳结果，橙色加粗数值表示次优结果。注意：在GitHub上可能无法显示颜色，但在本地Markdown编辑器中可正常高亮。*


---


### 联合优化结果
![联合优化](./Figures/PaperAcademic/figure6.png)

### 📊 Table 2: 基于配准与融合方法联合优化的详细定量比较结果  
*(红色高亮为最优值，橙色高亮为次优值)*  


### 🚗 RoadScene Dataset
| Method        | $Q_{MI}$ ↑ | $Q_{CE}$ ↓ | $Q_{VIF}$ ↑ | $Q_{AB/F}$ ↑ | $Q_{CB}$ ↑ | $Q_{CV}$ ↓ |
| ------------- | ---------- | ---------- | ----------- | ------------ | ---------- | ---------- |
| UMF-CMGR      | 2.0545 | 0.6979 | 0.4103 | 0.2118 | 0.3604 | 2090.1031 |
| SuperFusion   | <span style="color:orange; font-weight:bold">2.8808</span> | 1.0122 | <span style="color:orange; font-weight:bold">0.6070</span> | <span style="color:orange; font-weight:bold">0.3385</span> | 0.4000 | <span style="color:orange; font-weight:bold">736.4983</span> |
| MURF          | 1.5161 | 1.3275 | 0.2981 | 0.2716 | 0.3873 | 1537.4943 |
| SemLA         | 2.0849 | 1.1658 | 0.4083 | 0.2145 | 0.2783 | 2245.0144 |
| RFIVF         | 1.7874 | 0.8726 | 0.2359 | 0.1355 | 0.3749 | 2321.7069 |
| IMF           | 2.3733 | 0.7157 | 0.5042 | 0.2620 | 0.3986 | 1297.7976 |
| IVFWSR        | 1.9872 | <span style="color:orange; font-weight:bold">0.5890</span> | 0.4095 | 0.2306 | 0.3894 | 1647.9761 |
| **Ours**      | <span style="color:red; font-weight:bold">4.3598</span> | <span style="color:red; font-weight:bold">0.4122</span> | <span style="color:red; font-weight:bold">0.7877</span> | <span style="color:red; font-weight:bold">0.3979</span> | <span style="color:red; font-weight:bold">0.4857</span> | <span style="color:red; font-weight:bold">636.7558</span> |


### 🏍️ M3FD Dataset
| Method        | $Q_{MI}$ ↑ | $Q_{CE}$ ↓ | $Q_{VIF}$ ↑ | $Q_{AB/F}$ ↑ | $Q_{CB}$ ↑ | $Q_{CV}$ ↓ |
| ------------- | ---------- | ---------- | ----------- | ------------ | ---------- | ---------- |
| UMF-CMGR      | 2.4425 | 1.2821 | 0.4351 | 0.1894 | 0.3909 | 1640.6968 |
| SuperFusion   | <span style="color:orange; font-weight:bold">2.9868</span> | 1.3335 | 0.5982 | <span style="color:orange; font-weight:bold">0.4098</span> | 0.4199 | <span style="color:orange; font-weight:bold">885.7659</span>  |
| MURF          | 2.3557 | 1.3906 | 0.5393 | 0.3821 | <span style="color:orange; font-weight:bold">0.4251</span> | 1295.2825 |
| SemLA         | 2.5120 | 1.4842 | 0.5268 | 0.2949 | 0.3613 | 1521.0582 |
| RFIVF         | 1.9926 | <span style="color:orange; font-weight:bold">1.1140</span> | 0.4355 | 0.1939 | 0.4191 | 1608.0113 |
| IMF           | 2.5745 | 1.3291 | 0.5360 | 0.2602 | 0.4301 | 900.7963 |
| IVFWSR        | 2.2129 | 1.1828 | 0.4608 | 0.2375 | 0.4354 | 1424.2831|
| **Ours**      | <span style="color:red; font-weight:bold">4.1585</span> | <span style="color:red; font-weight:bold">0.8221</span> | <span style="color:red; font-weight:bold">0.8083</span> | <span style="color:red; font-weight:bold">0.4641</span> | <span style="color:red; font-weight:bold">0.4426</span> | <span style="color:red; font-weight:bold">766.2438</span> |


### 🌆 MSRS Dataset
| Method        | $Q_{MI}$ ↑ | $Q_{CE}$ ↓ | $Q_{VIF}$ ↑ | $Q_{AB/F}$ ↑ | $Q_{CB}$ ↑ | $Q_{CV}$ ↓ |
| ------------- | ---------- | ---------- | ----------- | ------------ | ---------- | ---------- |
| UMF-CMGR      | 1.8601 | 2.3082 | 0.3749 | 0.1881 | 0.3797 | 1351.6438 |
| SuperFusion   | <span style="color:orange; font-weight:bold">2.5990</span> | 1.8040 | 0.5599 | 0.3807 | 0.4285 | <span style="color:orange; font-weight:bold">666.8654</span> |
| MURF          | 2.5051 | 1.7401 | <span style="color:orange; font-weight:bold">0.6658| <span style="color:orange; font-weight:bold">0.4578</span> | <span style="color:orange; font-weight:bold">0.4871</span> | 986.7504 |
| SemLA         | 2.4467 | 1.6006| 0.5975 | 0.3265 | 0.4326 | 1230.6355 |
| RFIVF         | 1.3331 | 0.8183| 0.3786 | 0.1586 | 0.3265 | 2168.4381 |
| IMF           | 2.1115 | 1.6757 | 0.5201 | 0.2755 | 0.4488 | 783.3043 |
| IVFWSR        | 2.2742 | <span style="color:red; font-weight:bold">0.4942</span> | 0.4793 | 0.2242 | 0.4520 | 862.8701 |
| **Ours**      | <span style="color:red; font-weight:bold">4.2775</span> | <span style="color:orange; font-weight:bold">0.5673</span> | <span style="color:red; font-weight:bold">0.8594</span> | <span style="color:red; font-weight:bold">0.5742</span> | <span style="color:red; font-weight:bold">0.5226</span> | <span style="color:red; font-weight:bold">466.9209</span> |


📌 *注：每一列中，红色加粗数值表示最佳结果，橙色加粗数值表示次优结果。注意：在GitHub上可能无法显示颜色，但在本地Markdown编辑器中可正常高亮。*

---


### 仅对比配准性能的比较结果
![配准性能](./Figures/PaperAcademic/figure7.png)

### 参数分析：配准+融合
![参数分析reg+fus](./Figures/PaperAcademic/figure8.png)

### 参数分析：联合优化
![参数分析联合优化](./Figures/PaperAcademic/figure9.png)

### 模型：参数量+计算量
```python
cd ./ME-PMA
python -m Model.Architecture.RegImageFusModel
```


## 引用
如果您使用了本项目的代码,请引用我们的论文:
```
@article{xxx_2025_ME-PMA,
  title={Joint Multi-View Embedding with Progressive Multi-Scale Alignment for Unaligned Infrared-Visible Image Fusion},
  author={xxx},
  journal={xxx},
  volume={xx},
  number={x},
  pages={x--x},
  year={2025}
}
```

## 许可证
本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。


## 联系
感谢您的评审以及关注，如遇任何问题请联系我们邮箱：yida_myth@163.com（评审完成后会进一步完善项目，为各位读者提供帮助）


## Star History

<div style="text-align: center;">
<picture>
  <source
    media="(prefers-color-scheme: dark)"
    srcset="
      https://api.star-history.com/svg?repos=yidamyth/ME-PMA&type=Date&theme=dark
    "
  />
  <source
    media="(prefers-color-scheme: light)"
    srcset="
      https://api.star-history.com/svg?repos=yidamyth/ME-PMA&type=Date
    "
  />
  <img
    alt="Star History Chart"
    src="https://api.star-history.com/svg?repos=yidamyth/ME-PMA&type=Date"
  />
</picture>
</div>
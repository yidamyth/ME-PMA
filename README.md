<div align="center">

# **Joint Multi-View Embedding with Progressive Multi-Scale Alignment for Unaligned Infrared-Visible Image Fusion**

<br>

<!-- ==== Paper & Code Badges ==== -->
[![Status](https://img.shields.io/badge/Status-Accepted-brightgreen?style=for-the-badge)](https://doi.org/10.1016/j.inffus.2025.103960)
[![Journal](https://img.shields.io/badge/Information%20Fusion-128(2026)%20103960-orange?style=for-the-badge&logo=Elsevier&logoColor=white)](https://doi.org/10.1016/j.inffus.2025.103960)

[![DOI](https://img.shields.io/badge/DOI|PDF-10.1016%2Fj.inffus.2025.103960-blue?style=for-the-badge)](https://doi.org/10.1016/j.inffus.2025.103960)
[![Code](https://img.shields.io/badge/Code-ME--PMA-black?style=for-the-badge&logo=github)](https://github.com/yidamyth/ME-PMA)

<br>

## 👨‍💻 **Authors**

| Name | ORCID |
|------|--------|
| **Yida Chen (a)** | [![ORCID](https://img.shields.io/badge/0009--0009--8320--4669-A6CE39?style=flat&logo=ORCID&logoColor=white)](https://orcid.org/0009-0009-8320-4669) |
| **Yafei Zhang (a)** | [![ORCID](https://img.shields.io/badge/0000--0003--2347--5642-A6CE39?style=flat&logo=ORCID&logoColor=white)](https://orcid.org/0000-0003-2347-5642) |
| **Huafeng Li (a, ✉ corresponding author)** | [![ORCID](https://img.shields.io/badge/0000--0003--2462--6174-A6CE39?style=flat&logo=ORCID&logoColor=white)](https://orcid.org/0000-0003-2462-6174) |
| **Zhengtao Yu (a)** | [![ORCID](https://img.shields.io/badge/0000--0003--1094--5668-A6CE39?style=flat&logo=ORCID&logoColor=white)](https://orcid.org/0000-0003-1094-5668) |
| **Yu Liu (b)** | [![ORCID](https://img.shields.io/badge/0000--0003--2211--3535-A6CE39?style=flat&logo=ORCID&logoColor=white)](https://orcid.org/0000-0003-2211-3535) |

<br>

🏫 **Affiliations**

**a** Faculty of Information Engineering and Automation, Kunming University of Science and Technology, Kunming, Yunnan, 650500, China  
**b** Department of Biomedical Engineering, Hefei University of Technology, Hefei, Anhui, 230009, China  

<br>


</div>


<div align="center">

[![Ubuntu 20.04.5](https://img.shields.io/badge/OS-Ubuntu%2020.04.5-yellow?style=flat-square)](https://ubuntu.com/)
![CUDA 12.0](https://img.shields.io/badge/CUDA-V12.0-FF6B6B?logo=nvidia&logoColor=white&style=flat-square-4c1)
![NVIDIA-SMI 550.107.02](https://img.shields.io/badge/NVIDIA--SMI-550.107.02-76B900?logo=nvidia&logoColor=white&style=flat-square-4c1)
![Python 3.9.18](https://img.shields.io/badge/Python-3.9.18-blue?logo=python&logoColor=white&style=flat-square-4c1)
![PyTorch 1.12.1](https://img.shields.io/badge/PyTorch-1.12.1-EE4C2C?logo=pytorch&logoColor=white&style=flat-square-4c1)
![Torchvision 0.13.1](https://img.shields.io/badge/Torchvision-0.13.1-ff69b4?logo=pytorch&logoColor=white&style=flat-square-4c1)
![Multimodal Image Fusion](https://img.shields.io/badge/Image-Fusion-4c1)
![Image Registration](https://img.shields.io/badge/Image-Registration-ffcc99?style=flat-square-4c1)

</div>


<div style="background-color: #fff8c5; color: #000000; padding: 10px; border-left: 4px solid #f0ad4e; border-radius: 4px;">
  <strong> Our paper has been accepted for publication in Information Fusion (2026). We would like to express our sincere gratitude to the reviewers, editors, and collaborators for their constructive feedback and continuous support.
</strong>
</div>

<br>

<div align="center">
<h3>
  <a href="./README_CN.md">README-Chinese</a> |
  <a href="./README.md">README-English</a>
</h3>
</div>



## 1. Introduction
We propose an end-to-end infrared-visible image fusion network (ME-PMA) with feature-level registration, featuring:

- End-to-end registration and fusion for unaligned scenarios
- Progressive multi-scale feature alignment with multi-view embedding
- Superior performance across datasets with single model weights

### Network Architecture
![Network Structure](./Figures/PaperAcademic/figure1.png)

Key Components:
- **Feature Encoder**: SFE, UIB_Block, and Restormer
- **MSPA**: Multi-Scale Progressive Alignment module
- **Feature Decoder**: FFCM fusion and FRRB reconstruction



### MSPA Module
![Module Structure](./Figures/PaperAcademic/figure2.png)
- Restormer_Corr: Global feature extraction with local correlation
- UIB_CA: Channel attention for local features
- Reg_flow: Multi-view registration flow prediction

## 2. Environment Setup

### Base Configuration
```bash
git clone https://github.com/yidamyth/ME-PMA.git
cd ME-PMA

# Create conda environment
conda create -n ME-PMA python=3.9.18
conda activate ME-PMA

# Install PyTorch
pip install torch==1.12.1+cu113
pip install torchvision==0.13.1+cu113

# Install dependencies
pip install -r requirements.txt
```

## 3. Data Preparation
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

## 4. Testing

End-to-end feature-level registration and fusion results (input images from unaligned scenarios)
```bash
python test_phase2.py
# Save to: ./DataSet/IVIF/RoadScene/RoadS_test/Results/UnAligned/
```

Direct fusion results, without using the registration module (input images from aligned scenarios)
```bash
python test.py
# Save to: ./DataSet/IVIF/RoadScene/RoadS_test/Results/Aligned/
```

> You can switch datasets to get results from different datasets. The default is test_path['RoadScene'] or ['M3FD'] or ['MSRS']; the same model weights are used for all different dataset tests.

## 5. Training

### First Stage Training (Fusion Network)
```bash

# 1. Find python location
which python
# ouput: /home/yida/anaconda3/envs/ME-PMA/bin/python

# 2. Edit conda path
vim run.sh

# 3. Switch to your anaconda3 conda path
eval "$(/home/your_user_name_xxx/anaconda3/bin/conda shell.bash hook)"

# 4. Save vim

# 5. Run
sh ./run.sh

# 6. Check logs
tail -f ./Logs/nohup/2024-1119-1001_time.log

# 7. Run program in background, can exit terminal
# Model save path: ./Model/Parameters/24-1119-1001/

# 8. Exit program
control + z
```

### Second Stage Training (Registration Network)
```bash
# 1. Edit conda path
vim run_phase2.sh
eval "$(/home/your_user_name_xxx/anaconda3/bin/conda shell.bash hook)"

# Load first stage model path
phase2_model_id='24-1119-1001'
phase2_ModelPath='./Model/Parameters/24-1119-1001/RegImageFusModel-best.pth'
# Save vim

# 2. Run
sh ./run_phase2.sh

# 3. Check logs
tail -f ./Logs/nohup/2024-1119-1355_time.log

# 4. Exit program
control + z
```

## 6. Experiment Results

### Registration + Fusion Comparison Visualization
![Registration + Fusion](./Figures/PaperAcademic/figure3.png)



### Joint Optimization Comparison Visualization
![Joint Optimization](./Figures/PaperAcademic/figure4.png)



### Evaluation Metrics

### Fusion Evaluation Metrics
- $Q_{CE↓}$
- $Q_{MI↑}$
- $Q_{VIF↑}$
- $Q_{AB/F↑}$
- $Q_{CB↑}$
- $Q_{CV↓}$

You can get our detailed quantitative evaluation metrics, using the following example:
```python
python ./Util/metrics_fus.py
```



### Registration Evaluation Metrics
- $Q_{MI↑}$
- $Q_{MS-SSIM↑}$
- $Q_{NCC↑}$

You can get our detailed quantitative evaluation metrics, using the following example:
```python
python ./Util/metrics_reg.py
```

> For convenience, the provided metric scripts allow you to directly reproduce the results reported in the paper.

> We provide the complete fusion results for direct and unbiased metric evaluation. For convenience, resized input images are also included to help reproduce the results. While the overall outputs remain consistent, minor pixel-level differences may occur due to the resizing operation.

> Note that the registration evaluation metrics are the average of the three datasets.


### Experiment Results Visualization


### Registration + Fusion Optimization Results
![Registration + Fusion](./Figures/PaperAcademic/figure5.png)


### 📊 Table 1: Quantitative Comparison of Registration + Fusion Methods — Detailed Results
*(Red bold = best, Orange bold = second-best)*  


<p align="center">
  <img src="./Figures/PaperAcademic/figure5-1.png" alt="Registration + Fusion Quantitative Comparison">
</p>


📌 *Note: In each column, values in **red bold** are the best, and values in **orange bold** are the second-best.*


### Joint Optimization Results
![Joint Optimization](./Figures/PaperAcademic/figure6.png)

### 📊 Table 2: Quantitative Comparison of Joint Optimization Methods — Detailed Results
*(Red bold = best, Orange bold = second-best)*  

<p align="center">
  <img src="./Figures/PaperAcademic/figure6-1.png" alt="Registration + Fusion Quantitative Comparison">
</p>


📌 *Note: In each column, values in **red bold** are the best, and values in **orange bold** are the second-best.*



### Only Registration Performance Comparison Results
![Registration Performance](./Figures/PaperAcademic/figure7.png)



### Parameter Analysis: Joint Optimization
![Parameter Analysis Joint Optimization](./Figures/PaperAcademic/figure9.png)



### Object Detection and Semantic Segmentation

**Object Detection**: https://github.com/yidamyth/IF_Yolov8

**Semantic Segmentation**: https://github.com/yidamyth/IF_DeepLabv3



### Model: Parameter Quantity + Calculation
```python
cd ./ME-PMA
python -m Model.Architecture.RegImageFusModel
```

## 🥰 Acknowledgments

The overall architecture of this project was independently designed by the author@Yida Chen. However, parts of the implementation reference the following excellent open-source works:

### Code References
- **CDDFuse: Correlation-Driven Dual-Branch Feature Decomposition for Multi-Modality Image Fusion** 
  - (CVPR 2023) https://github.com/haozixiang1228/MMIF-CDDFuse  

- **Correlation-aware Coarse-to-Fine MLPs for Deformable Medical Image Registration**
  - (CVPR 2024) https://github.com/MungoMeng/Registration-CorrMLP  

- **MobileNetv4 Implementations**  
  - MobileNetv4-1: https://github.com/jiaowoguanren0615/MobileNetV4  
  - MobileNetv4-2: https://github.com/jaiwei98/MobileNetV4-pytorch  

### Metric Implementation
- **Analysis of Quality Objective Assessment Metrics for Visible and Infrared Image Fusion**
  - (Journal of Image and Graphics 2023) https://github.com/sunbinuestc/VIF-metrics-analysis  

### Experimental Visualization Tools
- **MulimgViewer** (for local detail visualization)
  - https://github.com/nachifur/MulimgViewer  

We sincerely appreciate the open-source community for providing valuable tools, resources, and inspiration that greatly supported the development of this project.


## 😘 Citation
If this work benefits your research, a citation to our paper would be greatly appreciated:
```
@article{2026_ME-PMA,
  title   = {Joint multi-view embedding with progressive multi-scale alignment for unaligned infrared-visible image fusion},
  author  = {Chen, Yida and Zhang, Yafei and Li, Huafeng and Yu, Zhengtao and Liu, Yu},
  journal = {Information Fusion},
  volume  = {128},
  pages   = {103960},
  year    = {2026},
  doi     = {10.1016/j.inffus.2025.103960}
}
```


## 🔖 License
This project uses the MIT License. See [LICENSE](LICENSE) file.


## 😃 Contact
Thank you for your attention. If you have any questions, please contact us by email at yida_myth@163.com. We will get back to you as soon as possible, and you may also raise your questions through the Issues page.


## 🙌 Star History
[![Star History Chart](https://api.star-history.com/svg?repos=yidamyth/ME-PMA&type=Date)](https://star-history.com/#yidamyth/ME-PMA&Date)

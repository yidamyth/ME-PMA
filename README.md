# Joint Multi-View Embedding with Progressive Multi-Scale Alignment for Unaligned Infrared-Visible Image Fusion



[![Ubuntu 20.04.5](https://img.shields.io/badge/OS-Ubuntu%2020.04.5-yellow?style=flat-square)](https://ubuntu.com/)
![CUDA 12.0](https://img.shields.io/badge/CUDA-V12.0-FF6B6B?logo=nvidia&logoColor=white&style=flat-square-4c1)
![NVIDIA-SMI 550.107.02](https://img.shields.io/badge/NVIDIA--SMI-550.107.02-76B900?logo=nvidia&logoColor=white&style=flat-square-4c1)
![Python 3.9.18](https://img.shields.io/badge/Python-3.9.18-blue?logo=python&logoColor=white&style=flat-square-4c1)
![PyTorch 1.12.1](https://img.shields.io/badge/PyTorch-1.12.1-EE4C2C?logo=pytorch&logoColor=white&style=flat-square-4c1)
![Torchvision 0.13.1](https://img.shields.io/badge/Torchvision-0.13.1-ff69b4?logo=pytorch&logoColor=white&style=flat-square-4c1)
![Multimodal Image Fusion](https://img.shields.io/badge/Image-Fusion-4c1)
![Image Registration](https://img.shields.io/badge/Image-Registration-ffcc99?style=flat-square-4c1)

<div style="background-color: #fff8c5; color: #000000; padding: 10px; border-left: 4px solid #f0ad4e; border-radius: 4px;">
  <strong>	Our paper is currently under peer review. We sincerely appreciate your interest and support. The README will be further improved after acceptance.</strong>
</div>

<br>

[README-Chinese](./README_CN.md) | [README-English](./README.md) 


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

## Experiment Results

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

> For convenience, you can directly perform metric tests to get results from the paper. However, specific details will be further supplemented after acceptance of the paper;

> Note that the registration evaluation metrics are the average of the three datasets.

### Experiment Results Visualization


### Registration + Fusion Optimization Results
![Registration + Fusion](./Figures/PaperAcademic/figure5.png)

### Joint Optimization Results
![Joint Optimization](./Figures/PaperAcademic/figure6.png)




### Only Registration Performance Comparison Results
![Registration Performance](./Figures/PaperAcademic/figure7.png)

### Parameter Analysis: Registration + Fusion
![Parameter Analysis reg+fus](./Figures/PaperAcademic/figure8.png)

### Parameter Analysis: Joint Optimization
![Parameter Analysis Joint Optimization](./Figures/PaperAcademic/figure9.png)

### Model: Parameter Quantity + Calculation
```python
cd ./ME-PMA
python -m Model.Architecture.RegImageFusModel
```


## Citation
If you use this project's code, please cite our paper:
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

## License
This project uses the MIT License. See [LICENSE](LICENSE) file.


## Contact
Thank you for your review and attention. If you have any questions, please contact us by email: yida_myth@163.com (We will further improve the project after acceptance to provide help for you)


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
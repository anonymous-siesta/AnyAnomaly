# AnyAnomaly: Zero-Shot Customizable Video Anomaly Detection with LVLM 
This paper has been submitted to ICCV '25.

## Description
Video Anomaly Detection (VAD) is a critical task in video analysis and surveillance within computer vision. However, existing VAD models rely on learned normal patterns, making them difficult to apply across diverse environments. As a result, users must retrain models or develop separate AI models for new environments, which requires expertise in machine learning, high-performance hardware, and extensive data collection, limiting the practical usability of VAD. **To address these challenges, this study proposes Customizable Video Anomaly Detection (C-VAD) and the AnyAnomaly model. C-VAD considers user-defined text as an abnormal event and detects frames containing the specified event in a video.** We implement C-VAD effectively using a Context-aware VQA approach without fine-tuning a Large Vision Language Model (LVLM). To validate the effectiveness of the proposed model, we constructed a C-VAD dataset and demonstrated the superiority of AnyAnomaly. Furthermore, despite adopting a zero-shot approach, our method achieves competitive performance on VAD benchmarks. The code for this study will be released after publication.<br/><br/>
<img width="850" alt="fig-1" src="https://github.com/user-attachments/assets/938d46a3-56cc-4cd4-8900-84bdfbd64b98">  

## Context-aware VQA
Comparison of the proposed model with the baseline. Both models perform C-VAD, but the baseline operates with frame-level VQA, whereas the proposed model employs a segment-level Context-Aware VQA.
**Context-Aware VQA is a method that performs VQA by utilizing additional contexts that describe an image.** To enhance the object analysis and action understanding capabilities of LVLM, we propose Position Context and Temporal Context.<br/>  
<img width="850" alt="fig-2" src="https://github.com/user-attachments/assets/272487fb-e626-47f7-b5d4-00b5119c84d3">  


## Qualitative Evaluation 
- Anomaly Detection in diverse scenarios
  
|         Text              |Demo  |
|:--------------:|:-----------:|
| **Jumping-Falling<br/>-Pickup** |![c5](https://github.com/user-attachments/assets/9110c14b-0999-45ca-9bd0-9319ceed883c)|
| **Bicycle-<br/>Running** |![c6](https://github.com/user-attachments/assets/c40de3ef-f8f2-462e-9bc7-447fa31be729)|
| **Bicycle-<br/>Stroller** |![c7](https://github.com/user-attachments/assets/f3cc1bbd-ea8e-4473-a84c-4b692e74ddb4)|

- Anomaly Detection in complex scenarios

|         Text              |Demo  |
|:--------------:|:-----------:|
| **Driving outside<br/> lane** |![c4](https://github.com/user-attachments/assets/4996b407-0ca1-4232-85d6-1f5efbad4ab5)|
| **People and car<br/> accident** |![c1](https://github.com/user-attachments/assets/eed1252a-59dd-412c-ab3d-00b71d7bb852)|
| **Jaywalking** |![c2](https://github.com/user-attachments/assets/f6205335-bebc-4df2-bd40-d06cc62d2a56)|
| **Walking<br/> drunk** |![c3](https://github.com/user-attachments/assets/eb450801-b700-419d-a920-2799553c2452)|

## 1. Requirements and Installation For ChatUniVi
- ```Chat-UniVi```: [[GitHub]](https://github.com/PKU-YuanGroup/Chat-UniVi)
- weights: Chat-UniVi 7B [[Huggingface]](https://huggingface.co/Chat-UniVi/Chat-UniVi/tree/main), Chat-UniVi 13B [[Huggingface]](https://huggingface.co/Chat-UniVi/Chat-UniVi-13B/tree/main)
- Install required packages:
```bash
git clone https://github.com/PKU-YuanGroup/Chat-UniVi
cd Chat-UniVi
conda create -n chatunivi python=3.10 -y
conda activate chatunivi
pip install --upgrade pip
pip install -e .
pip install numpy==1.24.3

# Download the Model (Chat-UniVi 7B)
mkdir weights
cd weights
sudo apt-get install git-lfs
git lfs install
git lfs clone https://huggingface.co/Chat-UniVi/Chat-UniVi

# Download extra packages
cd ../../
pip install -r requirements.txt
```


## Command
- ```avenue type```: [too_close, bicycle, throwing, running, dancing]
- ```shtech type```: [car, bicycle, fighting, throwing, hand_truck, running, skateboarding, falling, jumping, loitering, motorcycle]
```Shell
# Baseline model (Chat-UniVi) → C-ShT
python -u vad_chatunivi.py --dataset=shtech --type=falling
# proposed model (AnyAomaly) → C-ShT
python -u vad_proposed_chatunivi.py --dataset=shtech --type=falling 
# proposed model (AnyAomaly) → ShT
python -u ovad_proposed_chatunivi.py --dataset=shtech
```

## 2. Requirements and Installation For MiniCPM
- ```MiniCPM```: [[GitHub]](https://github.com/OpenBMB/MiniCPM-V.git)
- Install required packages:
```bash
git clone https://github.com/OpenBMB/MiniCPM-V.git
cd MiniCPM-V
conda create -n MiniCPM-V python=3.10 -y
conda activate MiniCPM-V
pip install -r requirements.txt

# Download extra packages
cd ../
pip install -r requirements.txt
```

## Command
- ```avenue type```: [too_close, bicycle, throwing, running, dancing]
- ```shtech type```: [car, bicycle, fighting, throwing, hand_truck, running, skateboarding, falling, jumping, loitering, motorcycle]
# Baseline model (MiniCPM) → C-ShT
python -u vad_MiniCPM.py --dataset=shtech --type=falling --model_path=MiniCPM-Llama3-V-2_5
# proposed model (AnyAomaly) → C-ShT
python -u vad_proposed_MiniCPM.py --dataset=shtech --type=falling --model_path=MiniCPM-Llama3-V-2_5
# proposed model (AnyAomaly) → ShT
python -u ovad_proposed_MiniCPM.py --dataset=shtech
```


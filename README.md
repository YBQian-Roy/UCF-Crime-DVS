<div align="center">
<img src="./figures/caption1.png" width="600">

**The First Event-based Video Anomaly Detection Benchmark Dataset**


------

<p align="center">
</p>

</div>

> **UCF-Crime-DVS: A Novel Event-Based Dataset for Video Anomaly Detection with Spiking Neural Networks.**
> Yuanbin Qian, Shuhan Ye, Chong Wang, Xiaojie Cai, Jiangbo Qian, Jiafei Wu (2025)

### :dart: Abstract
Video anomaly detection plays a significant role in intelligent surveillance systems. To enhance model's anomaly recognition ability, previous works have typically involved RGB, optical flow, and text features. Recently, dynamic vision sensors (DVS) have emerged as a promising technology, which capture visual information as discrete events with a very high dynamic range and temporal resolution. It reduces data redundancy and enhances the capture capacity of moving objects compared to conventional camera. To introduce this rich dynamic information into the surveillance field, we created the first DVS video anomaly detection benchmark, namely UCF-Crime-DVS. To fully utilize this new data modality, a multi-scale spiking fusion network (MSF) is designed based on spiking neural networks (SNNs). This work explores the potential application of dynamic information from event data in video anomaly detection. Our experiments demonstrate the effectiveness of our framework on UCF-Crime-DVS and its superior performance compared to other models, establishing a new baseline for SNN-based weakly supervised video anomaly detection.

### :fire: What's New
- [2024.12.09] Our paper is accepted by AAAI-2025 !!!
- [2024.12.12] The raw event file and pre-trained feature of UCF-Crime-DVS are all released.
- [2025.3.5] The feature extractor and UCF-Crime-DVS dataset code are all released.
- [2025.4.23] Links to onedrive for Pre-trained features and event frames are released.
- [2025.8.7] The model and training code are all released.

### :video_camera: UCF-Crime-DVS Samples
<p align="center">
  <img src="./figures/dataset.png" alt="UCF-Crime-DVS_samples" width="800"/>
</p>

### :dvd: Dataset Download
<p align="center">
  <img src="./figures/table.png" alt="Overview of various DVS datasets" width="800"/>
</p>

* **Download from Baidu Disk**: 
```
  [Pre-trained feature] 链接: https://pan.baidu.com/s/1atbxowEhdooTS9lZsKSy5g?pwd=uaba 提取码: uaba 
  [Event frame] 链接: https://pan.baidu.com/s/1BpA6z3DIuSyba2wts5M_gg?pwd=h48m 提取码: h48m 
  [Npz Event file] 链接: https://pan.baidu.com/s/1b0xSVk0tzdq19WNGSDY_og?pwd=5d8g 提取码: 5d8g 
  [Raw Event file]  链接：https://pan.baidu.com/s/10KsxHj1wQQxuanAPfBZNzQ?pwd=8v2x 提取码: 8v2x 
```

* **Download from Onedrive**: 
```
[Pre-trained feature] Link: https://1drv.ms/u/c/d7df95ea2b5a40f1/EQ6gHgQE6RFOlPJLRdH21QABhtyfMCiOfYzBVabFpHxtvQ
[Event frame] Link: https://1drv.ms/u/c/d7df95ea2b5a40f1/ES7KKwU4hR9KmyvY9P06ft0BvfH-AQKCbAlWygapt4ALNw
[Npz Event file] Link: https://1drv.ms/f/c/d7df95ea2b5a40f1/EgV-dSfAofxOiC4Z049EoFUByi5ViNoDd-3NyTtsSMH2EA
[Raw Event file] Link: To be updated...
```
### :gem: Framework
Our proposed multi-scale spiking fusion module (MSF) captures both multi-resolution local spiking dependencies (light green block) within individual clip, and global spiking dependencies (light yellow block) between event clips. Finally, these dependencies are seamlessly integrated based on the unique characteristics of the spiking feature (light blue block).
<p align="center">
  <img src="./figures/framework.png" alt="Framework" width="800"/>
</p>

### 🔨: Feature extraction
Requirements: timm==0.6.12; cupy==11.4.0; torch==1.12.1; spikingjelly==0.0.0.0.12; pyyaml; metavision

The above features use the [Spikingformer](https://github.com/zhouchenlin2096/Spikingformer)  to extract. You can download the pre-training checkpoit [here](https://pan.baidu.com/s/1NnSrQ06M8JSIdUjENZnCFA?pwd=u3pd), password: u3pd.
```
python extractor.py
```

### 🔨: Training
Requirements: CUDA; numpy; tqdm; torchvision; timm==0.6.12; cupy==11.4.0; torch==1.12.1; spikingjelly==0.0.0.0.12;
```
python main.py
```




### 💘 Acknowledgements
We thank the [SpikingJelly](https://github.com/fangwei123456/spikingjelly), [Spikingformer](https://github.com/zhouchenlin2096/Spikingformer) and [AR-Net](https://github.com/wanboyang/Anomaly_AR_Net_ICME_2020) for a quickly implement.

### :newspaper: Citation 
If you find this project helpful for your research, please consider citing the following BibTeX entry.
```bibtex
@inproceedings{qian2025ucf,
  title={UCF-Crime-DVS: A Novel Event-Based Dataset for Video Anomaly Detection with Spiking Neural Networks},
  author={Qian, Yuanbin and Ye, Shuhan and Wang, Chong and Cai, Xiaojie and Qian, Jiangbo and Wu, Jiafei},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={6},
  pages={6577--6585},
  year={2025}
}

@misc{qian2025ucfcrimedvsnoveleventbaseddataset,
      title={UCF-Crime-DVS: A Novel Event-Based Dataset for Video Anomaly Detection with Spiking Neural Networks}, 
      author={Yuanbin Qian and Shuhan Ye and Chong Wang and Xiaojie Cai and Jiangbo Qian and Jiafei Wu},
      year={2025},
      eprint={2503.12905},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.12905}, 
}
```


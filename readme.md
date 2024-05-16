<h1 align="center">UniChest: Conquer-and-Divide Pre-training for Multi-Source Chest X-Ray Classification</h1>

<p align="center">
    <a href="https://arxiv.org/abs/2310.17622"><img src="https://img.shields.io/badge/arXiv-2310.17622-b31b1b.svg" alt="Paper"></a>
    <a href="https://openreview.net/forum?id=geLARFEK8O"><img src="https://img.shields.io/badge/OpenReview-NeurIPS'23 Spotlight-blue" alt="Paper"></a>
    <a href="https://github.com/MediaBrain-SJTU/Geometric-Harmonization"><img src="https://img.shields.io/badge/Github-GH-brightgreen?logo=github" alt="Github"></a>
    <a href="https://neurips.cc/media/PosterPDFs/NeurIPS%202023/70835.png?t=1699436032.259549"> <img src="https://img.shields.io/badge/Poster-grey?logo=airplayvideo&logoColor=white" alt="Poster"></a>
</p>


<h2 align="center">IEEE Transactions on Medical Imaging, 2024</h2>



* 💻 [Project Website](https://tianjiedai.github.io/unichest/)
* 📖 [Paper Link (Early Access)](https://ieeexplore.ieee.org/abstract/document/10478603)
* 📁 [CSV File Link](https://drive.google.com/file/d/1LMiipnq-EouN2_wguSTfwCTBKREMKikP/view?usp=sharing)

**Abstract**: Vision-Language Pre-training (VLP) that utilizes the multi-modal information to promote the training efficiency and effectiveness, has achieved great success in vision recognition of natural domains and shown promise in medical imaging diagnosis for the Chest X-Rays (CXRs). However, current works mainly pay attention to the exploration on single dataset of CXRs, which locks the potential of this powerful paradigm on larger hybrid of multi-source CXRs datasets. We identify that although blending samples from the diverse sources offers the advantages to improve the model generalization, it is still challenging to maintain the consistent superiority for the task of each source due to the existing heterogeneity among sources. To handle this dilemma, we design a Conquer-and-Divide pre-training framework, termed as UniChest, aiming to make full use of the collaboration benefit of multiple sources of CXRs while reducing the negative influence of the source heterogeneity. Specially, the ``Conquer" stage in UniChest encourages the model to sufficiently capture multi-source common patterns, and the ``Divide" stage helps squeeze personalized patterns into different small experts (query networks). We conduct thorough experiments on many benchmarks, e.g., ChestX-ray14, CheXpert, Vindr-CXR, Shenzhen, Open-I and SIIM-ACR Pneumothorax, verifying the effectiveness of UniChest over a range of baselines, and release our codes and pre-training models at https://github.com/Elfenreigen/UniChest.

**Keywords**: Self-Supervised Learning, Long-Tailed Learning, Category-Level Uniformity


## Pre-training



### Conquer Stage
```
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 train.py --main_ratio 1 --bias_ratio 0 --moe_ratio 0 --output_dir --aws_output_dir
```

### Divide Stage
```
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 train.py --main_ratio 0.5 --bias_ratio 0.5 --moe_ratio 1 --output_dir --aws_output_dir --finetune
```

### Arguments

- `--output_dir` is the directory to save logs
- `--aws_output_dir` is the directory to save checkpoints
- `--finetune` is the path of the checkpoint of the _Conquer Stage_

### Pre-trained model weights
The pre-trained model can be downloaded from [google drive](https://drive.google.com/file/d/1V91ppG1M-IZcSFDyTBa4FNnMST9_vnkV/view?usp=sharing).

## Testing
```
python test.py --main_ratio 0.5 --bias_ratio 0.5 --aws_output_dir --test_data --save_result_dir
```

### Arguments

- `--aws_output_dir` is the path of the checkpoint
- `--test_data` is dataset name
- `--save_result_dir` is the path to save ground truth and prediction results

## Citation

If you find our work inspiring or use our codebase in your research, please consider giving a star ⭐ and a citation.
```
@ARTICLE{10478603,
  author={Dai, Tianjie and Zhang, Ruipeng and Hong, Feng and Yao, Jiangchao and Zhang, Ya and Wang, Yanfeng},
  journal={IEEE Transactions on Medical Imaging}, 
  title={UniChest: Conquer-and-Divide Pre-training for Multi-Source Chest X-Ray Classification}, 
  year={2024},
  volume={},
  number={},
  pages={1-1},
  keywords={Diseases;Medical diagnostic imaging;Training;X-ray imaging;MIMICs;Self-supervised learning;Visualization;Chest X-Rays;Medical Imaging Diagnosis;Conquer and Divide;Vision-Language Pre-training},
  doi={10.1109/TMI.2024.3381123}}
```


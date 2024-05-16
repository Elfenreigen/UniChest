#  [IEEE Transactions on Medical Imaging] UniChest: Conquer-and-Divide Pre-training for Multi-Source Chest X-Ray Classification
This is the official repository of **UniChest** (IEEE-TMI Accepted). 
* 💻 [Project Website](https://tianjiedai.github.io/unichest/)
* 📖 [Paper Link (Early Access)](https://ieeexplore.ieee.org/abstract/document/10478603)
* 📁 [CSV File Link](https://drive.google.com/file/d/1LMiipnq-EouN2_wguSTfwCTBKREMKikP/view?usp=sharing)
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


<!--
### Structure of code

```shell
├── config
│   └── config.yaml
├── dataset
│   ├── dataset_entity.py
│   ├── randaugment.py
│   └── test_dataset.py
├── engine
│   ├── test.py
│   └── train.py
├── factory
│   ├── loss.py
│   ├── metric.py
│   └── utils.py
├── models
│   ├── clip_tqn.py
│   ├── tokenization_bert.py
│   └── transformer_decoder.py
├── optim
│   ├── __init__.py
│   ├── adafactor.py
│   ├── adahessian.py
│   ├── adamp.py
│   ├── adamw.py
│   ├── lookahead.py
│   ├── nadam.py
│   ├── novograd.py
│   ├── optim_factory.py
│   ├── radam.py
│   ├── rmsprop_tf.py
│   └── sgdp.py
├── schedular
│   ├── __init__.py
│   ├── cosine_lr.py
│   ├── plateau_lr.py
│   ├── schedular.py
│   ├── schedular_factory.py
│   ├── step_lr.py
│   └── tanh_lr.py
├── readme.md
├── LICENSE
├── test.py
└── train.py

```
-->

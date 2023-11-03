#  [In Submission] UniChest

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

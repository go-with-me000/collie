"""
一个使用CoLLie训练Moss的实例。
"""
import sys
sys.path.append('..')
import os
import json
import torch

from transformers import AutoTokenizer

from collie.config import CollieConfig

from collie.data import CollieDatasetForTraining
from collie.data import CollieDataLoader

from collie.optim.lomo import Lomo

from collie.controller.trainer import Trainer
from collie.controller.evaluator import EvaluatorForPerplexity, EvaluatorForGeneration

from collie.models.moss import MossForCausalLM

from collie.utils.monitor import StepTimeMonitor, TGSMonitor, MemoryMonitor, LossMonitor, EvalMonitor
from collie.metrics import DecodeMetric, PPLMetric, BleuMetric
from collie.module import GPTLMLoss

# 1. 设置路径
# 1.1 预训练模型路径
pretrained_model = "/remote-home/share/MOSS_7B_Base/"

# 2. 设置配置
# 2.1 加载配置
config = CollieConfig.from_pretrained(pretrained_model)
config.tp_size = 2
config.dp_size = 2
config.pp_size = 1
config.train_epochs = 1
config.eval_per_n_steps = 0
config.eval_per_n_epochs = 1 
config.train_micro_batch_size = 2
config.eval_batch_size = 1
config.ds_config = {
        "fp16": {
            "enabled": True
        },
        "zero_allow_untested_optimizer": True,
        "zero_force_ds_cpu_optimizer": False,
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": False
            }
        }
}

# 3. 设置tokenizer
tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

# 4. 加载数据集
train_dataset = [
    {
        'input': 'Collie is a python package for',
        'output': 'finetuning large language models.'
    } for _ in range(100)
]
train_dataset = CollieDatasetForTraining(train_dataset, tokenizer)
eval_dataset = train_dataset[:32]

# 5. 加载预训练模型
model = MossForCausalLM.from_pretrained("/remote-home/share/MOSS_7B_Base/", config=config)

# 6. 设置优化器
optimizer = Lomo(
    model,
    lr = 0.001,
    clip_grad_norm = 5.0
)

# 7. 添加监视器
monitors = [
    StepTimeMonitor(config),
    TGSMonitor(config),
    MemoryMonitor(config),
    LossMonitor(config),
    EvalMonitor(config)
]

# 8. 添加Evaluator
evaluator_ppl = EvaluatorForPerplexity(
    model = model,
    config = config,
    dataset = eval_dataset,
    monitors = [
        EvalMonitor(config)
    ],
    metrics = {
        'ppl': PPLMetric()
    }
)
evaluator_decode = EvaluatorForGeneration(
    model = model,
    config = config,
    tokenizer = tokenizer,
    dataset = eval_dataset,
    monitors = [
        EvalMonitor(config)
    ],
    metrics = {
        'decode': DecodeMetric()
    }

)

# 9. 实例化trainer
trainer = Trainer(
    model = model,
    config = config,
    loss_fn = GPTLMLoss(-100),
    optimizer = optimizer,
    train_dataset = train_dataset,
    monitors = monitors,
    evaluators = [evaluator_ppl, evaluator_decode]
)

# 10. 训练/验证
trainer.train()
# srun -p llm --quotatype=spot --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --cpus-per-task=8 python finetune_llama_for_classification.py
#  Command CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:29402 --nnodes=1 --nproc_per_node=4 finetune_moss_for_training.py
import sys
import os
sys.path.append("../..")
import torch
from datasets import load_dataset
from transformers import LlamaTokenizer, GenerationConfig
from collie import Trainer, EvaluatorForPerplexity, CollieConfig, PPLMetric, \
    DecodeMetric, CollieDatasetForTraining, CollieDatasetForGeneration, \
    LossMonitor, TGSMonitor, MemoryMonitor, EvalMonitor, GradioProvider, \
        EvaluatorForGeneration, LRMonitor, BleuMetric, LlamaForCausalLM, \
            CheckpointCallback, env
from peft import LoraConfig, TaskType
# torch.cuda.set_per_process_memory_fraction(0.3, env.local_rank)
# Prepare training config
torch.cuda.empty_cache()
config = CollieConfig.from_pretrained("/remote-home/share/MOSS_7B_Base/")
config.dp_size = 2
config.train_micro_batch_size = 8
config.gradient_accumulation_steps = 1
config.eval_batch_size = 1
config.eval_per_n_steps = 100
config.train_epochs = 100
config.use_flash = False
config.ds_config = {
    "monitor_config": {
        "enabled": True,
        "tensorboard": {
            "enabled": True,
            "output_path": "./ds_logs/",
            "job_name": "full_finetuning_moss_7b"
        }
    },
    "fp16": {
        "enabled": True
    },
    "zero_optimization": {
        "stage": 3
    }
}
config.seed = 1024
config.peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
# Prepare training dataset
# train_dataset = [
#     {
#         "input": f"""<s>下面是描述任务的指令，并与提供进一步符合上下文的输入。请编写适当完成请求的响应。
#
# ### 指令：
# {sample["instruction"]}
#
# ### 输入：
# {sample["input"]}
#
# ### 响应：
# """ if len(sample["input"].strip()) != 0 else f"""<s>下面是描述任务的指令。请编写适当完成请求的响应。
#
# ### 指令：
# {sample["instruction"]}
#
# ### 响应：
# """,
#         "output": f"{sample['output']}</s>"
#     } for sample in load_dataset("Chinese-Vicuna/guanaco_belle_merge_v1.0", split="train[:-100]")
# ]
train_dataset = [
    {
        'input': 'Collie is a python package for',
        'output': 'finetuning large language models.'
    } for _ in range(100)
]
# Prepare perplexity evaluation dataset
ratio = 0.001
eval_dataset_ppl, train_dataset = train_dataset[:int(
    len(train_dataset) * ratio)], train_dataset[int(len(train_dataset) * ratio):]
# Prepare generation based evaluation dataset
# eval_dataset_bleu = [
#     {
#         "text": f"""<s>下面是描述任务的指令，并与提供进一步符合上下文的输入。请编写适当完成请求的响应。
#
# ### 指令：
# {sample["instruction"]}
#
# ### 输入：
# {sample["input"]}
#
# ### 响应：
# """ if len(sample["input"].strip()) != 0 else f"""<s>下面是描述任务的指令。请编写适当完成请求的响应。
#
# ### 指令：
# {sample["instruction"]}
#
# ### 响应：
# """,
#         "target": " ".join(f"{sample['output']}</s>".split())
#     } for sample in load_dataset("Chinese-Vicuna/guanaco_belle_merge_v1.0", split="train[-50:]")
# ]
# Prepare model
model = LlamaForCausalLM.from_pretrained(
    "/remote-home/share/MOSS_7B_Base/", config=config)
model.enable_input_require_grads()
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5) 
tokenizer = LlamaTokenizer.from_pretrained(
    "/remote-home/share/MOSS_7B_Base/", add_bos_token=True, padding_left=True)
# Convert to CoLLie Dataset
train_dataset = CollieDatasetForTraining(train_dataset,
                                          tokenizer=tokenizer)
eval_dataset_ppl = CollieDatasetForTraining(eval_dataset_ppl,
                                            tokenizer=tokenizer)
# eval_dataset_bleu = CollieDatasetForGeneration(eval_dataset_bleu,
#                                                tokenizer=tokenizer)
# Prepare Evaluator
evaluator_ppl = EvaluatorForPerplexity(
    model=model,
    config=config,
    dataset=eval_dataset_ppl,
    monitors=[
        EvalMonitor(config)
    ],
    metrics={
        "ppl": PPLMetric(gather_result=True)
    },
)
# evaluator_bleu = EvaluatorForGeneration(
#     model=model,
#     config=config,
#     dataset=eval_dataset_bleu,
#     monitors=[
#         EvalMonitor(config)
#     ],
#     metrics={
#         "bleu": BleuMetric(gather_result=True, ngram=1),
#         "decode": DecodeMetric()
#     },
#     generation_config=GenerationConfig(
#         eos_token_id=tokenizer.eos_token_id,
#         pad_token_id=tokenizer.pad_token_id,
#         max_new_tokens=100,
#         use_cache=False
#     )
# )
# Prepare Trainer
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    config=config,
    optimizer=optimizer,
    train_dataset=train_dataset,
    monitors=[
        LossMonitor(config),
        TGSMonitor(config),
        MemoryMonitor(config),
        LRMonitor(config)
    ],
    data_provider=GradioProvider(tokenizer, port=12888, stream=True,
                                 generation_config=GenerationConfig(
                                     eos_token_id=tokenizer.eos_token_id,
                                     pad_token_id=tokenizer.pad_token_id,
                                     max_new_tokens=250,
                                 )),
    evaluators=[
        evaluator_ppl],
    callbacks=[
        CheckpointCallback(
            folder="./lora/adaptors",  every_n_epochs=1, last=True, every_n_batches=100, adapter_name="default")
    ]
)
# Command: torchrun --standalone --nproc_per_node=4 sft_lora.py
trainer.train()
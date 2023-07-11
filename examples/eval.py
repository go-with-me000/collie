import argparse
# import openai
import os
from typing import List
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import numpy as np
import pandas as pd
import torch
from tqdm import trange
from transformers import LlamaTokenizer
from peft import PeftModel
from collie import CollieConfig, LlamaForCausalLM
from crop import crop
import csv

# openai.api_key = "INSERTYOURKEYHERE"
choices = ["A", "B", "C", "D"]
model = None
tokenizer = None

def softmax(x):
    z = x - max(x)
    numerator = np.exp(z)
    denominator = np.sum(numerator)
    softmax = numerator / denominator
    return softmax


def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt


def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(format_subject(subject))
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt


def get_logits(inputs: List[str]):
    tokens = tokenizer.batch_encode_plus(inputs,
                                         return_tensors='pt',
                                         padding=True,
                                         truncation=True).to('cuda')

    outputs = model(tokens.input_ids)
    return outputs[0], {'tokens': tokens}


def get_ppl(texts: List[str], mask_length=None):
    outputs, inputs = get_logits(texts)
    shift_logits = outputs[..., :-1, :].contiguous()

    shift_labels = inputs['tokens']['input_ids'][..., 1:].contiguous()

    loss_fct = torch.nn.CrossEntropyLoss(
        reduction='none', ignore_index=tokenizer.pad_token_id)
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)).view(shift_labels.size())

    if mask_length is not None:
        mask = torch.zeros_like(shift_labels)  # [batch,seqlen]
        for i in range(len(mask)):
            for j in range(mask_length[i] - 1, len(mask[i])):
                mask[i][j] = 1
        loss = loss * mask

    lens = (inputs['tokens']['input_ids'] !=
            tokenizer.pad_token_id).sum(-1).cpu().numpy()
    if mask_length is not None:
        lens -= np.array(mask_length)
    ce_loss = loss.sum(-1).cpu().detach().numpy() / lens
    return ce_loss


def eval(args, subject, dev_df, test_df):
    ppl = []
    answers = []
    predictions = []
    for choice in choices:
        prompt_list = []
        sub_ppl_list = []
        for i in range(test_df.shape[0]):
            # get prompt and make sure it fits
            k = args.ntrain
            prompt_end = format_example(test_df, i, include_answer=False)
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end
            import pdb;pdb.set_trace()
            while crop(prompt) != prompt:
                k -= 1
                train_prompt = gen_prompt(dev_df, subject, k)
                prompt = train_prompt + prompt_end
            prompt = prompt + choice
            prompt_list.append(prompt)
            if choice == 'A':
                answer = test_df.iloc[i, test_df.shape[1] - 1]
                answers.append(answer)
        batch_size = 8
        for idx in trange(0, len(prompt_list), batch_size):
            sub_prompt_list = prompt_list[idx:idx + batch_size]
            with torch.no_grad():
                sub_res = get_ppl(sub_prompt_list).tolist()
                for res, prompt in zip(sub_res, sub_prompt_list):
                    sub_ppl_list.append(res)
        ppl.append(sub_ppl_list)
    ppl = list(zip(*ppl))
    for single_ppl in ppl:
        predictions.append(choices[single_ppl.index(min(single_ppl))])
    correct = 0
    for pred, ref in zip(predictions, answers):
        if pred == ref:
            correct = correct + 1
    acc = correct / len(predictions) * 100
    print("ACC:", acc)
    return acc


def main(args):
    subjects = sorted(
        [f.split("_test.csv")[0] for f in os.listdir(os.path.join(args.data_dir, "test")) if "_test.csv" in f])
    all_acc = []
    model_path = "/mnt/petrelfs/share_data/zhangshuo/model/MOSS_7B_Base"
    global model
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16,)
    model.to("cuda").eval()
    adapater_path = args.adaptor
    adapater_name = os.path.basename(adapater_path)
    adapater_name = "/mnt/petrelfs/chenkeyu1/program/collie/examples/logs/"+adapater_name
    model = PeftModel.from_pretrained(model, adapater_path)
    global tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(
        model_path, add_bos_token=True, padding_left=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    for index,subject in enumerate(subjects):
        print("Index:",index,"  Nowadays:",subject)
        dev_df = pd.read_csv(os.path.join(args.data_dir, "dev", subject + "_dev.csv"), header=None)[:args.ntrain]
        test_df = pd.read_csv(os.path.join(args.data_dir, "test", subject + "_test.csv"), header=None)
        acc = eval(args, subject, dev_df, test_df)
        all_acc.append(acc)
    weighted_acc = np.mean(all_acc)
    save_result(subjects,all_acc,weighted_acc,adapater_name)
    print("Average accuracy: {:.3f}".format(weighted_acc))

def save_result(subjects,all_acc,avg,file):
    file = file+"/data.csv"
    data = all_acc + [avg]
    subjects = subjects+["Average"]

    # 创建一个CSV文件并写入数据
    with open(file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Subject', 'Value'])  # 写入标题行
        for i in range(len(subjects)):
            writer.writerow([subjects[i], data[i]])  # 写入每行的数据
    print(f"数据已成功写入 {file} 文件。")

def rename_files(folder_path, old_name='adapter.bin', new_name='adapter_model.bin'):
    convert_keys(folder_path, old_name)
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file == old_name:
                file_path = os.path.join(root, file)
                new_file_path = os.path.join(root, new_name)
                os.rename(file_path, new_file_path)

def convert_keys(folder_path,name ="adapter.bin"):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file == name:
                file_path = os.path.join(root, file)
                s: dict = torch.load(file_path)
                for key in list(s.keys()):
                    s[key.replace("base_model", "base_model.model")] = s.pop(key)
                torch.save(s, file_path)


if __name__ == "__main__":
    folder_path = '/mnt/petrelfs/chenkeyu1/program/collie/lora_weights/'
    data_path = "/mnt/petrelfs/chenkeyu1/program/openAGIEval/data/mmlu"
    rename_files(folder_path)


    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--data_dir", "-d", type=str, default=data_path)
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--adaptor", "-a", type=str)
    args = parser.parse_args()
    main(args)

# srun -p llm --quotatype=spot --gres=gpu:1  python eval.py 2>&1 | tee log.txt
import argparse
# import openai
import csv
import os
import numpy as np
import pandas as pd
import time
from transformers import LlamaTokenizer, LlamaForCausalLM, LlamaConfig
import torch
from peft import PeftModel


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

folder_path = '/mnt/petrelfs/chenkeyu1/program/collie/lora_weights/'
rename_files(folder_path)
model_path = "/mnt/petrelfs/share_data/zhangshuo/model/MOSS_7B_Base"
choices = ["A", "B", "C", "D"]
config = LlamaConfig.from_pretrained(model_path)
model = LlamaForCausalLM.from_pretrained(
    model_path, config=config, device_map="auto")
tokenizer = LlamaTokenizer.from_pretrained(
    model_path, add_bos_token=True, padding_left=True)
adapater_path = "/mnt/petrelfs/chenkeyu1/program/collie/lora_weights/adaptor_07_11/epoch_2"
model = PeftModel.from_pretrained(model, adapater_path)

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


def eval(args, subject, engine, dev_df, test_df):
    cors = []
    all_probs = []
    answers = choices[:test_df.shape[1] - 2]

    for i in range(test_df.shape[0]):
        # get prompt and make sure it fits
        k = args.ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end

        # while crop(prompt) != prompt:
        #     k -= 1
        #     train_prompt = gen_prompt(dev_df, subject, k)
        #     prompt = train_prompt + prompt_end

        label = test_df.iloc[i, test_df.shape[1] - 1]

        inputs = tokenizer(prompt, return_tensors="pt")
        inputs.input_ids[:, 0] = 1
        with torch.no_grad():
            outputs = model.generate(
                # inputs.input_ids.cuda(),
                # attention_mask=inputs.attention_mask.cuda(),
                input_ids=inputs.input_ids.cuda(),
                attention_mask=inputs.attention_mask.cuda(),
                # max_length=512,
                # do_sample=do_sample,
                # top_k=4,
                # top_p=top_p,
                # temperature=temperature,
                # repetition_penalty=penalty,
                num_return_sequences=1,
                max_new_tokens=1,
                eos_token_id=tokenizer.sp_model.eos_id(),
                bos_token_id=tokenizer.sp_model.bos_id())
            response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        lprobs = []
        for ans in answers:
            # try:
            #     lprobs.append(c["choices"][0]["logprobs"]["top_logprobs"][-1][" {}".format(ans)])
            if response[-1] == ans:
                lprobs.append(1.0)
            else:
                lprobs.append(0.0)
            # except:
            #     print("Warning: {} not found. Artificially adding log prob of -100.".format(ans))
            #     lprobs.append(-100)
        pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(lprobs)]
        probs = softmax(np.array(lprobs))

        cor = pred == label
        cors.append(cor)
        all_probs.append(probs)


    acc = np.mean(cors)

    cors = np.array(cors)

    all_probs = np.array(all_probs)
    print("Average accuracy {:.3f} - {}".format(acc, subject))
    with open('results/07_11_epoch_2_results.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        # 将生成结果作为一行写入CSV文件
        writer.writerow([subject,acc])
    return cors, acc, all_probs


def main(args):
    engines = args.engine
    subjects = sorted(
        [f.split("_test.csv")[0] for f in os.listdir(os.path.join(args.data_dir, "test")) if "_test.csv" in f])
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    for engine in engines:
        if not os.path.exists(os.path.join(args.save_dir, "results_{}".format(engine))):
            os.mkdir(os.path.join(args.save_dir, "results_{}".format(engine)))

    print(subjects)
    print(args)
    # import pdb;pdb.set_trace()
    for engine in engines:
        print(engine)
        all_cors = []

        for subject in subjects:
            dev_df = pd.read_csv(os.path.join(args.data_dir, "dev", subject + "_dev.csv"), header=None)[:args.ntrain]
            test_df = pd.read_csv(os.path.join(args.data_dir, "test", subject + "_test.csv"), header=None)

            cors, acc, probs = eval(args, subject, engine, dev_df, test_df)
            all_cors.append(cors)

            test_df["{}_correct".format(engine)] = cors
            for j in range(probs.shape[1]):
                choice = choices[j]
                test_df["{}_choice{}_probs".format(engine, choice)] = probs[:, j]
            test_df.to_csv(os.path.join(args.save_dir, "results_{}".format(engine), "{}.csv".format(subject)),
                           index=None)

        weighted_acc = np.mean(np.concatenate(all_cors))
        print("Average accuracy: {:.3f}".format(weighted_acc))




if __name__ == "__main__":

    folder_path = '/mnt/petrelfs/chenkeyu1/program/collie/lora_weights/'
    data_path = "/mnt/petrelfs/chenkeyu1/program/openAGIEval/data/mmlu"
    rename_files(folder_path)
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--data_dir", "-d", type=str, default=data_path)
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    # parser.add_argument("--engine", "-e", choices=["davinci", "curie", "babbage", "ada"],
    #                     default=["davinci", "curie", "babbage", "ada"], nargs="+")
    parser.add_argument("--engine", "-e", choices=["davinci", "curie", "babbage", "ada"],
                        default=["davinci"], nargs="+")
    args = parser.parse_args()

    main(args)

# srun -p llm --quotatype=spot --gres=gpu:1  python evaluate.py
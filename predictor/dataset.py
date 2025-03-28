import os
import datasets
import argparse
import transformers
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset

def extract_prompt(example):
    conversation = example['conversation']
    user_content = ''
    for i, sentence in enumerate(conversation):
        if sentence['role'] == 'user':
            if i > 0:
                user_content += '\n'
            user_content += sentence['content']
        else:
            break
    assistant_content = ''
    for j in range(i, len(conversation)):
        sentence = conversation[j]
        if sentence['role'] == 'assistant':
            if j > i:
                assistant_content += '\n'
            assistant_content += conversation[j]['content']
        else:
            break
    example['prompt'] = user_content
    encoded_response = vicuna_tokenizer(assistant_content, truncation=False)
    example['num_tokens'] = len(encoded_response['input_ids'])
    if task_type == 0:
        example['labels'] = len(encoded_response['input_ids'])
    else:
        for i, thresh in enumerate(multi_cls_thresholds):
            if len(encoded_response['input_ids']) < thresh:
                example['labels'] = i
                break
    return example

def tokenize_function(example):
    example = bert_tokenizer(example["prompt"], truncation=False)
    if len(example['input_ids']) >= 512:
        if FLAG_HEAD_TAIL:
            example['input_ids'] = example['input_ids'][: 128] + example['input_ids'][-384: ]
            example['token_type_ids'] = example['token_type_ids'][: 128] + example['token_type_ids'][-384: ]
            example['attention_mask'] = example['attention_mask'][: 128] + example['attention_mask'][-384: ]
        else:
            example['input_ids'] = example['input_ids'][-512: ]
            example['token_type_ids'] = example['token_type_ids'][-512: ]
            example['attention_mask'] = example['attention_mask'][-512: ]
    return example

def calc_percentile(dataset):
    output_token_lengths = []
    for sample in dataset:
        output_token_lengths.append(sample['num_tokens'])
    s = pd.Series(output_token_lengths)
    print(s.describe(percentiles=[.25, .5, .75, .99]))
    return dataset

def preprocess_dataset(dataset):
    dataset = dataset.remove_columns(['openai_moderation', 'redacted', 'language', 'conversation_id', 'turn'])
    new_sentence_column = [''] * len(dataset)
    dataset = dataset.add_column('prompt', new_sentence_column)
    new_label_column = [0] * len(dataset)
    dataset = dataset.add_column('labels', new_label_column)
    if task_type != 0:
        new_length_column = [0] * len(dataset)
        dataset = dataset.add_column('num_tokens', new_length_column)

    # Extract the user prompt(s) and the corresponding response length
    dataset = dataset.map(extract_prompt, remove_columns=['conversation'])
    print('Num samples before filtering: ', len(dataset))
    if task_type == 0:
        dataset = dataset.filter(lambda example: example["labels"] > 1 and example["labels"] < 512)
    else:
        dataset = dataset.filter(lambda example: example["num_tokens"] > 1 and example["num_tokens"] < 512)
    print('Num samples after filtering: ', len(dataset))
    dataset = dataset.remove_columns(['model'])
    dataset = dataset.map(tokenize_function, batched=False, remove_columns=['prompt'])
    return dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='llama-2-7b-chat')
    parser.add_argument('--head_tail', action='store_true', default=False)
    parser.add_argument('--task_type', type=int, default=2)
    parser.add_argument('--data_size', type=int, default=1000)
    args = parser.parse_args()

    # 0: regression; 1: binary classification; 2: multi-class classification;
    task_type = args.task_type
    FLAG_HEAD_TAIL = args.head_tail
    # cls_threshold = 328
    if task_type == 1:
        multi_cls_thresholds = [128, 400, 1000000]
    else:
        multi_cls_thresholds = [32, 128, 256, 400, 1000000]
    dataset_name = 'lmsys/lmsys-chat-1m'
    model_name = 'bert-base-uncased'
    vicuna_tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf", legacy=False) 
    bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
    bert_tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
    selected_data_size = 1000 * args.data_size
    dataset_path = 'data/lmsys_dataset'

    dataset = load_dataset(dataset_name, split='train')
    dataset = dataset.select(range(selected_data_size))
    dataset = dataset.filter(lambda example: example["model"] == args.model)
    dataset = dataset.shuffle(seed=1)
    dataset = preprocess_dataset(dataset)

    if task_type != 0:
        dataset = calc_percentile(dataset)
    dataset.set_format("torch")

    os.makedirs('./data', exist_ok=True)
    dataset.save_to_disk(dataset_path)
    print('Saved dataset to ' + dataset_path)

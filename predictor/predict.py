import os
import time
import torch
import argparse
import datasets
import transformers
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer, DataCollatorWithPadding

from model import split_dataset, BertClassificationModel, BertRegressionModel

@torch.inference_mode()
def predict(model, dataloader, device):
    predicted_labels = []
    actual_lengths = []
    latencies = []
    print_model_names = []
    for batch in dataloader:
        start_time = time.time()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        predictions = model(input_ids=input_ids, attention_mask=attention_mask)
        batch_size = len(input_ids)
        batch_latency = (time.time() - start_time) / batch_size
        latencies.extend([batch_latency] * batch_size)
        if TASK_TYPE in {0, 3, 4}:
            lengths = batch['num_tokens']
        else:
            predictions = torch.argmax(predictions, dim=-1)
            lengths = batch['num_tokens']
        predicted_labels.extend(predictions.cpu().numpy())
        actual_lengths.extend(lengths.numpy())
    df = pd.DataFrame({
        'actual_length': actual_lengths,
        'predicted_label': predicted_labels,
        'latency': latencies
    })
    df.index.name = "sample_id"
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='llama-2-7b-chat')
    parser.add_argument('--task_type', type=int, default=2)
    parser.add_argument('--data_size', type=int, default=1000)
    parser.add_argument('--dataset_path', type=str, default='./data/lmsys_dataset')
    args = parser.parse_args()

    # 0: regression; 1: binary classification; 2: multi-class classification; 3: multi-class ordinal classification; 4: bi-class ordinal classification; 
    TASK_TYPE = args.task_type
    dataset_path = args.dataset_path
    model_name = 'bert-base-uncased'
    num_classes = 3 if (TASK_TYPE == 1 or TASK_TYPE == 4) else 5
    output_filename = 'predictions_' + args.model.lower() + '.csv'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
    bert_tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
    
    test_batch_size = 100
    dataset = datasets.load_from_disk(dataset_path)
    print(f'Loaded dataset from ' + dataset_path)
    train_dataset, validation_dataset, test_dataset = split_dataset(dataset)
    data_collator = DataCollatorWithPadding(tokenizer = bert_tokenizer, padding="max_length", max_length=512, return_tensors="pt")
    test_dataloader = DataLoader(test_dataset, shuffle = False, batch_size = test_batch_size, collate_fn = data_collator)

    config = AutoConfig.from_pretrained(model_name)
    # regression or ordinal classification
    if TASK_TYPE == 0 or TASK_TYPE == 3 or TASK_TYPE == 4:
        model = BertRegressionModel(config, model_name, hidden_dim=128).to(device)
    # classification
    elif TASK_TYPE == 1 or TASK_TYPE == 2:
        model = BertClassificationModel(config, model_name, hidden_dim=128, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load('/home/zhaosh56/BatchingLLM/predictor/models/llama-2-7b-chat.pth', weights_only=True))
    model.to(device)

    # Inference
    print("Start inference...")
    df = predict(model, test_dataloader, device)
    os.makedirs('./results', exist_ok=True)
    df.to_csv('./results/' + output_filename)
    print('Saved results to ./results/' + output_filename)

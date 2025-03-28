import os
import time
import datasets
import argparse
import evaluate
import numpy as np
import transformers
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
from transformers import AutoConfig, AutoTokenizer, BertModel, DataCollatorWithPadding

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model import split_dataset, BertClassificationModel, BertRegressionModel

torch.cuda.set_device(3)

def train(model, criterion, optimizer, train_dataloader, validation_dataloader, num_epochs, device):
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = transformers.get_scheduler('linear', optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps,)

    for epoch in tqdm(range(num_epochs)):
        training_loss = 0
        model.train()

        if epoch == 30:
            for param in model.bert.parameters():
                param.requires_grad = False
            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-4

        for batch in train_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            if TASK_TYPE == 0:
                labels = batch['num_tokens'].to(device)
            else:
                labels = batch['labels'].to(device)
            if TASK_TYPE == 0 or TASK_TYPE == 3 or TASK_TYPE == 4:
                loss = criterion(output, labels.float())
            else:
                loss = criterion(output, labels)
            optimizer.zero_grad()
            
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            training_loss += loss.item()

        print(f"Training loss for epoch {epoch}: {training_loss / len(train_dataloader)}")
        if TASK_TYPE == 0:
            validation_metrics = eval_regression(model, validation_dataloader, device)
        elif TASK_TYPE == 3 or TASK_TYPE == 4:
            validation_metrics = eval_regression(model, validation_dataloader, device)
            validation_metrics = validation_metrics | eval_classification(model, validation_dataloader, device)
        else:
            validation_metrics = eval_classification(model, validation_dataloader, device)
        print(f'Validation loss after epoch {epoch}: ')
        for k, v in validation_metrics.items():
            print(f'{k}: {v:.4f}', end='\t')
        print(' ')
    print("Finished training.")

@torch.inference_mode()
def eval_classification(model, dataloader, device):
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1", average="macro")
    precision_metric = evaluate.load("precision", average="macro")
    recall_metric = evaluate.load("recall", average="macro")

    labels = []
    predictions = []
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        label = batch['labels'].to(device)

        if TASK_TYPE != 3 and TASK_TYPE != 4:
            prediction = torch.argmax(output, dim=-1)
        else:
            prediction = torch.round(output).type(torch.LongTensor)
            for i in range(len(prediction)):
                if prediction[i] >= num_classes:
                    prediction[i] = num_classes - 1
                elif prediction[i] < 0:
                    prediction[i] = 0
        labels.extend(label)
        predictions.extend(prediction)
    metric = accuracy_metric.compute(references=labels, predictions=predictions) | \
        f1_metric.compute(references=labels, predictions=predictions, average='macro') | \
        precision_metric.compute(references=labels, predictions=predictions, average='macro') | \
        recall_metric.compute(references=labels, predictions=predictions, average='macro')
    return metric

@torch.inference_mode()
def eval_regression(model, dataloader, device):
    l1loss = nn.L1Loss()
    mseloss = nn.MSELoss()

    l1err = 0
    mse = 0
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        prediction = model(input_ids=input_ids, attention_mask=attention_mask)
        if TASK_TYPE == 0:
            labels = batch['num_tokens'].to(device)
        else:
            labels = batch['labels'].to(device)
        l1err += l1loss(prediction, labels.type_as(prediction))
        mse += mseloss(prediction, labels.type_as(prediction))

    metric = {'L1 error': l1err.item() / len(dataloader), 'MSE': mse.item() / len(dataloader)}
    return metric


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='llama-2-7b-chat')
    parser.add_argument('--l1_loss', action='store_true', default=False)
    parser.add_argument('--task_type', type=int, default=2)
    parser.add_argument('--data_size', type=int, default=1000)
    parser.add_argument('--dataset_path', type=str, default='./data/lmsys_dataset')
    args = parser.parse_args()

    # 0: regression; 1: binary classification; 2: multi-class classification; 3: multi-class ordinal classification; 4: bi-class ordinal classification; 
    TASK_TYPE = args.task_type
    FLAG_L1_LOSS = args.l1_loss
    dataset_path = args.dataset_path
    selected_data_size = 1000 * args.data_size
    output_filename = 'predictions_' + args.model.lower() + '.csv'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # set the training model and classes
    model_name = 'bert-base-uncased'
    num_classes = 3 if (TASK_TYPE == 1 or TASK_TYPE == 4) else 5
    # init the toknier for the LLM and BERT
    bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
    bert_tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

    # training parameters
    num_epochs = 50
    train_batch_size = 64
    test_batch_size = 512
    lr = 1e-5
    # load the dataset for model training
    dataset = datasets.load_from_disk(dataset_path)
    print(f'Loaded dataset from ' + dataset_path)
    # set the dataloader for training
    train_dataset, validation_dataset, test_dataset = split_dataset(dataset)
    data_collator = DataCollatorWithPadding(tokenizer = bert_tokenizer, padding="max_length", max_length=512, return_tensors="pt")
    train_dataloader = DataLoader(train_dataset, shuffle = False, batch_size = train_batch_size, collate_fn = data_collator)
    validation_dataloader = DataLoader(validation_dataset, shuffle = True, batch_size = train_batch_size, collate_fn = data_collator)
    test_dataloader = DataLoader(test_dataset, shuffle = False, batch_size = test_batch_size, collate_fn = data_collator)
    config = AutoConfig.from_pretrained(model_name)

    # prepare the weights for the loss function
    weights = []
    if TASK_TYPE == 1 or TASK_TYPE == 2:
        for i in range(num_classes):
            n_samples_for_label_i = len(dataset.filter(lambda example: example["labels"] == i)['labels'])
            print('Number of samples for class ' + str(i) + ': ' + str(n_samples_for_label_i))
            if n_samples_for_label_i == 0:
                weights.append(0.0)
            else:
                weights.append(1.0 / n_samples_for_label_i)

    # regression or ordinal classification
    if TASK_TYPE == 0 or TASK_TYPE == 3 or TASK_TYPE == 4:
        model = BertRegressionModel(config, model_name, hidden_dim=128).to(device)
        if FLAG_L1_LOSS:
            criterion = nn.L1Loss()
        else:
            criterion = nn.MSELoss()
    # classification
    elif TASK_TYPE == 1 or TASK_TYPE == 2:
        model = BertClassificationModel(config, model_name, hidden_dim=128, num_classes=num_classes).to(device)
        criterion = nn.NLLLoss(weight=torch.tensor(weights).to(device))
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr)

    # Training
    print("Start training...")
    train(model, 
        criterion, 
        optimizer, 
        train_dataloader, 
        validation_dataloader, 
        num_epochs, 
        device)

    os.makedirs('./models', exist_ok=True)
    torch.save(model.state_dict(), './models/' + args.model + '.pth')

    if TASK_TYPE == 0:
        validation_metrics = eval_regression(model, validation_dataloader, device)
    elif TASK_TYPE == 3 or TASK_TYPE == 4:
        validation_metrics = eval_regression(model, validation_dataloader, device)
        validation_metrics = validation_metrics | eval_classification(model, validation_dataloader, device)
    else:
        validation_metrics = eval_classification(model, validation_dataloader, device)
    print(f'Validation metrics after training:')
    for k, v in validation_metrics.items():
        print(f'{k}: {v:.4f}')

    if TASK_TYPE == 0:
        validation_metrics = eval_regression(model, test_dataloader, device)
    elif TASK_TYPE == 3 or TASK_TYPE == 4:
        validation_metrics = eval_regression(model, test_dataloader, device)
        validation_metrics = validation_metrics | eval_classification(model, test_dataloader, device)
    else:
        validation_metrics = eval_classification(model, test_dataloader, device)
    print(f'Metrics on test set:')
    os.makedirs('./metrics', exist_ok=True)
    with open('./metrics/' + output_filename.split('.')[0] + '.txt', 'a') as f:
        for k, v in validation_metrics.items():
            f.write(f'{k}: {v:.4f}\n')
            print(f'{k}: {v:.4f}')
    print("Finished training.")

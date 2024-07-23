import torch
from transformers import BertForSequenceClassification, RobertaForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import numpy as np
import time
import datetime
import plotly.express as px
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import os

def initialize_model(model_type, learning_rate, eps, device):
    if model_type == "Bert":
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    elif model_type == "Roberta":
        model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)
    else:
        raise ValueError("Invalid model type. Choose 'Bert' or 'Roberta'.")

    model.to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=eps)
    return model, optimizer

def create_scheduler(optimizer, train_dataloader, epochs):
    total_steps = len(train_dataloader) * epochs
    return get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

def save_model(model, model_dir, model_name):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model.save_pretrained(os.path.join(model_dir, model_name))

def train_model(model, train_dataloader, validation_dataloader, optimizer, scheduler, epochs, device, log_dir):
    loss_values = []
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, "training_log.txt")
    with open(log_file, "w") as f:
        for epoch_i in range(epochs):
            print(f'======== Epoch {epoch_i + 1} / {epochs} ========')
            f.write(f'======== Epoch {epoch_i + 1} / {epochs} ========\n')
            print('Training...')
            f.write('Training...\n')
            t0 = time.time()
            total_loss = 0
            model.train()

            for step, batch in enumerate(train_dataloader):
                if step % 40 == 0 and not step == 0:
                    elapsed = format_time(time.time() - t0)
                    print(f'  Batch {step}  of  {len(train_dataloader)}.    Elapsed: {elapsed}.')
                    f.write(f'  Batch {step}  of  {len(train_dataloader)}.    Elapsed: {elapsed}.\n')

                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                token_type_ids = batch['token_type_ids']
                labels = batch['label']

                model.zero_grad()
                outputs = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs[0]
                total_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

            avg_train_loss = total_loss / len(train_dataloader)
            loss_values.append(avg_train_loss)
            print(f"  Average training loss: {avg_train_loss:.2f}")
            f.write(f"  Average training loss: {avg_train_loss:.2f}\n")
            print(f"  Training epoch took: {format_time(time.time() - t0)}")
            f.write(f"  Training epoch took: {format_time(time.time() - t0)}\n")
            print("Running Validation...")
            f.write("Running Validation...\n")
            evaluate_model(model, validation_dataloader, device, f)

    plot_loss_curve(loss_values, model.config._name_or_path)

def evaluate_model(model, dataloader, device, log_file):
    t0 = time.time()
    model.eval()
    eval_accuracy = 0
    nb_eval_steps = 0

    for batch in dataloader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        token_type_ids = batch['token_type_ids']
        labels = batch['label']

        with torch.no_grad():
            outputs = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        
        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        label_ids = labels.to('cpu').numpy()
        eval_accuracy += flat_accuracy(logits, label_ids)
        nb_eval_steps += 1

    accuracy = eval_accuracy / nb_eval_steps
    print(f"  Accuracy: {accuracy:.2f}")
    log_file.write(f"  Accuracy: {accuracy:.2f}\n")
    print(f"  Evaluation took: {format_time(time.time() - t0)}")
    log_file.write(f"  Evaluation took: {format_time(time.time() - t0)}\n")

def test_model(model, test_dataloader, device, log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, "test_log.txt")
    with open(log_file, "w") as f:
        print(f'Predicting labels for {len(test_dataloader.dataset)} test sentences...')
        f.write(f'Predicting labels for {len(test_dataloader.dataset)} test sentences...\n')
        model.eval()
        predictions, true_labels = [], []

        for batch in test_dataloader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            token_type_ids = batch['token_type_ids']
            labels = batch['label']

            with torch.no_grad():
                outputs = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
            
            logits = outputs[0]
            predictions.append(logits.detach().cpu().numpy())
            true_labels.append(labels.to('cpu').numpy())

        print('    DONE.')
        f.write('    DONE.\n')
        generate_report(predictions, true_labels, f)

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def format_time(elapsed):
    elapsed_rounded = int(round(elapsed))
    return str(datetime.timedelta(seconds=elapsed_rounded))

def plot_loss_curve(loss_values, model_name):
    df = pd.DataFrame(loss_values, columns=['Loss'])
    fig = px.line(df, x=df.index, y='Loss', title=f'{model_name} Training Loss Curve', labels={'x':'Epoch', 'Loss':'Loss'})
    fig.show()

def generate_report(predictions, true_labels, log_file):
    predictions_labels = np.argmax(np.concatenate(predictions, axis=0), axis=1).flatten()
    flat_true_labels = np.concatenate(true_labels, axis=0)
    report = classification_report(flat_true_labels, predictions_labels)
    confusion = confusion_matrix(flat_true_labels, predictions_labels)
    print(report)
    print(confusion)
    log_file.write(report + "\n")
    log_file.write(str(confusion) + "\n")

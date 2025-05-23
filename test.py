import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import pandas as pd
import random
from transformers import BertTokenizer, BertModel
from utils import get_dataloader, iter_product
from model import CustomBERT
from config import test_config
from easydict import EasyDict as edict

device = torch.device('cuda')


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def test_model(dataloader, model, threshold):
    print("---Start Test!---")
    model.eval()
    predictions, true_labels = [], []
    all_input_ids = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch["input_ids"].to(device)
            head_token_idx = batch["head_token_idx"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, head_token_idx, attention_mask)
            preds = (torch.softmax(outputs, dim=1)[:, 1] >= threshold)

            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            all_input_ids.extend(input_ids.cpu().numpy())

    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average="weighted")

    return accuracy, f1, predictions, true_labels, all_input_ids


def test(log):
    set_seed(log.param.SEED)

    tokenizer = BertTokenizer.from_pretrained(log.param.model_type)
    model = CustomBERT(log.param.model_type, hidden_dim=log.param.hidden_size, e=log.param.e).to(device)

    if "ihc" in log.param.dataset or "SST" in log.param.dataset:
        test_loader = get_dataloader(f"./data/{log.param.dataset}/test.tsv", tokenizer, ner_tagger=None, use_ner=False, batch_size=16, shuffle=False)
    else:
        test_loader = get_dataloader(f"./data/{log.param.dataset}/test.csv", tokenizer, ner_tagger=None, use_ner=False, batch_size=16, shuffle=False)

    ckpt = torch.load(f"./save/{log.param.model_path}/best_model.pth")
    threshold = ckpt["threshold"]
        
    model.load_state_dict(ckpt["model"])
    model.to(device)

    test_accuracy, test_f1, predictions, true_labels, all_input_ids = test_model(test_loader, model, threshold)

    print(f"Dataset: {log.param.dataset}")
    print(f"Test Accuracy: {test_accuracy*100:.2f}")
    print(f"Test F1-Score: {test_f1*100:.2f}")
    print(f"Wrong predictions saved to ./save/{log.param.dataset}/wrong_predictions.csv")


if __name__ == '__main__':
    tuning_param = test_config.tuning_param
    
    param_list = [test_config.param[i] for i in tuning_param]
    param_list = [tuple(tuning_param)] + list(iter_product(*param_list)) ## [(param_name),(param combinations)]

    for param_com in param_list[1:]: # as first element is just name
        log = edict()
        log.param = test_config.param

        for num,val in enumerate(param_com):
            log.param[param_list[0][num]] = val

    test(log)
import torch
import torch.nn as nn
from transformers import BertModel, BertConfig


class ILCBertClassifier(nn.Module):
    def __init__(self, bert_model_name: str):
        super(ILCBertClassifier, self).__init__()
        
        self.bert = BertModel.from_pretrained(bert_model_name, output_hidden_states=True)
        hidden_size = self.bert.config.hidden_size
        num_layers = self.bert.config.num_hidden_layers

        self.ilc_classifiers = nn.ModuleList([
            nn.Linear(hidden_size, 2)
            for _ in range(num_layers - 1)
        ])

    def forward(self, input_ids, labels=None):
        outputs = self.bert(input_ids=input_ids, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # tuple: (layer0, layer1, ..., layerN)

        # Collect logits from each intermediate layer classifier
        logits_list = []
        for idx, classifier in enumerate(self.ilc_classifiers, start=1):
            # hidden_states[idx] corresponds to the output of the idx-th layer
            cls_emb = hidden_states[idx][:, 0, :]  # [CLS] token embedding
            logits = classifier(cls_emb)
            logits_list.append(logits)

        # If labels provided, compute combined loss
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            losses = [loss_fct(logits, labels) for logits in logits_list]
            total_loss = sum(losses)
            return total_loss, logits_list

        # Otherwise return all logits for layer-specific evaluation
        return logits_list


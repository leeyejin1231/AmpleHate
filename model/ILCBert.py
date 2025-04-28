# model_ilc_binary.py
import torch, torch.nn as nn
from transformers import AutoModel
from typing import List, Optional

class BERT_ILC_Binary(nn.Module):
    def __init__(
        self,
        bert_name: str = "skt/kobert-base-v1",
        probe_layers: Optional[List[int]] = None,
        reg_type: str = "l1", reg_weight: float = 1e-4
    ):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_name, output_hidden_states=True)
        state_dict = torch.load("save/hateval/best_model.pth", map_location="cpu")
        state_dict = {k.replace("bert.", ""): v for k, v in state_dict.items()
                    if k.startswith("bert.")}
        _ = self.bert.load_state_dict(state_dict, strict=False) 
        
        for p in self.bert.parameters():
            p.requires_grad = False

        h = self.bert.config.hidden_size
        num_hidden = self.bert.config.num_hidden_layers           # 12
        if probe_layers is None:
            # probe_layers = list(range(0, num_hidden-1))           # 0~10  (L-2)
            probe_layers = [8, 9, 10, 11]

        self.probe_layers = probe_layers
        self.probes = nn.ModuleDict({str(l): nn.Linear(h, 1) for l in probe_layers})
        self.reg_type, self.reg_weight = reg_type, reg_weight

    def forward(self, input_ids, attention_mask):
        hs = self.bert(input_ids, attention_mask).hidden_states   # tuple len 13
        logits_dict = {}
        mask = attention_mask.bool()

        for l in self.probe_layers:
            rep = hs[l+1]                                   # (B,T,H)
            pooled = (rep * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
            logits_dict[l] = self.probes[str(l)](pooled).squeeze(-1)
        return logits_dict

        # for l in self.probe_layers:
        #     cls = hs[l+1][:, 0, :]
        #     logits_dict[l] = self.probes[str(l)](cls).squeeze(-1) # (B,)
        # return logits_dict
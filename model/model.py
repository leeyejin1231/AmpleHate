import torch
import torch.nn as nn
from transformers import BertModel, AutoModel

class HeadAttention(nn.Module):
    def __init__(self, hidden_dim, head_dim):
        super(HeadAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.head_dim = head_dim

        self.softmax = nn.Softmax(dim=-1)

        self.W_q = nn.Linear(hidden_dim, head_dim, bias=False)
        self.W_k = nn.Linear(hidden_dim, head_dim, bias=False)
        self.W_v = nn.Linear(hidden_dim, head_dim, bias=False)

    def forward(self, cls_embedding, head_token_embedding):
        Q_h = self.W_q(cls_embedding)   # [CLS] Query
        K_h = self.W_k(head_token_embedding)  # Target Token Key
        V_h = self.W_v(cls_embedding)  # [CLS] Value

        attention_scores = torch.matmul(Q_h, K_h.T) / (self.head_dim ** 0.5)
        attention_scores = attention_scores.float()
        attention_weights = self.softmax(attention_scores)

        output = torch.matmul(attention_weights, V_h)
        return output
    

class CustomBERT(nn.Module):
    def __init__(self, bert_model_name, hidden_dim, e=1e-3):
        super(CustomBERT, self).__init__()

        self.bert = AutoModel.from_pretrained(bert_model_name)
        self.hidden_dim = hidden_dim
        self.e = e

        # Head-Attention
        self.head_attention = HeadAttention(hidden_dim, hidden_dim)

        # Classifier
        self.classifier = nn.Linear(hidden_dim, 2)  # non-hate(0) / hate(1) 

    def forward(self, input_ids, head_token_idx, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask, output_hidden_states=True)

        cls_embedding = outputs.last_hidden_state[:, 0, :]

        hidden_dim = outputs.last_hidden_state.shape[-1]
        expanded_idx = head_token_idx.unsqueeze(-1).expand(-1, -1, hidden_dim)  # [batch, max_num_head, hidden_dim]
        head_token_embeddings = torch.gather(outputs.last_hidden_state, 1, expanded_idx)  # [batch, max_num_head, hidden_dim]

        outputs_list = []
        for i in range(head_token_embeddings.shape[1]):
            output = self.head_attention(cls_embedding, head_token_embeddings[:, i, :])
            outputs_list.append(output)
        
        head_attention_output = sum(outputs_list)
        final_embedding = cls_embedding + head_attention_output * self.e

        logits = self.classifier(final_embedding)
        return logits
    
    


    
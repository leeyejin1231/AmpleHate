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
        Q_h = self.W_q(cls_embedding)   # [CLS]의 Query
        K_h = self.W_k(head_token_embedding)  # 특정 토큰의 Key
        V_h = self.W_v(cls_embedding)  # [CLS]의 Value

        attention_scores = torch.matmul(Q_h, K_h.T) / (self.head_dim ** 0.5)
        attention_scores = attention_scores.float()
        attention_weights = self.softmax(attention_scores)

        output = torch.matmul(attention_weights, V_h)
        return output
    

class LinearHeadAttention(nn.Module):
    def __init__(self, head_dim):
        super(HeadAttention, self).__init__()
        self.head_dim = head_dim

        self.W = nn.Linear(head_dim, head_dim, bias=False)

    def forward(self, _, head_token_embedding):
        return self.W(head_token_embedding)
    

class CustomBERT(nn.Module):
    def __init__(self, bert_model_name, hidden_dim, e=1e-3):
        super(CustomBERT, self).__init__()

        self.bert = AutoModel.from_pretrained(bert_model_name)
        self.hidden_dim = hidden_dim
        self.e = e

        # Head-Attention 추가
        self.head_attention = HeadAttention(hidden_dim, hidden_dim)
        # self.head_attention = LinearHeadAttention(hidden_dim)

        # 최종 분류기
        self.classifier = nn.Linear(hidden_dim, 2)  # non-hate(0) / hate(1) 이진 분류

    def forward(self, input_ids, head_token_idx, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask, output_hidden_states=True)

        cls_embedding = outputs.last_hidden_state[:, 0, :]

        # Head-Token 위치 추출 (여러 개)
        hidden_dim = outputs.last_hidden_state.shape[-1]
        expanded_idx = head_token_idx.unsqueeze(-1).expand(-1, -1, hidden_dim)  # [batch, max_num_head, hidden_dim]
        head_token_embeddings = torch.gather(outputs.last_hidden_state, 1, expanded_idx)  # [batch, max_num_head, hidden_dim]

        head_attention_output = self.head_attention(cls_embedding, head_token_embeddings[:, 0, :])

        final_embedding = cls_embedding + head_attention_output * self.e

        # outputs_list = []
        # for i in range(head_token_embeddings.shape[1]):
        #     output = self.head_attention(cls_embedding, head_token_embeddings[:, i, :])
        #     outputs_list.append(output)
        
        # head_attention_output = sum(outputs_list)
        # final_embedding = cls_embedding + head_attention_output * self.e

        logits = self.classifier(final_embedding)
        return logits
    
    


    
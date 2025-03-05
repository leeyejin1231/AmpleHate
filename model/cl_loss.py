import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLossCosine(nn.Module):
    def __init__(self, margin=0.5):
        """
        Contrastive Loss with Cosine Similarity.
        margin: 다른 클래스 간 최소 거리
        """
        super(ContrastiveLossCosine, self).__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        """
        embeddings: (batch_size, embedding_dim) → 모델의 출력 임베딩
        labels: (batch_size) → 정답 라벨
        """
        batch_size = embeddings.size(0)

        # Cosine Similarity 계산 (1 - Cosine Distance)
        cosine_sim = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=-1)  # (batch_size, batch_size)

        # Label 매트릭스 생성 (같은 라벨이면 0, 다른 라벨이면 1)
        labels = labels.unsqueeze(1)  # (batch_size, 1)
        label_matrix = (labels != labels.T).float()  # (batch_size, batch_size), 같은 클래스=0, 다른 클래스=1

        # Contrastive Loss 계산 (Cosine Similarity 기준)
        positive_loss = (1 - label_matrix) * (1 - cosine_sim)  # 같은 클래스 → Cosine Similarity 높아야 함
        negative_loss = label_matrix * F.relu(cosine_sim - self.margin)  # 다른 클래스 → Cosine Similarity 낮아야 함

        # 평균 Loss 반환
        loss = (positive_loss + negative_loss).sum() / (batch_size * (batch_size - 1))  # 전체 쌍에 대해 평균
        return loss
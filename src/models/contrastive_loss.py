from enum import Enum
from typing import Iterable, Dict
import torch.nn.functional as F
from torch import nn, Tensor
from src.models.transformer_model import Transformer


class SiameseDistanceMetric(Enum):
    """
    The metric for the contrastive loss
    """

    EUCLIDEAN = lambda x, y: F.pairwise_distance(x, y, p=2)
    MANHATTAN = lambda x, y: F.pairwise_distance(x, y, p=1)
    COSINE_DISTANCE = lambda x, y: 1 - F.cosine_similarity(x, y)


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss. Expects as input two texts and a label of either 0 or 1. If the label == 1,
    then the distance between the two embeddings is reduced. If the label == 0, then the distance
    between the embeddings is increased.
    """

    def __init__(
        self,
        model: Transformer,
        distance_metric=SiameseDistanceMetric.COSINE_DISTANCE,
        margin: float = 0.5,
    ):
        super(ContrastiveLoss, self).__init__()
        self.distance_metric = distance_metric
        self.margin = margin
        self.model = model

    def forward(
        self,
        sentence_features: Iterable[Dict[str, Tensor]],
        labels: Tensor,
    ):
        reps = [
            self.model(sentence_feature)["sentence_embedding"]
            for sentence_feature in sentence_features
        ]
        assert len(reps) == 2
        rep_anchor, rep_other = reps
        distances = self.distance_metric(rep_anchor, rep_other)
        losses = 0.5 * (
            labels.float() * distances.pow(2)
            + (1 - labels).float() * F.relu(self.margin - distances).pow(2)
        )
        return losses.mean()

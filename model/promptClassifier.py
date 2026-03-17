import torch
import torch.nn as nn
from transformers import AutoModel

"""
Extremely simple classification model with inputs passed through bert and then passed through a linear and sigmoid layer.
"""
class PromptClassifier(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.bert = None
        self.bert_large = None

        if config.model == "bert":
            self.bert = AutoModel.from_pretrained("bert-base-uncased")
            self.classifier = nn.Linear(768, 1)
        else:
            self.bert_large = AutoModel.from_pretrained("bert-large-uncased")
            self.classifier = nn.Linear(1024, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):

        if self.bert:
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        else:
            outputs = self.bert_large(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

        cls_embedding = outputs.last_hidden_state[:,0,:]

        logits = self.classifier(cls_embedding)

        probabilities = self.sigmoid(logits)

        return probabilities

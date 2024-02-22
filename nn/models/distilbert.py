import torch
from torch import nn, Tensor
from transformers import DistilBertPreTrainedModel, DistilBertModel


class HuggingFaceDistilBertForClassification(DistilBertPreTrainedModel):
    def __init__(self, distilbert: DistilBertModel, num_labels: int = 2):
        self.num_labels = num_labels
        self.config = distilbert.config

        super().__init__(self.config)

        self.distilbert = DistilBertModel(self.config)
        self.pre_classifier = nn.Linear(self.config.dim, self.config.dim)
        self.classifier = nn.Linear(self.config.dim, self.num_labels)
        self.dropout = nn.Dropout(self.config.seq_classif_dropout)

        self.post_init()

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        input_ids = input_ids.to(torch.long)
        attention_mask = attention_mask.to(torch.long)

        outputs = self.distilbert(
            input_ids,
            attention_mask=attention_mask,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
        )

        hidden_state = outputs[0]  # (bs * num_choices, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs * num_choices, dim)
        pooled_output = self.pre_classifier(pooled_output)  # (bs * num_choices, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs * num_choices, dim)
        pooled_output = self.dropout(pooled_output)  # (bs * num_choices, dim)
        logits = self.classifier(pooled_output)

        return logits

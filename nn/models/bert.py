"""
All models should implement compute method for inference.

HF - huggingface
BFP - bert for pretraining
BP - bert pretrained
"""

import torch
from torch import nn, Tensor
from transformers.models.bert import BertPreTrainedModel, BertModel


class HuggingFaceBertForClassification(BertPreTrainedModel):
    def __init__(self, bert: BertModel, num_labels=2):
        self.num_labels = num_labels
        self.config = bert.config

        super().__init__(self.config)
        self.bert = bert

        classifier_dropout = (
            self.config.classifier_dropout
            if self.config.classifier_dropout is not None
            else self.config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self, input_ids: Tensor, token_type_ids: Tensor, attention_mask: Tensor
    ) -> Tensor:
        input_ids = input_ids.to(torch.long)
        token_type_ids = token_type_ids.to(torch.long)
        attention_mask = attention_mask.to(torch.long)

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
        )

        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)

        return self.classifier(pooled_output)

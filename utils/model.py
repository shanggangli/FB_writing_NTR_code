from typing import Text
import pandas as pd
import numpy as np

import copy
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer, AutoModel, AutoConfig


class TextModel(nn.Module):
    def __init__(self,model_name = None,num_labels = 1):
        super(TextModel,self).__init__()
        config = AutoConfig.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)# 768
        self.drop_out = nn.Dropout(0.1)
        self.drop_out1 = nn.Dropout(0.1)
        self.drop_out2 = nn.Dropout(0.2)
        self.drop_out3 = nn.Dropout(0.3)
        self.drop_out4 = nn.Dropout(0.4)
        self.drop_out5 = nn.Dropout(0.5)
        self.output = nn.Linear(config.hidden_size,num_labels)

        if 'deberta-v2-xxlarge' in model_name:
            self.model.embeddings.requires_grad_(False)
            self.model.encoder.layer[:24].requires_grad_(False) # 冻结24/48
        if 'deberta-v2-xlarge' in model_name:
            self.model.embeddings.requires_grad_(False)
            self.model.encoder.layer[:14].requires_grad_(False) # 冻结12/24
        
        if 'funnel-transformer-xlarge' in model_name:
            self.model.embeddings.requires_grad_(False)
            self.model.encoder.layer[:1].requires_grad_(False) # 冻结1/3

    def forward(self,input_ids,attention_mask,labels = None):
        if 'gpt' in self.model.name_or_path:
            emb = self.model(input_ids)[0]
        else:
            emb = self.model(input_ids,attention_mask)[0]

        preds1 = self.output(self.dropout1(emb))
        preds2 = self.output(self.dropout2(emb))
        preds3 = self.output(self.dropout3(emb))
        preds4 = self.output(self.dropout4(emb))
        preds5 = self.output(self.dropout5(emb))
        preds = (preds1 + preds2 + preds3 + preds4 + preds5) / 5

        logits = torch.softmax(preds,dim = -1)
        if labels is not None:
            loss = self.get_loss(preds,labels,attention_mask)
            return loss,logits
        else:
            return logits

    def get_loss(self,outputs,targets,attention_mask):
        loss_fct =nn.CrossEntropyLoss()

        active_loss = attention_mask.reshape(-1) == 1
        active_logits = outputs.reshape(-1,outputs.shape[-1])
        true_labels = targets.reshape(-1)
        idxs = np.where(active_loss.cpu().numpy()==1)[0]
        active_logits = active_logits[idxs]
        true_labels = true_labels[idxs].to(torch.long)

        loss = loss_fct(active_logits,true_labels)

        return loss





import os
from pathlib import Path
import torch
import torchvision.models as models
from torch import nn, optim
from src.entity.config_entity import PrepareBaseModelConfig


class PrepareBaseModel(nn.Module):
    def __init__(self, config):
        super(PrepareBaseModel, self).__init__()
        self.config = config
        self.params_freeze_all = self.config.params_freeze_all
        self.params_freeze_till = self.config.params_freeze_till  # Corrected this line
        self.model = self.get_base_model()
        self.full_model, self.optimizer, self.loss_fn = self.update_base_model()

    def get_base_model(self):
        # Load the pre-trained ResNet101 model
        model = models.resnet101(pretrained=self.config.params_weights)

        # If not including the fully connected top layer
        if not self.config.params_include_top:
            model = nn.Sequential(*list(model.children())[:-2])
        return model

    def _prepare_full_model(self, model, classes, freeze_all, freeze_till, learning_rate):
        if freeze_all:
            for param in model.parameters():
                param.requires_grad = False
        elif freeze_till is not None and freeze_till > 0:
            ct = 0
            for child in model.children():
                ct += 1
                if ct < freeze_till:
                    for param in child.parameters():
                        param.requires_grad = False

        if isinstance(model, nn.Sequential):
            in_features = 2048  # Beware reader: This is specific to ResNet101
            model = nn.Sequential(
                model,
                nn.Flatten(),
                nn.Linear(in_features, classes),
                nn.Softmax(dim=1)
            )
        else:
            # Replace the last fully connected layer
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, classes)
            model.add_module('softmax', nn.Softmax(dim=1))

        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        loss_fn = nn.CrossEntropyLoss()
        return model, optimizer, loss_fn

    def update_base_model(self):
        full_model, optimizer, loss_fn = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=self.config.params_freeze_all,
            freeze_till=self.config.params_freeze_till,
            learning_rate=self.config.params_learning_rate
        )
        return full_model, optimizer, loss_fn

    @staticmethod
    def save_model(path, model):
        torch.save(model.state_dict(), path)

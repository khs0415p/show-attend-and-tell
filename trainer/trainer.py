import torch
import torch.nn as nn
import pandas as pd
import torchvision.transforms as transforms

from .base import BaseTrainer
from utils.train_utils import get_dataloader



class Trainer(BaseTrainer):
    def __init__(self, config, device) -> None:
        super().__init__(config, device)

        self.bleu_score_list = []
        self.meteor_score_list = []

        # dataloaders
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            )
        transform = transforms.Compose([transforms.Resize((config.image_size, config.image_size)), transforms.ToTensor(), normalize])
        self.test_transform = transforms.Compose([transforms.Resize((config.encoded_image_size * 18, config.encoded_image_size * 18)), transforms.ToTensor()])
        self.dataloader, self.tokenizer = get_dataloader(config, transform) # {'train': dataloader, 'valid': dataloader}

        # main process
        self.rank_zero = True if not self.ddp or (self.ddp and device == 0) else False

        # initialize trainer
        self._init_trainer()

        # criterion
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)

    def regularization_loss(self, predictions, captions, all_attention_scores):
        """
        Args:
            predictions: (batch, seq, vocab size)
            captions: (batch, seq)
            all_attention_scores: (batch, seq, num_pixels)
        """
        loss = self.cross_entropy(predictions[:, :-1, :].reshape(-1, predictions.size(-1)), captions[:, 1:].reshape(-1))
        loss += self.config.regularization_lambda * ((1. - torch.sum(all_attention_scores, dim=2)) ** 2).mean()
        return loss

    def _training_step(self, model_inputs):
        """
        Args:
            model_inputs: data of batch
        Return:
            (Tensor loss): loss
        """
        # model_inputs / 'captions', 'images'
        encoder_output = self.encoder(model_inputs['images'])
        predictions, captions, all_attention_scores = self.decoder(encoder_output, model_inputs['captions'])
        loss = self.regularization_loss(predictions, model_inputs['captions'], all_attention_scores)

        self._backward_step(loss)

        return loss.item()


    @torch.no_grad()
    def _validation_step(self, model_inputs):
        """
        Args:
            model_inputs: data of batch
        Return:
            (Tensor loss): loss
        """
        encoder_output = self.encoder(model_inputs['images'])
        predictions, captions, all_attention_scores = self.decoder(encoder_output, model_inputs['captions'])
        loss = self.regularization_loss(predictions, model_inputs['captions'], all_attention_scores)

        return loss.item(), predictions

    @torch.no_grad()
    def _test_step(self, model_inputs):
        encoder_output = self.encoder(model_inputs['images'])
        predictions, captions, all_attention_scores = self.decoder(encoder_output, model_inputs['captions'])
        loss = self.regularization_loss(predictions, model_inputs['captions'], all_attention_scores)

        return loss.item(), predictions, all_attention_scores
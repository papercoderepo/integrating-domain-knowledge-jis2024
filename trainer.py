import logging

import torch
from torch import nn
from torch.utils import data
from tqdm import tqdm


class ModelTrainer:
    def __init__(
        self, model, criterion, optimizer, dataloader, device, epochs, scheduler
    ):
        logging.info("Initializing model trainer.")
        self.dataloader = dataloader
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.epochs = epochs
        self.scheduler = scheduler

    def train_model(self):
        for epoch in range(self.epochs):
            print(f"Training epoch {epoch+1}:")
            logging.info(f"Training epoch {epoch+1}:")
            self.train_epoch()
        return self.model

    def train_epoch(self):
        self.model.train()
        for inputs, labels in tqdm(self.dataloader):
            self.train_batch(inputs, labels)
        self.scheduler.step()
        # self.schedulers[1].step()

    def train_batch(self, inputs, labels):
        self.optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            loss = self.calculate_batch_loss(inputs, labels)
            logging.info(f"Current loss is {loss}.")
            self._backpropagate(loss)

    def calculate_batch_loss(self, inputs, labels):
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        # logging.info(labels)
        outputs = self.model(inputs)
        # logging.info(outputs)
        return self.criterion(outputs, labels)

    def _backpropagate(self, loss):
        loss.backward()
        self.optimizer.step()


class LossSumTrainer(ModelTrainer):

    # It's expected from dataloader to bring two lists, for inputs and outputs
    def __init__(
        self, model, criterion, optimizer, dataloader, device, epochs, scheduler
    ):
        super().__init__(
            model, criterion, optimizer, dataloader, device, epochs, scheduler
        )

    def train_epoch(self):
        self.model.train()
        for inputs, labels in tqdm(self.dataloader):
            self.train_batch(inputs, labels)
        self.scheduler.step()
        # self.schedulers[1].step()

    def train_batch(self, inputs_list, labels_list):
        self.optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            loss = self.__calculate_loss_sum(inputs_list, labels_list)
            logging.info(f"Current loss is {loss}.")
            self._backpropagate(loss)

    def __calculate_loss_sum(self, inputs_list, labels_list):
        loss = 0
        for inputs, labels in zip(inputs_list, labels_list):
            loss += self.calculate_batch_loss(inputs, labels)
        return loss

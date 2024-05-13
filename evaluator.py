import logging
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from tqdm import tqdm


class ModelEvaluator:
    def __init__(self, model, criterion, device):
        logging.info("Initializing model evaluator.")
        self.model = model.to(device)
        self.device = device
        self.criterion = criterion

    def _initialize_metrics(self):
        self.predictions = []
        self.labels = []
        self.entries = defaultdict(int)
        self.hits = defaultdict(int)
        self.total_loss = 0

    def evaluate(self, dataloader):
        self._initialize_metrics()
        with torch.set_grad_enabled(False):
            return self._evaluate(dataloader)

    def _evaluate(self, dataloader):
        for inputs, labels in tqdm(dataloader):
            self._run_batch(inputs, labels)
        self._normalize_metrics(dataloader)
        num_labels = 20  # dataloader.dataset.num_labels
        accuracies = self._calculate_accuracies(num_labels)

        f1_score_weight = 1 / np.unique(self.labels, return_counts=True)[1]
        return (
            accuracies,
            self.total_loss,
            np.mean(accuracies),
            f1_score(
                self.labels,
                self.predictions,
                average="weighted",
                sample_weight=f1_score_weight[self.labels],
            ),
        )

    def _run_batch(self, inputs, labels):
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        outputs = self.model(inputs)
        self._update_metrics(outputs, labels)

    def _update_metrics(self, outputs, labels):
        _, predictions = torch.max(outputs, 1)
        self._update_loss(outputs, labels)
        self._update_hits(predictions, labels)
        self.predictions.extend(list(map(lambda x: x.item(), predictions)))
        self.labels.extend(list(map(lambda x: x.item(), labels)))

    def _update_loss(self, outputs, labels):
        loss = self.criterion(outputs, labels)
        self.total_loss += loss.item()

    def _update_hits(self, predictions, labels):
        for prediction, label in zip(predictions, labels):
            if prediction == label:
                self.hits[label.item()] += 1
            self.entries[label.item()] += 1

    def _normalize_metrics(self, dataloader):
        self.total_loss /= len(dataloader)

    def _calculate_accuracies(self, num_labels):
        accuracies = np.zeros(num_labels)
        for label in range(len(accuracies)):
            hits, total = self.hits.get(label, 0), self.entries.get(label)
            if not total:
                continue
            accuracies[label] = hits / total
        return accuracies

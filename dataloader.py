from dataclasses import dataclass
from itertools import cycle

import numpy as np
import torch
from sklearn import preprocessing
from torch.utils import data
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, random_split

domain_size = []


class DataLoaderWrapper:
    def __init__(
        self,
        dataset,
        batch_size,
        balance_domains=False,
        balance_labels=False,
        shuffle_batches=False,
        random_sum=False,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.balance_domains = balance_domains
        self.balance_labels = balance_labels
        self.shuffle_batches = shuffle_batches
        self.random_sum = random_sum
        if self.balance_domains:
            self.batch_size = batch_size // len(dataset._domains)
        if self.random_sum:
            self.batch_size = batch_size // len(dataset._domains)
        self.dataloader = self.create_dataloader(self.dataset)
        self.label_enc = self.init_label_enc()
        self.generator = self.choose_generator()

    def init_label_enc(self):
        le = preprocessing.LabelEncoder()
        le.fit(
            [
                "f1",
                "f2",
                "f3",
                "f4",
                "f5",
                "f6",
                "f7",
                "f8",
                "f9",
                "f10",
                "m1",
                "m2",
                "m3",
                "m4",
                "m5",
                "m6",
                "m7",
                "m8",
                "m9",
                "m10",
            ]
        )
        return le

    def create_dataloader(self, dataset):
        if self.balance_domains:
            return self._create_domain_dataloaders(dataset)
        return self.create_single_dataloader(dataset)

    def create_single_dataloader(self, dataset):
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def _create_domain_dataloaders(self, dataset):
        return {
            domain: self.create_single_dataloader(domain_dataset)
            for domain, domain_dataset in dataset.domain_datasets.items()
        }

    def _create_sampler(self, dataset):
        # labels are balanced by default
        return None
        if not self.balance_labels:
            return None
        target = dataset.dataframe.label.values
        weight = 1 / np.unique(target, return_counts=True)[1]
        return WeightedRandomSampler(weight[target], len(target))

    def __len__(self):
        if self.balance_domains:
            num_iterations = len(max(self.dataloader.values(), key=len))
            return num_iterations
        return len(self.dataloader)

    def __iter__(self):
        return self.choose_generator()

    def __next__(self):
        return next(self.generator)

    def choose_generator(self):
        if self.balance_domains:
            return self._domains_iter()
        return self._get_generator()

    def _get_generator(self):
        for inputs, labels in self.dataloader:
            if self.shuffle_batches:
                inputs, labels = self._apply_shuffling(inputs, labels)
            labels = self.label_enc.transform(labels)
            labels = torch.from_numpy(labels)
            yield inputs, labels

    def _domains_iter(self):
        num_iterations = len(max(self.dataloader.values(), key=len))
        cycle_dataloaders = [
            cycle(dataloader) for dataloader in self.dataloader.values()
        ]
        for _ in range(num_iterations):
            yield self._get_domain_balanced_item(cycle_dataloaders)

    def _get_domain_balanced_item(self, cycle_dataloaders):
        iteration_inputs, iteration_labels = [], []
        for cycle_dataloader in cycle_dataloaders:
            inputs, labels = next(cycle_dataloader)
            labels = self.label_enc.transform(labels)
            labels = torch.from_numpy(labels)
            # print(inputs, labels)
            iteration_inputs.append(inputs)
            iteration_labels.append(labels)
        return torch.cat(iteration_inputs), torch.cat(iteration_labels)

    def _apply_shuffling(self, inputs, labels):
        idx = torch.randperm(inputs.shape[0])
        inputs = inputs[idx].view(inputs.size())
        labels = labels[idx].view(labels.size())
        return inputs, labels


class LossSumDataloader(DataLoaderWrapper):
    def __init__(
        self,
        dataset,
        batch_size,
        balance_domains=False,
        balance_labels=False,
        shuffle_batches=False,
    ):
        super().__init__(
            dataset,
            batch_size,
            balance_domains,
            balance_labels,
            shuffle_batches,
        )

    def _get_domain_balanced_item(self, cycle_dataloaders):
        iteration_inputs, iteration_labels = [], []
        for cycle_dataloader in cycle_dataloaders:
            inputs, labels = next(cycle_dataloader)
            labels = self.label_enc.transform(labels)
            labels = torch.from_numpy(labels)
            iteration_inputs.append(inputs)
            iteration_labels.append(labels)
        return iteration_inputs, iteration_labels

    def _apply_shuffling(self, inputs, labels):
        if self.shuffle_batches:
            shuffled_inputs, shuffled_labels = [], []
            for domain_inputs, domain_labels in zip(inputs, labels):
                domain_inputs, domain_labels = super()._apply_shuffling(
                    domain_inputs, domain_labels
                )
                shuffled_inputs.append(domain_inputs)
                shuffled_labels.append(domain_labels)
            return shuffled_inputs, shuffled_labels
        return inputs, labels


class RandomSumDataloader(DataLoaderWrapper):
    def __init__(
        self,
        dataset,
        batch_size,
        balance_domains=False,
        balance_labels=False,
        shuffle_batches=False,
        random_sum=False,
        seed=0,
    ):
        self.seed = seed

        super().__init__(
            dataset,
            batch_size,
            balance_domains,
            balance_labels,
            shuffle_batches,
            random_sum=random_sum,
        )
        self.dataloader = self.create_dataloader(self.dataset)
        self.label_enc = self.init_label_enc()
        self.generator = self.choose_generator()

    def choose_generator(self):
        if self.balance_domains:
            return self._domains_iter()
        return self._get_generator()

    def _domains_iter(self):
        num_iterations = len(max(self.dataloader.values(), key=len))
        cycle_dataloaders = [
            cycle(dataloader) for dataloader in self.dataloader.values()
        ]
        for _ in range(num_iterations):
            yield self._get_domain_balanced_item(cycle_dataloaders)

    def _get_domain_balanced_item(self, cycle_dataloaders):
        iteration_inputs, iteration_labels = [], []
        # print(cycle_dataloaders)
        for cycle_dataloader in cycle_dataloaders:
            inputs, labels = next(cycle_dataloader)
            # print(inputs, labels)
            labels = self.label_enc.transform(labels)
            labels = torch.from_numpy(labels)
            iteration_inputs.append(inputs)
            iteration_labels.append(labels)
        return iteration_inputs, iteration_labels

    def _apply_shuffling(self, inputs, labels):
        if self.shuffle_batches:
            shuffled_inputs, shuffled_labels = [], []
            for domain_inputs, domain_labels in zip(inputs, labels):
                domain_inputs, domain_labels = super()._apply_shuffling(
                    domain_inputs, domain_labels
                )
                shuffled_inputs.append(domain_inputs)
                shuffled_labels.append(domain_labels)
            return shuffled_inputs, shuffled_labels
        return inputs, labels

    def create_single_dataloader(self, dataset):
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def create_dataloader(self, dataset):
        if self.balance_domains:
            return self._create_domain_dataloaders(dataset)
        return self.create_single_dataloader(dataset)

    def _create_domain_dataloaders(self, dataset):
        fractions = [
            len(subset) / len(dataset) for subset in dataset.domain_datasets.values()
        ]

        splits = random_split(
            dataset, fractions, generator=torch.Generator().manual_seed(self.seed)
        )

        return {
            domain: self.create_single_dataloader(domain_dataset)
            for domain, domain_dataset in zip(dataset.domain_datasets.keys(), splits)
        }

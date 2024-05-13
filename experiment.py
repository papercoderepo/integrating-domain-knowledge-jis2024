import json
import logging
import random
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import numpy as np
import torch

from dataloader import DataLoaderWrapper, LossSumDataloader, RandomSumDataloader
from dataset import Folds, MultiDomainDataset
from evaluator import ModelEvaluator
from trainer import LossSumTrainer, ModelTrainer


@dataclass(kw_only=True)
class ExperimentConfig:
    seed: Any = None
    batch_size: Any = None
    epochs: Any = None
    device: Any = None
    architecture: Any = None
    lr: Any = None
    optimizer: Any = None
    criterion: Any = None
    lr_scheduler: Any = None
    domains: Any = None
    spectrogram_config: Any = None


class ExperimentType(Enum):

    SEQUENTIAL = auto()
    INVERSE_SEQUENTIAL = auto()
    SOUP = auto()
    BALANCED = auto()
    LOSS_SUM = auto()
    RANDOM_SUM = auto()

    def __repr__(self):
        cls_name = self.__class__.__name__
        return f"{cls_name}.{self.name}"


class ExperimentHandler:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.lr_scheduler = None

        self.test_domain_dataloaders = {}

        self.train_sequential_dataloaders = {}
        self.train_soup_dataloader = None
        self.train_balanced_dataloader = None
        self.train_loss_sum_dataloader = None
        self.train_random_dataloader = None

        self.experiment_results = {}

        self.experiment_map = {
            ExperimentType.SEQUENTIAL: self.sequential_experiment,
            ExperimentType.INVERSE_SEQUENTIAL: self.inv_sequential_experiment,
            ExperimentType.SOUP: self.soup_experiment,
            ExperimentType.BALANCED: self.balanced_experiment,
            ExperimentType.LOSS_SUM: self.loss_sum_experiment,
            ExperimentType.RANDOM_SUM: self.random_sum_experiment,
        }

    # assess initializing using singleDomainDataloader
    def initialize_single_dataloaders(self, fold):
        for domain in self.config.domains:
            dataset = MultiDomainDataset(
                [domain],
                fold,
                self.spectrogram_config,
            )
            dataloader_wrapper = DataLoaderWrapper(dataset, self.config.batch_size)
            if fold == Folds.TRAIN:
                self.train_sequential_dataloaders[domain] = dataloader_wrapper
            else:
                self.test_domain_dataloaders[domain] = dataloader_wrapper

    def initialize_soup_dataloader(self):
        dataset = MultiDomainDataset(
            self.config.domains,
            Folds.TRAIN,
            self.spectrogram_config,
        )
        dataloader_wrapper = DataLoaderWrapper(dataset, self.config.batch_size)
        self.train_soup_dataloader = dataloader_wrapper

    def initialize_random_dataloader(self):
        dataset = MultiDomainDataset(
            self.config.domains,
            Folds.TRAIN,
            self.spectrogram_config,
        )
        dataloader_wrapper = RandomSumDataloader(
            dataset,
            self.config.batch_size,
            balance_domains=True,
            balance_labels=False,
            shuffle_batches=True,
            seed=self.config.seed,
            random_sum=True,
        )
        self.train_random_dataloader = dataloader_wrapper

    def train_model(self, model, dataloader_wrapper):
        trainer = ModelTrainer(
            model,
            self.criterion,
            self.optimizer,
            dataloader_wrapper,
            self.config.device,
            self.config.epochs,
            self.lr_scheduler,
        )
        return trainer.train_model()

    def train_loss_sum_model(self, model, dataloader_wrapper):
        trainer = LossSumTrainer(
            model,
            self.criterion,
            self.optimizer,
            dataloader_wrapper,
            self.config.device,
            self.config.epochs,
            self.lr_scheduler,
        )
        return trainer.train_model()

    def evaluate_domains(self, model, train_type):
        evaluator = ModelEvaluator(model, self.criterion, self.config.device)
        experiment_acc = []
        experiment_f1 = []
        self.eval_logger.info(f"======================")
        self.eval_logger.info(f"Experiment: {train_type}")
        domain_amount = len(self.test_domain_dataloaders)

        for domain, dataloader in self.test_domain_dataloaders.items():
            print(f"Evaluating Model on {domain}:")
            logging.info(f"Evaluating model on {domain}.")
            self.eval_logger.info(f"Domain: {domain}")
            accuracies, loss, accuracy, f1_score = evaluator.evaluate(dataloader)
            experiment_acc.append(accuracy)
            experiment_f1.append(f1_score)
            metrics = {
                f"Label Accuracy {str(label).zfill(2)}": acc
                for label, acc in enumerate(accuracies)
            }
            print(f1_score)

            self.eval_logger.info(f"domain f1-score: {f1_score}")
            self.eval_logger.info(
                f"domain average acc: {sum(accuracies) / domain_amount}"
            )
            self.eval_logger.info(f"class metrics on {domain}: {metrics}")

        self.eval_logger.info(f"----general metrics----")
        self.eval_logger.info(
            f"experiment f1-score: {sum(experiment_f1) / domain_amount}"
        )
        self.eval_logger.info(f"experiment acc: {sum(experiment_acc) / domain_amount}")

        result = {
            "average f1-score": sum(experiment_f1) / domain_amount,
            "domain f1-score": {
                domain.name: score
                for domain, score in zip(
                    self.test_domain_dataloaders.keys(), experiment_f1
                )
            },
            "seed": self.config.seed
            # "average accuracy": ...
            # "average precision": ...
            # "average recall": ...
        }

        # implement jsonfy and write to results folder
        self.experiment_results[train_type.name] = result

    def sequential_experiment(self, model):
        self.initialize_single_dataloaders(Folds.TRAIN)
        for domain in self.config.domains:
            logging.info(f"Now training sequential experiment domain: {domain}")
            print(f"Training on domain: {domain}")
            model = self.train_model(model, self.train_sequential_dataloaders[domain])
        return model

    def inv_sequential_experiment(self, model):
        self.initialize_single_dataloaders(Folds.TRAIN)
        for domain in self.config.domains[::-1]:
            logging.info(f"Now training inverse sequential experiment domain: {domain}")
            print(f"Training on domain: {domain}")
            model = self.train_model(model, self.train_sequential_dataloaders[domain])
        return model

    def soup_experiment(self, model):
        self.initialize_soup_dataloader()
        logging.info("Now training soup experiment:")
        model = self.train_model(model, self.train_soup_dataloader)
        return model

    def initialize_balanced_dataloader(self):
        dataset = MultiDomainDataset(
            self.config.domains,
            Folds.TRAIN,
            self.spectrogram_config,
        )
        dataloader_wrapper = DataLoaderWrapper(
            dataset,
            self.config.batch_size,
            balance_domains=True,
            balance_labels=False,
            shuffle_batches=True,
        )
        self.train_balanced_dataloader = dataloader_wrapper

    def balanced_experiment(self, model):
        self.initialize_balanced_dataloader()
        logging.info("Now training balanced experiment:")
        model = self.train_model(model, self.train_balanced_dataloader)
        return model

    def initialize_loss_sum_dataloader(self):
        dataset = MultiDomainDataset(
            self.config.domains,
            Folds.TRAIN,
            self.spectrogram_config,
        )
        dataloader_wrapper = LossSumDataloader(
            dataset,
            self.config.batch_size,
            balance_domains=True,
            balance_labels=False,
            shuffle_batches=True,
        )
        self.train_loss_sum_dataloader = dataloader_wrapper

    def loss_sum_experiment(self, model):
        self.initialize_loss_sum_dataloader()
        logging.info("Now training loss sum experiment:")
        model = self.train_loss_sum_model(model, self.train_loss_sum_dataloader)
        return model

    def random_sum_experiment(self, model):
        self.initialize_random_dataloader()
        logging.info("Now training random sum experiment:")
        model = self.train_loss_sum_model(model, self.train_random_dataloader)
        return model

    def initialize_config(self):
        self.model = self.config.architecture(20).to(self.config.device)
        self.optimizer = self.config.optimizer(
            self.model.parameters(), lr=self.config.lr
        )
        self.criterion = self.config.criterion()
        self.lr_scheduler = self.config.lr_scheduler(self.optimizer, gamma=0.9)
        self.spectrogram_config = self.config.spectrogram_config()
        self.eval_logger = setup_logger("eval_logger", "results.log")

    def run_experiment(self, train_type):
        start_time = time.perf_counter()
        self.seed_experiment(self.config.seed)
        self.initialize_config()

        model = self.model
        train_routine = self.experiment_map[train_type]
        trained_model = train_routine(model)

        self.initialize_single_dataloaders(Folds.TEST)
        self.evaluate_domains(trained_model, train_type)
        end_time = time.perf_counter()
        self.experiment_results[train_type.name]["elapsed time"] = end_time - start_time

    def seed_experiment(self, seed):
        # cpu variables
        np.random.seed(seed)
        torch.manual_seed(seed)

        # python variables
        random.seed(seed)

        # cuda variables and config
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_logger(name, log_file, level=logging.INFO):
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(filename)s - %(message)s"
    )

    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

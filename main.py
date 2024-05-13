import json
import logging

import torch
from torch import nn, optim

from dataloader import DataLoaderWrapper
from dataset import (
    AudioSpectrogramConfig,
    DomainDataset,
    Domains,
    Folds,
    MultiDomainDataset,
)
from evaluator import ModelEvaluator
from experiment import ExperimentConfig, ExperimentHandler, ExperimentType
from model import CNNNetwork, ResNetWrapper
from trainer import ModelTrainer


def main():
    print("Attached container successfully, starting application...")
    logging.basicConfig(
        filename="execution.log",
        filemode="w",
        format="%(asctime)s - %(levelname)s - %(filename)s - %(message)s",
        level=logging.DEBUG,
    )

    SEED_START = 322
    REPETITIONS = 10

    for exp_i in range(REPETITIONS):
        logging.info(f"**Running REPETITION: {exp_i}.**")
        config = ExperimentConfig(
            seed=SEED_START + exp_i,
            batch_size=256,
            epochs=10,
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            architecture=ResNetWrapper,
            lr=1e-1,
            optimizer=optim.Adam,
            criterion=nn.CrossEntropyLoss,
            lr_scheduler=optim.lr_scheduler.ExponentialLR,
            domains=list(Domains),
            spectrogram_config=AudioSpectrogramConfig,
        )
        print(f"device is {config.device}")
        experiment = ExperimentHandler(config)

        for type in list(ExperimentType):
            logging.info(f"Running experiment: {type}.")
            experiment.run_experiment(type)

        result = experiment.experiment_results
        json_object = json.dumps(result, indent=4)
        with open(f"results/{str(config.seed)}.json", "w") as outfile:
            outfile.write(json_object)


if __name__ == "__main__":
    main()

import logging
import os
from dataclasses import dataclass
from enum import Enum, auto

import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset
from torchvision import transforms


class Domains(Enum):
    CLEAN = auto()
    IPAD_BALCONY1 = auto()
    IPAD_BEDROOM1 = auto()
    IPAD_CONFROOM1 = auto()
    IPAD_CONFROOM2 = auto()
    IPADFLAT_CONFROOM1 = auto()
    IPADFLAT_OFFICE1 = auto()
    IPAD_LIVINGROOM1 = auto()
    IPAD_OFFICE1 = auto()
    IPAD_OFFICE2 = auto()
    IPHONE_BALCONY1 = auto()
    IPHONE_BEDROOM1 = auto()
    IPHONE_LIVINGROOM1 = auto()
    PRODUCED = auto()

    def __repr__(self):
        cls_name = self.__class__.__name__
        return f"{cls_name}.{self.name}"


class Folds(Enum):
    TRAIN = auto()
    VALIDATION = auto()
    TEST = auto()


@dataclass
class AudioSpectrogramConfig:
    audio_size = 5
    target_sample_rate = 22050
    num_samples = target_sample_rate * audio_size

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=target_sample_rate, n_fft=1024, hop_length=512, n_mels=64
    )


class DomainDataset(Dataset):
    def __init__(self, domain: Domains, fold: Folds, spectrogram_config):
        self.domain = domain
        self.fold = fold
        self.directory = self.get_dataset_directory()
        self.dataframe = self.create_dataframe()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.spectrogram_config = spectrogram_config

    def get_dataset_directory(self):
        return os.path.join(
            "daps_split", self.domain.name.lower(), self.fold.name.lower()
        )

    def create_dataframe(self, save_to_file: bool = False) -> pd.DataFrame:

        dataframe = pd.DataFrame(columns=["Domain", "Fold", "File", "SpeakerID"])
        self.populate_dataframe(dataframe)
        logging.info(
            f"Creating dataframe for {self.directory}. Size is {len(dataframe)}."
        )

        if save_to_file:
            dataframe.to_csv(self.directory + "/metadata.csv")

        return dataframe

    def populate_dataframe(self, dataframe):
        for file in os.listdir(self.directory):  # [:200]:  # SAMPLE FOR TEST
            if file[-4:] != ".wav":
                continue

            speaker_id = file.split("_")[0]
            dataframe.loc[len(dataframe)] = [
                self.domain.name.lower(),
                self.fold.name.lower(),
                file,
                speaker_id,
            ]

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        _, _, file, label = self.dataframe.iloc[index]  # .values

        audio_path = os.path.join(self.directory, file)

        spectrogram = self.load_spectrogram(audio_path)
        transformed_spectrogram = self.apply_transformations(spectrogram)

        return transformed_spectrogram, label

    def load_spectrogram(self, audio_path):
        signal, sr = torchaudio.load(audio_path)
        signal.to(torch.double)
        signal = signal.to(self.device)
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        signal = signal.to(self.device)
        mel_spec = self.spectrogram_config.mel_spectrogram.to(self.device)
        signal = mel_spec(signal)
        return signal

    def _cut_if_necessary(self, signal):
        if signal.shape[1] > self.spectrogram_config.num_samples:
            signal = signal[:, : self.spectrogram_config.num_samples]
        return signal

    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.spectrogram_config.num_samples:
            num_missing_samples = self.spectrogram_config.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

    def _resample_if_necessary(self, signal, sr):
        if sr != self.spectrogram_config.target_sample_rate:
            resampler = torchaudio.transforms.Resample(
                sr, self.spectrogram_config.target_sample_rate
            )
            resampler.to(self.device)
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def apply_transformations(self, spectrogram):
        return spectrogram  # test
        transform = transforms.Compose([transforms.Resize(224)])
        return transform(spectrogram)


class MultiDomainDataset(Dataset):
    def __init__(self, domains, fold, spectrogram_config):
        self.fold = fold
        self._domains = domains
        self.spectrogram_config = spectrogram_config
        self.domain_datasets = self._create_domain_datasets()
        self.dataframe = self._create_merged_dataframe()

    @property
    def domains(self):
        return [domain.name for domain in self._domains]

    def _create_domain_datasets(self):
        datasets = dict()
        for domain in self._domains:
            datasets[domain.name.lower()] = DomainDataset(
                domain, self.fold, self.spectrogram_config
            )
        return datasets

    def _create_merged_dataframe(self):
        domain_dataframes = [
            dataset.dataframe for dataset in self.domain_datasets.values()
        ]
        dataframe = pd.concat(domain_dataframes)
        return dataframe.sample(frac=1).reset_index(drop=True)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        domain, _, file, label = self.dataframe.iloc[index]

        directory = self.domain_datasets[domain].get_dataset_directory()
        audio_path = os.path.join(directory, file)

        spectrogram = self.domain_datasets[domain].load_spectrogram(audio_path)
        transformed_spectrogram = self.domain_datasets[domain].apply_transformations(
            spectrogram
        )

        return transformed_spectrogram, label

import torch
import lightning as L
from torch.utils.data import DataLoader, random_split

from src.dataloaders.dataset import AudioDataset


class AudioDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset_path: str,
        sample_rate: int = 24000,
        crop_length: int = 16384,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        n_mels: int = 100,
        fmin: float = 0.0,
        fmax: float = 12000.0,
        batch_size: int = 16,
        num_workers: int = 4,
        val_split: float = 0.05,
    ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage=None):
        dataset = AudioDataset(
            dataset_path=self.hparams.dataset_path,
            sample_rate=self.hparams.sample_rate,
            crop_length=self.hparams.crop_length,
            n_fft=self.hparams.n_fft,
            hop_length=self.hparams.hop_length,
            win_length=self.hparams.win_length,
            n_mels=self.hparams.n_mels,
            fmin=self.hparams.fmin,
            fmax=self.hparams.fmax,
        )
        val_len = max(1, int(len(dataset) * self.hparams.val_split))
        train_len = len(dataset) - val_len
        self.train_ds, self.val_ds = random_split(
            dataset,
            [train_len, val_len],
            generator=torch.Generator().manual_seed(42),
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            pin_memory=True,
            persistent_workers=self.hparams.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            persistent_workers=self.hparams.num_workers > 0,
        )

from typing import List, Optional

import torch
import torch.nn.functional as F
import torchaudio
import lightning as L
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality

from src.discriminator import Discriminator
from src.model import Vocos


# -------------------------
# Loss functions
# -------------------------
def d_hinge_loss(real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    return torch.mean(F.relu(1.0 - real)) + torch.mean(F.relu(1.0 + fake))


def g_hinge_loss(fake: torch.Tensor) -> torch.Tensor:
    return torch.mean(F.relu(1.0 - fake))


# -------------------------
# Lightning Module
# -------------------------
class AudioPLModule(L.LightningModule):
    def __init__(
        self,
        generator: Optional[Vocos] = None,
        discriminator: Optional[Discriminator] = None,
        sample_rate: int = 24000,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        n_mels: int = 100,
        fmin: float = 0.0,
        fmax: float = 12000.0,
        lr: float = 2e-4,
        betas: List[float] = [0.9, 0.999],
        weight_decay: float = 0.01,
        lambda_mel: float = 45.0,
        lambda_feat: float = 2.0,
        lambda_adv: float = 1.0,
        grad_clip: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["generator", "discriminator"])

        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay
        self.lambda_mel = lambda_mel
        self.lambda_feat = lambda_feat
        self.lambda_adv = lambda_adv
        self.grad_clip = grad_clip

        self.model = generator if generator is not None else Vocos()
        self.disc = discriminator if discriminator is not None else Discriminator()
        self.automatic_optimization = False

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
            f_min=fmin,
            f_max=fmax,
        )

        # Hann window registered as buffer so it moves with the model to CUDA
        self.register_buffer("window", torch.hann_window(win_length))

        self.pesq_metric = PerceptualEvaluationSpeechQuality(fs=16000, mode="wb")

    # -------------------------
    # Helpers
    # -------------------------
    def _istft(
        self, re: torch.Tensor, im: torch.Tensor, length: int = None
    ) -> torch.Tensor:
        """Complex STFT -> waveform [B, T]"""
        spec = torch.complex(re, im)
        return torch.istft(
            spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            length=length,
        )

    def _mel(self, wav: torch.Tensor) -> torch.Tensor:
        """Waveform [B, T] or [B, 1, T] -> log-mel [B, n_mels, T_mel]"""
        if wav.dim() == 3:
            wav = wav.squeeze(1)
        return torch.log(torch.clamp(self.mel_transform(wav), min=1e-5))

    def _pesq(self, wav1: torch.Tensor, wav2: torch.Tensor) -> torch.Tensor:
        """Compute PESQ score between two waveforms [B, T] or [B, 1, T]"""
        if wav1.dim() == 3:
            wav1 = wav1.squeeze(1)
        if wav2.dim() == 3:
            wav2 = wav2.squeeze(1)

        wav1 = torchaudio.functional.resample(wav1, self.sample_rate, 16000)
        wav2 = torchaudio.functional.resample(wav2, self.sample_rate, 16000)

        return self.pesq_metric(wav1, wav2)

    # -------------------------
    # Optimizers
    # -------------------------
    def configure_optimizers(self):
        betas = tuple(self.betas)
        opt_g = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            betas=betas,
            weight_decay=self.weight_decay,
        )
        opt_d = torch.optim.AdamW(
            self.disc.parameters(),
            lr=self.lr,
            betas=betas,
            weight_decay=self.weight_decay,
        )

        # cosine decay over total training steps
        T_max = self.trainer.estimated_stepping_batches
        sch_g = CosineAnnealingLR(opt_g, T_max=T_max)
        sch_d = CosineAnnealingLR(opt_d, T_max=T_max)

        return [opt_g, opt_d], [sch_g, sch_d]

    # -------------------------
    # Training
    # -------------------------
    def training_step(self, batch, batch_idx):
        opt_g, opt_d = self.optimizers()
        sch_g, sch_d = self.lr_schedulers()

        mel, wav = batch  # mel: [B, n_mels, T_mel], wav: [B, 1, crop_length]
        wav_len = wav.shape[-1]

        # generator: mel -> complex STFT -> waveform
        re, im = self.model(mel)
        wav_fake = self._istft(re, im, length=wav_len)  # [B, T]
        wav_fake_3d = wav_fake.unsqueeze(1)  # [B, 1, T]

        # ---- Discriminator step ----
        opt_d.zero_grad()
        r_out, f_out, _, _ = self.disc(wav, wav_fake_3d.detach())
        # r_out / f_out: list[disc] of list[sub_disc] of Tensor
        loss_d = sum(
            d_hinge_loss(r, f)
            for r_list, f_list in zip(r_out, f_out)
            for r, f in zip(r_list, f_list)
        )
        self.manual_backward(loss_d)
        self.clip_gradients(
            opt_d,
            gradient_clip_val=self.grad_clip,
            gradient_clip_algorithm="norm",
        )
        opt_d.step()
        sch_d.step()

        # ---- Generator step ----
        opt_g.zero_grad()
        r_out, f_out, fmap_r, fmap_f = self.disc(wav, wav_fake_3d)

        loss_adv = sum(g_hinge_loss(f) for f_list in f_out for f in f_list)

        # fmap_r / fmap_f: list[disc] of list[sub_disc] of list[layer] of Tensor
        loss_feat = sum(
            F.l1_loss(r.detach(), f)
            for disc_r, disc_f in zip(fmap_r, fmap_f)
            for sub_r, sub_f in zip(disc_r, disc_f)
            for r, f in zip(sub_r, sub_f)
        )

        # mel reconstruction: L1 on log-mel (Lmel in paper)
        mel_fake = self._mel(wav_fake)
        t = min(mel.shape[-1], mel_fake.shape[-1])
        loss_mel = F.l1_loss(mel[..., :t], mel_fake[..., :t])

        loss_g = (
            self.lambda_adv * loss_adv
            + self.lambda_feat * loss_feat
            + self.lambda_mel * loss_mel
        )
        self.manual_backward(loss_g)
        self.clip_gradients(
            opt_g,
            gradient_clip_val=self.grad_clip,
            gradient_clip_algorithm="norm",
        )
        opt_g.step()
        sch_g.step()

        train_pesq = self._pesq(wav_fake, wav)

        self.log_dict(
            {
                "train/loss_g": loss_g,
                "train/loss_d": loss_d,
                "train/loss_adv": loss_adv,
                "train/loss_feat": loss_feat,
                "train/loss_mel": loss_mel,
                "train/pesq": train_pesq,
            },
            prog_bar=True,
            on_step=True,
            on_epoch=False,
        )

    # -------------------------
    # Validation
    # -------------------------
    def validation_step(self, batch, batch_idx):
        mel, wav = batch
        wav_len = wav.shape[-1]

        re, im = self.model(mel)
        wav_fake = self._istft(re, im, length=wav_len)
        wav_fake_3d = wav_fake.unsqueeze(1)

        mel_fake = self._mel(wav_fake)
        t = min(mel.shape[-1], mel_fake.shape[-1])
        val_mel = F.l1_loss(mel[..., :t], mel_fake[..., :t])

        _, f_out, fmap_r, fmap_f = self.disc(wav, wav_fake_3d)

        val_adv = sum(g_hinge_loss(f) for f_list in f_out for f in f_list)
        val_feat = sum(
            F.l1_loss(r.detach(), f)
            for disc_r, disc_f in zip(fmap_r, fmap_f)
            for sub_r, sub_f in zip(disc_r, disc_f)
            for r, f in zip(sub_r, sub_f)
        )

        val_pesq = self._pesq(wav_fake, wav)

        self.log_dict(
            {
                "val/loss_mel": val_mel,
                "val/loss_adv": val_adv,
                "val/loss_feat": val_feat,
                "val/pesq": val_pesq,
            },
            on_epoch=True,
        )
        return val_mel

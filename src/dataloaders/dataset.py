import os
import random
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset


class AudioDataset(Dataset):
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
    ):
        valid_extensions = (".wav", ".mp3", ".ogg", ".flac")
        self.files = sorted(
            os.path.join(root, filename)
            for root, _, files in os.walk(dataset_path)
            for filename in files
            if filename.lower().endswith(valid_extensions)
        )
        self.sr = sample_rate
        self.crop_length = crop_length

        self.mel_tf = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
            f_min=fmin,
            f_max=fmax,
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # wav, sr = torchaudio.load(self.files[idx])
        wav, sr = sf.read(self.files[idx], dtype="float32")
        wav = torch.from_numpy(wav).unsqueeze(0)  # [1, T]

        if sr != self.sr:
            wav = torchaudio.functional.resample(wav, sr, self.sr)

        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        # random gain augmentation: -1 to -6 dBFS (paper §3)
        gain_db = random.uniform(-6.0, -1.0)
        wav = wav * (10.0 ** (gain_db / 20.0))

        # random crop / pad to fixed length
        length = wav.shape[-1]
        if length < self.crop_length:
            wav = torch.nn.functional.pad(wav, (0, self.crop_length - length))
        else:
            start = random.randint(0, length - self.crop_length)
            wav = wav[:, start : start + self.crop_length]

        mel = self.mel_tf(wav)  # [1, n_mels, T]
        mel = torch.log(torch.clamp(mel, min=1e-5))

        return mel.squeeze(0), wav  # [n_mels, T_mel], [1, crop_length]

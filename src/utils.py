import os
import pathlib
import sys

import soundfile as sf
import torch
import torchaudio
from safetensors.torch import load_file
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.model import Vocos
from src.pretraining import AudioPLModule


# Find audio files from file or directory
def find_audio_files(path):
    exts = (".wav", ".mp3", ".ogg", ".flac")

    if os.path.isfile(path) and path.endswith(exts):
        return [path]

    if os.path.isdir(path):
        files = []
        for r, _, fs in os.walk(path):
            files += [os.path.join(r, f) for f in fs if f.endswith(exts)]
        return sorted(files)

    raise FileNotFoundError(path)


# Load Vocos model from checkpoint
def load_model(path, device, config):
    ext = pathlib.Path(path).suffix.lower()

    # safetensors or torch checkpoint
    if ext == ".safetensors":
        ckpt = load_file(path)
    else:
        ckpt = torch.load(path, map_location=device)

        # full Lightning checkpoint
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state = ckpt["state_dict"]

            # detect LightningModule (AudioPLModule)
            is_lightning = any("model." in k or "disc." in k for k in state.keys())
            if is_lightning:
                print(f"Loading Lightning checkpoint from {path}")
                pl_model = AudioPLModule.load_from_checkpoint(path, map_location=device)
                return pl_model.model.to(device).eval()

    # plain Vocos state_dict
    model = Vocos(**config)
    model.load_state_dict(ckpt, strict=True)
    return model.to(device).eval()


# Load and normalize waveform
def load_audio(path, target_sr):
    wav, sr = sf.read(path, dtype="float32")  # [T]
    wav = torch.from_numpy(wav).unsqueeze(0)  # [1, T]
    # wav, sr = torchaudio.load(path)

    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)

    wav = wav.mean(0, keepdim=True) if wav.shape[0] > 1 else wav
    return wav.unsqueeze(0) if wav.dim() == 1 else wav


def torch_mel(cfg, device):
    to_mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=cfg.sr,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop,
        win_length=cfg.win,
        n_mels=cfg.n_mels,
        f_min=cfg.fmin,
        f_max=cfg.fmax,
    ).to(device)

    window = torch.hann_window(cfg.win, device=device)

    return to_mel, window


# ISTFT from real/imag
def istft(re, im, cfg, window, length, device):
    spec = torch.complex(re, im)
    window = window.to(device)
    return torch.istft(
        spec,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop,
        win_length=cfg.win,
        window=window,
        length=length,
    )


# PESQ computation
def pesq_score(self, wav1: torch.Tensor, wav2: torch.Tensor) -> torch.Tensor:
    """Compute PESQ score between two waveforms [B, T] or [B, 1, T]"""
    pesq_metric = PerceptualEvaluationSpeechQuality(fs=16000, mode="wb")

    try:
        if wav1.dim() == 3:
            wav1 = wav1.squeeze(1)
        if wav2.dim() == 3:
            wav2 = wav2.squeeze(1)

        wav1 = torchaudio.functional.resample(wav1, self.sample_rate, 16000)
        wav2 = torchaudio.functional.resample(wav2, self.sample_rate, 16000)

        return pesq_metric(wav1, wav2)

    except Exception:
        return torch.tensor(0.0, device=wav1.device)


def export_onnx(model, cfg, out_path, device):
    model = model.to(device).eval()

    # [1, n_mels, frames]
    dummy = torch.randn(1, cfg.n_mels, 123, device=device)

    torch.onnx.export(
        model,
        (dummy,),
        out_path,
        input_names=["mel"],
        output_names=["real", "imag"],
        dynamic_axes={
            "mel": {2: "frames"},
            "real": {2: "frames"},
            "imag": {2: "frames"},
        },
        opset_version=17,
        dynamo=True,
    )

    print(f"[ONNX] exported -> {out_path}")

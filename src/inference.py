import argparse
import os
import sys

import soundfile as sf
import torch
import torchaudio
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
    ckpt = torch.load(path, map_location=device)

    # Case 1: full Lightning checkpoint
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]

        # detect LightningModule (AudioPLModule)
        is_lightning = any("model." in k or "disc." in k for k in state.keys())
        if is_lightning:
            pl_model = AudioPLModule.load_from_checkpoint(path, map_location=device)
            return pl_model.model.to(device).eval()

    # Case 2: plain Vocos state_dict
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


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--model", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--out", default="results/inference")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    p.add_argument("--sr", type=int, default=24000)
    p.add_argument("--n_fft", type=int, default=1024)
    p.add_argument("--hop", type=int, default=256)
    p.add_argument("--win", type=int, default=1024)
    p.add_argument("--n_mels", type=int, default=100)
    p.add_argument("--fmin", type=float, default=0.0)
    p.add_argument("--fmax", type=float, default=12000.0)

    cfg = p.parse_args()

    os.makedirs(cfg.out, exist_ok=True)
    device = torch.device(cfg.device)

    vocos_config = {
        "in_channels": cfg.n_mels,
        "channels": 512,
        "hidden_channels": 1536,
        "out_channels": cfg.n_fft + 2,  # 2*(n_fft//2+1)
        "num_layers": 8,
    }

    model = load_model(cfg.model, device, vocos_config)

    # Mel spectrogram extractor
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

    files = find_audio_files(cfg.input)
    total = 0.0

    for f in files:
        wav = load_audio(f, cfg.sr).to(device)
        ref = wav.squeeze(0)

        mel = to_mel(wav).to(device)

        # Inference
        with torch.no_grad():
            re, im = model(mel)
            out = istft(re, im, cfg, window, wav.shape[-1], device).squeeze(0)

        print(f"Input: {f}, Ref shape: {ref.shape}, Out shape: {out.shape}")

        score = pesq_score(ref, out.cpu(), cfg.sr)
        total += score
        print(f"{f}: PESQ={score:.4f}")

        out_path = os.path.join(cfg.out, os.path.basename(f).replace(".", "_recon."))
        sf.write(out_path, out.cpu().numpy(), cfg.sr)

    print(f"Avg PESQ: {total / len(files):.4f}")


if __name__ == "__main__":
    main()

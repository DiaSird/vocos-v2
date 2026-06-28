import argparse
import os
import sys

import soundfile as sf
import torch
import torchaudio
from pesq import pesq
from torch.package import PackageImporter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.model import Vocos


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
def load_model(path, device):
    if path.endswith(".pt"):
        try:
            m = PackageImporter(path).load_pickle("", "Vocos")
            return m.to(device).eval()
        except Exception:
            pass

    ckpt = torch.load(path, map_location=device)

    if isinstance(ckpt, torch.nn.Module):
        return ckpt.to(device).eval()

    if isinstance(ckpt, dict):
        if "model" in ckpt and isinstance(ckpt["model"], torch.nn.Module):
            return ckpt["model"].to(device).eval()

        state = ckpt.get("state_dict", ckpt)
        state = {k.replace("model.", ""): v for k, v in state.items()}

        model = Vocos()
        model.load_state_dict(state)
        return model.to(device).eval()

    raise ValueError("Unsupported checkpoint format")


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
def istft(re, im, cfg, window, length):
    spec = torch.complex(re, im)
    return torch.istft(
        spec,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop,
        win_length=cfg.win,
        window=window,
        length=length,
    )


# PESQ computation
def pesq_score(ref, deg, sr):
    ref = ref.cpu().numpy().astype("float32")
    deg = deg.cpu().numpy().astype("float32")

    l = min(ref.shape[-1], deg.shape[-1])
    mode = "wb" if sr >= 16000 else "nb"

    return pesq(sr, ref[..., :l], deg[..., :l], mode)


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

    model = load_model(cfg.model, device)

    # Mel spectrogram extractor
    mel = torchaudio.transforms.MelSpectrogram(
        cfg.sr, cfg.n_fft, cfg.hop, cfg.win, cfg.n_mels, cfg.fmin, cfg.fmax
    )

    window = torch.hann_window(cfg.win, device=device)

    files = find_audio_files(cfg.input)
    total = 0.0

    for f in files:
        wav = load_audio(f, cfg.sr).to(device)
        ref = wav.squeeze(0)

        # Forward pass
        m = mel(wav).unsqueeze(0).to(device)

        with torch.inference_mode():
            re, im = model(m)
            out = istft(re, im, cfg, window, wav.shape[-1]).squeeze(0)

        score = pesq_score(ref, out.cpu(), cfg.sr)
        total += score

        out_path = os.path.join(cfg.out, os.path.basename(f).replace(".", "_recon."))
        sf.write(out_path, out.unsqueeze(0).cpu().numpy(), cfg.sr)

        print(f"{f}: PESQ={score:.4f}")

    print(f"Avg PESQ: {total / len(files):.4f}")


if __name__ == "__main__":
    main()

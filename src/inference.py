import argparse
import os
import pathlib
import sys

import soundfile as sf
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import (
    find_audio_files,
    load_model,
    load_audio,
    pesq_score,
    istft,
    export_onnx,
    torch_mel,
)


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--model", required=True)
    p.add_argument("--input", default="tests/000001_260.wav")
    p.add_argument("--out", default="results")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    p.add_argument("--sr", type=int, default=24000)
    p.add_argument("--n_fft", type=int, default=1024)
    p.add_argument("--hop", type=int, default=256)
    p.add_argument("--win", type=int, default=1024)
    p.add_argument("--n_mels", type=int, default=100)
    p.add_argument("--fmin", type=float, default=0.0)
    p.add_argument("--fmax", type=float, default=12000.0)
    p.add_argument("--export_onnx", action="store_true")

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

    if cfg.model.endswith(".onnx"):
        raise ValueError("ONNX model is not supported in PyTorch inference script")

    if cfg.export_onnx:
        onnx_path = str(pathlib.Path(cfg.model).with_suffix(".onnx"))
        export_onnx(model, cfg, onnx_path, device)

    # Mel spectrogram extractor
    to_mel, window = torch_mel(cfg, device)

    files = find_audio_files(cfg.input)
    total = 0.0

    for f in files:
        wav = load_audio(f, cfg.sr).to(device)
        ref = wav.squeeze(0)

        mel = to_mel(wav).to(device)
        mel = torch.log(torch.clamp(mel, min=1e-5))

        # Inference
        with torch.no_grad():
            re, im = model(mel)
            out = istft(re, im, cfg, window, wav.shape[-1], device).squeeze(0)

        print(f"Input: {f}, Ref shape: {ref.shape}, Out shape: {out.shape}")

        score = pesq_score(ref, out.cpu(), sr=cfg.sr)
        total += score
        print(f"{f}: PESQ={score:.4f}")

        out_path = os.path.join(cfg.out, os.path.basename(f).replace(".", "_recon."))
        sf.write(out_path, out.cpu().numpy(), cfg.sr)

    print(f"Avg PESQ: {total / len(files):.4f}")


if __name__ == "__main__":
    main()

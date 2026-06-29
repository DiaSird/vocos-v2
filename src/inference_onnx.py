import argparse
import os

import numpy as np
import soundfile as sf
import torch
import onnxruntime as ort

from utils import find_audio_files, load_audio, pesq_score, istft, torch_mel


def load_onnx(path, device):
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if device == "cuda"
        else ["CPUExecutionProvider"]
    )

    return ort.InferenceSession(path, providers=providers)


def infer(session, mel):
    outputs = session.run(
        None,
        {"mel": mel.cpu().numpy().astype(np.float32)},
    )

    re = torch.from_numpy(outputs[0])
    im = torch.from_numpy(outputs[1])

    return re, im


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--model", required=True)  # .onnx
    p.add_argument("--input", required=True)
    p.add_argument("--out", default="results")

    p.add_argument("--sr", type=int, default=24000)
    p.add_argument("--n_fft", type=int, default=1024)
    p.add_argument("--hop", type=int, default=256)
    p.add_argument("--win", type=int, default=1024)

    p.add_argument("--n_mels", type=int, default=100)
    p.add_argument("--fmin", type=float, default=0.0)
    p.add_argument("--fmax", type=float, default=12000.0)

    cfg = p.parse_args()

    os.makedirs(cfg.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load ONNX
    session = load_onnx(cfg.model, device)

    # audio files
    files = find_audio_files(cfg.input)

    total = 0.0
    for f in files:
        wav, sr = load_audio(f, cfg.sr)
        wav = wav.to(device)

        # reference
        ref = wav.squeeze(0)

        # mel
        to_mel, window = torch_mel(cfg, device)

        # ONNX inference
        mel = to_mel(wav)
        re, im = infer(session, mel)

        # ISTFT
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

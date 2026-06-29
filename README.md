# Vocos-V2

Vocos-V2 is a neural vocoder based on the Vocos architecture with a ConvNeXt V2 backbone. It predicts the complex STFT spectrum directly from a log Mel spectrogram and reconstructs the waveform using ISTFT.

## Features

* ConvNeXt V2 backbone
* Global Response Normalization (GRN)
* Direct complex STFT prediction
* Compatible with pretrained Vocos weights
* Supports both `.pt` and `.safetensors` checkpoints
* ONNX export support

---

## Model

```
Log Mel Spectrogram
        │
        ▼
    Conv1d
        │
        ▼
 ConvNeXt V2 × N
        │
        ▼
 LayerNorm
        │
        ▼
    1×1 Conv
        │
        ▼
Real + Imaginary STFT
        │
        ▼
      ISTFT
```

Default configuration:

| Parameter       | Value |
| --------------- | ----: |
| Mel bins        |   100 |
| Channels        |   512 |
| Hidden channels |  1536 |
| ConvNeXt blocks |     8 |
| FFT size        |  1024 |

---

## Installation

```bash
git clone https://github.com/<user>/vocos-v2.git
cd vocos-v2

python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

---

## Training

Edit the configuration file and start training.

```bash
python src/train.py fit --config configs/config_template.yaml
```

The best checkpoint will be saved automatically.

---

## Inference

```bash
python src/inference.py \
    --model checkpoints/vocos_best.safetensors \
    --input sample.wav
```

`.pt`, `.pth`, `.ckpt`, and `.safetensors` checkpoints are supported.

---

## Pretrained Weights

You can initialize the model from the official Hugging Face Vocos checkpoint.

- [Vocos-mel-24khz](https://huggingface.co/hf-audio/vocos-mel-24khz): MIT Licence

```
checkpoints/
└── model.safetensors
```

Convert the checkpoint:

```bash
python src/finetuning.py
```

This converts the original checkpoint into the Vocos-V2 format.

---

## ONNX Export (Optional)

```bash
python src/inference.py \
    --model checkpoints/vocos_best.safetensors \
    --input sample.wav \
    --export-onnx
```

Example output:

```
vocos-v2.onnx
```

## ONNX Inference (Optional)

```bash
python src/inference_onnx.py --model vocos-v2.onnx --input sample.wav
```

---

## Project Structure

```
.
├── configs/
├── checkpoints/
├── src/
│   ├── train.py
│   ├── inference.py
│   ├── export_onnx.py
│   ├── convert_checkpoint.py
│   └── model.py
├── requirements.txt
└── README.md
```

---

## Checkpoint Format

Training automatically saves checkpoints in SafeTensors format.

```
checkpoints/
├── vocos_best.safetensors
└── last.safetensors
```

---

## License

[MIT](LICENSE)

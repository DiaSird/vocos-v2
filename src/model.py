import torch
import torch.nn as nn


class GRN1d(nn.Module):
    """
    Global Response Normalization 1D
    """

    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1))
        self.eps = eps

    def forward(self, x):
        # x: [B, C, T]
        gx = torch.norm(x, p=2, dim=2, keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)

        return x * (self.gamma * nx + 1) + self.beta


class ConvNeXt1dV2(nn.Module):
    """
    ConvNeXt1dV2 block with GRN:
    https://github.com/facebookresearch/ConvNeXt-V2/blob/main/models/convnextv2.py
    """

    def __init__(self, channels: int, hidden_channels: int, layer_scale: float = 1.0):
        super().__init__()

        # depthwise conv1d
        self.dw_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=7,
            padding=3,
            groups=channels,
        )
        self.norm = nn.LayerNorm(channels)

        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden_channels),  # pointwise/1x1 convs
            nn.GELU(),
            nn.Linear(hidden_channels, channels),
        )
        self.grn = GRN1d(channels)

        self.layer_scale = nn.Parameter(torch.full((channels,), layer_scale))

    def forward(self, x):
        residual = x

        x = self.dw_conv(x)

        x = x.transpose(1, 2)  # [B, T, C]
        x = self.norm(x)
        x = self.mlp(x)
        x = self.grn(x.transpose(1, 2)).transpose(1, 2)
        x = x * self.layer_scale
        x = x.transpose(1, 2)

        return residual + x


class Vocos(nn.Module):
    def __init__(
        self,
        in_channels: int = 100,
        channels: int = 512,
        hidden_channels: int = 1536,
        out_channels: int = 1026,
        num_layers: int = 8,
    ):
        super().__init__()

        self.pad = nn.ReflectionPad1d((1, 0))
        self.in_conv = nn.Conv1d(in_channels, channels, kernel_size=7, padding=3)
        self.norm = nn.LayerNorm(channels)

        layer_scale = 1.0 / num_layers
        self.layers = nn.Sequential(
            *[
                ConvNeXt1dV2(channels, hidden_channels, layer_scale)
                for _ in range(num_layers)
            ]
        )
        self.norm_last = nn.LayerNorm(channels)

        self.out_conv = nn.Conv1d(
            channels,
            out_channels,
            kernel_size=1,
        )

    def forward(self, x):
        x = self.pad(x)
        x = self.in_conv(x)
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)

        x = self.layers(x)
        x = self.norm_last(x.transpose(1, 2)).transpose(1, 2)

        x = self.out_conv(x)

        # x = [B, 2F, T]
        log_mag, phase = torch.chunk(x, chunks=2, dim=1)

        mag = torch.exp(log_mag)
        mag = torch.clamp(mag, max=1e2)

        re = mag * torch.cos(phase)
        im = mag * torch.sin(phase)

        return re, im


if __name__ == "__main__":
    import torchaudio
    import soundfile as sf

    # parameters
    audio_seconds = 2
    config_stft = {
        "sample_rate": 16000,
        "n_fft": 1024,
        "hop_length": 256,
        "win_length": 1024,
        "n_mels": 128,
        "power": 1.0,
    }

    # dummy wave
    sr = int(config_stft["sample_rate"])
    waveform = torch.randn(1, sr * audio_seconds)

    # mel-spectrogram
    to_mel = torchaudio.transforms.MelSpectrogram(**config_stft)
    mel = to_mel(waveform)
    mel = torch.log(torch.clamp(mel, min=1e-5))
    print("Mel:", mel.shape)

    # model
    model = Vocos(
        in_channels=128,
        channels=512,
        hidden_channels=1536,
        out_channels=config_stft["n_fft"] + 2,  # 2*(n_fft//2+1)
        num_layers=8,
    )
    model.eval()

    with torch.no_grad():
        re, im = model(mel)

    print("Real:", re.shape)
    print("Imag:", im.shape)

    # complex spectrogram
    spec = torch.complex(re, im)  # [B, F, T]
    print("Spec:", spec.shape)

    # ISTFT -> waveform
    wav = torch.istft(
        spec,
        n_fft=config_stft["n_fft"],
        hop_length=config_stft["hop_length"],
        win_length=config_stft["win_length"],
        length=waveform.shape[-1],
    )
    print("Wave:", wav.shape)

    # save wav
    wav_np = wav.squeeze().cpu().numpy()
    sf.write("out.wav", wav_np, sr)
    print("Saved: out.wav")

    torch.onnx.export(
        model,
        (mel,),
        "vocos-v2.onnx",
        opset_version=17,
        dynamo=True,
        input_names=["mel"],
        output_names=["real", "imag"],
    )

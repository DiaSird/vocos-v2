import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm

LRELU_SLOPE = 0.1


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        mrd = MultiResolutionDiscriminator()
        mpd = MultiPeriodDiscriminator()

        self.discriminators = nn.ModuleList([mrd, mpd])

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor):
        r_out, f_out, r_fmap, f_fmap = [], [], [], []

        for d in self.discriminators:
            r, f, fr, fg = d(y, y_hat)

            r_out.append(r)
            f_out.append(f)
            r_fmap.append(fr)
            f_fmap.append(fg)

        return r_out, f_out, r_fmap, f_fmap


def make_conv(in_ch: int, out_ch: int, k: tuple, s: tuple = (1, 1), p: tuple = (0, 0)):
    return weight_norm(nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p))


# =========================================================
# MRD - Multi Resolution Discriminator
# =========================================================
def stft_pad(n_fft: int, hop_length: int) -> tuple:
    pad = n_fft - hop_length
    return pad // 2, pad - pad // 2


def spectrogram(x: torch.Tensor, resolution: tuple):
    n_fft, hop, win = resolution
    x = x.squeeze(1)
    x = F.pad(x, stft_pad(n_fft, hop), mode="reflect")

    window = torch.ones(win, device=x.device, dtype=x.dtype)

    spec = torch.stft(
        x,
        n_fft=n_fft,
        hop_length=hop,
        win_length=win,
        window=window,
        center=False,
        return_complex=True,
    )

    return torch.abs(spec)


class DiscriminatorR(nn.Module):
    LAYERS = (
        ((3, 9),),
        ((3, 9),),
        ((3, 9),),
        ((3, 9),),
        ((3, 3), (1, 1), (1, 1)),
    )

    def __init__(self, resolution: tuple, multi: float = 1.0):
        super().__init__()

        self.resolution = resolution
        self.convs = nn.ModuleList()

        in_ch = 1
        chs = int(32 * multi)

        # k: kernel size, s: stride, p: padding
        for k, *args in self.LAYERS:
            s, p = args if args else ((1, 2), (1, 4))
            self.convs.append(make_conv(in_ch, chs, k, s, p))
            in_ch = chs

        self.conv_post = make_conv(chs, 1, (3, 3), 1, (1, 1))

    def forward(self, wav: torch.Tensor):
        x = spectrogram(wav, self.resolution).unsqueeze(1)

        fmap = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), LRELU_SLOPE, inplace=True)
            fmap.append(x)

        x = self.conv_post(x)
        fmap.append(x)

        return torch.flatten(x, 1), fmap


class MultiResolutionDiscriminator(nn.Module):
    def __init__(
        self,
        resolutions: tuple = ((1024, 120, 600), (2048, 240, 1200), (512, 50, 240)),
        multi: float = 1.0,
    ):
        super().__init__()
        self.discriminators = nn.ModuleList(
            DiscriminatorR(r, multi) for r in resolutions
        )

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor):
        r_out, f_out, r_fmap, f_fmap = [], [], [], []

        for d in self.discriminators:
            r, fr = d(y)
            f, fg = d(y_hat)

            r_out.append(r)
            f_out.append(f)
            r_fmap.append(fr)
            f_fmap.append(fg)

        return r_out, f_out, r_fmap, f_fmap


# =========================================================
# MPD - Multi Period Discriminator
# =========================================================
class DiscriminatorP(nn.Module):
    LAYERS = (
        (32, (5, 1), 3),
        (128, (5, 1), 3),
        (512, (5, 1), 3),
        (1024, (5, 1), 3),
        (1024, (5, 1), 1),
    )

    def __init__(self, period: int):
        super().__init__()

        self.period = period
        self.convs = nn.ModuleList()

        in_ch = 1

        # k: kernel size, s: stride
        for out_ch, k, s in self.LAYERS:
            self.convs.append(make_conv(in_ch, out_ch, k, s, (k[0] // 2, 0)))
            in_ch = out_ch

        self.conv_post = make_conv(1024, 1, (3, 1), 1, (1, 0))

    def forward(self, wav: torch.Tensor):
        b, c, t = wav.shape

        if t % self.period != 0:
            pad = self.period - (t % self.period)
            wav = F.pad(wav, (0, pad), mode="reflect")

        x = wav.view(b, c, -1, self.period)

        fmap = []

        for conv in self.convs:
            x = F.leaky_relu(conv(x), LRELU_SLOPE, inplace=True)
            fmap.append(x)

        x = self.conv_post(x)
        fmap.append(x)

        return torch.flatten(x, 1), fmap


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, periods=(2, 3, 5, 7, 11)):
        super().__init__()
        self.discriminators = nn.ModuleList(DiscriminatorP(p) for p in periods)

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor):
        r_out, f_out, r_fmap, f_fmap = [], [], [], []

        for d in self.discriminators:
            r, fr = d(y)
            f, fg = d(y_hat)

            r_out.append(r)
            f_out.append(f)
            r_fmap.append(fr)
            f_fmap.append(fg)

        return r_out, f_out, r_fmap, f_fmap


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    y = torch.randn(2, 1, 16000, device=device)
    y_hat = torch.randn(2, 1, 16000, device=device)

    model = Discriminator().to(device).eval()

    with torch.no_grad():
        r, f, fr, fg = model(y, y_hat)

    print("MRD+MPD count:", len(r))
    print("real shapes:", [len(x) for x in r])
    print("fake shapes:", [len(x) for x in f])
    print("feature blocks:", len(fr))

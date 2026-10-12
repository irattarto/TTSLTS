"""RVC v2 (768-dim, with pitch) hyperparameters for each sample rate, as in the original project."""

from __future__ import annotations

from dataclasses import dataclass

CHECKPOINT = "checkpoint.pt"  # resumable training state in a work folder
SEGMENT_SAMPLES = 12800  # audio samples the generator is trained on per example
INTER_CHANNELS = 192
HIDDEN_CHANNELS = 192
FILTER_CHANNELS = 768
N_HEADS = 2
N_LAYERS = 6
KERNEL_SIZE = 3
P_DROPOUT = 0
RESBLOCK = "1"
RESBLOCK_KERNEL_SIZES = [3, 7, 11]
RESBLOCK_DILATION_SIZES = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
UPSAMPLE_INITIAL_CHANNEL = 512
SPK_EMBED_DIM = 109
GIN_CHANNELS = 256

LEARNING_RATE = 1e-4
LR_DECAY = 0.999875
BETAS = (0.8, 0.99)
EPS = 1e-9
C_MEL = 45.0
C_KL = 1.0


@dataclass(frozen=True)
class SampleRateConfig:
    sample_rate: int
    hop: int
    n_fft: int  # also the window length
    n_mels: int
    upsample_rates: tuple[int, ...]
    upsample_kernel_sizes: tuple[int, ...]

    @property
    def name(self) -> str:
        return f"{self.sample_rate // 1000}k"

    @property
    def spec_channels(self) -> int:
        return self.n_fft // 2 + 1

    def model_args(self, is_half: bool) -> dict:
        """Keyword arguments of ``SynthesizerTrnMs768NSFsid``."""
        return {
            "spec_channels": self.spec_channels,
            "segment_size": SEGMENT_SAMPLES // self.hop,
            "inter_channels": INTER_CHANNELS,
            "hidden_channels": HIDDEN_CHANNELS,
            "filter_channels": FILTER_CHANNELS,
            "n_heads": N_HEADS,
            "n_layers": N_LAYERS,
            "kernel_size": KERNEL_SIZE,
            "p_dropout": P_DROPOUT,
            "resblock": RESBLOCK,
            "resblock_kernel_sizes": RESBLOCK_KERNEL_SIZES,
            "resblock_dilation_sizes": RESBLOCK_DILATION_SIZES,
            "upsample_rates": list(self.upsample_rates),
            "upsample_initial_channel": UPSAMPLE_INITIAL_CHANNEL,
            "upsample_kernel_sizes": list(self.upsample_kernel_sizes),
            "spk_embed_dim": SPK_EMBED_DIM,
            "gin_channels": GIN_CHANNELS,
            "sr": self.sample_rate,
            "is_half": is_half,
        }

    def checkpoint_config(self) -> list:
        """The ``config`` list stored in a shared ``.pth`` voice (read by ``rvc_pth``)."""
        a = self.model_args(False)
        return [
            a["spec_channels"],
            a["segment_size"],
            a["inter_channels"],
            a["hidden_channels"],
            a["filter_channels"],
            a["n_heads"],
            a["n_layers"],
            a["kernel_size"],
            a["p_dropout"],
            a["resblock"],
            a["resblock_kernel_sizes"],
            a["resblock_dilation_sizes"],
            a["upsample_rates"],
            a["upsample_initial_channel"],
            a["upsample_kernel_sizes"],
            a["spk_embed_dim"],
            a["gin_channels"],
            a["sr"],
        ]


SAMPLE_RATES: dict[int, SampleRateConfig] = {
    32000: SampleRateConfig(32000, 320, 1024, 80, (10, 8, 2, 2), (20, 16, 4, 4)),
    40000: SampleRateConfig(40000, 400, 2048, 125, (10, 10, 2, 2), (16, 16, 4, 4)),
    48000: SampleRateConfig(48000, 480, 2048, 128, (12, 10, 2, 2), (24, 20, 4, 4)),
}

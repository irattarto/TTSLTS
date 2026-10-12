"""Batches of prepared clips for training."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ttslts.train.hparams import SampleRateConfig
from ttslts.train.mel import spectrogram

MAX_FRAMES = 900


class VoiceDataset(Dataset):
    def __init__(self, work: Path, cfg: SampleRateConfig):
        self.work = work
        self.cfg = cfg
        self.stems = sorted(
            (p.stem for p in (work / "wavs").glob("*.npy")),
            key=int,
        )

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, index: int):
        name = f"{self.stems[index]}.npy"
        wav = torch.from_numpy(np.load(self.work / "wavs" / name)).float()
        phone = np.repeat(np.load(self.work / "feats" / name), 2, axis=0)  # 50 -> 100 fps
        pitch = np.load(self.work / "f0" / name)
        pitchf = np.load(self.work / "f0nsf" / name)
        spec = spectrogram(wav[None], self.cfg.n_fft, self.cfg.hop)[0]
        frames = min(spec.shape[-1], len(phone), len(pitch), len(pitchf), MAX_FRAMES)
        return (
            spec[:, :frames],
            wav[None, : frames * self.cfg.hop],
            torch.from_numpy(phone[:frames]).float(),
            torch.from_numpy(pitch[:frames]).long(),
            torch.from_numpy(pitchf[:frames]).float(),
        )


def collate(batch):
    """Zero-pad a list of examples to the longest one and stack them."""
    n = len(batch)
    frames = max(item[0].shape[-1] for item in batch)
    samples = max(item[1].shape[-1] for item in batch)
    spec = torch.zeros(n, batch[0][0].shape[0], frames)
    wave = torch.zeros(n, 1, samples)
    phone = torch.zeros(n, frames, batch[0][2].shape[1])
    pitch = torch.zeros(n, frames, dtype=torch.long)
    pitchf = torch.zeros(n, frames)
    spec_lengths = torch.zeros(n, dtype=torch.long)
    for i, (s, w, ph, p, pf) in enumerate(batch):
        spec[i, :, : s.shape[-1]] = s
        wave[i, :, : w.shape[-1]] = w
        phone[i, : ph.shape[0]] = ph
        pitch[i, : p.shape[0]] = p
        pitchf[i, : pf.shape[0]] = pf
        spec_lengths[i] = s.shape[-1]
    return phone, spec_lengths, pitch, pitchf, spec, spec_lengths.clone(), wave

"""The RVC v2 training loop (generator + multi-period discriminator, as in the original project)."""

from __future__ import annotations

import logging
import random
import time
import warnings
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ttslts.train import hparams as hp
from ttslts.train.dataset import VoiceDataset, collate
from ttslts.train.losses import discriminator_loss, feature_loss, generator_loss, kl_loss
from ttslts.train.mel import mel_spectrogram, spec_to_mel
from ttslts.train.rvc_model import commons
from ttslts.train.rvc_model.models import MultiPeriodDiscriminatorV2, SynthesizerTrnMs768NSFsid

log = logging.getLogger(__name__)

CHECKPOINT = hp.CHECKPOINT


class Cancelled(Exception):
    pass


def pick_device(use_gpu: bool) -> torch.device:
    return torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")


def _load_pretrained(net, path: Path) -> None:
    state = torch.load(path, map_location="cpu", weights_only=True)
    net.load_state_dict(state["model"])


def export_pth(net_g, cfg: hp.SampleRateConfig, path: Path, info: str) -> None:
    """Write the voice the way RVC shares it (half precision, no training-only encoder)."""
    state = OrderedDict(
        (k, v.detach().cpu().half()) for k, v in net_g.state_dict().items() if "enc_q" not in k
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(
        {
            "weight": state,
            "config": cfg.checkpoint_config(),
            "info": info,
            "sr": cfg.name,
            "f0": 1,
            "version": "v2",
        },
        tmp,
    )
    tmp.replace(path)


def train(
    work: Path,
    cfg: hp.SampleRateConfig,
    pretrained_g: Path,
    pretrained_d: Path,
    epochs: int,
    batch_size: int,
    save_every: int,
    pth: Path,
    on_save: Callable[[Path, int], None],
    on_progress: Callable[[dict], None],
    cancelled: Callable[[], bool],
    use_gpu: bool = True,
    fp16: bool = True,
    resume: bool = True,
    seed: int = 1234,
) -> None:
    """Train on the prepared clips in ``work`` and write the voice to ``pth``.

    ``on_save(pth, epoch)`` is called whenever a checkpoint and the ``.pth`` are written.
    Raises ``Cancelled`` after saving if ``cancelled()`` turns true.
    """
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", message=".*stft with return_complex.*")
    warnings.filterwarnings("ignore", message=".*lr_scheduler.step.*")
    random.seed(seed)
    torch.manual_seed(seed)
    device = pick_device(use_gpu)
    amp = fp16 and device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    dataset = VoiceDataset(work, cfg)
    if len(dataset) == 0:
        raise ValueError("There are no prepared clips to train on.")
    batch_size = max(1, min(batch_size, len(dataset)))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)

    net_g = SynthesizerTrnMs768NSFsid(**cfg.model_args(is_half=amp)).to(device)
    net_d = MultiPeriodDiscriminatorV2().to(device)
    optim_args = {"lr": hp.LEARNING_RATE, "betas": hp.BETAS, "eps": hp.EPS}
    optim_g = torch.optim.AdamW(net_g.parameters(), **optim_args)
    optim_d = torch.optim.AdamW(net_d.parameters(), **optim_args)

    start_epoch = 1
    checkpoint = work / CHECKPOINT
    if resume and checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        net_g.load_state_dict(saved["g"])
        net_d.load_state_dict(saved["d"])
        optim_g.load_state_dict(saved["optim_g"])
        optim_d.load_state_dict(saved["optim_d"])
        start_epoch = int(saved["epoch"]) + 1
        log.info("Resuming after epoch %d", start_epoch - 1)
    else:
        _load_pretrained(net_g, pretrained_g)
        _load_pretrained(net_d, pretrained_d)
    sched_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hp.LR_DECAY)
    sched_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hp.LR_DECAY)
    for _ in range(start_epoch - 1):
        sched_g.step()
        sched_d.step()
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    seg_frames = hp.SEGMENT_SAMPLES // cfg.hop
    started = time.monotonic()

    def save(done_epoch: int) -> None:
        tmp = checkpoint.with_suffix(".tmp")
        torch.save(
            {
                "g": net_g.state_dict(),
                "d": net_d.state_dict(),
                "optim_g": optim_g.state_dict(),
                "optim_d": optim_d.state_dict(),
                "epoch": done_epoch,
            },
            tmp,
        )
        tmp.replace(checkpoint)
        export_pth(net_g, cfg, pth, f"Trained in TTSLTS, {done_epoch} epochs")
        on_save(pth, done_epoch)

    net_g.train()
    net_d.train()
    for epoch in range(start_epoch, epochs + 1):
        totals = {"disc": 0.0, "gen": 0.0, "mel": 0.0, "kl": 0.0}
        for step, batch in enumerate(loader, 1):
            if cancelled():
                if epoch > start_epoch:
                    save(epoch - 1)
                raise Cancelled
            phone, phone_len, pitch, pitchf, spec, spec_len, wave = (t.to(device) for t in batch)
            sid = torch.zeros(phone.shape[0], dtype=torch.long, device=device)

            with torch.autocast("cuda", enabled=amp):
                y_hat, ids_slice, _x_mask, z_mask, (_z, z_p, m_p, logs_p, _m_q, logs_q) = net_g(
                    phone, phone_len, pitch, pitchf, spec, spec_len, sid
                )
                with torch.autocast("cuda", enabled=False):
                    mel = spec_to_mel(spec.float(), cfg.n_fft, cfg.n_mels, cfg.sample_rate)
                    y_mel = commons.slice_segments(mel, ids_slice, seg_frames)
                    y_hat_mel = mel_spectrogram(
                        y_hat.float().squeeze(1), cfg.n_fft, cfg.n_mels, cfg.sample_rate, cfg.hop
                    )
                if amp:
                    y_hat_mel = y_hat_mel.half()
                target = commons.slice_segments(wave, ids_slice * cfg.hop, hp.SEGMENT_SAMPLES)
                d_real, d_fake, _, _ = net_d(target, y_hat.detach())
                with torch.autocast("cuda", enabled=False):
                    loss_d = discriminator_loss(d_real, d_fake)
            optim_d.zero_grad()
            scaler.scale(loss_d).backward()
            scaler.step(optim_d)

            with torch.autocast("cuda", enabled=amp):
                d_real, d_fake, fmap_r, fmap_g = net_d(target, y_hat)
                with torch.autocast("cuda", enabled=False):
                    loss_mel = F.l1_loss(y_mel, y_hat_mel) * hp.C_MEL
                    loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hp.C_KL
                    loss_fm = feature_loss(fmap_r, fmap_g)
                    loss_gen = generator_loss(d_fake)
                    loss_g = loss_gen + loss_fm + loss_mel + loss_kl
            optim_g.zero_grad()
            scaler.scale(loss_g).backward()
            scaler.step(optim_g)
            scaler.update()

            for key, value in (
                ("disc", loss_d),
                ("gen", loss_g),
                ("mel", loss_mel),
                ("kl", loss_kl),
            ):
                totals[key] += float(value.detach())
            on_progress(
                {
                    "epoch": epoch,
                    "epochs": epochs,
                    "step": step,
                    "steps": len(loader),
                    "elapsed": time.monotonic() - started,
                }
            )

        sched_g.step()
        sched_d.step()
        on_progress(
            {
                "epoch": epoch,
                "epochs": epochs,
                "step": len(loader),
                "steps": len(loader),
                "elapsed": time.monotonic() - started,
                "losses": {k: v / len(loader) for k, v in totals.items()},
            }
        )
        if epoch % save_every == 0 and epoch < epochs:
            save(epoch)
    save(epochs)

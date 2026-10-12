"""Runs the real training loop for two epochs on a tiny synthetic voice (CPU, random weights)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ttslts.train import hparams, mel, trainer  # noqa: E402
from ttslts.train.rvc_model.models import (  # noqa: E402
    MultiPeriodDiscriminatorV2,
    SynthesizerTrnMs768NSFsid,
)
from ttslts.vc import rvc_pth  # noqa: E402
from ttslts.vc.torch_zip import load_object  # noqa: E402

CFG = hparams.SAMPLE_RATES[40000]


def make_work(tmp_path, clips=3, seconds=1.2):
    work = tmp_path / "work"
    for sub in ("wavs", "feats", "f0", "f0nsf"):
        (work / sub).mkdir(parents=True)
    rng = np.random.default_rng(0)
    n = int(seconds * CFG.sample_rate)
    for i in range(clips):
        t = np.arange(n) / CFG.sample_rate
        np.save(work / "wavs" / f"{i}.npy", (0.3 * np.sin(2 * np.pi * 180 * t)).astype(np.float32))
        np.save(
            work / "feats" / f"{i}.npy", rng.standard_normal((n // 800, 768)).astype(np.float32)
        )
        frames = n // 400
        np.save(work / "f0" / f"{i}.npy", np.full(frames, 100, dtype=np.int64))
        np.save(work / "f0nsf" / f"{i}.npy", np.full(frames, 180.0, dtype=np.float32))
    return work


def make_pretrained(tmp_path):
    torch.manual_seed(0)
    g = SynthesizerTrnMs768NSFsid(**CFG.model_args(is_half=False))
    d = MultiPeriodDiscriminatorV2()
    torch.save({"model": g.state_dict()}, tmp_path / "g.pth")
    torch.save({"model": d.state_dict()}, tmp_path / "d.pth")
    return tmp_path / "g.pth", tmp_path / "d.pth"


def run(work, pretrained, epochs, **kw):
    saved, progress = [], []
    trainer.train(
        work=work,
        cfg=CFG,
        pretrained_g=pretrained[0],
        pretrained_d=pretrained[1],
        epochs=epochs,
        batch_size=2,
        save_every=1,
        pth=work.parent / "voice.pth",
        on_save=lambda path, epoch: saved.append(epoch),
        on_progress=progress.append,
        cancelled=kw.pop("cancelled", lambda: False),
        use_gpu=False,
        **kw,
    )
    return saved, progress


def test_mel_filterbank_matches_the_slaney_scale():
    bank = mel.mel_filterbank(40000, 2048, 125)
    assert bank.shape == (125, 1025) and bank.min() >= 0
    assert np.all(bank.sum(axis=1) > 0)
    assert 0.0 < bank[0].nonzero()[0][0] < 5  # starts at the lowest bins


def test_trains_exports_a_voice_the_app_can_import_and_resumes(tmp_path):
    work = make_work(tmp_path)
    pretrained = make_pretrained(tmp_path)

    saved, progress = run(work, pretrained, epochs=2)
    assert saved == [1, 2]
    last = progress[-1]
    assert last["epoch"] == 2 and all(np.isfinite(v) for v in last["losses"].values())

    # The exported file is a normal RVC v2 voice: the app's own importer accepts it.
    state, config = rvc_pth.validate(load_object(str(tmp_path / "voice.pth")))
    assert config.sample_rate == 40000 and config.spk_embed_dim == 109
    assert not any("enc_q" in k for k in state)
    onnx_path = tmp_path / "voice.onnx"
    assert rvc_pth.convert(str(tmp_path / "voice.pth"), str(onnx_path)) == 40000
    assert onnx_path.stat().st_size > 1_000_000

    # Continuing picks up after the last finished epoch instead of starting over.
    saved, progress = run(work, pretrained, epochs=3, resume=True)
    assert saved == [3] and progress[0]["epoch"] == 3


def test_cancel_saves_and_stops(tmp_path):
    work = make_work(tmp_path)
    pretrained = make_pretrained(tmp_path)
    seen = []

    def cancelled():
        seen.append(1)
        return len(seen) > 3  # during the second epoch (two steps per epoch)

    with pytest.raises(trainer.Cancelled):
        run(work, pretrained, epochs=5, cancelled=cancelled)
    assert (work.parent / "voice.pth").exists() and (work / trainer.CHECKPOINT).exists()


@pytest.mark.parametrize("rate", sorted(hparams.SAMPLE_RATES))
def test_segment_is_a_whole_number_of_hops(rate):
    # The generator emits segment // hop frames of hop samples; the target slice must match.
    cfg = hparams.SAMPLE_RATES[rate]
    assert cfg.segment_samples % cfg.hop == 0
    assert cfg.model_args(False)["segment_size"] * cfg.hop == cfg.segment_samples

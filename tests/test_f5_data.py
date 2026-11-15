import numpy as np
import pytest

from ttslts.train import f5_data


def test_only_sure_sentences_of_a_usable_length_are_kept():
    assert f5_data.usable_segment("Jó napot.", 2.0, -0.2, 0.1, 1.5)
    assert not f5_data.usable_segment("Jó napot.", 0.6, -0.2, 0.1, 1.5)  # too short
    assert not f5_data.usable_segment("Jó napot.", 20.0, -0.2, 0.1, 1.5)  # too long
    assert not f5_data.usable_segment("Jó napot.", 2.0, -0.9, 0.1, 1.5)  # Whisper unsure
    assert not f5_data.usable_segment("la la la", 2.0, -0.2, 0.1, 3.0)  # stuck repeating
    assert not f5_data.usable_segment(" ", 2.0, -0.2, 0.1, 1.5)


def test_transcripts_are_compared_on_letters_and_written_out_numbers():
    assert f5_data.agreement("Három órakor jött.", "3 órakor jött") == 1.0
    assert f5_data.agreement("A Sziklasziget", "A szikla sziget") >= 0.9
    assert f5_data.agreement("Reggel indultak.", "Este értek haza.") < 0.7


def test_the_narrator_is_the_largest_voice_group():
    rng = np.random.default_rng(0)
    narrator, other = rng.normal(size=16), rng.normal(size=16)

    def around(v, n):
        x = v + 0.3 * rng.normal(size=(n, 16))
        return x / np.linalg.norm(x, axis=1, keepdims=True)

    embs = np.vstack([around(narrator, 70), around(other, 30)])
    sims, core = f5_data.narrator_core(embs)
    assert set(core) <= set(range(70))  # the group is all narrator
    assert sims[:70].mean() > sims[70:].mean() + 0.3


def _clip(sim, f0, wobble, agree=1.0, core=True, dur=3.0):
    return {"sim": sim, "f0": f0, "wobble": wobble, "agree": agree, "core": core, "dur": dur}


def test_selection_keeps_the_narrator_steady_and_well_transcribed():
    clips = [_clip(0.9 - i * 0.001, 110, 50 + i) for i in range(40)]
    clips += [
        _clip(0.4, 110, 50, core=False),  # another voice
        _clip(0.9, 180, 50, core=False),  # the narrator doing a high character voice
        _clip(0.9, 110, 50, agree=0.8, core=False),  # suspect transcript
    ]
    report = f5_data.select(clips, "normal")
    assert not clips[40]["narrator"] and not clips[41]["narrator"]
    assert clips[42]["steady"] and clips[42]["transcript"] == "check" and not clips[42]["keep"]
    assert not clips[39]["steady"]  # the most trembling quarter is left out
    assert clips[0]["keep"]
    assert report["kept"] == sum(c["keep"] for c in clips)
    assert report["suspect"] == 1
    loose = f5_data.select([dict(c) for c in clips], "loose")["kept"]
    strict = f5_data.select([dict(c) for c in clips], "strict")["kept"]
    assert loose > report["kept"] > strict


def test_selection_without_a_narrator_keeps_nothing():
    assert f5_data.select([_clip(0.9, None, None, core=True)])["kept"] == 0


def test_a_sentence_is_cut_at_the_pauses_around_it():
    quiet = np.ones(300, bool)
    quiet[50:100] = False  # the end of the sentence before
    quiet[110:200] = False  # the sentence
    quiet[230:280] = False  # the next one
    # Whisper put the start 0.2 s early (inside the one before) and the end 0.2 s early
    start, end, soft = f5_data.cut_edges(quiet, 1.0, 1.8, 2.3)
    assert start == pytest.approx(1.07, abs=0.011)  # 30 ms before the sentence
    assert end == pytest.approx(2.08, abs=0.011)  # after the first 80 ms pause
    assert not soft


def test_a_sentence_read_on_without_a_pause_ends_at_whispers_end():
    quiet = np.zeros(300, bool)
    quiet[:100] = True
    start, end, soft = f5_data.cut_edges(quiet, 1.0, 2.0, 2.5)
    assert soft and end == pytest.approx(2.03, abs=0.011)
    assert start == pytest.approx(0.97, abs=0.011)


def test_quiet_frames_follow_the_loudness():
    sr = 24000
    t = np.arange(sr) / sr
    audio = np.concatenate([np.zeros(sr // 2), 0.5 * np.sin(2 * np.pi * 200 * t)]).astype(
        np.float32
    )
    quiet = f5_data.quiet_frames(audio, sr)
    assert quiet[:45].all() and not quiet[60:].any()


def test_a_trembling_pitch_wobbles_more_than_a_steady_one():
    t = np.arange(200)
    steady = 110 * 2 ** (np.sin(t / 30) / 12)  # a slow melody
    shaky = steady * 2 ** (0.4 * np.sin(t * 2.5) / 12)  # plus a fast tremble
    assert f5_data.wobble(shaky) > 3 * f5_data.wobble(steady)
    assert f5_data.wobble(np.zeros(100)) is None  # no voice
    assert f5_data.median_pitch(steady) == pytest.approx(110, rel=0.05)


def test_clips_left_out_are_grouped_by_why_and_the_choice_is_saved(tmp_path):
    clips = [_clip(0.9 - i * 0.001, 110, 50 + i) for i in range(40)]
    clips += [_clip(0.4, 110, 50, core=False), _clip(0.9, 110, 50, agree=0.8, core=False)]
    for n, c in enumerate(clips):
        c.update(name=f"000_{n:05d}", text=f"mondat {n}")
    report = f5_data.select(clips, "normal")
    g = f5_data.groups(clips)
    assert (
        clips[40] in g["other_voice"] and clips[41] in g["unsure"] and clips[35] in g["trembling"]
    )
    assert len(g["kept"]) == report["kept"]
    f5_data.save_selection(tmp_path, "normal", clips, report)
    saved = f5_data.load_selection(tmp_path)
    assert saved["strictness"] == "normal" and len(saved["kept"]) == report["kept"]
    assert f5_data.load_selection(tmp_path / "none") is None

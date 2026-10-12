import sys

from ttslts.train import runtime


def fake_runtime(tmp_path, ready=True):
    root = tmp_path / "runtime"
    python = runtime._python_exe(root)
    python.parent.mkdir(parents=True, exist_ok=True)
    python.write_bytes(b"")
    if ready:
        (root / "ready.json").write_text("{}")
    return root


def packaged(monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)


def test_packaged_app_needs_the_environment(tmp_path, monkeypatch):
    packaged(monkeypatch)
    assert not runtime.is_installed(tmp_path / "nothing")
    assert not runtime.is_installed(fake_runtime(tmp_path, ready=False))
    (tmp_path / "runtime" / "ready.json").unlink(missing_ok=True)
    root = tmp_path / "other"
    (root / "ready.json").parent.mkdir(parents=True)
    (root / "ready.json").write_text("{}")
    assert not runtime.is_installed(root)  # marker without an interpreter
    assert runtime.is_installed(fake_runtime(tmp_path))


def test_packaged_app_runs_jobs_with_the_private_python(tmp_path, monkeypatch):
    packaged(monkeypatch)
    root = fake_runtime(tmp_path)
    assert runtime.python_command(root) == [str(runtime._python_exe(root))]


def test_packaged_sources_come_from_the_bundle(tmp_path, monkeypatch):
    packaged(monkeypatch)
    (tmp_path / "pysrc").mkdir()
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    assert runtime.source_dir() == tmp_path / "pysrc"


def test_source_dir_from_a_checkout_has_the_package():
    assert (runtime.source_dir() / "ttslts" / "train" / "run.py").exists()


def test_embedded_python_reads_sources_from_its_pth_file(tmp_path, monkeypatch):
    packaged(monkeypatch)
    root = fake_runtime(tmp_path)
    pth = runtime._python_exe(root).parent / "python312._pth"
    pth.write_text("python312.zip\n.\nimport site\n")
    assert runtime.prepare_for_run(root) == {}
    runtime.prepare_for_run(root)  # running again does not pile up lines
    lines = pth.read_text().splitlines()
    assert sum("pysrc" in ln for ln in lines) == 1
    assert str(runtime.source_dir()) in lines[-1]


def test_other_pythons_get_a_pythonpath(tmp_path, monkeypatch):
    packaged(monkeypatch)
    root = fake_runtime(tmp_path)
    assert runtime.prepare_for_run(root) == {"PYTHONPATH": str(runtime.source_dir())}


def test_consent_mentions_the_download_size():
    assert "GB" in runtime.consent_text()

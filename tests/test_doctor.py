import copy
import io
import json
from pathlib import Path

from expressive_audiobook.config import _merge_config, load_config
from expressive_audiobook.doctor import Doctor, run_doctor
from expressive_audiobook.ui import Ui

ROOT = Path(__file__).resolve().parents[1]


def _config(**sections):
    return _merge_config(copy.deepcopy(load_config(ROOT / "src" / "expressive_audiobook" / "default.toml")), sections)


def _levels(doctor, label):
    doctor.run_all()
    return [item.level for _, item in doctor.findings if item.label == label]


def _isolate(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # no .env here
    monkeypatch.setenv("AUDIOBOOK_HOME", str(tmp_path / "home"))
    for name in ("GEMINI_API_KEY", "ELEVENLABS_API_KEY"):
        monkeypatch.delenv(name, raising=False)


def _down(url, headers=None, timeout=3.0):
    raise OSError("connection refused")


def test_missing_gemini_key_is_a_problem_only_when_gemini_is_used(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    assert _levels(Doctor(_config(), http=_down), "GEMINI_API_KEY") == ["fail"]
    local = _config(text={"provider": "openai", "model": "m"}, tts={"provider": "http"})
    assert _levels(Doctor(local, http=_down), "GEMINI_API_KEY") == ["info"]


def test_elevenlabs_key_matters_only_for_elevenlabs_effects(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    assert _levels(Doctor(_config(), http=_down), "ELEVENLABS_API_KEY") == ["info"]
    eleven = _config(sound={"sfx_provider": "elevenlabs"})
    assert _levels(Doctor(eleven, http=_down), "ELEVENLABS_API_KEY") == ["fail"]


def test_keys_are_never_printed(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    monkeypatch.setenv("GEMINI_API_KEY", "SECRET-VALUE-123")
    out = io.StringIO()
    code = run_doctor(_config(), Ui(out, color=False), http=_down)
    assert "SECRET-VALUE-123" not in out.getvalue()
    assert "GEMINI_API_KEY" in out.getvalue() and code == 0


def test_text_server_down_and_model_missing(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    config = _config(text={"provider": "openai", "model": "qwen3:14b"}, tts={"provider": "http"})
    assert _levels(Doctor(config, http=_down), "text server") == ["fail"]

    listing = json.dumps({"data": [{"id": "llama3:8b"}]}).encode()
    assert _levels(Doctor(config, http=lambda *a, **k: (200, listing)), "text.model") == ["fail"]
    have = json.dumps({"data": [{"id": "qwen3:14b"}]}).encode()
    assert _levels(Doctor(config, http=lambda *a, **k: (200, have)), "text.model") == ["ok"]


def test_orpheus_not_installed_gives_the_install_command(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    config = _config(tts={"provider": "orpheus"})
    doctor = Doctor(config, http=_down)
    doctor.run_all()
    found = [item for _, item in doctor.findings if item.label == "Orpheus install"]
    assert found and found[0].level == "fail" and "models install orpheus" in found[0].fix


def test_exit_code_reflects_problems(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    out = io.StringIO()
    assert run_doctor(_config(), Ui(out, color=False), http=_down) == 1
    assert "1 problem to fix" in out.getvalue()
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    assert run_doctor(_config(), Ui(io.StringIO(), color=False), http=_down) == 0


def test_online_check_reports_a_rejected_key(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    monkeypatch.setenv("GEMINI_API_KEY", "bad")
    doctor = Doctor(_config(), online=True, http=lambda *a, **k: (403, b"{}"))
    assert _levels(doctor, "Gemini API") == ["fail"]


def test_doctor_command_runs(monkeypatch, tmp_path, capsys):
    _isolate(monkeypatch, tmp_path)
    from expressive_audiobook.cli import main

    assert main(["doctor"]) == 1
    assert "GEMINI_API_KEY" in capsys.readouterr().out

"""What the package promises: a light base import, routes from an argument, no hidden inputs, and
a wheel that needs no Rust toolchain."""
import pathlib
import shutil
import subprocess
import sys
import zipfile

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
PYTHON = ROOT / "python"


def run(*args, cwd=PYTHON):
    return subprocess.run([sys.executable, *args], cwd=cwd, capture_output=True, text=True, timeout=300)


def test_the_base_import_does_not_load_torch():
    code = (
        "import sys, copper_policy, copper_policy.wire, copper_policy.runner, "
        "copper_policy.calibration, copper_policy.policies\n"
        "assert 'torch' not in sys.modules, 'torch was imported'\n"
    )
    result = run("-c", code)
    assert result.returncode == 0, result.stderr


def test_the_routes_come_from_one_prefix():
    from copper_policy import runner

    keys = runner.Keys("arm_left")
    assert (keys.obs, keys.action, keys.img, keys.exec, keys.infer) == (
        "arm_left/obs", "arm_left/action", "arm_left/img", "arm_left/exec", "arm_left/infer")
    default = runner.Keys()
    assert default.obs == "vla/obs" and default.infer == "vla/infer"


def test_the_command_line_takes_the_key_prefix():
    result = run("-m", "copper_policy", "--help")
    assert result.returncode == 0, result.stderr
    assert "--key-prefix" in result.stdout


def test_the_package_reads_no_environment_variables():
    offenders = [
        f.name
        for f in (PYTHON / "copper_policy").glob("*.py")
        if any(token in f.read_text() for token in ("os.environ", "getenv"))
    ]
    assert offenders == []


def test_the_wheel_is_pure_python_with_the_documented_dependencies(tmp_path):
    pytest.importorskip("setuptools")
    pytest.importorskip("wheel")
    # Build from a copy, so no build directory or egg-info appears in the crate.
    src = tmp_path / "src"
    shutil.copytree(PYTHON / "copper_policy", src / "python" / "copper_policy",
                    ignore=shutil.ignore_patterns("__pycache__"))
    for name in ("pyproject.toml", "README.md"):
        shutil.copy(ROOT / name, src / name)
    out = tmp_path / "dist"
    out.mkdir()
    code = f"from setuptools import build_meta; print(build_meta.build_wheel({str(out)!r}))"
    result = subprocess.run([sys.executable, "-c", code], cwd=src, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stderr[-2000:]
    (wheel,) = out.glob("copper_policy-*.whl")
    assert wheel.name.endswith("-py3-none-any.whl"), "a wheel with no compiled code"
    with zipfile.ZipFile(wheel) as z:
        names = z.namelist()
        assert {"copper_policy/wire.py", "copper_policy/runner.py", "copper_policy/rtc.py"} <= set(names)
        assert not [n for n in names if n.endswith((".rs", ".so", ".pyd"))]
        meta = z.read(next(n for n in names if n.endswith("METADATA"))).decode()
    requires = [l for l in meta.splitlines() if l.startswith("Requires-Dist")]
    assert any("eclipse-zenoh" in l and "extra" not in l for l in requires)
    assert any("numpy" in l and "extra" not in l for l in requires)
    assert not any("torch" in l and "extra" not in l for l in requires), "torch only under extras"
    assert any('extra == "flow"' in l and "torch" in l for l in requires)
    assert any('extra == "act"' in l and "lerobot" in l for l in requires)

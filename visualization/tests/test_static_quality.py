import subprocess
import sys


def test_visualization_src_is_pyflakes_clean():
    result = subprocess.run(
        [sys.executable, "-m", "pyflakes", "visualization/src"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr

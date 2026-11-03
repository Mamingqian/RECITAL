"""Copy only installed Python, NumPy, and their ELF dependencies for bwrap."""
from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import sysconfig
import tempfile

import numpy as np

from .models import IsolationError, MAX_MESSAGE_BYTES


class RuntimeBundle:
    """A private, read-only runtime tree, containing neither repo nor home mounts."""

    def __init__(self, directory: str | Path | None = None):
        self._temporary = tempfile.TemporaryDirectory(prefix="synthesis-runtime-", dir=directory)
        self.path = Path(self._temporary.name)
        try:
            self._build()
        except BaseException:
            self.close()
            raise

    def _build(self) -> None:
        if os.name != "posix" or not shutil.which("bwrap"):
            raise IsolationError("Linux bubblewrap is required; no parent fallback")
        self.bwrap = shutil.which("bwrap")
        version = f"python{sys.version_info.major}.{sys.version_info.minor}"
        (self.path / "bin").mkdir()
        shutil.copy2(sys.executable, self.path / "bin/python")
        stdlib = Path(sysconfig.get_path("stdlib"))
        target = self.path / "lib" / version
        shutil.copytree(stdlib, target, ignore=shutil.ignore_patterns(
            "site-packages", "__pycache__", "test", "tests", "idlelib", "tkinter",
            "turtledemo", "ensurepip", "config-*", "*.pyc", "*.a"))
        site = target / "site-packages"
        site.mkdir()
        numpy_path = Path(np.__file__).parent
        shutil.copytree(numpy_path, site / "numpy", ignore=shutil.ignore_patterns(
            "__pycache__", "tests", "test", "*.pyc", "*.a"))
        numpy_libs = numpy_path.parent / "numpy.libs"
        if numpy_libs.exists():
            shutil.copytree(numpy_libs, site / "numpy.libs")
        shutil.copy2(Path(__file__).with_name("worker.py"), self.path / "worker.py")
        deps = self.path / "deps"
        deps.mkdir()
        # ldd is run exclusively on trusted installed executable/library files.
        candidates = [Path(sys.executable), *stdlib.joinpath("lib-dynload").glob("*.so"),
                      *numpy_path.rglob("*.so"), *numpy_path.rglob("*.so.*")]
        if numpy_libs.exists():
            candidates.extend(numpy_libs.glob("*.so*"))
        seccomp = None
        listing = subprocess.run(["ldconfig", "-p"], capture_output=True, text=True,
                                 check=True, timeout=10).stdout
        for line in listing.splitlines():
            if "libseccomp.so.2 " in line:
                seccomp = Path(line.split("=>", 1)[1].strip())
                break
        if seccomp is None:
            raise IsolationError("libseccomp.so.2 is required")
        candidates.append(seccomp)
        libraries: dict[str, Path] = {"libseccomp.so.2": seccomp}
        for candidate in candidates:
            result = subprocess.run(["ldd", str(candidate)], capture_output=True,
                                    text=True, timeout=10)
            for line in result.stdout.splitlines():
                match = re.search(r"(?:=>\s*)?(/[^\s]+)", line)
                if match:
                    library = Path(match.group(1))
                    libraries[library.name] = library
                elif "not found" in line:
                    raise IsolationError(f"missing runtime dependency: {line.strip()}")
        for name, source in libraries.items():
            shutil.copy2(source, deps / name)
        loaders = sorted(deps.glob("ld-linux*"))
        if len(loaders) != 1:
            raise IsolationError("could not identify ELF interpreter")
        self.loader = f"/runtime/deps/{loaders[0].name}"
        for path in sorted(self.path.rglob("*")):
            if path.is_file():
                path.chmod(0o555 if os.access(path, os.X_OK) else 0o444)
        self.version = version

    def command(self, *, max_message_bytes: int = MAX_MESSAGE_BYTES) -> list[str]:
        return [self.bwrap, "--unshare-all", "--die-with-parent", "--new-session",
                "--cap-drop", "ALL", "--ro-bind", str(self.path),
                "/runtime", "--dev", "/dev", "--tmpfs", "/tmp", "--chdir", "/tmp",
                "--setenv", "PATH", "/runtime/bin", "--setenv", "HOME", "/tmp",
                "--setenv", "PYTHONDONTWRITEBYTECODE", "1",
                "--setenv", "OPENBLAS_NUM_THREADS", "1",
                "--setenv", "OMP_NUM_THREADS", "2",
                "--setenv", "MKL_NUM_THREADS", "1",
                self.loader, "--library-path", "/runtime/deps",
                "/runtime/bin/python", "-I", "-S", "/runtime/worker.py",
                str(max_message_bytes)]

    def close(self) -> None:
        self._temporary.cleanup()

"""Create/repair a private environment without requiring the host's ensurepip.

Embedded verbatim into the Colab notebook. Standard library only; neither the
kernel's packages nor previous experiment journals are changed.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request

PIP_VERSION = "25.2"
PIP_BOOTSTRAP_URL = "https://bootstrap.pypa.io/pip/zipapp/pip-25.2.pyz"
PIP_BOOTSTRAP_SHA256 = "769a10214aca718618e7014ec8e70b36288fe9d49a9d7ba3185ea3917e84a4d3"


def verified_bootstrap(cache):
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    destination = cache / "pip-25.2.pyz"
    if destination.exists():
        data = destination.read_bytes()
    else:
        with urllib.request.urlopen(PIP_BOOTSTRAP_URL, timeout=60) as response:
            data = response.read(10 * 1024**2)
    if hashlib.sha256(data).hexdigest() != PIP_BOOTSTRAP_SHA256:
        raise RuntimeError("Le fichier d'installation pip ne correspond pas à l'empreinte prévue.")
    if not destination.exists():
        temporary = destination.with_suffix(".download")
        temporary.write_bytes(data)
        temporary.replace(destination)
    return destination


def environment_python(destination):
    return Path(destination) / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def assert_private_environment(python, destination):
    code = "import json,sys; print(json.dumps([sys.prefix,sys.base_prefix]))"
    prefix, base = json.loads(subprocess.check_output([str(python), "-I", "-c", code], text=True))
    if Path(prefix).resolve() != Path(destination).resolve() or Path(prefix).resolve() == Path(base).resolve():
        raise RuntimeError("L'interpréteur ne pointe pas vers l'environnement privé attendu.")


def ensure_environment(destination, cache=None):
    destination = Path(destination).absolute()
    python = environment_python(destination)
    if not python.exists():
        # This option never invokes ensurepip. It also safely completes a partial directory.
        subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(destination)], check=True)
    assert_private_environment(python, destination)
    healthy = subprocess.run([str(python), "-I", "-m", "pip", "--version"], capture_output=True, text=True)
    if healthy.returncode:
        bootstrap = verified_bootstrap(cache or destination.parent / "menia-pip-bootstrap-v1")
        subprocess.run([str(python), "-I", str(bootstrap), "--isolated", "install", "--no-input",
                        "--disable-pip-version-check", "--only-binary=:all:",
                        "--index-url", "https://pypi.org/simple", f"pip=={PIP_VERSION}"], check=True)
    subprocess.run([str(python), "-I", "-m", "pip", "--version"], check=True)
    return python

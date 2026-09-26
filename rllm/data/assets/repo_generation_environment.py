"""Read-only package/toolchain probe, runnable in benchmark images (stdlib only)."""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from urllib.parse import urlsplit, urlunsplit


def redact(text):
    def clean(match):
        raw = match.group(0)
        try:
            url = urlsplit(raw)
            host = url.hostname or ""
            if url.port:
                host += ":" + str(url.port)
            return urlunsplit((url.scheme, host, url.path, "", ""))
        except ValueError:
            return "<redacted-url>"
    return re.sub(r"""https?://[^\s'"<>]+""", clean, text)


def run(argv, timeout=25):
    try:
        result = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                universal_newlines=True, timeout=timeout)  # noqa: UP021 - Python 3.6 images
        output = redact(result.stdout)
        return {"exit_code": result.returncode, "output": output[:32000], "truncated": len(output) > 32000}
    except subprocess.TimeoutExpired:
        return {"exit_code": 124, "output": "environment probe timed out", "truncated": False}
    except OSError as exc:
        return {"exit_code": 127, "output": type(exc).__name__, "truncated": False}


def main():
    data = {
        "schema_version": 1, "path": os.environ.get("PATH", ""),
        "tools": {name: shutil.which(name) for name in ("python", "python3", "pip", "pytest")},
        "python": run(["python", "-c", "import sys; print(sys.executable); print(sys.version)"]),
        "pip": run(["pip", "--version"]),
        "packages": run(["pip", "freeze", "--all"]),
        "editable": run(["pip", "list", "--editable", "--format=json"]),
    }
    # Query the same pip's configured index, with its proxy/index settings.
    # No dependency install, no index override and no isolation bypass.
    if "--check-network" in sys.argv:
        # An index page can work while its wheel host is blocked. Download a
        # wheel through this image's pip/resolver; do not install or reuse cache.
        data["network_environment"] = {
            key: redact(value) for key, value in os.environ.items()
            if key.lower() in ("http_proxy", "https_proxy", "all_proxy", "no_proxy",
                               "pip_index_url", "pip_extra_index_url", "pip_no_index", "pip_find_links")
        }
        with tempfile.TemporaryDirectory(prefix="rllm-package-probe-") as destination:
            data["package_index"] = run(
                ["pip", "--disable-pip-version-check", "download", "--no-deps",
                 "--only-binary=:all:", "--no-cache-dir", "--dest", destination,
                 "--retries", "0", "--timeout", "10", "setuptools"], timeout=30)
    print(json.dumps(data))


if __name__ == "__main__":
    main()

"""Filesystem-only compatibility for pristine NL2Repo images (Python 3.6+).

Run before untrusted source is overlaid, with -I -S: never execute setup.py,
site hooks, or generated code while holding setup privileges.
"""
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import sysconfig


def normalize(value):
    return re.sub(r"[-_.]+", "-", value).lower()


def remove_stale_editables(site_dirs, package_name):
    repairs = []
    for directory in site_dirs:
        for link in sorted(Path(directory).glob("*.egg-link")):
            if link.is_symlink() or normalize(link.stem) != normalize(package_name):
                continue
            content = link.read_text()
            lines = content.splitlines()
            if not lines:
                continue
            source = Path(lines[0])
            # Only obsolete absolute build roots of this target distribution.
            # Do not change live installations, dependencies, or workspace files.
            if source.parent != Path("/") or source.name in ("", "workspace") or source.exists() or source.is_symlink():
                continue
            pth = link.parent / "easy-install.pth"
            if pth.is_symlink():
                raise ValueError("refusing a symlinked easy-install.pth")
            if pth.is_file():
                text = pth.read_text()
                kept = [line for line in text.splitlines(keepends=True) if line.strip() != str(source)]
                if "".join(kept) != text:
                    pth.write_text("".join(kept))
            link.unlink()
            repairs.append({"kind": "stale_editable_removed", "metadata_path": str(link), "old_source": str(source)})
    return repairs


def expose_image_user_site(home, system_site, version):
    """Restore access to an image-owned user install under capability-free UID 0."""
    home = Path(home)
    local = home / ".local"
    site = local / "lib" / ("python" + version) / "site-packages"
    pytest = local / "bin" / "pytest"
    if not site.is_dir() or not pytest.is_file() or home.is_symlink() or local.is_symlink():
        raise ValueError("expected preinstalled pyautogui image user toolchain is missing")
    # Traverse the home, and read/execute the installed toolchain. Do not grant
    # write permissions or copy/install a newer pytest/dependency version.
    home.chmod(stat.S_IMODE(home.stat().st_mode) | stat.S_IXOTH)
    for root, dirs, files in os.walk(str(local), followlinks=False):
        for path in [Path(root)] + [Path(root) / name for name in files]:
            if path.is_symlink():
                continue
            mode = stat.S_IMODE(path.stat().st_mode)
            extra = stat.S_IROTH | (stat.S_IXOTH if path.is_dir() or mode & stat.S_IXUSR else 0)
            path.chmod(mode | extra)
    destination = Path(system_site) / "rllm_nl2repo_image_user_site.pth"
    if destination.is_symlink():
        raise ValueError("refusing a symlinked user-site registration")
    destination.write_text(str(site) + "\n")
    return {"kind": "image_user_site_exposed", "site": str(site), "pytest": str(pytest)}


def setup_distribution_name(path):
    tree = ast.parse(Path(path).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and (getattr(node.func, "id", None) == "setup" or getattr(node.func, "attr", None) == "setup"):
            for item in node.keywords:
                if item.arg == "name":
                    try:
                        value = ast.literal_eval(item.value)
                    except (ValueError, TypeError):
                        continue
                    if isinstance(value, str):
                        return value
    return None


def prepare(config):
    workdir = Path(config["workdir"])
    task_id = config.get("instance_id", "")
    if task_id == "boto" and (workdir / "setup.py").is_file():
        name = setup_distribution_name(workdir / "setup.py")
        if name == "boto3":
            return {"schema_version": 1, "ok": False, "reason": "nl2repo_asset_contract_mismatch",
                    "detail": "boto task image contains golden boto3 packaging; a matching benchmark asset is required",
                    "setup_sha256": hashlib.sha256((workdir / "setup.py").read_bytes()).hexdigest()}
    paths = sysconfig.get_paths()
    sites = sorted(set(paths[key] for key in ("purelib", "platlib")))
    repairs = remove_stale_editables(sites, config.get("package_name", ""))
    if task_id == "pyautogui":
        repairs.append(expose_image_user_site("/home/appuser", paths["purelib"], "%d.%d" % sys.version_info[:2]))
    return {"schema_version": 1, "ok": True, "repairs": repairs}


if __name__ == "__main__":
    try:
        result = prepare(json.loads(sys.argv[1]))
    except Exception as exc:
        result = {"schema_version": 1, "ok": False, "reason": "nl2repo_image_preparation_failed",
                  "detail": type(exc).__name__ + ": " + str(exc)}
    print(json.dumps(result))

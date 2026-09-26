"""Prepare package-manager permissions in repository-generation task images."""

# MiniSandbox already isolates and drops capabilities for every command.
# APT's additional _apt UID switch needs SETUID/SETGID, which it deliberately
# lacks. Configure APT's existing user setting; never elevate agent commands.
_APT_SETUP = r'''set -e
if command -v apt-get >/dev/null 2>&1; then
    mkdir -p /etc/apt/apt.conf.d
    printf 'APT::Sandbox::User "root";\n' > /etc/apt/apt.conf.d/99rllm-minisandbox
    for directory in /var/lib/apt/lists /var/cache/apt/archives; do
        mkdir -p "$directory/partial"
        chown -R 0:0 -- "$directory"
        chmod -R u+rwX -- "$directory"
    done
fi
'''


def configure_repo_generation_package_manager(sandbox, *, timeout: float = 300.0) -> None:
    """Prepare image-owned APT paths without installing or downloading anything."""
    setup = getattr(sandbox, "exec_setup", None)
    if callable(setup):
        setup(_APT_SETUP, timeout=timeout, user="root")



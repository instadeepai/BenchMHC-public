"""Module to define functions relative to RAM."""

import resource
from pathlib import Path

from bench_mhc.utils.logging import system

log = system.get(__name__)


def set_container_ram_limit() -> None:
    """Sets the RAM limit for the current process in a Docker container.

    This function is useful in containerized environments where we want to use the container's RAM
    limit rather than the host machine's RAM limit. It ensures that the Python process respects the
    container's memory constraints.
    """
    # The former path corresponds to cgroup v2 standard, the latter to cgroup v1
    for path in ["/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"]:
        if Path(path).exists():
            with open(path) as limit:
                try:
                    mem = int(limit.read())
                    resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
                    log.info(f"RAM limit set to {mem:,} bytes.")
                    break
                except ValueError:
                    pass  # When the limit is not an int but "max\n", the limit is not set

"""bench-mhc Python package."""

from dotenv import load_dotenv

# Set the variables defined in .env as environment variables
# Set it before importing the utils to ensure the environment variables are set
load_dotenv()

import bench_mhc.utils.pl_namespace  # noqa: E402, F401 ensure polars namespaces are registered
from bench_mhc.utils.ram import set_container_ram_limit  # noqa: E402

set_container_ram_limit()

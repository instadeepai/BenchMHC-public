"""Unit tests related to bench_mhc/utils/mhc.py."""

import json

from bench_mhc.constants import ALLELE_MAPPING_PATH
from bench_mhc.utils.mhc import format_allele_name


def test_the_shipped_mapping_survives_formatting() -> None:
    """Test every allele in the pseudo-sequence mapping formats to itself.

    The mapping's keys are already in the format `format_allele_name` produces, so
    formatting one again has to return it unchanged. Where it does not, the name a model
    looks up is no longer a key of the mapping it looks it up in, and the lookup raises.

    This is a property of the installed `mhcgnomes`, not of this repository: it holds up to
    mhcgnomes 3.18.0 and breaks from 3.19.0, which adds a leading zero to the gene field of
    macaque alleles (`Mamu__B01` becomes `Mamu__B001`). `pyproject.toml` pins 3.18.0; this
    is what notices if bumping that pin moves the names again.
    """
    mapping = json.loads(ALLELE_MAPPING_PATH.read_text())
    assert mapping, f"no alleles read from {ALLELE_MAPPING_PATH}"

    renamed = {
        allele: formatted
        for allele in mapping
        if (formatted := format_allele_name(allele)) != allele
    }

    assert not renamed, (
        f"{len(renamed)} of {len(mapping)} alleles no longer format to themselves, "
        f"{sum(1 for name in renamed.values() if name not in mapping)} of them to a name "
        f"the mapping does not have: {dict(list(renamed.items())[:3])}"
    )

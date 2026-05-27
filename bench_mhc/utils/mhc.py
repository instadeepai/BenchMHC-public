"""Module to define utils functions linked to MHC."""

import mhcgnomes

from bench_mhc.constants import SEPARATOR
from bench_mhc.utils.logging import system

log = system.get(__name__)

MHC2_ALLELE_PAIR_SEPARATOR = "="


def format_allele_name(allele: str) -> str:
    """Format an MHC allele name.

    The following format is used: '{species}__{compact_parsed_allele_string}'.

    Args:
        allele: (Maybe) un-formatted allele name.

    Returns:
        The formatted allele name or the allele in case it cannot be parsed by mhcgnomes.

    >>> format_allele_name('A0101')
    'HLA__A0101'

    >>> format_allele_name('HLA-A:02:01')
    'HLA__A0201'

    >>> format_allele_name('HLA__A0201')
    'HLA__A0201'

    >>> format_allele_name('HLA-DRB1*02:01')
    'HLA__DRB10201'

    >>> format_allele_name('DRA1*01:01-DRB1*02:01')
    'HLA__DRA0101=DRB10201'

    >>> format_allele_name('DRA1*01:01-DRB1*02:01')
    'HLA__DRA0101=DRB10201'

    >>> format_allele_name('DRA1*01:01=DRB1*02:01')
    'HLA__DRA0101=DRB10201'

    >>> format_allele_name('DPA10301-DPB10305')
    'HLA__DPA10301=DPB10305'

    >>> format_allele_name('DRA10101=DRB10201')
    'HLA__DRA10101=DRB10201'

    >>> format_allele_name('HLA-DQA10301_DQB10305')
    'HLA__DQA10301=DQB10305'

    >>> format_allele_name('DQB10631')
    'HLA__DQB10631'

    >>> format_allele_name('DPB10305')
    'HLA__DPB10305'

    >>> format_allele_name('BoLA-2*04601')
    'BoLA__204601'

    >>> format_allele_name('BoLA__2*04601')
    'BoLA__204601'

    >>> format_allele_name('Mamu-A2:0511')
    'Mamu__A20511'

    >>> format_allele_name('Mamu_A2:0511')
    'Mamu__A20511'

    >>> format_allele_name('unparsable')
    'unparsable'
    """
    allele = (
        allele.replace(SEPARATOR, "-").replace("_", "-").replace(MHC2_ALLELE_PAIR_SEPARATOR, "-")
    )

    try:
        parsed_allele = mhcgnomes.parse(allele)
        compact_allele = (
            parsed_allele.compact_string().replace("*", "").replace("-", MHC2_ALLELE_PAIR_SEPARATOR)
        )

        return f"{parsed_allele.species_prefix}{SEPARATOR}{compact_allele}"

    except mhcgnomes.ParseError:
        log.warning(f"Allele '{allele}' could not be parsed and hence won't be formatted.")

        return allele

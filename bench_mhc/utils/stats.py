"""Module used to track statistics."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bench_mhc.utils.io import save_json
from bench_mhc.utils.logging import system

log = system.get(__name__)


@dataclass(frozen=True)
class StatsConfig:
    """Configuration for the statistics.

    Attributes:
        item_name: Name of the item being tracked.
        process_name: Name of the process being tracked.
    """

    item_name: str
    process_name: str


class BaseStats:
    """Base class for tracking statistics.

    Attributes:
        config: Configuration for the statistics.
        _per_kmer_stats: Dictionary mapping kmer length to statistics for that length.
        _global_stats: Dictionary containing the global statistics.
    """

    def __init__(self, total_items: int, config: StatsConfig) -> None:
        """Initialize the statistics tracker.

        Args:
            total_items: Total number of items to process.
            config: Configuration for the statistics.
        """
        self.config = config
        self._per_kmer_stats: dict[int, dict[str, Any]] = {}
        self._global_stats: dict[str, Any] = {
            f"total_{self.config.item_name}": total_items,
            f"num_{self.config.item_name}_processed": 0,
        }

    @property
    def total_items(self) -> int:
        """Get total number of items."""
        return self._global_stats[f"total_{self.config.item_name}"]

    @property
    def num_items_processed(self) -> int:
        """Get number of items processed."""
        return self._global_stats[f"num_{self.config.item_name}_processed"]

    @num_items_processed.setter
    def num_items_processed(self, value: int) -> None:
        """Set number of items processed."""
        self._update_global_stats({f"num_{self.config.item_name}_processed": value})

    def _update_kmer_stats(self, kmer_length: int, stats_update: dict[str, Any]) -> None:
        """Update kmer statistics.

        Args:
            kmer_length: The length of the kmer to update statistics for.
            stats_update: Dictionary containing the statistics to update.
        """
        if kmer_length not in self._per_kmer_stats:
            self._per_kmer_stats[kmer_length] = {}
        self._per_kmer_stats[kmer_length].update(stats_update)

    def _update_global_stats(self, stats_update: dict[str, Any]) -> None:
        """Update global statistics.

        Args:
            stats_update: Dictionary containing the statistics to update.
        """
        self._global_stats.update(stats_update)

    def display_statistics(self, kmer_length: int | None) -> None:
        """Display the statistics in a formatted manner.

        Args:
            kmer_length: The length of the kmer to display statistics for. If None,
                display global statistics.
        """
        log.info(f"{self.config.process_name} statistics:")
        log.info("--------------------------------")

        if kmer_length is None:
            self._display_stats_section(
                title="Global statistics",
                stats=self._global_stats,
            )
        else:
            if kmer_length in self._per_kmer_stats:
                log.info("")
                log.info("Per-kmer statistics:")
                self._display_stats_section(
                    title=f"{kmer_length}-mers",
                    stats=self._per_kmer_stats[kmer_length],
                    indent="  ",
                )

        log.info("--------------------------------")

    def _display_stats_section(self, title: str, stats: dict[str, Any], indent: str = "") -> None:
        """Display a statistics section with formatted output.

        Args:
            title: Title of the statistics section.
            stats: Dictionary containing the statistics.
            indent: Indentation string for nested display.
        """
        log.info(f"{indent}{title}:")

        total = stats[f"total_{self.config.item_name}"]
        log.info(f"{indent}  Total {self.config.item_name}: {total:_}")

        self._display_additional_stats(stats, indent)

    def _display_additional_stats(self, stats: dict[str, Any], indent: str = "") -> None:
        """Hook for subclasses to display additional, specific statistics."""
        pass  # pragma: no cover

    def to_dict(self) -> dict[str, Any]:
        """Convert statistics to a dictionary format.

        Returns:
            Dictionary containing the global statistics and per-kmer statistics.
        """
        result = self._global_stats.copy()
        result["per_kmer_stats"] = self._per_kmer_stats

        return result

    def save_to_file(self, file_path: Path) -> None:
        """Save the statistics to a file.

        Args:
            file_path: Path to the file where the statistics will be saved.
        """
        save_json(self.to_dict(), file_path)


class SourceProteinMatchingStats(BaseStats):
    """Track statistics for the source protein matching process.

    This class inherits from BaseStats and extends it to track statistics about matching
    peptides to their source proteins in the human proteome and SwissProt databases.
    """

    def __init__(self, total_peptides: int) -> None:
        """Initialize the statistics tracker.

        Args:
            total_peptides: Total number of peptides to process.
        """
        config = StatsConfig(item_name="peptides", process_name="Source protein matching")
        super().__init__(total_peptides, config)

        super()._update_global_stats(
            {
                "total_unmatched_human_proteome_peptides": 0,
                "total_unmatched_swissprot_human_proteome_peptides": 0,
            }
        )

    @property
    def total_unmatched_human_proteome_peptides(self) -> int:
        """Get total number of unmatched human proteome."""
        return self._global_stats["total_unmatched_human_proteome_peptides"]

    @property
    def total_unmatched_swissprot_human_proteome_peptides(self) -> int:
        """Get total number of unmatched SwissProt."""
        return self._global_stats["total_unmatched_swissprot_human_proteome_peptides"]

    def update_kmer_stats(
        self,
        kmer_length: int,
        peptides_in_kmer: int,
        matched_peptides: int,
        unmatched_peptides: int,
        unmatched_after_blastp: int,
    ) -> None:
        """Update statistics for a specific kmer length.

        Args:
            kmer_length: Length of the kmer being processed.
            peptides_in_kmer: Total number of peptides for this kmer length.
            matched_peptides: Number of peptides matched to human proteome.
            unmatched_peptides: Number of peptides unmatched to human proteome.
            unmatched_after_blastp: Number of peptides unmatched after BLASTP.
        """
        stats_update = {
            "total_peptides": peptides_in_kmer,
            "matched_peptides": matched_peptides,
            "unmatched_peptides": unmatched_peptides,
            "unmatched_after_blastp": unmatched_after_blastp,
        }
        super()._update_kmer_stats(kmer_length, stats_update)

    def update_global_stats(
        self,
        unmatched_human_proteome_peptides: int,
        unmatched_swissprot_human_proteome_peptides: int,
        num_peptides_processed: int,
    ) -> None:
        """Update global statistics.

        Args:
            unmatched_human_proteome_peptides: Number of unmatched human proteome peptides.
            unmatched_swissprot_human_proteome_peptides: Number of unmatched SwissProt human
                proteome peptides.
            num_peptides_processed: Number of peptides processed.
        """
        stats_update = {
            "total_unmatched_human_proteome_peptides": (
                self.total_unmatched_human_proteome_peptides + unmatched_human_proteome_peptides
            ),
            "total_unmatched_swissprot_human_proteome_peptides": (
                self.total_unmatched_swissprot_human_proteome_peptides
                + unmatched_swissprot_human_proteome_peptides
            ),
            "num_peptides_processed": num_peptides_processed,
        }
        super()._update_global_stats(stats_update)

    def _display_additional_stats(self, stats: dict[str, Any], indent: str = "") -> None:
        """Display the detailed matching statistics for peptides."""
        total = stats["total_peptides"]
        if total == 0:
            return

        # If the stats are per-kmer
        if "matched_peptides" in stats:
            matched = stats["matched_peptides"]
            unmatched = stats["unmatched_peptides"]
            unmatched_after_blastp = stats["unmatched_after_blastp"]
        # If the stats are global
        else:
            unmatched = self.total_unmatched_human_proteome_peptides
            matched = self.total_items - unmatched
            unmatched_after_blastp = self.total_unmatched_swissprot_human_proteome_peptides

        matched_pct = (matched / total) * 100
        unmatched_pct = (unmatched / total) * 100
        pseudo_matched_pct = ((unmatched - unmatched_after_blastp) / total) * 100
        no_matches_pct = (unmatched_after_blastp / total) * 100

        log.info(
            f"{indent}  Peptides matched to human proteome: " f"{matched:_} ({matched_pct:.2f}%)"
        )
        log.info(
            f"{indent}  Peptides unmatched to human proteome: "
            f"{unmatched:_} ({unmatched_pct:.2f}%)"
        )
        log.info(
            f"{indent}  Unmatched human proteome peptides pseudo-matched in SwissProt + human "
            f"proteome: {unmatched - unmatched_after_blastp:_} ({pseudo_matched_pct:.2f}%)"
        )
        log.info(
            f"{indent}  Peptides without any pseudo match: "
            f"{unmatched_after_blastp:_} ({no_matches_pct:.2f}%)"
        )

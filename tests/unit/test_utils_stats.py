"""Unit tests for bench_mhc/utils/stats.py."""

from unittest.mock import MagicMock
from unittest.mock import patch

from bench_mhc.utils.stats import SourceProteinMatchingStats


class TestSourceProteinMatchingStats:
    """Test the SourceProteinMatchingStats class."""

    def test_init(self) -> None:
        """Test the specific initialization of SourceProteinMatchingStats."""
        stats = SourceProteinMatchingStats(1000)

        assert stats._global_stats["total_peptides"] == 1000
        assert stats._global_stats["num_peptides_processed"] == 0
        assert stats._global_stats["total_unmatched_human_proteome_peptides"] == 0
        assert stats._global_stats["total_unmatched_swissprot_human_proteome_peptides"] == 0
        assert stats._per_kmer_stats == {}

    def test_update_kmer_stats(self) -> None:
        """Test updating kmer statistics with specific fields."""
        stats = SourceProteinMatchingStats(1000)

        stats.update_kmer_stats(
            kmer_length=9,
            peptides_in_kmer=500,
            matched_peptides=400,
            unmatched_peptides=100,
            unmatched_after_blastp=20,
        )
        expected_stats = {
            "total_peptides": 500,
            "matched_peptides": 400,
            "unmatched_peptides": 100,
            "unmatched_after_blastp": 20,
        }
        assert stats._per_kmer_stats[9] == expected_stats

        stats.update_kmer_stats(
            kmer_length=10,
            peptides_in_kmer=300,
            matched_peptides=200,
            unmatched_peptides=100,
            unmatched_after_blastp=20,
        )
        expected_stats = {
            "total_peptides": 300,
            "matched_peptides": 200,
            "unmatched_peptides": 100,
            "unmatched_after_blastp": 20,
        }
        assert stats._per_kmer_stats[10] == expected_stats
        assert len(stats._per_kmer_stats) == 2

    def test_update_global_stats(self) -> None:
        """Test updating global statistics with accumulation logic."""
        stats = SourceProteinMatchingStats(1000)

        stats.update_global_stats(
            unmatched_human_proteome_peptides=100,
            unmatched_swissprot_human_proteome_peptides=20,
            num_peptides_processed=500,
        )

        assert stats.total_unmatched_human_proteome_peptides == 100
        assert stats.total_unmatched_swissprot_human_proteome_peptides == 20
        assert stats.num_items_processed == 500

    @patch("bench_mhc.utils.stats.log")
    def test_display_statistics_global(self, mock_log: MagicMock) -> None:
        """Test displaying global statistics with percentage calculations."""
        # Test case where total == 0
        stats = SourceProteinMatchingStats(0)
        stats.display_statistics(kmer_length=None)
        log_calls = [call[0][0] for call in mock_log.info.call_args_list]
        expected_log_calls = [
            "Source protein matching statistics:",
            "--------------------------------",
            "Global statistics:",
            "  Total peptides: 0",
            "--------------------------------",
        ]
        assert log_calls == expected_log_calls

        # Test case where total > 0
        mock_log.reset_mock()
        stats = SourceProteinMatchingStats(1000)

        stats.update_global_stats(
            unmatched_human_proteome_peptides=200,
            unmatched_swissprot_human_proteome_peptides=50,
            num_peptides_processed=1000,
        )
        stats.display_statistics(kmer_length=None)

        # Check that log.info was called with the expected messages
        log_calls = [call[0][0] for call in mock_log.info.call_args_list]
        expected_log_calls = [
            "Source protein matching statistics:",
            "--------------------------------",
            "Global statistics:",
            "  Total peptides: 1_000",
            "  Peptides matched to human proteome: 800 (80.00%)",
            "  Peptides unmatched to human proteome: 200 (20.00%)",
            (
                "  Unmatched human proteome peptides pseudo-matched in SwissProt "
                "+ human proteome: 150 (15.00%)"
            ),
            "  Peptides without any pseudo match: 50 (5.00%)",
            "--------------------------------",
        ]

        assert log_calls == expected_log_calls

    @patch("bench_mhc.utils.stats.log")
    def test_display_statistics_per_kmer(self, mock_log: MagicMock) -> None:
        """Test displaying per-kmer statistics with percentages."""
        stats = SourceProteinMatchingStats(1000)

        stats.update_kmer_stats(9, 500, 400, 100, 20)
        stats.display_statistics(kmer_length=9)

        # Check that log.info was called with the expected messages
        log_calls = [call[0][0] for call in mock_log.info.call_args_list]
        expected_log_calls = [
            "Source protein matching statistics:",
            "--------------------------------",
            "",
            "Per-kmer statistics:",
            "  9-mers:",
            "    Total peptides: 500",
            "    Peptides matched to human proteome: 400 (80.00%)",
            "    Peptides unmatched to human proteome: 100 (20.00%)",
            (
                "    Unmatched human proteome peptides pseudo-matched in SwissProt "
                "+ human proteome: 80 (16.00%)"
            ),
            "    Peptides without any pseudo match: 20 (4.00%)",
            "--------------------------------",
        ]

        assert log_calls == expected_log_calls

    @patch("bench_mhc.utils.stats.log")
    def test_display_statistics_percentage_calculations(self, mock_log: MagicMock) -> None:
        """Test that percentage calculations are correct in display."""
        stats = SourceProteinMatchingStats(1000)
        stats._global_stats["total_unmatched_human_proteome_peptides"] = 200
        stats._global_stats["total_unmatched_swissprot_human_proteome_peptides"] = 50

        stats.display_statistics(kmer_length=None)

        # Check that the percentages are calculated correctly
        log_calls = [call[0][0] for call in mock_log.info.call_args_list]

        # Find the percentage lines
        matched_line = next(
            line for line in log_calls if "Peptides matched to human proteome:" in line
        )
        unmatched_line = next(
            line for line in log_calls if "Peptides unmatched to human proteome:" in line
        )
        pseudo_matched_line = next(line for line in log_calls if "pseudo-matched" in line)
        no_matches_line = next(
            line for line in log_calls if "Peptides without any pseudo match:" in line
        )

        # Expected percentages:
        # matched: (1000-200)/1000 = 80%
        # unmatched: 200/1000 = 20%
        # pseudo_matched: (200-50)/1000 = 15%
        # no_matches: 50/1000 = 5%

        assert "800 (80.00%)" in matched_line
        assert "200 (20.00%)" in unmatched_line
        assert "150 (15.00%)" in pseudo_matched_line
        assert "50 (5.00%)" in no_matches_line

    def test_to_dict(self) -> None:
        """Test converting statistics to dictionary with specific keys."""
        stats = SourceProteinMatchingStats(1000)

        stats.update_global_stats(
            unmatched_human_proteome_peptides=200,
            unmatched_swissprot_human_proteome_peptides=50,
            num_peptides_processed=800,
        )
        result = stats.to_dict()
        expected = {
            "total_peptides": 1000,
            "num_peptides_processed": 800,
            "total_unmatched_human_proteome_peptides": 200,
            "total_unmatched_swissprot_human_proteome_peptides": 50,
            "per_kmer_stats": {},
        }
        assert result == expected

    def test_statistics_accumulation(self) -> None:
        """Test that statistics accumulate correctly over multiple updates."""
        stats = SourceProteinMatchingStats(1000)

        updates = [(50, 10, 200), (30, 5, 400), (20, 3, 600)]

        for unmatched_human, unmatched_swissprot, processed in updates:
            stats.update_global_stats(
                unmatched_human_proteome_peptides=unmatched_human,
                unmatched_swissprot_human_proteome_peptides=unmatched_swissprot,
                num_peptides_processed=processed,
            )

        assert stats.total_unmatched_human_proteome_peptides == 100  # 50 + 30 + 20
        assert stats.total_unmatched_swissprot_human_proteome_peptides == 18  # 10 + 5 + 3
        assert stats.num_items_processed == 600  # 200 + 400 + 600

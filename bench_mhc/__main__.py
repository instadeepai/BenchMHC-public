"""bench-mhc entry point."""

import rich_click as click

from bench_mhc.cli.assign_protein_features_command import assign_protein_features
from bench_mhc.cli.calibrate_command import calibrate
from bench_mhc.cli.compute_nnalign_features_command import compute_nnalign_features
from bench_mhc.cli.evaluate_command import evaluate
from bench_mhc.cli.format_allele_feature_command import format_allele_feature
from bench_mhc.cli.generate_decoys_command import generate_decoys
from bench_mhc.cli.generate_performance_plot_command import generate_performance_plot
from bench_mhc.cli.predict_command import predict
from bench_mhc.cli.refine_peptides_command import refine_peptides
from bench_mhc.cli.train_command import train


@click.group()
@click.version_option()
def main() -> None:
    """bench-mhc is a CLI tool to model peptide-MHC presentation."""


main.add_command(compute_nnalign_features)
main.add_command(format_allele_feature)
main.add_command(train)
main.add_command(predict)
main.add_command(evaluate)
main.add_command(generate_performance_plot)
main.add_command(calibrate)
main.add_command(generate_decoys)
main.add_command(assign_protein_features)
main.add_command(refine_peptides)

if __name__ == "__main__":
    main()  # pragma: no cover

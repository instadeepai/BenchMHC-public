"""Module to define command line arguments."""

import functools
from collections.abc import Callable
from pathlib import Path
from typing import Any

import rich_click as click

from bench_mhc.constants import FLANK_SIZE
from bench_mhc.constants import MODEL_DIRECTORY
from bench_mhc.utils.click.callbacks import format_gpu_arg
from bench_mhc.utils.click.callbacks import validate_strict_positive
from bench_mhc.utils.click.types import CommaSepOrFileToSet
from bench_mhc.utils.click.types import RelativePath


def dataset_path(func: Callable) -> Callable:
    """Define the --dataset_path argument."""

    @click.option(
        "--dataset_path",
        "-d",
        type=click.Path(exists=True, path_type=Path),
        required=True,
        help="Path to the dataset.",
    )
    @functools.wraps(func)
    def _wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    return _wrapper


def output_file_path(func: Callable) -> Callable:
    """Define the --output_file_path argument."""

    @click.option(
        "--output_file_path",
        "-o",
        type=click.Path(path_type=Path),
        required=False,
        default=None,
        help=(
            "Path to the output file. "
            "If not provided, file provided with --dataset_path will be used."
        ),
    )
    @functools.wraps(func)
    def _wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    return _wrapper


def force(func: Callable) -> Callable:
    """Define the --force argument."""

    @click.option(
        "--force",
        "-f",
        type=bool,
        is_flag=True,
        help="Whether to override the output file if it already exists.",
    )
    @functools.wraps(func)
    def _wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    return _wrapper


def configuration_file_path(func: Callable) -> Callable:
    """Define the --configuration_file_path argument."""

    @click.option(
        "--configuration_file_path",
        "-c",
        type=click.Path(exists=True, path_type=Path),
        required=True,
        help="File path of the configuration to load.",
    )
    @functools.wraps(func)
    def _wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    return _wrapper


def experiment_name(func: Callable) -> Callable:
    """Define the --experiment_name argument."""

    @click.option(
        "--experiment_name",
        "-n",
        type=str,
        required=True,
        help=(
            "Name of the experiment which is used as a prefix of the model name "
            "(i.e. folder name): {experiment_name}__{%Y%m%d-%H%M%S}__{uuid}."
        ),
    )
    @functools.wraps(func)
    def _wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    return _wrapper


def training_path(func: Callable) -> Callable:
    """Define the --training_path argument."""

    @click.option(
        "--training_path",
        "-t",
        type=click.Path(exists=True, path_type=Path),
        required=True,
        help="Path to the single-allelic (SA) training dataset.",
    )
    @functools.wraps(func)
    def _wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    return _wrapper


def validation_path(func: Callable) -> Callable:
    """Define the --validation_path argument."""

    @click.option(
        "--validation_path",
        "-v",
        type=click.Path(exists=True, path_type=Path),
        required=True,
        help="Path to the single-allelic (SA) validation dataset.",
    )
    @functools.wraps(func)
    def _wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    return _wrapper


def ma_training_path(func: Callable) -> Callable:
    """Define the --ma_training_path argument."""

    @click.option(
        "--ma_training_path",
        "-ma_t",
        type=click.Path(exists=True, path_type=Path),
        required=False,
        help=(
            "Path to the multi-allelic (MA) training dataset. "
            "If provided, training is performed with iterative annotation on SA and MA data. "
            "The data must be in along format, i.e. the allele column is 'explode' such that "
            "each row is a peptide-allele combination."
        ),
    )
    @functools.wraps(func)
    def _wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    return _wrapper


def ma_validation_path(func: Callable) -> Callable:
    """Define the --ma_validation_path argument."""

    @click.option(
        "--ma_validation_path",
        "-ma_v",
        type=click.Path(exists=True, path_type=Path),
        required=False,
        help=(
            "Path to the multi-allelic (MA) validation dataset. "
            "The data must be in along format, i.e. the allele column is 'explode' such that "
            "each row is a peptide-allele combination."
        ),
    )
    @functools.wraps(func)
    def _wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    return _wrapper


def random_seed(func: Callable) -> Callable:
    """Define the --random_seed argument."""

    @click.option(
        "--random_seed",
        "-rs",
        type=int,
        required=False,
        default=None,
        help=(
            "Random seed which is set at the beginning of the training, for reproducibility "
            "between experiments."
        ),
    )
    @functools.wraps(func)
    def _wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    return _wrapper


def model_path(func: Callable) -> Callable:
    """Define the --model_path argument."""

    @click.option(
        "--model_path",
        "-m",
        type=CommaSepOrFileToSet(
            item_click_type=RelativePath(
                relative_to=MODEL_DIRECTORY, strict=False, exists=True, path_type=Path
            )
        ),
        required=True,
        multiple=True,
        help=(
            "Path(s) to the model(s). Multiple models can be provided in several ways: "
            "1. Multiple -m flags: -m path/to/model/1 -m path/to/model/2 ; "
            "2. Comma-separated list: -m 'path/to/model/1,path/to/model/2' ; "
            "3. Text file with one model per line: -m path/to/file/with/models.txt. "
            "The individual paths can be relative to the current working directory, "
            "or to the models/ directory."
        ),
    )
    @functools.wraps(func)
    def _wrapper(*args: Any, **kwargs: Any) -> Any:
        """Wrapper to define the --model_path argument.

        We flatten the sets of models into a single set in case models are provided with
        multiple flags.
        """
        model_paths = kwargs.get("model_path", [])
        if len(model_paths) > 1:
            kwargs["model_path"] = set().union(*model_paths)
        else:
            kwargs["model_path"] = model_paths[0]

        return func(*args, **kwargs)

    return _wrapper


def predictions_column_prefix(func: Callable) -> Callable:
    """Define the --predictions_column_prefix argument."""

    @click.option(
        "--predictions_column_prefix",
        "-p_pref",
        type=str,
        required=True,
        help="Prefix of the prediction column to append to the dataset, "
        "e.g. the output predictions will be saved under "
        "'{predictions_column_prefix}__{output_name}'.",
    )
    @functools.wraps(func)
    def _wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    return _wrapper


def batch_size(func: Callable) -> Callable:
    """Define the --batch_size argument."""

    @click.option(
        "--batch_size",
        "-bs",
        type=int,
        required=False,
        default=512,
        help="Batch size to use for inference.",
    )
    @functools.wraps(func)
    def _wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    return _wrapper


def num_workers(func: Callable) -> Callable:
    """Define the --num_workers argument."""

    @click.option(
        "--num_workers",
        type=int,
        required=False,
        default=0,
        help="Number of workers to use.",
    )
    @functools.wraps(func)
    def _wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    return _wrapper


def sa_warmup_epochs(func: Callable) -> Callable:
    """Define the --sa_warmup_epochs argument."""

    @click.option(
        "--sa_warmup_epochs",
        "-e",
        type=int,
        required=False,
        default=20,
        help=(
            "Number of epochs to pre-train (warmup) the model on SA data. "
            "Only relevant for iterative annotation."
        ),
    )
    @functools.wraps(func)
    def _wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    return _wrapper


def deconvolution_identifier(func: Callable) -> Callable:
    """Define the --deconvolution_identifier argument."""

    @click.option(
        "--deconvolution_identifier",
        "-d_id",
        type=str,
        default="deconvolution_identifier",
        required=False,
        help=(
            "Column name to use for the deconvolution. The latter is used to select one sample "
            "for each unique valued in the provided column, often using the sample yielding the "
            "highest score. Only relevant for iterative annotation."
        ),
    )
    @functools.wraps(func)
    def _wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    return _wrapper


def prediction_identifier(func: Callable) -> Callable:
    """Define the --prediction_identifier argument."""

    @click.option(
        "--prediction_identifier",
        "-p_id",
        type=str,
        required=True,
        help=(
            "Name of the column containing predictions. In case it contains the `pctrnk` "
            "sub-string, the column will be transformed into scores between [0,1] during the "
            "evaluation process, with the following transformation: `1 - rank/100`."
        ),
    )
    @functools.wraps(func)
    def _wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    return _wrapper


def deconvolution_output_name(func: Callable) -> Callable:
    """Define the --deconvolution_output_name argument."""

    @click.option(
        "--deconvolution_output_name",
        type=str,
        required=False,
        default="hit",
        help=(
            "Name of the output to consider for deconvolution (MA data annotation). "
            "It must correspond to a binary output variable of the model."
        ),
    )
    @functools.wraps(func)
    def _wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    return _wrapper


def use_prediction_score_rescaling(func: Callable) -> Callable:
    """Define the --use_prediction_score_rescaling argument."""

    @click.option(
        "--use_prediction_score_rescaling",
        required=False,
        is_flag=True,
        help=(
            "Whether to rescale predictions from training MA data, prior to deconvolution "
            "(MA data annotation). This applies only to positive training MA data, unless "
            "--use_max_negative_allele_annotation is used. In that case, all training MA data is "
            "rescaled. Only relevant for iterative annotation."
        ),
    )
    @functools.wraps(func)
    def _wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    return _wrapper


def reference_path(func: Callable) -> Callable:
    """Define the --reference_path argument."""

    @click.option(
        "--reference_path",
        "-r",
        type=click.Path(exists=True, path_type=Path),
        required=False,
        help="Path to the file containing reference data.",
    )
    @functools.wraps(func)
    def _wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    return _wrapper


def w_shift_parameter(func: Callable) -> Callable:
    """Define the --w_shift_parameter argument."""

    @click.option(
        "--w_shift_parameter",
        type=int,
        required=False,
        default=75,
        help=(
            "Shift value in the w calculation for the prediction score rescaling step. "
            "Only relevant for iterative annotation."
        ),
        callback=validate_strict_positive,
    )
    @functools.wraps(func)
    def _wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    return _wrapper


def z_score_threshold(func: Callable) -> Callable:
    """Define the --z_score_threshold argument."""

    @click.option(
        "--z_score_threshold",
        type=float,
        required=False,
        default=3.0,
        help=(
            "Threshold to use to filter out z-score outliers for the prediction score "
            "rescaling step. Only relevant for iterative annotation."
        ),
        callback=validate_strict_positive,
    )
    @functools.wraps(func)
    def _wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    return _wrapper


def target_identifier(func: Callable) -> Callable:
    """Define the --target_identifier argument."""

    @click.option(
        "--target_identifier",
        "-t_id",
        type=str,
        required=True,
        help="Name of the column containing target values.",
    )
    @functools.wraps(func)
    def _wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    return _wrapper


def output_directory(func: Callable) -> Callable:
    """Define the --output_directory argument."""

    def _create_directory(
        ctx: click.Context,  # noqa: ARG001
        param: click.Parameter,  # noqa: ARG001
        value: Path | None,
    ) -> Path | None:
        """Create the output directory if it doesn't exist.

        Args:
            ctx: Click context.
            param: Click parameter.
            value: The directory path value.

        Returns:
            The directory path value.
        """
        if value is not None:
            value.mkdir(parents=True, exist_ok=True)

        return value

    @click.option(
        "--output_directory",
        "-o",
        type=click.Path(path_type=Path),
        required=True,
        callback=_create_directory,
        help="Directory where to save the evaluation results.",
    )
    @functools.wraps(func)
    def _wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    return _wrapper


def allele_column_name(func: Callable) -> Callable:
    """Define the --allele_column_name argument."""

    @click.option(
        "--allele_column_name",
        "-a_id",
        type=str,
        required=False,
        default="allele",
        help="Name of the allele column.",
    )
    @functools.wraps(func)
    def _wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    return _wrapper


def gpus(func: Callable) -> Callable:
    """Define the --gpus argument."""

    @click.option(
        "--gpus",
        type=str,
        required=False,
        default=None,
        callback=format_gpu_arg,
        help="GPU configuration. Can be a single integer for GPU id or comma-separated "
        "integers for specific GPU indices. If not provided, falls back to CPU training.",
    )
    @functools.wraps(func)
    def _wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    return _wrapper


def with_frank_score(func: Callable) -> Callable:
    """Define the --with_frank_score argument."""

    @click.option(
        "--with_frank_score",
        type=bool,
        required=False,
        is_flag=True,
        help=(
            "Whether to evaluate with Frank Score. "
            "If provided, only the Frank score will be computed. "
            "Otherwise, all binary metrics will be computed."
        ),
    )
    @functools.wraps(func)
    def _wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    return _wrapper


def group_identifier(func: Callable) -> Callable:
    """Define the --group_identifier argument."""

    @click.option(
        "--group_identifier",
        "-g_id",
        type=str,
        required=False,
        default="allele",
        help=(
            "Name of the column with group identifiers to compute metrics per group. "
            "If not provided, metrics will try to be computed per allele."
        ),
    )
    @functools.wraps(func)
    def _wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    return _wrapper


def cd8_epitopes(func: Callable) -> Callable:
    """Define the --cd8_epitopes argument."""

    @click.option(
        "--cd8_epitopes",
        type=bool,
        required=False,
        is_flag=True,
        help="Whether CD8 epitopes data is the input.",
    )
    @functools.wraps(func)
    def _wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    return _wrapper


def ms_ligands(func: Callable) -> Callable:
    """Define the --ms_ligands argument."""

    @click.option(
        "--ms_ligands",
        type=bool,
        required=False,
        is_flag=True,
        help="Whether MS ligands data is the input.",
    )
    @functools.wraps(func)
    def _wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    return _wrapper


def alleles(func: Callable) -> Callable:
    """Define the --alleles argument."""

    @click.option(
        "--alleles",
        "-a",
        type=CommaSepOrFileToSet(),
        required=False,
        help=(
            "List of alleles to calibrate. Can be either: "
            "1. A path to a text file with one allele per line. "
            "2. Comma-separated list: -a 'allele1,allele2'"
        ),
    )
    @functools.wraps(func)
    def _wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    return _wrapper


def percentile_rank_directory(func: Callable) -> Callable:
    """Define the --percentile_rank_directory argument."""

    @click.option(
        "--percentile_rank_directory",
        "-pctrnk_dir",
        type=click.Path(exists=True, path_type=Path),
        required=False,
        help="Directory containing the percentile rank files.",
    )
    @functools.wraps(func)
    def _wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    return _wrapper


def hits_path(func: Callable) -> Callable:
    """Define the --hits_path argument."""

    @click.option(
        "--hits_path",
        type=click.Path(exists=True, path_type=Path),
        required=True,
        help="Path to CSV file containing hits with a 'peptide' and 'allele' column.",
    )
    @functools.wraps(func)
    def _wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    return _wrapper


def human_proteome_path(func: Callable) -> Callable:
    """Define the --human_proteome_path argument."""

    @click.option(
        "--human_proteome_path",
        type=click.Path(exists=True, path_type=Path),
        required=True,
        help="Path to `hg38.fasta` human proteome FASTA file containing human protein sequences.",
    )
    @functools.wraps(func)
    def _wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    return _wrapper


def num_decoys(func: Callable) -> Callable:
    """Define the --num_decoys argument."""

    @click.option(
        "--num_decoys",
        type=int,
        required=True,
        help="Number of decoys to generate per hit (hit:decoy ratio).",
    )
    @functools.wraps(func)
    def _wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    return _wrapper


def swissprot_db_path(func: Callable) -> Callable:
    """Define the --swissprot_db_path argument."""

    @click.option(
        "--swissprot_db_path",
        type=click.Path(exists=True, path_type=Path),
        required=True,
        help=(
            "Path to SwissProt database for BLASTP search. "
            "You can download the swissprot database from: "
            "https://ftp.ebi.ac.uk/pub/databases/uniprot/current_release/knowledgebase/complete/"
        ),
    )
    @functools.wraps(func)
    def _wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    return _wrapper


def with_columns(func: Callable) -> Callable:
    """Define the --with_columns argument."""

    @click.option(
        "--with_columns",
        "-w_cols",
        type=CommaSepOrFileToSet(),
        required=False,
        default=None,
        help=(
            "Optional columns to copy from hits to decoys. Can be either: "
            "1. A path to a text file with one column name per line. "
            "2. Comma-separated list: --with_columns 'col1,col2,col3'. "
            "3. Value 'all' to return all columns. "
        ),
    )
    @functools.wraps(func)
    def _wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    return _wrapper


def peptides_path(func: Callable) -> Callable:
    """Define the --peptides_path argument."""

    @click.option(
        "--peptides_path",
        type=click.Path(exists=True, path_type=Path),
        required=True,
        help="Path to CSV file containing peptides with a 'peptide' column.",
    )
    @functools.wraps(func)
    def _wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    return _wrapper


def flank_size(func: Callable) -> Callable:
    """Decorator used to define --flank_size parameter for a command line."""

    @click.option(
        "--flank_size",
        help="Size of the flanks to consider.",
        default=FLANK_SIZE,
        required=False,
        type=int,
    )
    @functools.wraps(func)
    def _wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    return _wrapper


def remove_multiple_matches(func: Callable) -> Callable:
    """Define the --remove_multiple_matches argument."""

    @click.option(
        "--remove_multiple_matches",
        type=bool,
        required=False,
        is_flag=True,
        help=(
            "Whether match with multiple source proteins is allowed. If not, one protein is "
            "randomly sampled with its associated flanks and species."
        ),
    )
    @functools.wraps(func)
    def _wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    return _wrapper


def remove_imperfect_matching(func: Callable) -> Callable:
    """Define the --remove_imperfect_matching argument."""

    @click.option(
        "--remove_imperfect_matching",
        type=bool,
        required=False,
        is_flag=True,
        help=(
            "Whether we remove peptides with imperfect match. Mutually exclusive with "
            "'nullify_imperfect_flanks'"
        ),
    )
    @functools.wraps(func)
    def _wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    return _wrapper


def nullify_imperfect_flanks(func: Callable) -> Callable:
    """Define the --nullify_imperfect_flanks argument."""

    @click.option(
        "--nullify_imperfect_flanks",
        type=bool,
        required=False,
        is_flag=True,
        help=(
            "Whether we replace imperfect flanks with padding token. Mutually exclusive with "
            "'remove_imperfect_matching'."
        ),
    )
    @functools.wraps(func)
    def _wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    return _wrapper


def peptides_only(func: Callable) -> Callable:
    """Define the --peptides_only argument."""

    @click.option(
        "--peptides_only",
        type=bool,
        required=False,
        is_flag=True,
        help=(
            "Build reference set from peptides and alleles. "
            "If set, creates Cartesian product of peptides and alleles. "
            "If not set, uses prebuilt reference set directly from file."
        ),
    )
    @functools.wraps(func)
    def _wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    return _wrapper

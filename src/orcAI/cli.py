import click
from pathlib import Path
from importlib.resources import files
from orcAI.train import train
from orcAI.predict import predict
from orcAI.spectrogram import create_spectrograms
from orcAI.annotation import create_label_arrays

ClickDirPathR = click.Path(
    exists=True, file_okay=False, readable=True, resolve_path=True, path_type=Path
)
ClickDirPathW = click.Path(
    exists=True, file_okay=False, writable=True, resolve_path=True, path_type=Path
)
ClickFilePathR = click.Path(
    exists=True, dir_okay=False, readable=True, resolve_path=True, path_type=Path
)
ClickFilePathW = click.Path(
    exists=False, dir_okay=False, writable=True, resolve_path=True, path_type=Path
)


@click.group(
    # fmt: off
    help="\n\b\n"
    + "            █████  " + click.style("Command line interface for ", bold=True) + click.style("orcAI", fg="blue", bold=True) + "\n"
    + "███ ███   ████████ " + "  a tool for \n"
    + "  ████  ████░██░░░ " + "  training, testing & applying AI models \n"
    + "    ██████████░░░  " + "  to detect acoustic signals in spectrograms generated from audio recordings.\n"
    + "     ░░██░░░░      \n"
    + "      ███ ██       " + "  Reference:" +"\n",
    # TODO: Add reference
    # fmt: on
    epilog="For further information see the help pages of the individual subcommands (e.g. "
    + click.style("orcai train --help", italic=True)
    + ") and/or visit: https://gitlab.ethz.ch/seb/orcai_test",
)
def cli():
    pass


@cli.command(
    name="train",
    help="Train a model on the given data",
    short_help="Train a model on the given data",
    no_args_is_help=True,
    epilog="For further information visit: https://gitlab.ethz.ch/seb/orcai_test",
)
@click.option(
    "--data_dir",
    "-d",
    help="Path to the directory containing the training, validation and test datasets",
    required=True,
    type=ClickDirPathR,
)
@click.option(
    "--output_dir",
    "-o",
    help="Path to the output directory",
    required=True,
    type=ClickDirPathW,
)
@click.option(
    "-m",
    "--model_parameter",
    "model_parameter",
    help="Path to a JSON file containing model specifications",
    type=ClickFilePathR,
    default=str(files("orcAI.defaults").joinpath("default_model_parameter.json")),
    show_default=True,
)
@click.option(
    "-lc",
    "--label_calls",
    "label_calls",
    help="Path to a JSON file containing calls for labeling",
    type=ClickFilePathR,
    default=str(files("orcAI.defaults").joinpath("default_calls.json")),
    show_default=True,
)
@click.option(
    "-lw",
    "--load_weights",
    is_flag=True,
    default=False,
    show_default=True,
    help="Load weights and continue fitting",
)
@click.option(
    "-tp",
    "--transformer_parallel",
    is_flag=True,
    default=False,
    show_default=True,
    help="Use transformer parallelization",
)
@click.option(
    "-v", "--verbosity", type=click.IntRange(0, 2), default=1, show_default=True
)
def cli_train(**kwargs):
    train(**kwargs)


@cli.command(
    name="predict",
    help="Predicts call annotations in the wav file at WAV_FILE_PATH.",
    short_help="Predicts call annotations.",
    no_args_is_help=True,
    epilog="For further information visit: https://gitlab.ethz.ch/seb/orcai_test",
)
@click.argument("wav_file_path", type=ClickFilePathR)
@click.option(
    "--model",
    "-m",
    "model_path",
    type=ClickDirPathR,
    default=files("orcAI.defaults").joinpath("orcai_Orca_1_0_0"),
    show_default="orcai_Orca_1_0_0",
    help="Path to the model directory.",
)
@click.option(
    "--output_file",
    "-o",
    type=ClickFilePathW,
    default=None,
    show_default="None",
    help="Path to the output file or None if the output file should be saved in the same directory as the wav file.",
)
@click.option(
    "--spectrogram_parameter",
    "-sp",
    type=ClickFilePathR,
    default=files("orcAI.defaults").joinpath("default_spectrogram_parameter.json"),
    show_default="default_spectrogram_parameter.json",
    help="Path to the spectrogram parameter file.",
)
@click.option(
    "--verbosity",
    "-v",
    type=click.IntRange(0, 2),
    default=2,
    show_default=True,
    help="Verbosity level.",
)
def cli_predict(**kwargs):
    predict(**kwargs)


@cli.command(
    name="create-spectrograms",
    help="Creates spectrograms for all files in SPECTROGRAM_TABLE_PATH",
    short_help="Creates spectrograms for all files in spectrogram_table",
    no_args_is_help=True,
    epilog="For further information visit: https://gitlab.ethz.ch/seb/orcai_test",
)
@click.argument(
    "recording_table_path",
    type=ClickFilePathR,
)
@click.option(
    "--base_dir",
    "-bd",
    type=ClickDirPathR,
    default=None,
    show_default="None",
    help="Base directory for the wav files. If not None entries in the wav_file column are interpreted as filenames searched for in base_dir and subfolders. If None the entries are interpreted as absolute paths.",
)
@click.option(
    "--output_dir",
    "-o",
    type=ClickDirPathW,
    required=True,
    help="Output directory for the spectrograms. Spectograms are stored in subdirectories named '<recording_name>/spectrogram'",
)
@click.option(
    "--spectrogram_parameter",
    "-sp",
    type=ClickFilePathR,
    default=files("orcAI.defaults").joinpath("default_spectrogram_parameter.json"),
    show_default="default_spectrogram_parameter.json",
    help="Path to the spectrogram parameter file.",
)
@click.option(
    "--label_calls",
    "-lc",
    type=ClickFilePathR,
    default=files("orcAI.defaults").joinpath("default_calls.json"),
    show_default="default_calls.json",
    help="Path to a JSON file containing calls for labeling.",
)
@click.option(
    "--exclude",
    "-e",
    is_flag=True,
    default=True,
    show_default=True,
    help="Exclude recordings without possible annotations.",
)
@click.option(
    "--verbosity",
    "-v",
    type=click.IntRange(0, 2),
    default=2,
    show_default=True,
    help="Verbosity level.",
)
def cli_create_spectrogram(**kwargs):
    create_spectrograms(**kwargs)


@cli.command(
    name="make-labels",
    help="Makes label arrays for all files in csv at RECORDING_TABLE_PATH",
    short_help="Makes label arrays for all files in recording_table",
    no_args_is_help=True,
    epilog="For further information visit: https://gitlab.ethz.ch/seb/orcai_test",
)
@click.argument(
    "recording_table_path",
    type=ClickFilePathR,
)
@click.option(
    "--base_dir",
    "-bd",
    type=ClickDirPathR,
    default=None,
    show_default="None",
    help="Base directory for the recording files. If not None entries in the recording column are interpreted as filenames searched for in base_dir and subfolders. If None the entries are interpreted as absolute paths.",
)
@click.option(
    "--output_dir",
    "-o",
    type=ClickDirPathW,
    required=True,
    help="Output directory for the labels. Labels are stored in subdirectories named '<recording>/labels'",
)
@click.option(
    "--label_calls",
    "-lc",
    type=ClickFilePathR,
    default=files("orcAI.defaults").joinpath("default_calls.json"),
    show_default="default_calls.json",
    help="Path to a JSON file containing calls for labeling.",
)
@click.option(
    "--call_equivalences",
    "-ce",
    type=ClickFilePathR,
    default=None,
    show_default="None",
    help="Optional path to a call equivalences file or a dictionary. A dictionary associating original call labels with new call labels.",
)
@click.option(
    "--verbosity",
    "-v",
    type=click.IntRange(0, 2),
    default=2,
    show_default=True,
    help="Verbosity level.",
)
def cli_make_label_arrays(**kwargs):
    create_label_arrays(**kwargs)

from pathlib import Path
from importlib.resources import files
import click

from orcAI.spectrogram import create_spectrograms
from orcAI.annotation import create_label_arrays
from orcAI.snippets import (
    create_snippet_table,
    create_tvt_snippet_tables,
    create_tvt_data,
)
from orcAI.train import train
from orcAI.predict import predict


class SpecialHelpOrder(click.Group):

    def __init__(self, *args, **kwargs):
        self.help_priorities = {}
        super(SpecialHelpOrder, self).__init__(*args, **kwargs)

    def get_help(self, ctx):
        self.list_commands = self.list_commands_for_help
        return super(SpecialHelpOrder, self).get_help(ctx)

    def list_commands_for_help(self, ctx):
        """reorder the list of commands when listing the help"""
        commands = super(SpecialHelpOrder, self).list_commands(ctx)
        return (
            c[1]
            for c in sorted(
                (self.help_priorities.get(command, 1), command) for command in commands
            )
        )

    def command(self, *args, **kwargs):
        """Behaves the same as `click.Group.command()` except capture
        a priority for listing command names in help.
        """
        help_priority = kwargs.pop("help_priority", 1)
        help_priorities = self.help_priorities

        def decorator(f):
            cmd = super(SpecialHelpOrder, self).command(*args, **kwargs)(f)
            help_priorities[cmd.name] = help_priority
            return cmd

        return decorator


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
    cls=SpecialHelpOrder,
)
def cli():
    pass


@cli.command(
    name="create-spectrograms",
    help="Creates spectrograms for all files in RECORDING_TABLE_PATH",
    short_help="Creates spectrograms for all files in recording table",
    no_args_is_help=True,
    epilog="For further information visit: https://gitlab.ethz.ch/seb/orcai_test",
    help_priority=0,
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
    help="Path to a JSON file containing spectrogram parameter.",
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
    name="create-labels",
    help="Makes label arrays for all files in csv at RECORDING_TABLE_PATH",
    short_help="Makes label arrays for all files in recording table",
    no_args_is_help=True,
    epilog="For further information visit: https://gitlab.ethz.ch/seb/orcai_test",
    help_priority=1,
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
def cli_create_label_arrays(**kwargs):
    create_label_arrays(**kwargs)


@cli.command(
    name="create-snippets",
    help="Creates snippet tables for all files in RECORDING_TABLE_PATH",
    short_help="Creates snippet tables for all files in recording table",
    no_args_is_help=True,
    epilog="For further information visit: https://gitlab.ethz.ch/seb/orcai_test",
    help_priority=2,
)
@click.argument(
    "recording_table_path",
    type=ClickFilePathR,
)
@click.option(
    "--recording_data_dir",
    "-rd",
    type=ClickDirPathR,
    required=True,
    help="Path to the directory containing the recording data.",
)
@click.option(
    "--snippet_parameter",
    "-sp",
    type=ClickFilePathR,
    default=files("orcAI.defaults").joinpath("default_snippet_parameter.json"),
    show_default="default_snippet_parameter.json",
    help="Path to a JSON file containing snippet parameter.",
)
@click.option(
    "-m",
    "--model_parameter",
    "model_parameter",
    help="Path to a JSON file containing model specifications",
    type=ClickFilePathR,
    default=str(files("orcAI.defaults").joinpath("default_model_parameter.json")),
    show_default="default_model_parameter.json",
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
    "--verbosity",
    "-v",
    type=click.IntRange(0, 2),
    default=2,
    show_default=True,
    help="Verbosity level.",
)
def cli_create_snippet_table(**kwargs):
    create_snippet_table(**kwargs)


@cli.command(
    name="create-tvt-snippets",
    help="Creates training, validation and test snippet tables",
    no_args_is_help=True,
    epilog="For further information visit: https://gitlab.ethz.ch/seb/orcai_test",
    help_priority=3,
)
@click.option(
    "--recording_data_dir",
    "-rd",
    type=ClickDirPathR,
    required=True,
    help="Path to the directory containing the recording data.",
)
@click.option(
    "--output_dir",
    "-o",
    type=ClickDirPathW,
    required=True,
    help="Path to the output directory.",
)
@click.option(
    "--snippet_table",
    "-st",
    type=ClickFilePathR,
    default=None,
    show_default="<recording_data_dir>/all_snippets.csv.gz",
    help="Path to the snippet table csv. If None, the snippet table is read from the <recording_data_dir>/all_snippets.csv.gz.",
)
@click.option(
    "--snippet_parameter",
    "-sp",
    type=ClickFilePathR,
    default=files("orcAI.defaults").joinpath("default_snippet_parameter.json"),
    show_default="default_snippet_parameter.json",
    help="Path to a JSON file containing snippet parameter.",
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
    "--verbosity",
    "-v",
    type=click.IntRange(0, 2),
    default=2,
    show_default=True,
    help="Verbosity level.",
)
def cli_create_tvt_snippet_tables(**kwargs):
    create_tvt_snippet_tables(**kwargs)


@cli.command(
    name="create-tvt-data",
    help="Creates training, validation and test data from snippet tables in TVT_DIR",
    short_help="Creates training, validation and test data from snippet tables",
    no_args_is_help=True,
    epilog="For further information visit: https://gitlab.ethz.ch/seb/orcai_test",
    help_priority=4,
)
@click.argument(
    "tvt_dir",
    type=ClickDirPathW,
    required=True,
)
@click.option(
    "-m",
    "--model_parameter",
    "model_parameter",
    help="Path to a JSON file containing model specifications",
    type=ClickFilePathR,
    default=str(files("orcAI.defaults").joinpath("default_model_parameter.json")),
    show_default="default_model_parameter.json",
)
@click.option(
    "--verbosity",
    "-v",
    type=click.IntRange(0, 2),
    default=2,
    show_default=True,
    help="Verbosity level.",
)
def cli_create_tvt_data(**kwargs):
    create_tvt_data(**kwargs)


@cli.command(
    name="train",
    help="Train a model on training, validation and test data",
    short_help="Train a model on data",
    no_args_is_help=True,
    epilog="For further information visit: https://gitlab.ethz.ch/seb/orcai_test",
    help_priority=5,
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
    show_default="default_model_parameter.json",
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
    "--verbosity",
    "-v",
    type=click.IntRange(0, 2),
    default=2,
    show_default=True,
    help="Verbosity level.",
)
def cli_train(**kwargs):
    train(**kwargs)


@cli.command(
    name="predict",
    help="Predicts call annotations in the wav file at WAV_FILE_PATH.",
    short_help="Predicts call annotations.",
    no_args_is_help=True,
    epilog="For further information visit: https://gitlab.ethz.ch/seb/orcai_test",
    help_priority=6,
)
@click.argument("wav_file_path", type=ClickFilePathR)
@click.option(
    "--model",
    "-m",
    "model_path",
    type=ClickDirPathR,
    default=files("orcAI.models").joinpath("orcai-V1"),
    show_default="orcai-V1",
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
    "--spectrogram_parameter_path",
    "-sp",
    type=ClickFilePathR,
    default=files("orcAI.defaults").joinpath("default_spectrogram_parameter.json"),
    show_default="default_spectrogram_parameter.json",
    help="Path to a JSON file containing spectrogram parameter.",
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

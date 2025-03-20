from pathlib import Path
from importlib.resources import files
from importlib.metadata import version
import rich_click as click

click.rich_click.COMMAND_GROUPS = {
    "orcai": [
        {
            "name": "Predicting calls",
            "commands": ["predict", "filter-predictions"],
        },
        {
            "name": "Training Models",
            "commands": [
                "create-spectrograms",
                "create-label-arrays",
                "create-snippet-table",
                "create-tvt-snippet-tables",
                "create-tvt-data",
            ],
        },
        {
            "name": "Helpers",
            "commands": ["init", "create-recording-table"],
        },
    ]
}


ClickDirPathR = click.Path(
    exists=True, file_okay=False, readable=True, resolve_path=True, path_type=Path
)
ClickDirPathW = click.Path(
    exists=True, file_okay=False, writable=True, resolve_path=True, path_type=Path
)
ClickDirPathWcreate = click.Path(
    exists=False, file_okay=False, writable=True, resolve_path=True, path_type=Path
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
    + "     ░░██░░░░      " + "Version: " + version("orcAI") + "\n"
    + "      ███ ██       " + "Reference: " + click.style("in preparation", italic=True) + "\n",
    # TODO: Add reference
    # fmt: on
    epilog="For further information see the help pages of the individual subcommands (e.g. "
    + click.style("orcai predict --help", italic=True)
    + ") and/or visit: https://gitlab.ethz.ch/tb/orcai",
)
@click.version_option()
def cli():
    pass


@cli.command(
    name="predict",
    help="Predicts call annotations from RECORDING_PATH. This can either be a path to a wav file or a recording table (created with create-recording-table) as .csv.",
    short_help="Predicts call annotations.",
    no_args_is_help=True,
    epilog="For further information visit: https://gitlab.ethz.ch/tb/orcai",
)
@click.argument("recording_path", type=ClickFilePathR)
@click.option(
    "--channel",
    "-c",
    type=int,
    default=1,
    show_default=1,
    help="Channel to use for prediction if running predicitons for a single file.",
)
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
    "--output_path",
    "-o",
    default="default",
    show_default="default",
    help="Path to the output file/folder or 'default' to save in the same directory as the wav file. None to not save predictions to disk.",
)
@click.option(
    "--save_prediction_probabilities",
    "-sp",
    is_flag=True,
    help="If True the prediction probabilities are saved to a file.",
)
@click.option(
    "--base_dir_recording",
    "-bdr",
    type=ClickDirPathW,
    default=None,
    show_default="None",
    help="Alternative base directory containing the recordings (possibly in subdirectories). If None the base directory is taken from the recording_table.",
)
@click.option(
    "--call_duration_limits",
    "-cdl",
    type=ClickFilePathR,
    default=None,
    show_default="None",
    help="Path to a JSON file containing call duration limits. None for no filtering based on call duration.",
)
@click.option(
    "--label_suffix",
    "-ls",
    default="*",
    show_default=True,
    help="Suffix to add to the label names.",
)
@click.option(
    "--verbosity",
    "-v",
    type=click.IntRange(0, 3),
    default=2,
    show_default=True,
    help="Verbosity level. 0: Errors only, 1: Warnings, 2: Info, 3: Debug",
)
def cli_predict(**kwargs):
    from orcAI.predict import predict

    predict(**kwargs)


@cli.command(
    name="filter-predictions",
    help="Filters predictions in the predictions file at PREDICTION_FILE_PATH.",
    short_help="Filters predictions.",
    no_args_is_help=True,
    epilog="For further information visit: https://gitlab.ethz.ch/tb/orcai",
)
@click.argument("predicted_labels", type=ClickFilePathR)
@click.option(
    "--call_duration_limits",
    "-cdl",
    type=ClickFilePathR,
    default=files("orcAI.defaults").joinpath("default_call_duration_limits.json"),
    show_default="default_call_duration_limits.json",
    help="Path to a JSON file containing call duration limits.",
)
@click.option(
    "--output_file",
    "-o",
    default="default",
    show_default="default",
    help="Path to the output file or 'default' to save in the same directory as the prediction file.",
)
@click.option(
    "--label_suffix",
    "-ls",
    default="orcai-V1",
    show_default="orcai-V1",
    help="Suffix to add to the label names.",
)
@click.option(
    "--verbosity",
    "-v",
    type=click.IntRange(0, 3),
    default=2,
    show_default=True,
    help="Verbosity level. 0: Errors only, 1: Warnings, 2: Info, 3: Debug",
)
def cli_filter_predictions(**kwargs):
    from orcAI.predict import filter_predictions

    filter_predictions(**kwargs)


@cli.command(
    name="init",
    help="Initializes a new orcAI project with PROJECT_NAME in PROJECT_DIR.",
    short_help="Initializes a new orcAI project.",
    no_args_is_help=True,
    epilog="For further information visit: https://gitlab.ethz.ch/tb/orcai",
)
@click.argument("project_dir", type=ClickDirPathW)
@click.argument("project_name", type=str)
@click.option(
    "--verbosity",
    "-v",
    type=click.IntRange(0, 3),
    default=2,
    show_default=True,
    help="Verbosity level. 0: Errors only, 1: Warnings, 2: Info, 3: Debug",
)
def cli_init_project(**kwargs):
    from orcAI.helpers import init_project

    init_project(**kwargs)


@cli.command(
    name="create-recording-table",
    help="Create a table of recordings in BASE_DIR_RECORDING for use with other orcAI functions.",
    short_help="Create a table of recordings.",
    no_args_is_help=True,
    epilog="For further information visit: https://gitlab.ethz.ch/tb/orcai",
)
@click.argument("base_dir_recording", type=ClickDirPathR)
@click.option(
    "--output_path",
    "-o",
    type=ClickFilePathW,
    default=None,
    show_default="BASE_DIR_RECORDING/recording_table.csv",
    help="Path to save the table of recordings. If none it is saved as recording_table.csv in base_dir_recording.",
)
@click.option(
    "--base_dir_annotation",
    "-bda",
    type=ClickDirPathR,
    default=None,
    show_default="None",
    help="Base directory containing the annotations (if different from base_dir_recording).",
)
@click.option(
    "--default_channel",
    "-dc",
    type=int,
    default=1,
    show_default=1,
    help="Default channel number for the recordings.",
)
@click.option(
    "--orcai_parameter",
    "-p",
    type=ClickFilePathR,
    default=None,
    show_default="None",
    help="Path to a JSON file containing OrcAI parameter. Only needed if preparing table for generating training data.",
)
@click.option(
    "--update_table",
    "-ut",
    type=ClickFilePathR,
    default=None,
    show_default="None",
    help="Path to a .csv file with a previous table of recordings to update.",
)
@click.option(
    "--update_paths",
    "-up",
    is_flag=True,
    help="If True the paths in the table to update are updated with the new paths. Only valid if update_table is not None.",
)
@click.option(
    "--exclude_patterns",
    "-ep",
    type=ClickFilePathR,
    default=None,
    show_default="None",
    help="Path to a JSON file containing filenames to exclude from the table or an array containing the same.",
)
@click.option(
    "--remove_duplicate_filenames",
    "-rdf",
    is_flag=True,
    help="Remove duplicate filenames from the table.",
)
@click.option(
    "--verbosity",
    "-v",
    type=click.IntRange(0, 3),
    default=2,
    show_default=True,
    help="Verbosity level. 0: Errors only, 1: Warnings, 2: Info, 3: Debug",
)
def cli_create_recordings_table(**kwargs):
    from orcAI.helpers import create_recording_table

    create_recording_table(**kwargs)


@cli.command(
    name="create-spectrograms",
    help="Creates spectrograms for all files in recording table at RECORDING_TABLE_PATH and writes them to OUTPUT_DIR.",
    short_help="Creates spectrograms.",
    no_args_is_help=True,
    epilog="For further information visit: https://gitlab.ethz.ch/tb/orcai",
)
@click.argument("recording_table_path", type=ClickFilePathR)
@click.argument("output_dir", type=ClickDirPathW)
@click.option(
    "--base_dir_recording",
    "-bdr",
    type=ClickDirPathR,
    default=None,
    show_default="None",
    help="Base directory for the wav files. If None the base_dir_recording is taken from the recording_table.",
)
@click.option(
    "--orcai_parameter",
    "-p",
    type=ClickFilePathR,
    default=files("orcAI.defaults").joinpath("default_orcai_parameter.json"),
    show_default="default_orcai_parameter.json",
    help="Path to the OrcAI parameter file.",
)
@click.option(
    "--include_not_annotated",
    "-en",
    is_flag=True,
    help="Include recordings without annotations.",
)
@click.option(
    "--include_no_possible_annotations",
    "-enp",
    is_flag=True,
    help="Include recordings without possible annotations.",
)
@click.option(
    "--overwrite",
    "-ow",
    is_flag=True,
    help="Recreate existing spectrograms.",
)
@click.option(
    "--verbosity",
    "-v",
    type=click.IntRange(0, 3),
    default=2,
    show_default=True,
    help="Verbosity level. 0: Errors only, 1: Warnings, 2: Info, 3: Debug",
)
def cli_create_spectrograms(**kwargs):
    from orcAI.spectrogram import create_spectrograms

    create_spectrograms(**kwargs)


@cli.command(
    name="create-label-arrays",
    help="Creates label arrays for all files in recording table at RECORDING_TABLE_PATH and writes them to OUTPUT_DIR.",
    short_help="Creates label arrays.",
    no_args_is_help=True,
    epilog="For further information visit: https://gitlab.ethz.ch/tb/orcai",
)
@click.argument("recording_table_path", type=ClickFilePathR)
@click.argument("output_dir", type=ClickDirPathW)
@click.option(
    "--base_dir_annotation",
    "-bda",
    type=ClickDirPathR,
    default=None,
    show_default="None",
    help="Base directory for the annotation files. If None the base_dir_annotation is taken from the recording_table.",
)
@click.option(
    "--orcai_parameter",
    "-p",
    type=ClickFilePathR,
    default=files("orcAI.defaults").joinpath("default_orcai_parameter.json"),
    show_default="default_orcai_parameter.json",
    help="Path to a JSON file containing orcai parameter.",
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
    "--overwrite",
    "-ow",
    is_flag=True,
    help="Recreate existing label arrays.",
)
@click.option(
    "--verbosity",
    "-v",
    type=click.IntRange(0, 3),
    default=2,
    show_default=True,
    help="Verbosity level. 0: Errors only, 1: Warnings, 2: Info, 3: Debug",
)
def cli_create_label_arrays(**kwargs):
    from orcAI.annotation import create_label_arrays

    create_label_arrays(**kwargs)


@cli.command(
    name="create-snippet-table",
    help="Creates a table of snippets for all files in recording table at RECORDING_TABLE_PATH and writes them to RECORDING_DATA_DIR.",
    short_help="Creates snippet table.",
    no_args_is_help=True,
    epilog="For further information visit: https://gitlab.ethz.ch/tb/orcai",
)
@click.argument("recording_table_path", type=ClickFilePathR)
@click.argument("recording_data_dir", type=ClickDirPathW)
@click.option(
    "--orcai_parameter",
    "-p",
    type=ClickFilePathR,
    default=files("orcAI.defaults").joinpath("default_orcai_parameter.json"),
    show_default="default_orcai_parameter.json",
    help="Path to the snippet parameter file.",
)
@click.option(
    "--verbosity",
    "-v",
    type=click.IntRange(0, 3),
    default=2,
    show_default=True,
    help="Verbosity level. 0: Errors only, 1: Warnings, 2: Info, 3: Debug",
)
def cli_create_snippet_table(**kwargs):
    from orcAI.snippets import create_snippet_table

    create_snippet_table(**kwargs)


@cli.command(
    name="create-tvt-snippet-tables",
    help="Creates snippet tables for training, validation and test datasets from recordings in RECORDING_DATA_DIR and saves them to OUTPUT_DIR.",
    short_help="Creates TVT snippet tables.",
    no_args_is_help=True,
    epilog="For further information visit: https://gitlab.ethz.ch/tb/orcai",
)
@click.argument("recording_data_dir", type=ClickDirPathR)
@click.argument("output_dir", type=ClickDirPathWcreate)
@click.option(
    "--snippet_table",
    "-st",
    type=ClickFilePathR,
    default=None,
    show_default="None",
    help="Path to the snippet table csv or the snippet table itself. None if the snippet table should be read from RECORDING_DATA_DIR/all_snippets.csv.gz.",
)
@click.option(
    "--orcai_parameter",
    "-p",
    type=ClickFilePathR,
    default=files("orcAI.defaults").joinpath("default_orcai_parameter.json"),
    show_default="default_orcai_parameter.json",
    help="Path to the snippet parameter file.",
)
@click.option(
    "--verbosity",
    "-v",
    type=click.IntRange(0, 3),
    default=2,
    show_default=True,
    help="Verbosity level. 0: Errors only, 1: Warnings, 2: Info, 3: Debug",
)
def cli_create_tvt_snippet_tables(**kwargs):
    from orcAI.snippets import create_tvt_snippet_tables

    create_tvt_snippet_tables(**kwargs)


@cli.command(
    name="create-tvt-data",
    help="Creates training, validation and test datasets from snippet tables in TVT_DIR.",
    short_help="Creates TVT datasets.",
    no_args_is_help=True,
    epilog="For further information visit: https://gitlab.ethz.ch/tb/orcai",
)
@click.argument("tvt_dir", type=ClickDirPathR)
@click.option(
    "--orcai_parameter",
    "-p",
    type=ClickFilePathR,
    default=files("orcAI.defaults").joinpath("default_orcai_parameter.json"),
    show_default="default_orcai_parameter.json",
    help="Path to the snippet parameter file.",
)
@click.option(
    "--overwrite",
    "-ow",
    is_flag=True,
    help="Recreate existing data.",
)
@click.option(
    "--verbosity",
    "-v",
    type=click.IntRange(0, 3),
    default=2,
    show_default=True,
    help="Verbosity level. 0: Errors only, 1: Warnings, 2: Info, 3: Debug",
)
def cli_create_tvt_data(**kwargs):
    from orcAI.snippets import create_tvt_data

    create_tvt_data(**kwargs)

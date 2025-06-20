from importlib.metadata import version
from importlib.resources import files
from pathlib import Path

import rich_click as click

from orcAI.auxiliary import Messenger

click.rich_click.STYLE_OPTIONS_PANEL_BOX = "SIMPLE"
click.rich_click.STYLE_COMMANDS_PANEL_BOX = "SIMPLE"
click.rich_click.STYLE_COMMANDS_PANEL_BORDER = "bold"
click.rich_click.STYLE_OPTIONS_PANEL_BORDER = "bold"
click.rich_click.MAX_WIDTH = 100

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
                "hpsearch",
                "train",
                "test",
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

INCLUDED_MODELS = [
    file.stem for file in files("orcAI.models").iterdir() if file.stem != ".DS_Store"
]


@click.group(
    help="\n\b\n"
    + "            █████  "
    + click.style("Command line interface for ", bold=True)
    + click.style("orcAI", fg="blue", bold=True)
    + "\n"
    + "███ ███   ████████ "
    + "  a tool for training, testing & applying\n"
    + "  ████  ████░██░░░ "
    + "  AI models to detect acoustic signals in\n"
    + "    ██████████░░░  "
    + "  spectrograms generated from audio recordings.\n"
    + "     ░░██░░░░      "
    + "Version: "
    + version("orcAI")
    + "\n"
    + "      ███ ██       "
    + "Reference: "
    + click.style("in preparation", italic=True)
    + "\n",
    # TODO: Add reference
    epilog="For further information see the help pages of the individual subcommands (e.g. "
    + click.style("orcai predict --help", italic=True)
    + ") and/or visit: https://github.com/ethz-tb/orcAI",
)
@click.version_option()
def cli():
    pass


@cli.command(
    name="predict",
    help="Predicts call annotations from RECORDING_PATH. This can either be a path to a wav file or a recording table (created with create-recording-table) as .csv.",
    short_help="Predicts call annotations.",
    no_args_is_help=True,
    epilog="For further information visit: https://github.com/ethz-tb/orcAI",
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
    type=click.Choice(INCLUDED_MODELS, case_sensitive=False),
    default="orcai-v1",
    show_default=True,
    help="Builtin model to use for prediction. Overriden if model_dir is given.",
)
@click.option(
    "--model_dir",
    "-md",
    "model_dir",
    type=ClickDirPathR,
    default=None,
    show_default="use builtin model",
    help="Path to a model directory.",
)
@click.option(
    "--output_path",
    "-o",
    default="default",
    show_default="default",
    help="Path to the output file/folder or 'default' to save in the same directory as the wav file. None to not save predictions to disk.",
)
@click.option(
    "--overwrite",
    "-ow",
    is_flag=True,
    help="Overwrite existing predictions.",
)
@click.option(
    "--save_probabilities",
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
    kwargs["msgr"] = Messenger(verbosity=kwargs["verbosity"], title="Predicting calls")
    from orcAI.predict import predict

    if kwargs["model_dir"] is None:
        kwargs["model_dir"] = files("orcAI.models").joinpath(kwargs["model"])
    del kwargs["model"]

    predict(**kwargs)


@cli.command(
    name="filter-predictions",
    help="Filters predictions in the predictions file at PREDICTION_FILE_PATH.",
    short_help="Filters predictions.",
    no_args_is_help=True,
    epilog="For further information visit: https://github.com/ethz-tb/orcAI",
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
    "--overwrite",
    "-ow",
    is_flag=True,
    help="Overwrite existing predictions.",
)
@click.option(
    "--label_suffix",
    "-ls",
    default="*",
    show_default="*",
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
    kwargs["msgr"] = Messenger(
        verbosity=kwargs["verbosity"], title="Filtering predictions"
    )
    from orcAI.predict import filter_predictions_file

    filter_predictions_file(**kwargs)


@cli.command(
    name="init",
    help="Initializes a new orcAI project with PROJECT_NAME in PROJECT_DIR.",
    short_help="Initializes a new orcAI project.",
    no_args_is_help=True,
    epilog="For further information visit: https://github.com/ethz-tb/orcAI",
)
@click.argument("project_dir", type=ClickDirPathWcreate)
@click.argument("project_name", type=str)
@click.option(
    "--parameter",
    "-p",
    type=ClickFilePathR,
    default=None,
    show_default=True,
    help="Path to a JSON file containing OrcAI parameter (or a subset of). Only needed if overwriting defaults.",
)
@click.option(
    "--verbosity",
    "-v",
    type=click.IntRange(0, 3),
    default=2,
    show_default=True,
    help="Verbosity level. 0: Errors only, 1: Warnings, 2: Info, 3: Debug",
)
def cli_init_project(**kwargs):
    kwargs["msgr"] = Messenger(
        verbosity=kwargs["verbosity"], title="Initializing project"
    )
    from orcAI.helpers import init_project

    init_project(**kwargs)


@cli.command(
    name="create-recording-table",
    help="Create a table of recordings in BASE_DIR_RECORDING for use with other orcAI functions.",
    short_help="Create a table of recordings.",
    no_args_is_help=True,
    epilog="For further information visit: https://github.com/ethz-tb/orcAI",
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
    help="Path to the OrcAI parameter file. Only needed if preparing table for generating training data.",
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
    kwargs["msgr"] = Messenger(
        verbosity=kwargs["verbosity"], title="Creating recording table"
    )
    from orcAI.helpers import create_recording_table

    create_recording_table(**kwargs)


@cli.command(
    name="create-spectrograms",
    help="Creates spectrograms for all files in recording table at RECORDING_TABLE_PATH and writes them to OUTPUT_DIR.",
    short_help="Creates spectrograms.",
    no_args_is_help=True,
    epilog="For further information visit: https://github.com/ethz-tb/orcAI",
)
@click.argument("recording_table_path", type=ClickFilePathR)
@click.argument("output_dir", type=ClickDirPathWcreate)
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
    kwargs["msgr"] = Messenger(
        verbosity=kwargs["verbosity"], title="Creating spectrograms"
    )
    from orcAI.spectrogram import create_spectrograms

    create_spectrograms(**kwargs)


@cli.command(
    name="create-label-arrays",
    help="Creates label arrays for all files in recording table at RECORDING_TABLE_PATH and writes them to OUTPUT_DIR.",
    short_help="Creates label arrays.",
    no_args_is_help=True,
    epilog="For further information visit: https://github.com/ethz-tb/orcAI",
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
    help="Path to the OrcAI parameter file.",
)
@click.option(
    "--call_equivalences",
    "-ce",
    type=ClickFilePathR,
    default=None,
    show_default="None",
    help="Optional path to a call equivalences file. A dictionary associating original call labels with new call labels.",
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
    kwargs["msgr"] = Messenger(
        verbosity=kwargs["verbosity"], title="Creating label arrays"
    )
    from orcAI.labels import create_label_arrays

    create_label_arrays(**kwargs)


@cli.command(
    name="create-snippet-table",
    help="Creates a table of snippets for all files in recording table at RECORDING_TABLE_PATH and writes them to RECORDING_DATA_DIR.",
    short_help="Creates snippet table.",
    no_args_is_help=True,
    epilog="For further information visit: https://github.com/ethz-tb/orcAI",
)
@click.argument("recording_table_path", type=ClickFilePathR)
@click.argument("recording_data_dir", type=ClickDirPathW)
@click.option(
    "--output_dir",
    "-o",
    type=ClickDirPathWcreate,
    default=None,
    show_default="None",
    help="Path to the output directory. None to save in the same directory as the recording data table.",
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
    "--verbosity",
    "-v",
    type=click.IntRange(0, 3),
    default=2,
    show_default=True,
    help="Verbosity level. 0: Errors only, 1: Warnings, 2: Info, 3: Debug",
)
def cli_create_snippet_table(**kwargs):
    kwargs["msgr"] = Messenger(
        verbosity=kwargs["verbosity"], title="Creating snippet table"
    )
    from orcAI.snippets import create_snippet_table

    create_snippet_table(**kwargs)


@cli.command(
    name="create-tvt-snippet-tables",
    help="Creates snippet tables for training, validation and test datasets and saves them to OUTPUT_DIR.",
    short_help="Creates TVT snippet tables.",
    no_args_is_help=True,
    epilog="For further information visit: https://github.com/ethz-tb/orcAI",
)
@click.argument("output_dir", type=ClickDirPathWcreate)
@click.option(
    "--snippet_table",
    "-st",
    type=ClickFilePathR,
    default=None,
    show_default="None",
    help="Path to the snippet table csv. None if the snippet table should be read from RECORDING_DATA_DIR/all_snippets.csv.gz.",
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
    "--create_unfiltered_test_snippets",
    "-uts",
    is_flag=True,
    help="If set, creates an additional test snippet table with unfiltered snippets",
)
@click.option(
    "--n_unfiltered_test_snippets",
    "-n_uts",
    type=int,
    default=None,
    show_default="None",
    help="Number of unfiltered test snippets. If None, an unfiltered sample of the same size as the training snippet table is created.",
)
@click.option(
    "--overwrite",
    "-ow",
    is_flag=True,
    help="Overwrite existing snippet tables.",
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
    kwargs["msgr"] = Messenger(
        verbosity=kwargs["verbosity"],
        title="Creating train, validation and test snippet tables",
    )
    from orcAI.snippets import create_tvt_snippet_tables

    create_tvt_snippet_tables(**kwargs)


@cli.command(
    name="create-tvt-data",
    help="Creates training, validation and test datasets from snippet tables in TVT_DIR.",
    short_help="Creates TVT datasets.",
    no_args_is_help=True,
    epilog="For further information visit: https://github.com/ethz-tb/orcAI",
)
@click.argument("tvt_dir", type=ClickDirPathR)
@click.option(
    "--orcai_parameter",
    "-p",
    type=ClickFilePathR,
    default=files("orcAI.defaults").joinpath("default_orcai_parameter.json"),
    show_default="default_orcai_parameter.json",
    help="Path to the OrcAI parameter file.",
)
@click.option(
    "--overwrite",
    "-ow",
    is_flag=True,
    help="Recreate existing data.",
)
@click.option(
    "--data_compression",
    "-dc",
    type=click.Choice(["GZIP", "None"], case_sensitive=False),
    default="GZIP",
    show_default=True,
    help="Data compression for datasets",
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
    kwargs["msgr"] = Messenger(
        verbosity=kwargs["verbosity"],
        title="Creating train, validation and test datasets",
    )
    if kwargs["data_compression"] == "None":
        kwargs["data_compression"] = None

    from orcAI.snippets import create_tvt_data

    create_tvt_data(**kwargs)


@cli.command(
    name="train",
    help="Trains a model on the training dataset in DATA_DIR and saves it to OUTPUT_DIR.",
    short_help="Trains a model.",
    no_args_is_help=True,
    epilog="For further information visit: https://github.com/ethz-tb/orcAI",
)
@click.argument("data_dir", type=ClickDirPathR)
@click.argument("output_dir", type=ClickDirPathW)
@click.option(
    "--orcai_parameter",
    "-p",
    type=ClickFilePathR,
    help="Path to the OrcAI parameter file.",
)
@click.option(
    "--data_compression",
    "-dc",
    type=click.Choice(["GZIP", "None"], case_sensitive=False),
    default="GZIP",
    show_default=True,
    help="Data compression of saved datasets",
)
@click.option(
    "--load_model",
    "-lm",
    is_flag=True,
    help="Load model from previous training.",
)
@click.option(
    "--verbosity",
    "-v",
    type=click.IntRange(0, 3),
    default=2,
    show_default=True,
    help="Verbosity level. 0: Errors only, 1: Warnings, 2: Info, 3: Debug",
)
def cli_train(**kwargs):
    kwargs["msgr"] = Messenger(
        verbosity=kwargs["verbosity"],
        title="Training model",
    )
    if kwargs["data_compression"] == "None":
        kwargs["data_compression"] = None

    from orcAI.train import train

    train(**kwargs)


@cli.command(
    name="test",
    help="Tests a model at MODEL_DIR on the test dataset in DATA_DIR and saves the results to OUTPUT_DIR.",
    short_help="Tests a model.",
    no_args_is_help=True,
    epilog="For further information visit: https://github.com/ethz-tb/orcAI",
)
@click.argument("model_dir", type=ClickDirPathR)
@click.argument("data_dir", type=ClickDirPathR)
@click.option(
    "--test_unfiltered",
    "-tu",
    is_flag=True,
    help="If set, the model is also tested on the unfiltered test dataset.",
)
@click.option(
    "--output_dir",
    "-o",
    type=ClickDirPathWcreate,
    default=None,
    show_default="None",
    help="Path to the output directory. None to save in the same directory as the model.",
)
@click.option(
    "--data_compression",
    "-dc",
    type=click.Choice(["GZIP", "None"], case_sensitive=False),
    default="GZIP",
    show_default=True,
    help="Data compression of saved datasets",
)
@click.option(
    "--verbosity",
    "-v",
    type=click.IntRange(0, 3),
    default=2,
    show_default=True,
    help="Verbosity level. 0: Errors only, 1: Warnings, 2: Info, 3: Debug",
)
def cli_test(**kwargs):
    kwargs["msgr"] = Messenger(
        verbosity=kwargs["verbosity"],
        title=f"Testing model {kwargs['model_dir'].name}",
    )
    if kwargs["data_compression"] == "None":
        kwargs["data_compression"] = None

    from orcAI.test import test_model

    test_model(**kwargs)


@cli.command(
    name="hpsearch",
    help="Performs hyperparameter search on the training dataset in DATA_DIR and saves the results to OUTPUT_DIR.",
    short_help="Performs hyperparameter search.",
    no_args_is_help=True,
    epilog="For further information visit: https://github.com/ethz-tb/orcAI",
)
@click.argument("data_dir", type=ClickDirPathR)
@click.argument("output_dir", type=ClickDirPathW)
@click.option(
    "--orcai_parameter",
    "-p",
    type=ClickFilePathR,
    default=files("orcAI.defaults").joinpath("default_orcai_parameter.json"),
    show_default="default_orcai_parameter.json",
    help="Path to the OrcAI parameter file.",
)
@click.option(
    "--hps_parameter",
    "-hp",
    type=ClickFilePathR,
    default=files("orcAI.defaults").joinpath("default_hps_parameter.json"),
    show_default="default_hps_parameter.json",
    help="Path to the hyperparameter search parameter file.",
)
@click.option(
    "--parallel",
    "-pl",
    is_flag=True,
    help="Run hyperparameter search on multiple GPUs in parallel.",
)
@click.option(
    "--data_compression",
    "-dc",
    type=click.Choice(["GZIP", "None"], case_sensitive=False),
    default="GZIP",
    show_default=True,
    help="Data compression of saved datasets",
)
@click.option(
    "--verbosity",
    "-v",
    type=click.IntRange(0, 3),
    default=2,
    show_default=True,
    help="Verbosity level. 0: Errors only, 1: Warnings, 2: Info, 3: Debug",
)
def cli_hpsearch(**kwargs):
    kwargs["msgr"] = Messenger(
        verbosity=kwargs["verbosity"],
        title="Hyperparameter search",
    )
    if kwargs["data_compression"] == "None":
        kwargs["data_compression"] = None
    from orcAI.hpsearch import hyperparameter_search

    hyperparameter_search(**kwargs)

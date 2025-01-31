import argparse
import pathlib

def orcai_cli():
    """
    Command line interface for ORCAI tool.
    """
    # create the top-level parser
    parser = argparse.ArgumentParser(
        description = """
            Command line interface for OrcAI, a tool for training, testing, and analyzing spectrograms.
            """
    )
    subparsers = parser.add_subparsers(
        title = "Subcommands",
        required = True,
        help = "valid subcommands, see orcAI subcommand -h for more Info.",
    )

    # hyperparameter search
    parser_hp_search = subparsers.add_parser(
        "hp_search", description = "run hyperparameter search"
    )
    parser_hp_search.add_argument(
        "-m", "--model_name", help = "name of model", type = str, required = True
    )
    parser_hp_search.add_argument(
        "-d",
        "--data_dir",
        help = "path to directories where train/val/test data is (data_dir with / at the end)",
        type = pathlib.Path,
        required = True,
    )
    parser_hp_search.add_argument(
        "-p",
        "--project_dir",
        help = "name of project_dir (where all data is stored in project_dir/model_name/), output_dir with / at the end",
        type = str,
        required = True,
    )
    parser_hp_search.set_defaults(func = _hp_search_cli)

    # train model
    parser_train = subparsers.add_parser("train", description = "train model")
    parser_train.add_argument(
        "-d",
        "--data_dir",
        help = "path to directories where train/val/test data is (data_dir with / at the end)",
        type = str,
        required = True,
    )
    parser_train.add_argument(
        "-m", "--model_name", help = "name of model", type = str, required = True
    )
    parser_train.add_argument(
        "-p",
        "--project_dir",
        help = "name of project_dir (where all data is stored in project_dir/model_name/), output_dir with / at the end",
        type = str,
        required = True,
    )
    parser_train.add_argument(
        "-lw",
        "--load_weights",
        help = "load weights and continue fitting",
        action = "store_true",
        default = False,
    )
    parser_train.set_defaults(func = _train_cli)

    # parse the args and call func
    args = parser.parse_args()
    args.func(args)


def _hp_search_cli(args):
    """helper function to unwrap environment passed by argparse"""
    hp_search(
        model_name = args.model_name,
        data_dir = args.data_dir,
        project_dir = args.project_dir
    )


def _train_cli(args):
    """helper function to unwrap environment passed by argparse"""
    train_model(
        model_name = args.model_name,
        data_dir = args.data_dir,
        project_dir = args.project_dir,
        load_weights = args.load_weights,
    )


def hp_search(model_name, data_dir, project_dir):
    print("🐳 PLACEHOLDER HP_SEARCH FUNCTION 🐳")
    print(f"Model Name: {model_name}")
    print(f"Data Directory: {data_dir}")
    print(f"Project Directory: {project_dir}")


def train_model(model_name, data_dir, project_dir, load_weights):
    print("🐳 PLACEHOLDER TRAIN FUNCTION 🐳")
    print(f"Model Name: {model_name}")
    print(f"Data Directory: {data_dir}")
    print(f"Project Directory: {project_dir}")
    print(f"Load Weights: {load_weights}")

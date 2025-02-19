import click
from .train import train
from .predict import predict
from .spectrogram import create_spectrograms


@click.group(
    help="\n\b\n"
    + "            █████  " + click.style("Command line interface for ", bold=True) + click.style("orcAI", fg="cyan", bold=True) + "\n"
    + "███ ███   ████████ " + "  a tool for \n"
    + "  ████  ████░██░░░ " + "  training, testing & applying AI models \n"
    + "    ██████████░░░  " + "  to detect acoustic signals in spectrograms generated from audio recordings.\n"
    + "     ░░██░░░░      \n"
    + "      ███ ██       \n",
    epilog="For further information see the help pages of the individual subcommands (e.g. "
        + click.style("orcai train --help", italic=True)
        + ") and/or visit: https://gitlab.ethz.ch/seb/orcai_test",
)
def cli():
    pass


cli.add_command(train)
cli.add_command(predict)
cli.add_command(create_spectrograms)

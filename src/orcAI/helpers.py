from pathlib import Path
from importlib.resources import files

from orcAI.auxiliary import Messenger, filter_recordings, read_json


def create_prediction_list(
    base_dir,
    call_duration_limits=files("orcAI.defaults").joinpath(
        "default_call_duration_limits.json"
    ),
    exclude=None,
    verbosity=2,
):
    msgr = Messenger(verbosity=verbosity)
    msgr.part("Creating list of wav files for prediction")
    wav_files = list(Path(base_dir).glob("**/*.wav"))
    if exclude is not None:
        wav_files = filter_recordings(wav_files, exclude, msgr=msgr)

    if isinstance(call_duration_limits, (Path | str)):
        call_duration_limits = read_json(call_duration_limits)
    msgr.debug("Call duration limits:")
    msgr.debug(call_duration_limits)

    pass

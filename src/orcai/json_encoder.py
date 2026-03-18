import json
from pathlib import Path
from typing import Any

import numpy as np


class JsonEncoderExt(json.JSONEncoder):
    """Custom JSON encoder to handle additional data types."""

    def default(self, o: Any):
        if isinstance(o, Path):
            return str(o)
        if isinstance(o, np.float32):
            return o.astype(np.float64)
        return super().default(o)

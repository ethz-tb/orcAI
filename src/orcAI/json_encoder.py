import json
from pathlib import Path

import numpy as np


class JsonEncoderExt(json.JSONEncoder):
    """Custom JSON encoder to handle additional data types."""

    def default(self, obj):
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, np.float32):
            return obj.astype(np.float64)
        return super().default(obj)

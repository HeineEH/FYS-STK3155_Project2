from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

ArrayF = NDArray[np.floating]
LayerParams  = tuple[ArrayF, ArrayF]
NetworkParams = list[LayerParams]
import hashlib
import random

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def reseed_rng(request):
    node_id = request.node.nodeid.encode("utf-8")
    seed = int(hashlib.sha256(node_id).hexdigest()[:8], 16)
    random.seed(seed)
    np.random.seed(seed)

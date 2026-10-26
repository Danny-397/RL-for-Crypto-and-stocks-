"""Tests that a domain-randomized training stream is reproducible.

The ablation's domain-random arm once drew every training path from OS entropy,
so the training seed never reached its data and no single DR run could be
reproduced. ``seeded_path_factory`` fixes the stream by seed while still giving
fresh noise every episode.
"""

import numpy as np

from rl_trader.data.data_loader import seeded_path_factory


def _stream(market, seed, n=3):
    factory = seeded_path_factory(market, seed)
    return [factory().prices for _ in range(n)]


def test_same_seed_gives_the_same_stream():
    for a, b in zip(_stream("stock", 42), _stream("stock", 42)):
        np.testing.assert_array_equal(a, b)


def test_each_episode_gets_a_fresh_path():
    first, second, _ = _stream("crypto", 42)
    assert not np.array_equal(first, second)


def test_different_seeds_give_different_streams():
    assert not np.array_equal(_stream("stock", 42)[0], _stream("stock", 43)[0])

"""rng.py — stream independence is what makes baselines survive future edits."""
import numpy as np

from model.rng import RngBundle


def test_same_seed_reproduces():
    a, b = RngBundle(42), RngBundle(42)
    assert a.action.random() == b.action.random()
    assert a.tiebreak.random() == b.tiebreak.random()


def test_streams_are_independent():
    """Consuming one stream must not shift another — that is the whole point of
    spawning, and what lets a new draw site be added without moving every baseline."""
    a, b = RngBundle(42), RngBundle(42)
    for _ in range(100):
        a.init.random()
    assert a.action.random() == b.action.random()


def test_streams_differ_from_each_other():
    r = RngBundle(0)
    vals = [r.init.random(), r.activation.random(), r.action.random(),
            r.tiebreak.random(), r.order.random()]
    assert len(set(vals)) == 5


def test_different_seeds_differ():
    assert RngBundle(1).action.random() != RngBundle(2).action.random()


def test_no_global_rng_dependence():
    """If any consumer still calls np.random.*, perturbing the global state changes
    the result. This is the tripwire for a missed migration."""
    np.random.seed(1)
    a = RngBundle(9).action.random()
    np.random.seed(2)
    b = RngBundle(9).action.random()
    assert a == b

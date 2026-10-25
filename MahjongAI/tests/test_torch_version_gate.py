"""T-064: the torch-version gate of the two broken-fix MPS memory controls in test_mps.py. CPU only, no marker (Linux CI
runs it): the gate is on from torch 2.13 by release number, and exactly the VERSION_GATED tests carry it, with exactly
LEAK_CONTROL_REASON (the suite gate allows those skips and no other)."""
import types

import pytest

from MahjongAI.tests import test_mps as T


@pytest.mark.parametrize("version, skipped", [
    ("2.5.1", False),  # as strings "2.5.1" >= "2.13": a str compare would skip the controls on 2.5.1
    ("2.12.1", False), ("2.13.0", True), ("2.13.1", True), ("2.13.0+cpu", True), ("2.14.0", True), ("3.0.0", True),
])
def test_the_leak_controls_are_skipped_from_torch_2_13(version, skipped):
    assert T.leak_controls_skipped(version) is skipped


def skipif_marks(module) -> dict:
    """{test name: [(args, kwargs) of each skipif mark]} of the module's test functions that carry one."""
    found = {}
    for name, fn in vars(module).items():
        marks = [(m.args, m.kwargs) for m in getattr(fn, "pytestmark", []) if m.name == "skipif"]
        if name.startswith("test_") and marks:
            found[name] = marks
    return found


def gate_errors(module) -> list:
    """Every way the module's skipif marks differ from "exactly VERSION_GATED, each with the one version gate"."""
    marks, want = skipif_marks(module), [((module.TORCH_2_13,), {"reason": module.LEAK_CONTROL_REASON})]
    errors = [f"{n}: skipif not in VERSION_GATED" for n in sorted(marks.keys() - module.VERSION_GATED)]
    errors += [f"{n}: VERSION_GATED but no skipif" for n in sorted(module.VERSION_GATED - marks.keys())]
    errors += [f"{n}: {m} != {want}" for n, m in sorted(marks.items()) if n in module.VERSION_GATED and m != want]
    return errors


def test_exactly_the_version_gated_controls_carry_the_gate():
    assert T.VERSION_GATED and not gate_errors(T), gate_errors(T)


def _fake(gated: dict, version_gated=("test_a", "test_b")):
    """A module with test_a, test_b, test_c; gated: {name: (condition, reason)} -> that test's skipif."""
    module = types.SimpleNamespace(TORCH_2_13=True, LEAK_CONTROL_REASON="why", VERSION_GATED=set(version_gated))
    for name in ("test_a", "test_b", "test_c"):
        def fn():  # a def, not a lambda: pytest.mark(...)(<lambda>) does not mark it
            pass
        if name in gated:
            fn = pytest.mark.skipif(gated[name][0], reason=gated[name][1])(fn)
        setattr(module, name, fn)
    return module


def test_the_gate_check_fails_a_wrong_mark():
    """Negative controls: the clean fake passes; a third gated test, a missing gate, a wrong reason, a wrong condition
    each fail."""
    ok = {"test_a": (True, "why"), "test_b": (True, "why")}
    assert not gate_errors(_fake(ok))
    broken = {"extra": {**ok, "test_c": (True, "why")}, "missing": {"test_a": ok["test_a"]},
              "reason": {**ok, "test_b": (True, "other")}, "condition": {**ok, "test_b": (False, "why")}}
    for case, gated in broken.items():
        assert gate_errors(_fake(gated)), case

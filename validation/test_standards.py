#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pytest entry point for the standards validation cases.

    pytest validation                 # all cases
    pytest validation -m "not numerical"

Informational cases are not collected; run ``run_validation.py`` to see them.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cases import NUMERICAL, all_cases, evaluate  # noqa: E402

CASES = [c for c in all_cases() if not c.informational]


def _param(case):
    marks = [pytest.mark.numerical] if case.kind == NUMERICAL else []
    return pytest.param(case, id=case.id, marks=marks)


@pytest.mark.parametrize("case", [_param(c) for c in CASES])
def test_against_standard(case):
    result = evaluate(case)
    if result.status == "ERROR":
        pytest.fail("{} raised {}".format(case.function, result.error))
    assert result.status == "PASS", (
        "{fn}: computed {comp:.6g} {unit}, {std} {clause} gives {ref:.6g} {unit} "
        "(deviation {dev:+.2%}, allowed {tol}). {note}".format(
            fn=case.function, comp=result.computed, unit=case.unit,
            std=case.standard, clause=case.clause, ref=case.reference,
            dev=result.deviation or 0.0, tol=case.tolerance_text(), note=case.note))

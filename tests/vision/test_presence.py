# Copyright (c), The Rusoil Software Development Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# author : Konstantin Ustiuzhanin
# date   : 2026-Oct-21
# ==============================================================================
"""The station gate: which boxes may mean "the part is there".

The numbers in these tests are the measured ones from
``assets/images/metal_sheet_test_image.jpg``: the model's best box is the table
(area 0.2481) and the workpiece is its second candidate (area 0.0721, confidence
0.1594). That is the failure the gate exists to fix, so it is the case pinned first.
"""

from __future__ import annotations

import pytest

from src.vision.detections import Detection
from src.vision.presence import (
    FULL_FRAME,
    PresenceGate,
    enabled,
    from_environment,
    parse_roi,
    select_presence,
)

#: The station region measured on the real photo, in normalised frame coordinates.
#: The top edge sits below the wall band: during the replay pan that band scores
#: 0.070-0.087, i.e. inside the range the real part occupies, so only geometry keeps
#: it out.  See doc/vision-pipeline.md section 19.2.
STATION = (0.15, 0.42, 0.85, 0.85)

#: The gate a deployment can actually run, as validated against all eight pan phases
#: of the real photo.
CALIBRATED = PresenceGate(
    roi=STATION,
    min_area=0.02,
    max_area=0.20,
    min_aspect=1.0,
    max_aspect=2.0,
    min_confidence=0.05,
)


def box(
    x: float,
    y: float,
    width: float,
    height: float,
    confidence: float = 0.2,
    label: str = "steel blank",
) -> Detection:
    return Detection(
        object_type=label,
        x_norm=x,
        y_norm=y,
        width_norm=width,
        height_norm=height,
        confidence=confidence,
    )


def table() -> Detection:
    """The model's favourite box on the real photo: the white table."""
    return box(0.188, 0.378, 0.590, 0.421, confidence=0.2459)


def sheet() -> Detection:
    """The workpiece itself: the model's second candidate."""
    return box(0.3385, 0.454, 0.311, 0.232, confidence=0.1594)


# -- the failure that started this ----------------------------------------------


def test_the_table_is_refused_as_too_large_to_be_the_part():
    gate = CALIBRATED
    assert gate.accepts(sheet())
    reason = gate.rejection(table())
    assert reason is not None
    assert "a table, not the part" in reason
    assert "0.2484" in reason


def test_the_sheet_is_chosen_even_though_the_table_scores_higher():
    chosen, report = select_presence([table(), sheet()], CALIBRATED)
    assert chosen is not None
    assert chosen.confidence == pytest.approx(0.1594)
    assert (chosen.x_norm, chosen.y_norm) == pytest.approx((0.3385, 0.454))
    assert report["present"] is True
    assert report["offered"] == 2
    assert report["accepted"] == 1


def test_a_floor_box_far_from_the_station_is_refused():
    # The floor tile in the bottom-left corner of the photo, which scores 0.1216.
    reason = CALIBRATED.rejection(box(0.30, 0.78, 0.29, 0.22, confidence=0.1216))
    assert reason is not None
    assert "inside the station region" in reason


def test_a_confident_but_off_specimen_box_is_refused():
    # `mirror` scores 0.5684 on the floor, so confidence alone would fire.
    reason = CALIBRATED.rejection(box(0.30, 0.78, 0.45, 0.234, confidence=0.5684))
    assert reason is not None
    assert "a table, not the part" in reason or "station region" in reason


def test_the_wall_band_is_refused_by_aspect_not_by_confidence():
    # Measured on the real photo across the replay pan: the wall/door band scores
    # 0.070-0.087, overlapping the part's own 0.085-0.099 low phases, so a
    # confidence floor cannot separate them.  Its 2.26 aspect is the discriminator.
    reason = CALIBRATED.rejection(box(0.436, 0.310, 0.433, 0.191, confidence=0.0788))
    assert reason is not None
    assert "aspect 2.27 outside 1.00..2.00" in reason


@pytest.mark.parametrize(
    "x,y,width,height,confidence",
    [
        (0.3385, 0.563, 0.314, 0.239, 0.1199),
        (0.3385, 0.569, 0.314, 0.239, 0.1041),
        (0.3385, 0.567, 0.314, 0.239, 0.1253),
        (0.3385, 0.563, 0.314, 0.239, 0.1111),
        (0.3385, 0.559, 0.314, 0.239, 0.1201),
        (0.3385, 0.557, 0.314, 0.239, 0.0895),
        (0.3385, 0.559, 0.314, 0.239, 0.0987),
        (0.3385, 0.567, 0.314, 0.239, 0.0851),
    ],
)
def test_every_pan_phase_of_the_stand_admits_the_part(x, y, width, height, confidence):
    # The replay source pans its still, so the part's score moves by 40% between
    # phases; the gate must answer "present" on all of them, not every other frame.
    assert CALIBRATED.accepts(box(x, y, width, height, confidence=confidence))


def test_the_part_is_found_whatever_else_the_model_offers():
    # One frame of the real stand, as the worker saw it: table, wall, floor, part.
    frame = [
        box(0.188, 0.378, 0.590, 0.421, confidence=0.2001),
        box(0.436, 0.310, 0.433, 0.191, confidence=0.0866),
        box(0.300, 0.780, 0.290, 0.220, confidence=0.1165),
        box(0.3385, 0.563, 0.314, 0.239, confidence=0.1199),
    ]
    chosen, report = select_presence(frame, CALIBRATED)
    assert chosen is not None
    assert chosen.confidence == pytest.approx(0.1199)
    assert report["present"] is True
    assert report["offered"] == 4
    assert report["accepted"] == 1


# -- individual rules -----------------------------------------------------------


def test_confidence_below_the_calibrated_threshold_is_refused():
    reason = CALIBRATED.rejection(box(0.4, 0.5, 0.3, 0.2, confidence=0.04))
    assert reason is not None
    assert "confidence 0.040 < 0.050" in reason


def test_a_tiny_box_is_refused_as_noise():
    reason = CALIBRATED.rejection(box(0.4, 0.5, 0.02, 0.02, confidence=0.4))
    assert reason is not None
    assert "too small to be the part" in reason


def test_a_tall_box_is_refused_by_aspect():
    reason = CALIBRATED.rejection(box(0.4, 0.4, 0.05, 0.4, confidence=0.4))
    assert reason is not None
    assert "aspect" in reason


def test_a_box_half_over_the_station_edge_is_still_accepted():
    # The interesting case: the part is being put down, so it hangs over the edge.
    half_out = box(0.72, 0.5, 0.30, 0.20, confidence=0.3)
    assert CALIBRATED.inside(half_out) == pytest.approx(0.43, abs=0.02)
    permissive = PresenceGate(**{**CALIBRATED.__dict__, "min_inside": 0.4})
    assert permissive.accepts(half_out)


# -- reporting -----------------------------------------------------------------


def test_no_boxes_says_so_rather_than_blaming_the_gate():
    chosen, report = select_presence([], CALIBRATED)
    assert chosen is None
    assert report["present"] is False
    assert report["offered"] == 0
    assert "no box at all" in report["reason"]


def test_when_nothing_is_accepted_the_reason_names_the_best_boxs_flaw():
    chosen, report = select_presence([table()], CALIBRATED)
    assert chosen is None
    assert report["present"] is False
    assert "a table, not the part" in report["reason"]


def test_the_report_is_json_friendly():
    _, report = select_presence([table(), sheet()], CALIBRATED)
    assert set(report) >= {"present", "offered", "accepted", "confidence", "reason"}
    assert isinstance(report["confidence"], float)
    assert isinstance(report["offered"], int)


def test_the_most_confident_acceptable_box_wins():
    dim = box(0.4, 0.5, 0.3, 0.2, confidence=0.12)
    bright = box(0.4, 0.5, 0.3, 0.2, confidence=0.18)
    chosen, report = select_presence([dim, bright], CALIBRATED)
    assert chosen is not None and chosen.confidence == pytest.approx(0.18)
    assert report["accepted"] == 2


def test_the_default_gate_is_loose_on_purpose():
    gate = PresenceGate()
    assert gate.roi == FULL_FRAME
    assert gate.min_confidence == 0.0
    assert gate.accepts(box(0.0, 0.0, 0.4, 0.4, confidence=0.01))


# -- configuration ------------------------------------------------------------


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
def test_truthy_switches(value):
    assert enabled(value) is True


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off", None, "maybe"])
def test_everything_else_is_off(value):
    assert enabled(value) is False


def test_presence_is_off_unless_the_deployment_asks_for_it():
    assert from_environment({}) is None
    assert from_environment({"PRESENCE_ENABLED": "0"}) is None


def test_the_environment_describes_the_measured_station():
    gate = from_environment(
        {
            "PRESENCE_ENABLED": "1",
            "PRESENCE_ROI": "0.15,0.42,0.85,0.85",
            "PRESENCE_MIN_AREA": "0.02",
            "PRESENCE_MAX_AREA": "0.20",
            "PRESENCE_MIN_ASPECT": "1.0",
            "PRESENCE_MAX_ASPECT": "2.0",
            "PRESENCE_MIN_CONFIDENCE": "0.05",
            "PRESENCE_MIN_INSIDE": "0.5",
        }
    )
    assert gate is not None
    assert gate.roi == pytest.approx(STATION)
    assert gate.min_confidence == pytest.approx(0.05)
    assert gate.accepts(sheet())
    assert not gate.accepts(table())


def test_an_unconfigured_region_is_the_whole_frame():
    gate = from_environment({"PRESENCE_ENABLED": "1"})
    assert gate is not None and gate.roi == FULL_FRAME


@pytest.mark.parametrize(
    "text",
    ["", "0.15,0.33,0.85", "a,b,c,d", "0.9,0.1,0.2,0.3", "-0.1,0,1,1", "0.5,0.5,0.5,0.9"],
)
def test_a_broken_region_is_an_error_not_the_full_frame(text):
    with pytest.raises(ValueError):
        parse_roi(text)


def test_semicolons_are_accepted_as_separators():
    assert parse_roi("0.1;0.2;0.3;0.4") == pytest.approx((0.1, 0.2, 0.3, 0.4))


def test_an_inverted_area_band_is_refused():
    with pytest.raises(ValueError, match="area bounds are empty"):
        from_environment(
            {"PRESENCE_ENABLED": "1", "PRESENCE_MIN_AREA": "0.30", "PRESENCE_MAX_AREA": "0.20"}
        )


def test_a_non_numeric_budget_is_refused():
    with pytest.raises(ValueError, match="PRESENCE_MAX_AREA is not a number"):
        from_environment({"PRESENCE_ENABLED": "1", "PRESENCE_MAX_AREA": "big"})

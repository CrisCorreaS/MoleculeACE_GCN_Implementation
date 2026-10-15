"""Check scientific figure inputs and the submission's export contract on CPU.

Synthetic data exercise aggregation, input checks and naming without loading a
model. Renderer checks use the saved public summary tables and benchmark inputs.
File tests use isolated temporary directories.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import statistics
import tempfile
import unittest
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axis import Axis
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from matplotlib.mathtext import MathTextParser
from matplotlib.patches import ArrowStyle
from matplotlib.table import Table
from matplotlib.text import Annotation, Text
from matplotlib.transforms import Affine2D, Bbox
import numpy as np
import pandas as pd

import plot_paper_figures as figures


EXPECTED_FIGURE_NAMES = (
    "paired_cliff_rmse",
    "paired_rmse",
    "paired_error_metrics",
    "cliff_rmse_strategy_context",
    "rmse_cliff_comparison",
    "rmse_cliff_comparison_overlay",
    "rmse_cliff_shifts_zoomed",
    "rmse_cliff_shifts_context",
    "cliff_error_difference_strategy_context",
    "cliff_error_difference_by_dataset",
    "dataset_errors_mean_sd",
    "dataset_errors_by_fold",
    "cliff_error_difference_training_size",
    "cliff_error_difference_training_size_comparison",
    "published_reproduced_agreement",
    "bland_altman_agreement",
    "published_reproduced_differences_by_dataset",
    "published_reproduced_difference_distribution",
    "published_strategy_ranking_context",
    "activity_cliff_example_datasets",
    "activity_cliff_overview",
    "dataset_comparison_overview",
    "gcn_results_overview",
    "dataset_error_comparison_table",
)


def collect_current_figures(project_root: Path) -> dict[str, Figure]:
    """Render the actual export dispatch from saved inputs without saving figures.

    Parameters
    ----------
    project_root : Path
        Submission root containing the public reference and summary tables.

    Returns
    -------
    dict
        Export names mapped to figures; callers must close the figures afterwards.
    """
    captured: dict[str, Figure] = {}

    def capture(
        writer: figures.FigureWriter,
        figure: Figure,
        name: str,
        title: str,
        caption: str,
    ) -> None:
        if name in captured:
            raise AssertionError(f"Duplicate figure export name: {name}")
        captured[name] = figure

    root = project_root.resolve()
    with tempfile.TemporaryDirectory(prefix="figure_layout_") as temporary:
        arguments = [
            "--published",
            str(root / "references/published/MoleculeACE_results.csv"),
            "--benchmark-dir",
            str(root / "code/MoleculeACE/Data/benchmark_data"),
            "--comparison",
            str(root / "results/gcn_full_v1/_analysis/comparison.csv"),
            "--per-fold",
            str(root / "results/gcn_full_v1/_summary/per_fold.csv"),
            "--output-dir",
            str(Path(temporary) / "captured"),
        ]
        with (
            patch.object(
                figures.FigureWriter, "save", autospec=True, side_effect=capture
            ),
            patch.object(figures, "write_captions"),
            patch.object(figures, "write_catalogue"),
            patch.object(figures, "write_manifest"),
            patch.object(pd.DataFrame, "to_csv"),
            patch("builtins.print"),
            plt.rc_context({"figure.max_open_warning": 0}),
        ):
            figures.main(arguments)
    return captured


def visible_text_bounds(figure: Figure) -> list[tuple[Text, Bbox]]:
    """Collect rendered text bounds, excluding hidden axes and off-range ticks.

    Annotation arrow bounds are deliberately excluded: only glyph layout is
    relevant to label collisions. The canvas must have been drawn first.
    """
    renderer = figure.canvas.get_renderer()
    candidates = list(figure.texts)
    legends = list(figure.legends)
    for axis in figure.axes:
        if not axis.get_visible():
            continue
        for child in axis.get_children():
            if not isinstance(child, Axis) and child.get_visible():
                candidates.extend(child.findobj(match=Text))
                if isinstance(child, Table):
                    candidates.extend(
                        cell.get_text()
                        for cell in child.get_celld().values()
                        if cell.get_visible()
                    )
        if axis.get_legend() is not None:
            legends.append(axis.get_legend())
        if not axis.axison:
            continue
        for coordinate in (axis.xaxis, axis.yaxis):
            if not coordinate.get_visible():
                continue
            candidates.extend([coordinate.label, coordinate.get_offset_text()])
            lower, upper = sorted(coordinate.get_view_interval())
            tolerance = (upper - lower) * 1e-10
            for tick in coordinate.get_major_ticks() + coordinate.get_minor_ticks():
                if (
                    tick.get_visible()
                    and lower - tolerance <= tick.get_loc() <= upper + tolerance
                ):
                    candidates.extend([tick.label1, tick.label2])
    for legend in legends:
        if legend.get_visible():
            candidates.extend(legend.get_texts())
            candidates.append(legend.get_title())
    found = []
    visited = set()
    for text in candidates:
        if id(text) in visited:
            continue
        visited.add(id(text))
        if (
            not text.get_visible()
            or not text.get_text().strip()
            or text.get_alpha() == 0
        ):
            continue
        box = Text.get_window_extent(text, renderer=renderer)
        if not np.isfinite(box.extents).all() or box.width <= 0 or box.height <= 0:
            continue
        if text.get_clip_on() and text.get_clip_box() is not None:
            if Bbox.intersection(box, text.get_clip_box()) is None:
                continue
        found.append((text, box))
    return found


def marker_bounds(axis: plt.Axes) -> list[Bbox]:
    """Return visible scatter and line marker footprints in display pixels.

    Scatter transforms already contain the point-area and DPI scaling after a
    canvas draw. Line markers use their path, marker size and edge width. Clipping
    is applied to the rendered footprint, not merely to the marker centre.
    """
    pixels_per_point = axis.figure.dpi / 72
    bounds = []
    for artist in [*axis.collections, *axis.lines]:
        if not artist.get_visible() or artist.get_alpha() == 0:
            continue
        footprints = []
        if isinstance(artist, PathCollection):
            offsets = np.ma.asarray(artist.get_offsets()).filled(np.nan)
            centres = artist.get_offset_transform().transform(offsets)
            paths = artist.get_paths()
            transforms = artist.get_transforms()
            widths = artist.get_linewidths()
            faces = artist.get_facecolors()
            edges = artist.get_edgecolors()
            for index, centre in enumerate(centres):
                if not np.isfinite(centre).all() or not paths:
                    continue
                has_face = len(faces) and faces[index % len(faces), 3] > 0
                has_edge = len(edges) and edges[index % len(edges), 3] > 0
                if not has_face and not has_edge:
                    continue
                transform = (
                    Affine2D(transforms[index % len(transforms)])
                    if len(transforms)
                    else Affine2D()
                )
                box = paths[index % len(paths)].get_extents(
                    transform + artist.get_transform()
                )
                edge = (
                    widths[index % len(widths)] * pixels_per_point / 2
                    if has_edge and len(widths)
                    else 0
                )
                footprints.append(
                    Bbox.from_extents(*(box.extents + np.tile(centre, 2))).padded(edge)
                )
        elif isinstance(artist, Line2D) and artist.get_marker() not in (
            None,
            "None",
            "none",
            "",
            " ",
        ):
            marker = MarkerStyle(artist.get_marker())
            shape = marker.get_path().get_extents(
                marker.get_transform()
                + Affine2D().scale(artist.get_markersize() * pixels_per_point)
            )
            centres = artist.get_transform().transform(artist.get_xydata())
            every = artist.get_markevery()
            if isinstance(every, int):
                centres = centres[::every]
            elif isinstance(every, slice):
                centres = centres[every]
            elif isinstance(every, tuple) and all(
                isinstance(item, int) for item in every
            ):
                centres = centres[every[0] :: every[1]]
            elif isinstance(every, list):
                centres = centres[every]
            edge = artist.get_markeredgewidth() * pixels_per_point / 2
            footprints.extend(
                Bbox.from_extents(*(shape.extents + np.tile(centre, 2))).padded(edge)
                for centre in centres
                if np.isfinite(centre).all()
            )
        for box in footprints:
            if artist.get_clip_on():
                clip = artist.get_clip_box()
                clip = axis.bbox if clip is None else clip
                box = Bbox.intersection(box, clip)
            if box is not None and box.width > 0 and box.height > 0:
                bounds.append(box)
    return bounds


def audit_tick_label_spacing(
    figure: Figure, minimum_gap_points: float = 2.0
) -> list[dict]:
    """Require physical separation between adjacent visible tick labels.

    Two points provide a clear gap of roughly 29% of the minimum 7 pt base
    type size. Distance is measured between rendered rectangles in both
    directions, so rotated labels are not judged by horizontal projection alone.
    Hidden and off-range ticks are excluded; opposite axis sides are separate.
    The canvas must have been drawn before calling this helper.
    """
    text_boxes = dict(visible_text_bounds(figure))
    findings = []
    for axis_index, axis in enumerate(figure.axes):
        if not axis.get_visible() or not axis.axison:
            continue
        for direction, coordinate in (("x", axis.xaxis), ("y", axis.yaxis)):
            if not coordinate.get_visible():
                continue
            lower, upper = sorted(coordinate.get_view_interval())
            tolerance = (upper - lower) * 1e-10
            ticks = [
                tick
                for tick in coordinate.get_major_ticks() + coordinate.get_minor_ticks()
                if tick.get_visible()
                and lower - tolerance <= tick.get_loc() <= upper + tolerance
            ]
            for side in ("label1", "label2"):
                labels = [
                    getattr(tick, side)
                    for tick in ticks
                    if getattr(tick, side) in text_boxes
                ]
                labels.sort(
                    key=lambda text: (
                        text_boxes[text].x0 if direction == "x" else text_boxes[text].y0
                    )
                )
                for left, right in zip(labels, labels[1:]):
                    first, second = text_boxes[left], text_boxes[right]
                    horizontal_gap = max(first.x0 - second.x1, second.x0 - first.x1, 0)
                    vertical_gap = max(first.y0 - second.y1, second.y0 - first.y1, 0)
                    gap_points = float(
                        np.hypot(horizontal_gap, vertical_gap) * 72 / figure.dpi
                    )
                    if gap_points + 1e-6 < minimum_gap_points:
                        findings.append(
                            {
                                "kind": "tick_labels_too_close",
                                "axis_index": axis_index,
                                "direction": direction,
                                "side": side,
                                "text": left.get_text(),
                                "other_text": right.get_text(),
                                "gap_points": gap_points,
                                "minimum_gap_points": minimum_gap_points,
                                "bounds": first.extents.tolist(),
                                "other_bounds": second.extents.tolist(),
                            }
                        )
    return findings


def audit_figure_layout(
    figure: Figure,
    minimum_font_size: float = 7.0,
    export_settings: dict | None = None,
) -> dict:
    """Audit export size, base fonts, text bounds, tick gaps and marker collisions.

    Returns JSON-compatible findings with display coordinates for diagnosis.
    Bounding-box intersections require more than half a display pixel in each
    dimension, allowing harmless numerical contact at an edge.
    Mathematical subscripts and superscripts may be smaller than the base font.
    """
    if export_settings is not None:
        dpi = export_settings.get("savefig.dpi", "figure")
        if dpi != "figure":
            figure.set_dpi(dpi)
    with plt.rc_context(export_settings or {}):
        figure.canvas.draw()
    text_bounds = visible_text_bounds(figure)
    findings = []
    canvas = figure.bbox
    if export_settings is not None and export_settings.get("savefig.bbox") == "tight":
        canvas = (
            figure.get_tightbbox(figure.canvas.get_renderer())
            .padded(export_settings.get("savefig.pad_inches", 0.1))
            .transformed(figure.dpi_scale_trans)
        )
    width_cm = canvas.width / figure.dpi * 2.54
    height_cm = canvas.height / figure.dpi * 2.54
    for dimension, actual, maximum in (
        ("width", width_cm, figures.MAX_FIGURE_WIDTH_CM),
        ("height", height_cm, figures.MAX_FIGURE_HEIGHT_CM),
    ):
        if actual > maximum + 1e-6:
            findings.append(
                {
                    "kind": "export_dimension_exceeded",
                    "dimension": dimension,
                    "actual_cm": actual,
                    "maximum_cm": maximum,
                }
            )
    for text, box in text_bounds:
        identity = {"text": text.get_text(), "bounds": box.extents.tolist()}
        if text.get_fontsize() < minimum_font_size - 1e-6:
            findings.append(
                {
                    "kind": "font_below_minimum",
                    **identity,
                    "font_size": text.get_fontsize(),
                }
            )
        if (
            box.x0 < canvas.x0 - 0.5
            or box.y0 < canvas.y0 - 0.5
            or box.x1 > canvas.x1 + 0.5
            or box.y1 > canvas.y1 + 0.5
        ):
            findings.append({"kind": "text_outside_canvas", **identity})
    for index, (left, left_box) in enumerate(text_bounds):
        for right, right_box in text_bounds[index + 1 :]:
            overlap = Bbox.intersection(left_box, right_box)
            if overlap is not None and overlap.width > 0.5 and overlap.height > 0.5:
                findings.append(
                    {
                        "kind": "text_overlap",
                        "text": left.get_text(),
                        "other_text": right.get_text(),
                        "bounds": overlap.extents.tolist(),
                    }
                )
    for axis in figure.axes:
        if not axis.get_visible():
            continue
        markers = marker_bounds(axis)
        annotations = set(axis.texts)
        for text, box in text_bounds:
            if text not in annotations:
                continue
            inside = Bbox.intersection(box, axis.bbox)
            if inside is None:
                continue
            overlaps = []
            for marker in markers:
                overlap = Bbox.intersection(inside, marker)
                if overlap is not None and overlap.width > 0.5 and overlap.height > 0.5:
                    overlaps.append(marker.extents.tolist())
            if overlaps:
                findings.append(
                    {
                        "kind": "annotation_over_marker",
                        "text": text.get_text(),
                        "bounds": box.extents.tolist(),
                        "marker_bounds": overlaps,
                    }
                )
    findings.extend(audit_tick_label_spacing(figure))
    return {
        "canvas_bounds": canvas.extents.tolist(),
        "design_canvas_bounds": figure.bbox.extents.tolist(),
        "export_width_cm": width_cm,
        "export_height_cm": height_cm,
        "text_count": len(text_bounds),
        "findings": findings,
    }


def audit_figure_notation(name: str, figure: Figure) -> list[dict]:
    """Require mathematical metric and count notation in the rendered labels."""
    labels = [text.get_text() for text, _ in visible_text_bounds(figure)]
    findings = [
        {"kind": "reversed_cliff_metric_label", "text": label}
        for label in labels
        if "Cliff RMSE" in label
    ]
    if name != "paired_rmse" and not any(
        re.search(r"RMSE\}?\$?\s*_\{", label) and "cliff" in label for label in labels
    ):
        findings.append({"kind": "missing_cliff_metric_subscript"})
    if name in {
        "cliff_error_difference_strategy_context",
        "cliff_error_difference_by_dataset",
        "dataset_comparison_overview",
    } and not any(
        re.search(r"r_\{", label)
        and label.count("RMSE") == 2
        and label.count("_{") >= 2
        and "cliff" in label
        for label in labels
    ):
        findings.append({"kind": "missing_nested_correlation_subscript"})
    if name == "dataset_error_comparison_table":
        for count in ("train", "cliff"):
            if not any(
                re.search(r"n_\{", label) and count in label for label in labels
            ):
                findings.append({"kind": "missing_count_subscript", "count": count})
    return findings


def audit_shift_annotation_clearance(
    figure: Figure, minimum_clearance_points: float = 1.0
) -> list[dict]:
    """Check shift labels against axes boundaries and actual arrow paths.

    The figure must already have been drawn. Leader arrows are checked too;
    their intentional approach to a label must retain the stated clearance.
    """
    renderer = figure.canvas.get_renderer()
    clearance = minimum_clearance_points * figure.dpi / 72
    findings = []
    for axis in figure.axes:
        annotations = [
            text
            for text in axis.texts
            if isinstance(text, Annotation) and text.get_visible()
        ]
        arrows = [
            text.arrow_patch
            for text in annotations
            if text.arrow_patch is not None and text.arrow_patch.get_visible()
        ]
        for text in annotations:
            if not text.get_text().strip():
                continue
            box = Text.get_window_extent(text, renderer=renderer).padded(clearance)
            if not (
                axis.bbox.contains(box.x0, box.y0)
                and axis.bbox.contains(box.x1, box.y1)
            ):
                findings.append(
                    {
                        "kind": "shift_label_near_axis_boundary",
                        "text": text.get_text(),
                        "bounds": box.extents.tolist(),
                    }
                )
            for arrow in arrows:
                path = arrow.get_path().transformed(arrow.get_transform())
                if path.intersects_bbox(box, filled=False):
                    findings.append(
                        {
                            "kind": "shift_label_near_arrow",
                            "text": text.get_text(),
                            "bounds": box.extents.tolist(),
                        }
                    )
                    break
    return findings


class FigureLayoutTests(unittest.TestCase):
    """Detect real renderer regressions and prove each check with bad fixtures."""

    def tearDown(self) -> None:
        plt.close("all")

    def tick_spacing_fixture(self, gap_points: float, rotation: float) -> Figure:
        """Place two rendered tick labels with a known positive physical gap."""
        figure, axis = plt.subplots(figsize=(4, 2), dpi=100)
        figure.subplots_adjust(bottom=0.3)
        axis.set(xlim=(0, 1), ylim=(0, 1), yticks=[])
        axis.set_xticks([0.3, 0.7], labels=["−0.1", "0.0"])
        axis.tick_params(axis="x", labelsize=7, labelrotation=rotation)
        figure.canvas.draw()
        labels = axis.get_xticklabels()
        left, right = [
            text.get_window_extent(figure.canvas.get_renderer()) for text in labels
        ]
        current_gap = right.x0 - left.x1
        shift = (gap_points * figure.dpi / 72 - current_gap) / axis.bbox.width
        axis.set_xticks([0.3, 0.7 + shift], labels=["−0.1", "0.0"])
        figure.canvas.draw()
        return figure

    def test_non_overlapping_tick_labels_still_require_clear_separation(self) -> None:
        for rotation in (0, 45):
            with self.subTest(rotation=rotation):
                figure = self.tick_spacing_fixture(1.0, rotation)
                report = audit_figure_layout(figure)
                self.assertNotIn(
                    "text_overlap", {item["kind"] for item in report["findings"]}
                )
                findings = [
                    item
                    for item in report["findings"]
                    if item["kind"] == "tick_labels_too_close"
                ]
                self.assertEqual(len(findings), 1)
                self.assertAlmostEqual(findings[0]["gap_points"], 1.0)
                self.assertGreater(findings[0]["gap_points"], 0)

    def test_well_separated_tick_labels_pass_with_or_without_rotation(self) -> None:
        for rotation in (0, 45):
            with self.subTest(rotation=rotation):
                figure = self.tick_spacing_fixture(3.0, rotation)
                self.assertEqual(audit_figure_layout(figure)["findings"], [])

    def test_strategy_categories_keep_equal_spacing_and_explained_abbreviations(
        self,
    ) -> None:
        root = Path(__file__).resolve().parents[2]
        published = figures.load_published(
            root / "references/published/MoleculeACE_results.csv"
        )
        counts = figures.load_benchmark_counts(
            root / "code/MoleculeACE/Data/benchmark_data"
        )
        comparison = figures.load_comparison(
            root / "results/gcn_full_v1/_analysis/comparison.csv"
        )
        table = figures.build_table(published, counts, comparison, comparison)
        writer = figures.FigureWriter(Path("unused"), False, [])
        with plt.rc_context(figures.STYLE):
            figures.f03_fig4a_context(writer, table, published)
            figures.f19_composite_figure4_layout(writer, table, published)
        for saved in writer.figures:
            with self.subTest(figure=saved["name"]):
                axis = saved["figure"].axes[0]
                ticks = axis.get_xticks()
                self.assertEqual(len(ticks), 10)
                np.testing.assert_allclose(np.diff(ticks), np.ones(9))
                labels = [
                    text.get_text().replace("\n", " ")
                    for text in axis.get_xticklabels()
                ]
                self.assertIn("GCN publ.", labels)
                self.assertIn("GCN repr.", labels)
                self.assertEqual(
                    labels.index("GCN repr."), labels.index("GCN publ.") + 1
                )
                self.assertRegex(saved["caption"], r"publ\..{0,30}published")
                self.assertRegex(saved["caption"], r"repr\..{0,30}reproduced")

    def test_every_current_figure_has_readable_non_overlapping_text(self) -> None:
        rendered = collect_current_figures(Path(__file__).resolve().parents[2])
        self.assertEqual(tuple(rendered), EXPECTED_FIGURE_NAMES)
        for name, figure in rendered.items():
            with self.subTest(figure=name):
                report = audit_figure_layout(figure, export_settings=figures.STYLE)
                report["findings"].extend(audit_figure_notation(name, figure))
                if name == "gcn_results_overview":
                    self.assertIn(
                        "Reproduced vs published",
                        [text.get_text() for text in figure.axes[2].texts],
                    )
                if name in {"rmse_cliff_shifts_zoomed", "rmse_cliff_shifts_context"}:
                    report["findings"].extend(audit_shift_annotation_clearance(figure))
                self.assertEqual(
                    report["findings"],
                    [],
                    json.dumps(report["findings"], ensure_ascii=False),
                )

    def test_outside_text_and_small_font_are_detected(self) -> None:
        figure = plt.figure(figsize=(3, 2))
        figure.text(1.05, 0.5, "Outside", fontsize=8)
        figure.text(0.2, 0.5, "Too small", fontsize=6)
        kinds = {item["kind"] for item in audit_figure_layout(figure)["findings"]}
        self.assertEqual(kinds, {"text_outside_canvas", "font_below_minimum"})

    def test_overlapping_text_is_detected(self) -> None:
        figure = plt.figure(figsize=(3, 2))
        figure.text(0.4, 0.5, "First label", fontsize=9)
        figure.text(0.41, 0.5, "Second label", fontsize=9)
        self.assertIn(
            "text_overlap",
            {item["kind"] for item in audit_figure_layout(figure)["findings"]},
        )

    def test_mathtext_subscripts_and_superscripts_may_be_smaller_than_base_font(
        self,
    ) -> None:
        figure = plt.figure(figsize=(3, 2))
        text = figure.text(
            0.2,
            0.5,
            r"$r_{\mathrm{RMSE},\mathrm{RMSE}_{\mathrm{cliff}}}^{2}$",
            fontsize=7,
        )
        parsed = MathTextParser("path").parse(
            text.get_text(), dpi=figure.dpi, prop=text.get_fontproperties()
        )
        self.assertLess(min(glyph[1] for glyph in parsed.glyphs), 7)
        self.assertEqual(audit_figure_layout(figure)["findings"], [])

    def test_export_size_accepts_the_physical_limits_including_padding(self) -> None:
        padding_inches = 0.04
        width_inches = figures.MAX_FIGURE_WIDTH_CM / 2.54 - 2 * padding_inches
        height_inches = figures.MAX_FIGURE_HEIGHT_CM / 2.54 - 2 * padding_inches
        figure = plt.figure(figsize=(width_inches, height_inches))
        figure.add_axes((0, 0, 1, 1)).set_axis_off()
        report = audit_figure_layout(
            figure,
            export_settings={
                "savefig.bbox": "tight",
                "savefig.pad_inches": padding_inches,
            },
        )
        self.assertAlmostEqual(report["export_width_cm"], figures.MAX_FIGURE_WIDTH_CM)
        self.assertAlmostEqual(report["export_height_cm"], figures.MAX_FIGURE_HEIGHT_CM)
        self.assertEqual(report["findings"], [])

    def test_export_size_rejects_width_or_height_beyond_the_limit(self) -> None:
        for dimension in ("width", "height"):
            with self.subTest(dimension=dimension):
                width_cm = figures.MAX_FIGURE_WIDTH_CM + (
                    0.02 if dimension == "width" else -1
                )
                height_cm = figures.MAX_FIGURE_HEIGHT_CM + (
                    0.02 if dimension == "height" else -1
                )
                padding_inches = 0.04
                figure = plt.figure(
                    figsize=(
                        width_cm / 2.54 - 2 * padding_inches,
                        height_cm / 2.54 - 2 * padding_inches,
                    )
                )
                figure.add_axes((0, 0, 1, 1)).set_axis_off()
                report = audit_figure_layout(
                    figure,
                    export_settings={
                        "savefig.bbox": "tight",
                        "savefig.pad_inches": padding_inches,
                    },
                )
                self.assertEqual(
                    [
                        item["dimension"]
                        for item in report["findings"]
                        if item["kind"] == "export_dimension_exceeded"
                    ],
                    [dimension],
                )

    def test_notation_guards_detect_reversed_labels_and_missing_subscripts(
        self,
    ) -> None:
        figure = plt.figure(figsize=(3, 2))
        figure.text(0.1, 0.8, "Cliff RMSE", fontsize=7)
        figure.text(0.1, 0.5, "n_train   n_cliff", fontsize=7)
        figure.canvas.draw()
        findings = audit_figure_notation("dataset_error_comparison_table", figure)
        self.assertEqual(
            {item["kind"] for item in findings},
            {
                "reversed_cliff_metric_label",
                "missing_cliff_metric_subscript",
                "missing_count_subscript",
            },
        )
        self.assertIn(
            "missing_nested_correlation_subscript",
            {
                item["kind"]
                for item in audit_figure_notation("dataset_comparison_overview", figure)
            },
        )

    def test_notation_guards_accept_nested_mathematical_subscripts(self) -> None:
        figure = plt.figure(figsize=(3, 2))
        figure.text(0.1, 0.8, r"RMSE$_{\mathrm{cliff}}$", fontsize=7)
        figure.text(0.1, 0.5, r"$n_{\mathrm{train}}$  $n_{\mathrm{cliff}}$", fontsize=7)
        figure.text(0.1, 0.2, r"$r_{\mathrm{RMSE,\ RMSE_{cliff}}}$", fontsize=7)
        figure.canvas.draw()
        self.assertEqual(
            audit_figure_notation("dataset_error_comparison_table", figure), []
        )
        self.assertEqual(
            audit_figure_notation("dataset_comparison_overview", figure), []
        )

    def test_tight_export_canvas_includes_external_text_but_not_excluded_text(
        self,
    ) -> None:
        figure = plt.figure(figsize=(3, 2))
        figure.text(0.2, 0.5, "Inside", fontsize=8)
        external = figure.text(1.1, 0.5, "External legend", fontsize=8)
        settings = {"savefig.bbox": "tight", "savefig.pad_inches": 0.04}
        self.assertEqual(
            audit_figure_layout(figure, export_settings=settings)["findings"], []
        )
        external.set_in_layout(False)
        self.assertIn(
            "text_outside_canvas",
            {
                item["kind"]
                for item in audit_figure_layout(figure, export_settings=settings)[
                    "findings"
                ]
            },
        )

    def test_scatter_and_line_marker_collisions_are_detected(self) -> None:
        for drawing in ("scatter", "line"):
            with self.subTest(drawing=drawing):
                figure, axis = plt.subplots(figsize=(3, 2))
                axis.set(xlim=(0, 1), ylim=(0, 1))
                if drawing == "scatter":
                    axis.scatter([0.5], [0.5], s=144)
                else:
                    axis.plot([0.5], [0.5], marker="D", markersize=12)
                axis.text(0.5, 0.5, "Collision", fontsize=9)
                self.assertIn(
                    "annotation_over_marker",
                    {item["kind"] for item in audit_figure_layout(figure)["findings"]},
                )

    def test_marker_footprints_use_display_sizes_and_partial_clipping(self) -> None:
        figure, axis = plt.subplots(figsize=(4, 3), dpi=100)
        axis.set(xlim=(0, 1), ylim=(0, 1))
        scatter = axis.scatter([0.5], [0.5], s=100, linewidths=0)
        (line,) = axis.plot([0.5], [0.5], marker="o", markersize=10, markeredgewidth=0)
        figure.canvas.draw()
        for box in marker_bounds(axis):
            self.assertAlmostEqual(box.width, 10 * 100 / 72)
            self.assertAlmostEqual(box.height, 10 * 100 / 72)
        scatter.set_offsets([[-0.01, 0.5]])
        line.set_data([-0.01], [0.5])
        axis.text(0, 0.5, "Clipped marker collision", fontsize=9, va="center")
        report = audit_figure_layout(figure)
        partial = marker_bounds(axis)
        self.assertEqual(len(partial), 2)
        for box in partial:
            self.assertAlmostEqual(box.x0, axis.bbox.x0)
            self.assertGreater(box.width, 0)
            self.assertLess(box.width, 10 * 100 / 72)
        self.assertIn(
            "annotation_over_marker", {item["kind"] for item in report["findings"]}
        )

    def test_table_cell_text_is_included_in_font_checks(self) -> None:
        figure, axis = plt.subplots(figsize=(4, 3))
        axis.set_axis_off()
        table = axis.table(cellText=[["Small table entry"]], loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(6)
        findings = audit_figure_layout(figure)["findings"]
        self.assertTrue(
            any(
                item["kind"] == "font_below_minimum"
                and item["text"] == "Small table entry"
                for item in findings
            )
        )

    def test_shift_labels_near_arrows_and_axis_boundaries_are_detected(self) -> None:
        figure, axis = plt.subplots(figsize=(4, 3))
        axis.set(xlim=(0, 1), ylim=(0, 1))
        axis.annotate("Beyond boundary", xy=(0.99, 0.8), fontsize=9)
        axis.annotate("Over arrow", xy=(0.45, 0.5), va="center", fontsize=9)
        axis.annotate(
            "", xy=(0.8, 0.5), xytext=(0.2, 0.5), arrowprops={"arrowstyle": "->"}
        )
        figure.canvas.draw()
        findings = audit_shift_annotation_clearance(figure)
        self.assertEqual(
            {item["kind"] for item in findings},
            {"shift_label_near_axis_boundary", "shift_label_near_arrow"},
        )

    def test_short_shift_connectors_keep_true_endpoints_without_arrowheads(
        self,
    ) -> None:
        figure, axis = plt.subplots(figsize=(4, 3), dpi=100)
        axis.set(xlim=(0, 1), ylim=(0, 1))
        table = pd.DataFrame(
            {
                "pub_rmse": [0.2, 0.3],
                "pub_cliff": [0.2, 0.3],
                "rep_rmse": [0.2005, 0.8],
                "rep_cliff": [0.2005, 0.7],
            }
        )
        figures.draw_shift_connectors(axis, table, short_threshold_points=5.0)
        figure.canvas.draw()
        self.assertEqual(len(axis.texts), 2)
        for index, annotation in enumerate(axis.texts):
            published = table.loc[index, ["pub_rmse", "pub_cliff"]].to_numpy()
            reproduced = table.loc[index, ["rep_rmse", "rep_cliff"]].to_numpy()
            np.testing.assert_allclose(annotation.xyann, published)
            np.testing.assert_allclose(annotation.xy, reproduced)
            arrow = annotation.arrow_patch
            expected_style = ArrowStyle("-" if index == 0 else "-|>")
            self.assertIs(type(arrow.get_arrowstyle()), type(expected_style))
            self.assertEqual(arrow.shrinkA, 0 if index == 0 else 2.5)
            self.assertEqual(arrow.shrinkB, 0 if index == 0 else 2.5)
            if index == 0:
                path = arrow.get_path().transformed(arrow.get_transform())
                np.testing.assert_allclose(
                    path.vertices[[0, -1]],
                    axis.transData.transform([published, reproduced]),
                )

    def test_arrow_extent_is_not_mistaken_for_label_overlap(self) -> None:
        figure, axis = plt.subplots(figsize=(4, 3))
        axis.set(xlim=(0, 1), ylim=(0, 1))
        axis.annotate(
            "Remote label",
            xy=(0.2, 0.2),
            xytext=(0.65, 0.8),
            arrowprops={"arrowstyle": "->"},
            fontsize=9,
        )
        axis.text(0.2, 0.25, "By arrow", fontsize=9)
        axis.scatter([0.2], [0.2], s=100)
        self.assertEqual(audit_figure_layout(figure)["findings"], [])

    def test_hidden_ticks_and_clipped_markers_do_not_create_findings(self) -> None:
        figure, axis = plt.subplots(figsize=(4, 3))
        axis.set(xlim=(0, 1), ylim=(0, 1))
        axis.set_xticks([-10, 0, 1, 10], labels=["off left", "0", "1", "off right"])
        axis.set_xlim(0, 1)
        axis.scatter([-0.5], [0.5], s=400)
        axis.text(-0.5, 0.5, "Hidden", visible=False, fontsize=1)
        report = audit_figure_layout(figure)
        self.assertEqual(report["findings"], [])
        self.assertEqual(marker_bounds(axis), [])


class FigureDataTests(unittest.TestCase):
    """Check quantities and invalid inputs independently of plotting."""

    def setUp(self) -> None:
        self.datasets = list(figures.ABBREVIATIONS)
        rows = []
        published_rows = []
        for index, dataset in enumerate(self.datasets):
            baseline = 0.6 + index * 0.01
            for fold in range(5):
                rows.append(
                    dict(
                        dataset=dataset,
                        fold=fold,
                        rmse=baseline + fold * 0.02,
                        rmse_cliff=baseline + 0.1 + fold * 0.04,
                    )
                )
            for algorithm, descriptor, offset in (
                ("GCN", "GRAPH", 0.0),
                ("SVM", "ECFP", 0.2),
            ):
                published_rows.append(
                    dict(
                        dataset=dataset,
                        algorithm=algorithm,
                        descriptor=descriptor,
                        rmse=baseline + offset,
                        cliff_rmse=baseline + offset + 0.15,
                    )
                )
        self.folds = pd.DataFrame(rows)
        self.folds["delta"] = self.folds["rmse_cliff"] - self.folds["rmse"]
        self.published_input = pd.DataFrame(published_rows)
        with patch.object(figures.pd, "read_csv", return_value=self.published_input):
            self.published = figures.load_published(Path("synthetic.csv"))
        self.summary = figures.summarise_folds(self.folds)
        self.counts = pd.DataFrame(
            {"n_train": 500, "n_test": 100, "n_test_cliff": 20},
            index=self.datasets,
        )

    def tearDown(self) -> None:
        plt.close("all")

    def test_comparison_table_orders_all_datasets_by_training_size_and_id(self) -> None:
        counts = self.counts.copy()
        counts["n_train"] = [200 + 100 * (index % 3) for index in range(30)]
        table = figures.build_table(self.published, counts, self.summary).iloc[::-1]
        original = table.copy(deep=True)
        writer = figures.FigureWriter(Path("unused"), False, [])
        with plt.rc_context(figures.STYLE):
            figures.f22_dataset_error_comparison_table(writer, table)
        labels = [
            text
            for text in writer.figures[0]["figure"].axes[0].texts
            if text.get_text() in self.datasets
        ]
        actual = [
            text.get_text()
            for text in sorted(labels, key=lambda text: text.get_position()[1])
        ]
        expected = sorted(
            self.datasets, key=lambda dataset: (counts.loc[dataset, "n_train"], dataset)
        )
        self.assertEqual(len(actual), 30)
        self.assertEqual(len(set(actual)), 30)
        self.assertEqual(actual, expected)
        self.assertGreater(counts["n_train"].duplicated().sum(), 0)
        pd.testing.assert_frame_equal(table, original)

    def test_comparison_table_counts_each_published_test_cliff_molecule_once(
        self,
    ) -> None:
        benchmark = pd.DataFrame(
            {
                "split": ["train"] * 3 + ["test"] * 5,
                "cliff_mol": [1, 1, 0, 1, 1, 1, 1, 0],
            }
        )
        with (
            patch.object(Path, "is_file", return_value=True),
            patch.object(figures.pd, "read_csv", return_value=benchmark),
        ):
            counts = figures.load_benchmark_counts(Path("synthetic"))
        self.assertTrue((counts["n_test_cliff"] == 4).all())
        self.assertTrue((counts["n_train"] == 3).all())
        table = figures.build_table(self.published, counts, self.summary)
        writer = figures.FigureWriter(Path("unused"), False, [])
        with plt.rc_context(figures.STYLE):
            figures.f22_dataset_error_comparison_table(writer, table)
        text = writer.figures[0]["figure"].axes[0].texts
        for row in range(30):
            cells = sorted(
                [item for item in text if item.get_position()[1] == row],
                key=lambda item: item.get_position()[0],
            )
            self.assertEqual(len(cells), 4)
            self.assertEqual(cells[-2].get_text(), "3")
            self.assertEqual(cells[-1].get_text(), "4")

    def test_comparison_table_plots_paired_difference_sd_and_cliff_sd(self) -> None:
        table = figures.build_table(self.published, self.counts, self.summary)
        writer = figures.FigureWriter(Path("unused"), False, [])
        with plt.rc_context(figures.STYLE):
            figures.f22_dataset_error_comparison_table(writer, table)
        axis_a, axis_b = writer.figures[0]["figure"].axes[1:]
        segments_a = [
            collection.get_segments()
            for collection in axis_a.collections
            if isinstance(collection, LineCollection)
        ]
        segments_b = [
            collection.get_segments()
            for collection in axis_b.collections
            if isinstance(collection, LineCollection)
        ]
        self.assertEqual(len(segments_a), 1)
        self.assertEqual(len(segments_b), 1)
        self.assertEqual(len(segments_a[0]), 30)
        self.assertEqual(len(segments_b[0]), 30)
        for index, dataset in enumerate(sorted(self.datasets)):
            folds = self.folds[self.folds["dataset"] == dataset]
            paired = (folds["rmse_cliff"] - folds["rmse"]).tolist()
            paired_mean = statistics.mean(paired)
            paired_sd = statistics.stdev(paired)
            independent_sd = (
                statistics.variance(folds["rmse_cliff"])
                + statistics.variance(folds["rmse"])
            ) ** 0.5
            self.assertNotAlmostEqual(paired_sd, independent_sd)
            np.testing.assert_allclose(
                segments_a[0][index][:, 0],
                [paired_mean - paired_sd, paired_mean + paired_sd],
            )
            published_cliff = self.published_input.loc[
                (self.published_input["dataset"] == dataset)
                & (self.published_input["algorithm"] == "GCN"),
                "cliff_rmse",
            ].iloc[0]
            cliff_difference = statistics.mean(folds["rmse_cliff"]) - published_cliff
            cliff_sd = statistics.stdev(folds["rmse_cliff"])
            np.testing.assert_allclose(
                segments_b[0][index][:, 0],
                [cliff_difference - cliff_sd, cliff_difference + cliff_sd],
            )
            np.testing.assert_allclose(segments_b[0][index][:, 1], [index, index])

        with plt.rc_context(figures.STYLE):
            figures.f15_differences_by_dataset(writer, table, sorted(self.datasets))
        for entry in writer.figures:
            with self.subTest(figure=entry["name"]):
                figure = entry["figure"]
                legend = figure.legends[0]
                labels = [text.get_text() for text in legend.get_texts()]
                difference_handle = legend.legend_handles[
                    labels.index("Difference (line to zero)")
                ]
                self.assertEqual(difference_handle.get_marker(), "s")
                self.assertEqual(difference_handle.get_linestyle(), "-")
                self.assertGreater(difference_handle.get_linewidth(), 0)
                self.assertIn(
                    "connect each difference estimate to zero", entry["caption"]
                )
                self.assertIn("Light-grey capped bars", entry["caption"])
                self.assertIn("fold SD", entry["caption"])
                if entry["name"] == "published_reproduced_differences_by_dataset":
                    self.assertIn(
                        "larger reproduced errors in (a, b)", entry["caption"]
                    )
                    self.assertIn(
                        "larger cliff-versus-whole-test error gap in (c)",
                        entry["caption"],
                    )
                difference_axes = (
                    figure.axes[2:]
                    if entry["name"] == "dataset_error_comparison_table"
                    else figure.axes
                )
                for axis in difference_axes:
                    points = next(
                        collection.get_offsets()
                        for collection in axis.collections
                        if isinstance(collection, PathCollection)
                    )
                    stems = [
                        line
                        for line in axis.lines
                        if len(line.get_xdata()) == 2
                        and line.get_ydata()[0] == line.get_ydata()[1]
                        and line.get_linestyle() == "-"
                        and line.get_marker() == "None"
                    ]
                    self.assertEqual(len(stems), len(points))
                    for stem, (difference, row) in zip(stems, points, strict=True):
                        np.testing.assert_allclose(stem.get_xdata(), [0, difference])
                        np.testing.assert_allclose(stem.get_ydata(), [row, row])

    def test_fold_means_and_sample_sd_preserve_paired_difference(self) -> None:
        row = self.summary.loc[self.datasets[0]]
        rmse = [0.6 + fold * 0.02 for fold in range(5)]
        cliff = [0.7 + fold * 0.04 for fold in range(5)]
        paired_difference = [b - a for a, b in zip(rmse, cliff)]
        self.assertAlmostEqual(row["rmse_mean"], statistics.mean(rmse))
        self.assertAlmostEqual(row["rmse_sd"], statistics.stdev(rmse))
        self.assertAlmostEqual(row["rmse_cliff_mean"], statistics.mean(cliff))
        self.assertAlmostEqual(row["delta_mean"], statistics.mean(paired_difference))
        self.assertAlmostEqual(row["delta_sd"], statistics.stdev(paired_difference))
        independent_sd = (statistics.variance(rmse) + statistics.variance(cliff)) ** 0.5
        self.assertNotAlmostEqual(row["delta_sd"], independent_sd)

    def test_published_duplicate_strategy_is_rejected(self) -> None:
        duplicated = pd.concat([self.published_input, self.published_input.iloc[:1]])
        with patch.object(figures.pd, "read_csv", return_value=duplicated):
            with self.assertRaisesRegex(figures.FigureDataError, "duplicated"):
                figures.load_published(Path("synthetic.csv"))

    def test_fold_table_requires_five_distinct_folds_for_every_dataset(self) -> None:
        for altered in (self.folds.iloc[:-1], self.folds.assign(fold=0)):
            with self.subTest(rows=len(altered)):
                with patch.object(figures.pd, "read_csv", return_value=altered):
                    with self.assertRaisesRegex(figures.FigureDataError, "30 datasets"):
                        figures.load_per_fold(Path("synthetic.csv"), {})

    def test_fold_identities_reject_relabelled_unknown_and_duplicated_pairs(
        self,
    ) -> None:
        shifted = self.folds.assign(fold=self.folds["fold"] + 1)
        fractional = self.folds.assign(fold=self.folds["fold"] + 0.5)
        unknown = self.folds.replace({"dataset": {self.datasets[0]: "UNKNOWN_Ki"}})
        duplicated = pd.concat([self.folds.iloc[1:], self.folds.iloc[[1]]])
        for altered in (shifted, fractional, unknown, duplicated, self.folds.iloc[:-1]):
            with self.subTest(values=altered.head().to_dict()):
                with tempfile.TemporaryDirectory() as temporary:
                    source = Path(temporary) / "folds.csv"
                    altered.to_csv(source, index=False)
                    with self.assertRaisesRegex(
                        figures.FigureDataError, "exactly once"
                    ):
                        figures.load_per_fold(source, {})

    def test_fold_scores_reject_nonfinite_negative_and_nonnumeric_values(self) -> None:
        for column in ("rmse", "rmse_cliff"):
            for invalid in (np.nan, np.inf, -np.inf, -0.1, "invalid"):
                with self.subTest(column=column, invalid=invalid):
                    altered = self.folds.astype({column: object})
                    altered.loc[0, column] = invalid
                    with tempfile.TemporaryDirectory() as temporary:
                        source = Path(temporary) / "folds.csv"
                        altered.to_csv(source, index=False)
                        with self.assertRaises(figures.FigureDataError):
                            figures.load_per_fold(source, {})

    def test_comparison_statistics_reject_nonfinite_and_negative_errors(self) -> None:
        columns = list(self.summary.columns)
        nonnegative = set(columns) - {"delta_mean"}
        for column in columns:
            invalid_values = [np.nan, np.inf, -np.inf, "invalid"]
            if column in nonnegative:
                invalid_values.append(-0.1)
            for invalid in invalid_values:
                with self.subTest(column=column, invalid=invalid):
                    altered = self.summary.astype({column: object})
                    altered.loc[self.datasets[0], column] = invalid
                    with tempfile.TemporaryDirectory() as temporary:
                        source = Path(temporary) / "comparison.csv"
                        altered.rename_axis("dataset").to_csv(source)
                        with self.assertRaises(figures.FigureDataError):
                            figures.load_comparison(source)
        negative_gap = self.summary.assign(delta_mean=-0.1)
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "comparison.csv"
            negative_gap.rename_axis("dataset").to_csv(source)
            observed = figures.load_comparison(source)
        np.testing.assert_allclose(observed["delta_mean"], -0.1)

    def test_published_errors_reject_nonfinite_negative_and_nonnumeric_values(
        self,
    ) -> None:
        for column in ("rmse", "cliff_rmse"):
            for invalid in (np.nan, np.inf, -np.inf, -0.1, "invalid"):
                with self.subTest(column=column, invalid=invalid):
                    altered = self.published_input.astype({column: object})
                    altered.loc[0, column] = invalid
                    with tempfile.TemporaryDirectory() as temporary:
                        source = Path(temporary) / "published.csv"
                        altered.to_csv(source, index=False)
                        with self.assertRaises(figures.FigureDataError):
                            figures.load_published(source)

    def test_benchmark_counts_reject_invalid_splits_and_nonbinary_labels(self) -> None:
        for column, invalid in (
            ("split", "validation"),
            ("split", None),
            ("cliff_mol", -1),
            ("cliff_mol", 2),
            ("cliff_mol", np.nan),
            ("cliff_mol", np.inf),
        ):
            benchmark = pd.DataFrame({"split": ["train", "test"], "cliff_mol": [0, 1]})
            benchmark = benchmark.astype({column: object})
            benchmark.loc[0, column] = invalid
            with self.subTest(column=column, invalid=invalid):
                with (
                    patch.object(Path, "is_file", return_value=True),
                    patch.object(figures.pd, "read_csv", return_value=benchmark),
                    self.assertRaisesRegex(
                        figures.FigureDataError, "binary cliff label"
                    ),
                ):
                    figures.load_benchmark_counts(Path("synthetic"))

    def test_fold_crosscheck_rejects_changed_gap_mean_and_paired_sd(self) -> None:
        for column in ("delta_mean", "delta_sd"):
            comparison = self.summary.copy()
            comparison.loc[self.datasets[0], column] += 0.01
            with (
                self.subTest(column=column),
                tempfile.TemporaryDirectory() as temporary,
            ):
                folder = Path(temporary)
                comparison.rename_axis("dataset").to_csv(folder / "comparison.csv")
                self.folds.to_csv(folder / "folds.csv", index=False)
                arguments = argparse.Namespace(
                    published=Path("published.csv"),
                    benchmark_dir=Path("synthetic"),
                    comparison=folder / "comparison.csv",
                    per_fold=folder / "folds.csv",
                    demo=False,
                    fold_columns="",
                )
                with (
                    patch.object(
                        figures, "load_published", return_value=self.published
                    ),
                    patch.object(
                        figures, "load_benchmark_counts", return_value=self.counts
                    ),
                    self.assertRaisesRegex(
                        figures.FigureDataError, "Per-fold " + column
                    ),
                ):
                    figures.load_figure_inputs(arguments, folder / "new_figures")
                self.assertFalse((folder / "new_figures").exists())

    def test_fold_crosscheck_detects_changed_pairing_with_equal_marginal_statistics(
        self,
    ) -> None:
        altered = self.folds.copy()
        selected = altered["dataset"] == self.datasets[0]
        altered.loc[selected, "rmse_cliff"] = altered.loc[
            selected, "rmse_cliff"
        ].to_numpy()[::-1]
        original = self.folds.loc[selected]
        changed = altered.loc[selected]
        for column in ("rmse", "rmse_cliff"):
            self.assertAlmostEqual(
                statistics.mean(original[column]), statistics.mean(changed[column])
            )
            self.assertAlmostEqual(
                statistics.stdev(original[column]), statistics.stdev(changed[column])
            )
        self.assertNotAlmostEqual(
            statistics.stdev(original["rmse_cliff"] - original["rmse"]),
            statistics.stdev(changed["rmse_cliff"] - changed["rmse"]),
        )
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            self.summary.rename_axis("dataset").to_csv(folder / "comparison.csv")
            altered.to_csv(folder / "folds.csv", index=False)
            arguments = argparse.Namespace(
                published=Path("published.csv"),
                benchmark_dir=Path("synthetic"),
                comparison=folder / "comparison.csv",
                per_fold=folder / "folds.csv",
                demo=False,
                fold_columns="",
            )
            with (
                patch.object(figures, "load_published", return_value=self.published),
                patch.object(
                    figures, "load_benchmark_counts", return_value=self.counts
                ),
                self.assertRaisesRegex(figures.FigureDataError, "Per-fold delta_sd"),
            ):
                figures.load_figure_inputs(arguments, folder / "new_figures")
            self.assertFalse((folder / "new_figures").exists())

    def test_comparison_requires_every_dataset_once(self) -> None:
        comparison = self.summary.rename_axis("dataset").reset_index()
        duplicated = pd.concat([comparison, comparison.iloc[:1]])
        for altered in (comparison.iloc[:-1], duplicated):
            with self.subTest(rows=len(altered)):
                with patch.object(figures.pd, "read_csv", return_value=altered):
                    with self.assertRaisesRegex(figures.FigureDataError, "30 datasets"):
                        figures.load_comparison(Path("synthetic.csv"))

    def test_assembled_differences_are_reproduced_minus_published(self) -> None:
        table = figures.build_table(self.published, self.counts, self.summary)
        self.assertEqual(set(table.index), set(self.datasets))
        for dataset in self.datasets:
            row = table.loc[dataset]
            self.assertAlmostEqual(row["diff_rmse"], row["rep_rmse"] - row["pub_rmse"])
            self.assertAlmostEqual(
                row["diff_cliff"], row["rep_cliff"] - row["pub_cliff"]
            )
            self.assertAlmostEqual(row["rep_delta"], row["rep_cliff"] - row["rep_rmse"])
            self.assertAlmostEqual(
                row["rep_delta_sd"], self.summary.loc[dataset, "delta_sd"]
            )

    def test_inconsistent_cliff_minus_all_difference_is_rejected(self) -> None:
        altered = self.summary.copy()
        altered.loc[self.datasets[0], "delta_mean"] *= -1
        with self.assertRaisesRegex(figures.FigureDataError, "Reproduced delta"):
            figures.build_table(self.published, self.counts, altered)

    def test_changed_benchmark_counts_are_rejected(self) -> None:
        comparison = self.summary.assign(n_train=501)
        with self.assertRaisesRegex(figures.FigureDataError, "n_train"):
            figures.build_table(self.published, self.counts, self.summary, comparison)

    def test_dataset_order_does_not_depend_on_reproduced_performance(self) -> None:
        table = figures.build_table(self.published, self.counts, self.summary)
        altered = table.copy()
        altered["rep_cliff"] = -altered["rep_cliff"]
        for rule in ("paper", "n_train", "alphabetical"):
            self.assertEqual(
                figures.dataset_order(table, rule), figures.dataset_order(altered, rule)
            )


class FigureExportTests(unittest.TestCase):
    """Protect canonical filenames and read-only handling of existing outputs."""

    def tearDown(self) -> None:
        plt.close("all")

    def test_all_approved_figure_names_are_unique_and_descriptive(self) -> None:
        tree = ast.parse(Path(figures.__file__).read_text(encoding="utf-8"))
        actual = []
        for function in tree.body:
            if isinstance(function, ast.FunctionDef) and function.name.startswith("f"):
                for statement in function.body:
                    if isinstance(statement, ast.Assign) and any(
                        isinstance(target, ast.Name) and target.id == "name"
                        for target in statement.targets
                    ):
                        actual.append(ast.literal_eval(statement.value))
        self.assertEqual(tuple(actual), EXPECTED_FIGURE_NAMES)
        self.assertEqual(len(set(actual)), len(EXPECTED_FIGURE_NAMES))
        for name in actual:
            self.assertRegex(name, r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")

    def test_existing_output_directory_is_rejected_before_reading_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            sentinel = folder / "preserved.txt"
            sentinel.write_bytes(b"original output")
            with patch.object(figures, "load_published") as load:
                with self.assertRaisesRegex(
                    figures.FigureDataError, "Output directory exists"
                ):
                    figures.load_figure_inputs(argparse.Namespace(), folder)
            load.assert_not_called()
            self.assertEqual(sentinel.read_bytes(), b"original output")
            self.assertEqual(list(folder.iterdir()), [sentinel])

    def test_main_passes_output_directory_directly_without_a_subfolder(self) -> None:
        parser = figures.build_argument_parser()
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "figures"
            arguments = parser.parse_args(
                [
                    "--published",
                    "published.csv",
                    "--benchmark-dir",
                    "datasets",
                    "--comparison",
                    "comparison.csv",
                    "--output-dir",
                    str(destination),
                ]
            )
            with patch.object(figures, "build_argument_parser", return_value=parser):
                with patch.object(parser, "parse_args", return_value=arguments):
                    with patch.object(
                        figures,
                        "load_figure_inputs",
                        side_effect=RuntimeError("stop before export"),
                    ) as load:
                        with self.assertRaisesRegex(RuntimeError, "stop before export"):
                            figures.main([])
            self.assertEqual(load.call_args.args[1], destination)
            self.assertFalse(destination.exists())

    def test_writer_and_caption_use_the_descriptive_name(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            writer = figures.FigureWriter(folder, False, ["png"])
            figure, axis = plt.subplots(figsize=(1, 1), dpi=20)
            axis.plot([0, 1], [0, 1])
            writer.save(figure, "paired_cliff_rmse", "Cliff RMSE", "Synthetic caption.")
            path = folder / "paired_cliff_rmse.png"
            self.assertTrue(path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
            self.assertEqual(writer.outputs[0]["file"], path.name)
            self.assertEqual(
                writer.outputs[0]["sha256"],
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )
            figures.write_captions(folder / "figure_captions.md", writer.figures, False)
            text = (folder / "figure_captions.md").read_text(encoding="utf-8")
            self.assertIn("## paired_cliff_rmse — Cliff RMSE", text)
            self.assertIn("Synthetic caption.", text)

    def _manifest_fixture(
        self, folder: Path
    ) -> tuple[argparse.Namespace, figures.FigureWriter]:
        """Create actual payload bytes for independent manifest-hash assertions."""
        published = folder / "published.csv"
        published.write_text("synthetic input\n", encoding="utf-8")
        benchmark = folder / "benchmark"
        benchmark.mkdir()
        for dataset in figures.ABBREVIATIONS:
            (benchmark / f"{dataset}.csv").write_text(dataset + "\n", encoding="utf-8")
        for name in (
            "figure_data.csv",
            "figure_captions.md",
            "figure_catalogue.pdf",
            "paired_cliff_rmse.png",
        ):
            (folder / name).write_bytes(("synthetic " + name).encode())
        arguments = argparse.Namespace(
            published=published,
            benchmark_dir=benchmark,
            comparison=None,
            per_fold=None,
            demo=False,
            order="paper",
        )
        writer = figures.FigureWriter(folder, False, ["png"])
        writer.outputs = [
            {
                "figure": "paired_cliff_rmse",
                "file": "paired_cliff_rmse.png",
                "sha256": hashlib.sha256(
                    (folder / "paired_cliff_rmse.png").read_bytes()
                ).hexdigest(),
            }
        ]
        return arguments, writer

    def test_manifest_records_all_benchmark_and_product_hashes_without_self_reference(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            arguments, writer = self._manifest_fixture(folder)
            figures.write_manifest(arguments, folder, ["CHEMBL214_Ki"], writer)
            manifest = json.loads(
                (folder / "figure_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["schema_version"], 2)
            self.assertEqual(manifest["outputs"], writer.outputs)
            self.assertEqual(
                manifest["inputs"]["published"]["sha256"],
                hashlib.sha256(arguments.published.read_bytes()).hexdigest(),
            )
            benchmark = manifest["inputs"]["benchmark_dir"]["files"]
            self.assertEqual(
                [row["dataset"] for row in benchmark], sorted(figures.ABBREVIATIONS)
            )
            self.assertEqual(len(benchmark), 30)
            for row in benchmark:
                self.assertEqual(row["file"], row["dataset"] + ".csv")
                self.assertEqual(
                    row["sha256"],
                    hashlib.sha256(
                        (arguments.benchmark_dir / row["file"]).read_bytes()
                    ).hexdigest(),
                )
            products = manifest["outputs"] + manifest["supporting_outputs"]
            self.assertEqual(
                {row["file"] for row in products},
                {
                    "paired_cliff_rmse.png",
                    "figure_data.csv",
                    "figure_captions.md",
                    "figure_catalogue.pdf",
                },
            )
            for row in products:
                self.assertEqual(
                    row["sha256"],
                    hashlib.sha256((folder / row["file"]).read_bytes()).hexdigest(),
                )
            self.assertNotIn("figure_manifest.json", {row["file"] for row in products})
            self.assertIn("excludes itself", manifest["hash_policy"])
            self.assertFalse((folder / "export_manifest.json").exists())

    def test_manifest_rejects_a_missing_benchmark_or_supporting_product(self) -> None:
        for missing in (
            "benchmark/" + next(iter(figures.ABBREVIATIONS)) + ".csv",
            "figure_data.csv",
            "figure_captions.md",
            "figure_catalogue.pdf",
        ):
            with (
                self.subTest(missing=missing),
                tempfile.TemporaryDirectory() as temporary,
            ):
                folder = Path(temporary)
                arguments, writer = self._manifest_fixture(folder)
                (folder / missing).unlink()
                with self.assertRaises(FileNotFoundError):
                    figures.write_manifest(arguments, folder, [], writer)
                self.assertFalse((folder / "figure_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()

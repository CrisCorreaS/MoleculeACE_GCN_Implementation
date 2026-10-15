"""Paper-style figures comparing the reproduced GCN_GRAPH with the corrected paper.

The figures adapt corrected Figures 4 and 5 of van Tilborg, Alenicheva and Grisoni
(correction DOI 10.1021/acs.jcim.3c01576). Only GCN_GRAPH was reproduced; every other
strategy is external published data, drawn in grey as context.

Each figure is drawn by a single function that holds everything about it: file name,
title, caption, size, panel titles, axis labels and drawing code. Colours, markers and
fonts are shared by every figure and defined once at the top of this file.

Inputs
    --published      corrected MoleculeACE_results.csv
    --benchmark-dir  corrected benchmark CSV folder
    --comparison     _analysis/comparison.csv of the campaign, or --demo
    --per-fold       _summary/per_fold.csv (optional; enables dataset_errors_by_fold)

Outputs
    <output-dir>/ with every figure as PDF, SVG and PNG, plus
    figure_captions.md, figure_catalogue.pdf, figure_data.csv and
    figure_manifest.json. The output directory must not exist yet.

Usage, from code/ (a single line in PowerShell)
    python plot_paper_figures.py
        --published ../references/published/MoleculeACE_results.csv
        --benchmark-dir MoleculeACE/Data/benchmark_data
        --comparison ../results/gcn_full_v1/_analysis/comparison.csv
        --per-fold ../results/gcn_full_v1/_summary/per_fold.csv
        --output-dir ../figures

--demo replaces the reproduction with seeded synthetic values for layout review.
Demo figures carry a watermark and must never be used as results.
"""

import argparse
import datetime
import hashlib
import json
import platform
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MultipleLocator
from matplotlib.text import Text
from matplotlib.transforms import Bbox

matplotlib.use("Agg")

PUBLISHED = {
    "colour": "#2F3C8C",
    "fill": "#B9BFE3",
    "edge": "#1A214D",
    "marker": "o",
    "size": 26,
    "label": "Published GCN (corrected)",
    "short": "Published",
}
REPRODUCED = {
    "colour": "#BC2351",
    "fill": "#ECC1CE",
    "edge": "#6D142F",
    "marker": "D",
    "size": 20,
    "label": "Reproduced GCN (this work)",
    "short": "Reproduction",
}
OTHER_STRATEGIES = {
    "colour": "#BDBDBD",
    "fill": "#EBEBEB",
    "edge": "#7F7F7F",
    "median": "#555555",
    "marker": "o",
    "label": "Other published strategies",
}
DIFFERENCE = {
    "colour": "#3F3F3F",
    "edge": "#1A1A1A",
    "marker": "s",
    "label": "Reproduced − published",
}
MAX_FIGURE_WIDTH_CM = 16.0
MAX_FIGURE_HEIGHT_CM = 22.0

INK = "#1A1A1A"
GUIDE_COLOUR = "#9E9E9E"
BAND_COLOUR = "#F4F4F4"

PUBLISHED_MARKER = {
    "marker": PUBLISHED["marker"],
    "facecolor": PUBLISHED["colour"],
    "edgecolor": PUBLISHED["edge"],
    "linewidth": 0.45,
}
REPRODUCED_MARKER = {
    "marker": REPRODUCED["marker"],
    "facecolor": REPRODUCED["colour"],
    "edgecolor": REPRODUCED["edge"],
    "linewidth": 0.45,
}
DIFFERENCE_MARKER = {
    "marker": DIFFERENCE["marker"],
    "facecolor": DIFFERENCE["colour"],
    "edgecolor": DIFFERENCE["edge"],
    "linewidth": 0.45,
}
OTHER_STRATEGIES_CLOUD = {
    "marker": "o",
    "facecolor": OTHER_STRATEGIES["colour"],
    "edgecolor": "none",
    "alpha": 0.55,
}
PUBLISHED_BOX = {
    "patch_artist": True,
    "showfliers": False,
    "manage_ticks": False,
    "medianprops": {"color": PUBLISHED["edge"], "lw": 1.1},
    "whiskerprops": {"color": PUBLISHED["edge"], "lw": 0.75},
    "capprops": {"color": PUBLISHED["edge"], "lw": 0.75},
    "boxprops": {
        "edgecolor": PUBLISHED["edge"],
        "facecolor": PUBLISHED["fill"],
        "lw": 0.8,
    },
}
REPRODUCED_BOX = {
    "patch_artist": True,
    "showfliers": False,
    "manage_ticks": False,
    "medianprops": {"color": REPRODUCED["edge"], "lw": 1.1},
    "whiskerprops": {"color": REPRODUCED["edge"], "lw": 0.75},
    "capprops": {"color": REPRODUCED["edge"], "lw": 0.75},
    "boxprops": {
        "edgecolor": REPRODUCED["edge"],
        "facecolor": REPRODUCED["fill"],
        "lw": 0.8,
    },
}
OTHER_STRATEGIES_BOX = {
    "patch_artist": True,
    "showfliers": False,
    "manage_ticks": False,
    "medianprops": {"color": OTHER_STRATEGIES["median"], "lw": 1.1},
    "whiskerprops": {"color": OTHER_STRATEGIES["edge"], "lw": 0.75},
    "capprops": {"color": OTHER_STRATEGIES["edge"], "lw": 0.75},
    "boxprops": {
        "edgecolor": OTHER_STRATEGIES["edge"],
        "facecolor": OTHER_STRATEGIES["fill"],
        "lw": 0.8,
    },
}
PUBLISHED_LEGEND_MARKER = {
    "ls": "none",
    "marker": PUBLISHED["marker"],
    "markersize": 5.2,
    "markerfacecolor": PUBLISHED["colour"],
    "markeredgecolor": PUBLISHED["edge"],
    "markeredgewidth": 0.45,
    "label": PUBLISHED["label"],
}
REPRODUCED_LEGEND_MARKER = {
    "ls": "none",
    "marker": REPRODUCED["marker"],
    "markersize": 5.2,
    "markerfacecolor": REPRODUCED["colour"],
    "markeredgecolor": REPRODUCED["edge"],
    "markeredgewidth": 0.45,
    "label": REPRODUCED["label"],
}
OTHER_STRATEGIES_LEGEND_MARKER = {
    "ls": "none",
    "marker": "o",
    "markersize": 5,
    "markerfacecolor": OTHER_STRATEGIES["colour"],
    "markeredgecolor": "none",
    "label": OTHER_STRATEGIES["label"],
}
DIFFERENCE_LEGEND_MARKER = {
    "ls": "none",
    "marker": DIFFERENCE["marker"],
    "markersize": 4.6,
    "markerfacecolor": DIFFERENCE["colour"],
    "markeredgecolor": DIFFERENCE["edge"],
    "markeredgewidth": 0.45,
    "label": DIFFERENCE["label"],
}
IDENTITY_LINE = {"color": INK, "lw": 0.8, "ls": (0, (4, 3)), "zorder": 1}
HALF_LOG_UNIT_LINE = {"color": "#C4C4C4", "lw": 0.8, "ls": (0, (4, 3)), "zorder": 1}
PANEL_LETTER = {"fontsize": 10.5, "fontweight": "bold", "ha": "left", "va": "bottom"}

SOURCE_DOI = "10.1021/acs.jcim.3c01576"

STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": 7,
    "axes.labelsize": 7.5,
    "axes.titlesize": 7.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "mathtext.default": "regular",
    "mathtext.fontset": "custom",
    "mathtext.rm": "sans",
    "mathtext.it": "sans:italic",
    "mathtext.bf": "sans:bold",
    "mathtext.cal": "sans",
    "axes.linewidth": 0.75,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "text.color": INK,
    "xtick.major.width": 0.75,
    "ytick.major.width": 0.75,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "figure.dpi": 150,
    "figure.max_open_warning": 0,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "svg.hashsalt": "gcn-paper-style",
    "axes.unicode_minus": True,
}

ABBREVIATIONS = {
    "CHEMBL1862_Ki": "ABL1",
    "CHEMBL1871_Ki": "AR",
    "CHEMBL2034_Ki": "GR",
    "CHEMBL2047_EC50": "FXR",
    "CHEMBL204_Ki": "Thrombin",
    "CHEMBL2147_Ki": "PIM1",
    "CHEMBL214_Ki": "5-HT1A",
    "CHEMBL218_EC50": "CB1",
    "CHEMBL219_Ki": "D4R",
    "CHEMBL228_Ki": "SERT",
    "CHEMBL231_Ki": "HRH1",
    "CHEMBL233_Ki": "MOR",
    "CHEMBL234_Ki": "D3R",
    "CHEMBL235_EC50": "PPARγ",
    "CHEMBL236_Ki": "DOR",
    "CHEMBL237_EC50": "KOR (a)",
    "CHEMBL237_Ki": "KOR (i)",
    "CHEMBL238_Ki": "DAT",
    "CHEMBL239_EC50": "PPARα",
    "CHEMBL244_Ki": "FX",
    "CHEMBL262_Ki": "GSK3",
    "CHEMBL264_Ki": "HRH3",
    "CHEMBL2835_Ki": "JAK1",
    "CHEMBL287_Ki": "SOR",
    "CHEMBL2971_Ki": "JAK2",
    "CHEMBL3979_EC50": "PPARδ",
    "CHEMBL4005_Ki": "PIK3CA",
    "CHEMBL4203_Ki": "CLK4",
    "CHEMBL4616_EC50": "GHSR",
    "CHEMBL4792_Ki": "OX2R",
}
DESCRIPTOR_NAMES = {
    "GRAPH": "graph",
    "SMILES": "SMILES",
    "TOKENS": "SMILES",
    "ECFP": "ECFP",
    "MACCS": "MACCS",
    "PHYSCHEM": "PhysChem",
    "WHIM": "WHIM",
}
PER_FOLD_COLUMN_CANDIDATES = {
    "dataset": ("dataset",),
    "fold": ("fold", "fold_index", "k"),
    "rmse": ("rmse", "rmse_official", "official_rmse", "test_rmse"),
    "rmse_cliff": (
        "rmse_cliff",
        "cliff_rmse",
        "rmse_cliff_official",
        "official_rmse_cliff",
        "test_rmse_cliff",
    ),
}


class FigureDataError(RuntimeError):
    """Inputs are missing, inconsistent or unsuitable for plotting."""


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_numeric_columns(
    table: pd.DataFrame,
    columns: list[str],
    nonnegative: list[str],
    description: str,
) -> None:
    """Reject missing/non-finite values and negative errors or sample SDs."""
    try:
        values = table[columns].to_numpy(dtype=float)
        unsigned_values = table[nonnegative].to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise FigureDataError(f"{description} contains non-numeric values") from error
    if not np.isfinite(values).all():
        raise FigureDataError(f"{description} contains missing or non-finite values")
    if (unsigned_values < 0).any():
        raise FigureDataError(f"{description} contains negative errors, SDs or counts")


def load_published(path: Path) -> pd.DataFrame:
    """Load the corrected published results, one row per dataset and strategy."""
    table = pd.read_csv(path)
    required = {"dataset", "algorithm", "descriptor", "rmse", "cliff_rmse"}
    if not required <= set(table.columns):
        raise FigureDataError(
            f"Published table lacks columns: {required - set(table.columns)}"
        )
    if table.duplicated(["dataset", "algorithm", "descriptor"]).any():
        raise FigureDataError("Published table has duplicated strategy rows")
    unknown = set(table["dataset"]) - set(ABBREVIATIONS)
    if unknown or table["dataset"].nunique() != 30:
        raise FigureDataError(f"Unexpected published datasets: {sorted(unknown)}")
    _validate_numeric_columns(
        table, ["rmse", "cliff_rmse"], ["rmse", "cliff_rmse"], "Published table"
    )
    table = table.copy()
    table["delta"] = table["cliff_rmse"] - table["rmse"]
    table["strategy"] = (
        table["algorithm"] + " (" + table["descriptor"].map(DESCRIPTOR_NAMES) + ")"
    )
    table["is_gcn"] = (table["algorithm"] == "GCN") & (table["descriptor"] == "GRAPH")
    if table["is_gcn"].sum() != 30:
        raise FigureDataError("Expected one published GCN_GRAPH row per dataset")
    return table


def load_benchmark_counts(folder: Path) -> pd.DataFrame:
    """Count training, test and cliff-test molecules in each benchmark CSV."""
    rows = []
    for dataset in ABBREVIATIONS:
        path = Path(folder) / (dataset + ".csv")
        if not path.is_file():
            raise FigureDataError(f"Missing benchmark dataset: {path}")
        data = pd.read_csv(path, usecols=["split", "cliff_mol"])
        if (
            set(data["split"]) != {"train", "test"}
            or not data["cliff_mol"].isin([0, 1]).all()
        ):
            raise FigureDataError(
                f"Invalid published split or binary cliff label in {path.name}"
            )
        is_test = data["split"] == "test"
        rows.append(
            {
                "dataset": dataset,
                "n_train": int((data["split"] == "train").sum()),
                "n_test": int(is_test.sum()),
                "n_test_cliff": int(data.loc[is_test, "cliff_mol"].sum()),
            }
        )
    return pd.DataFrame(rows).set_index("dataset")


def find_per_fold_column(columns: pd.Index, key: str, override: str | None) -> str:
    """Return the per-fold column for a key, from an override or the known names."""
    if override:
        if override not in columns:
            raise FigureDataError(f"Column {override!r} not in per-fold table")
        return override
    for name in PER_FOLD_COLUMN_CANDIDATES[key]:
        if name in columns:
            return name
    raise FigureDataError(
        f"Cannot find a {key!r} column in per-fold table; columns are {list(columns)}. "
        "Use --fold-columns."
    )


def load_per_fold(path: Path, overrides: dict[str, str]) -> pd.DataFrame:
    """Load the official fold-level RMSE and RMSE_cliff of every dataset."""
    table = pd.read_csv(path)
    names = {
        key: find_per_fold_column(table.columns, key, overrides.get(key))
        for key in PER_FOLD_COLUMN_CANDIDATES
    }
    folds = table[[names[key] for key in PER_FOLD_COLUMN_CANDIDATES]].copy()
    folds.columns = list(PER_FOLD_COLUMN_CANDIDATES)
    _validate_numeric_columns(
        folds, ["fold", "rmse", "rmse_cliff"], ["rmse", "rmse_cliff"], "Per-fold table"
    )
    fold_counts = folds.groupby("dataset")["fold"].nunique()
    if (
        set(folds["dataset"]) != set(ABBREVIATIONS)
        or len(folds) != 150
        or folds.duplicated(["dataset", "fold"]).any()
        or not folds["fold"].isin(range(5)).all()
        or (fold_counts != 5).any()
    ):
        raise FigureDataError(
            "Per-fold table must contain the expected 30 datasets x 5 folds "
            "(0 to 4), with each dataset-fold pair exactly once"
        )
    folds["delta"] = folds["rmse_cliff"] - folds["rmse"]
    return folds


def load_comparison(path: Path) -> pd.DataFrame:
    """Load the dataset-level reproduction means and sample SDs."""
    table = pd.read_csv(path)
    required = [
        "dataset",
        "rmse_mean",
        "rmse_sd",
        "rmse_cliff_mean",
        "rmse_cliff_sd",
        "delta_mean",
        "delta_sd",
    ]
    missing = [column for column in required if column not in table.columns]
    if missing:
        raise FigureDataError(f"Comparison table lacks columns: {missing}")
    if table["dataset"].duplicated().any() or set(table["dataset"]) != set(
        ABBREVIATIONS
    ):
        raise FigureDataError("Comparison table must hold each of the 30 datasets once")
    error_columns = [
        "rmse_mean",
        "rmse_sd",
        "rmse_cliff_mean",
        "rmse_cliff_sd",
        "delta_sd",
    ]
    published_columns = [
        column
        for column in ("published_rmse", "published_rmse_cliff")
        if column in table.columns
    ]
    _validate_numeric_columns(
        table,
        required[1:] + published_columns,
        error_columns + published_columns,
        "Comparison table",
    )
    return table.set_index("dataset")


def synthetic_reproduction(
    published: pd.DataFrame, seed: int = 20261002
) -> pd.DataFrame:
    """Make seeded stand-in fold scores for layout review; never a result."""
    generator = np.random.default_rng(seed)
    gcn = published[published["is_gcn"]].set_index("dataset").loc[list(ABBREVIATIONS)]
    rows = []
    for dataset, row in gcn.iterrows():
        shift_rmse, shift_cliff = (
            generator.normal(0.012, 0.035),
            generator.normal(0.018, 0.05),
        )
        for fold in range(5):
            noise = generator.normal(0, 0.035)
            rows.append(
                {
                    "dataset": dataset,
                    "fold": fold,
                    "rmse": row["rmse"] + shift_rmse + noise,
                    "rmse_cliff": row["cliff_rmse"]
                    + shift_cliff
                    + noise * 1.2
                    + generator.normal(0, 0.03),
                }
            )
    folds = pd.DataFrame(rows)
    folds["delta"] = folds["rmse_cliff"] - folds["rmse"]
    return folds


def summarise_folds(folds: pd.DataFrame) -> pd.DataFrame:
    """Average the five folds of each dataset and give their sample SD."""
    grouped = folds.groupby("dataset")
    return pd.DataFrame(
        {
            "rmse_mean": grouped["rmse"].mean(),
            "rmse_sd": grouped["rmse"].std(ddof=1),
            "rmse_cliff_mean": grouped["rmse_cliff"].mean(),
            "rmse_cliff_sd": grouped["rmse_cliff"].std(ddof=1),
            "delta_mean": grouped["delta"].mean(),
            "delta_sd": grouped["delta"].std(ddof=1),
        }
    )


def build_table(
    published: pd.DataFrame,
    counts: pd.DataFrame,
    reproduction: pd.DataFrame,
    comparison: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Join published GCN, reproduced GCN and context statistics per dataset."""
    gcn = published[published["is_gcn"]].set_index("dataset")
    by_dataset = published.groupby("dataset")
    pearson = by_dataset.apply(
        lambda rows: float(np.corrcoef(rows["rmse"], rows["cliff_rmse"])[0, 1])
    )
    table = pd.DataFrame(index=list(ABBREVIATIONS))
    table.index.name = "dataset"
    table["abbreviation"] = [ABBREVIATIONS[dataset] for dataset in table.index]
    table["type"] = [dataset.rsplit("_", 1)[1] for dataset in table.index]
    table = table.join(counts)
    table["pub_rmse"] = gcn["rmse"]
    table["pub_cliff"] = gcn["cliff_rmse"]
    table["pub_delta"] = gcn["delta"]
    reproduced = reproduction.loc[table.index]
    for source, target in (
        ("rmse_mean", "rep_rmse"),
        ("rmse_sd", "rep_rmse_sd"),
        ("rmse_cliff_mean", "rep_cliff"),
        ("rmse_cliff_sd", "rep_cliff_sd"),
        ("delta_mean", "rep_delta"),
        ("delta_sd", "rep_delta_sd"),
    ):
        table[target] = reproduced[source].astype(float)
    table["r_published_methods"] = pearson
    if comparison is not None:
        for column, target in (
            ("published_rmse", "pub_rmse"),
            ("published_rmse_cliff", "pub_cliff"),
        ):
            if column in comparison.columns:
                gap = (comparison.loc[table.index, column] - table[target]).abs().max()
                if gap > 1e-6:
                    raise FigureDataError(
                        f"{column} in comparison.csv differs from the published "
                        f"table by up to {gap:.2e}"
                    )
        if (
            "n_train" in comparison.columns
            and (comparison.loc[table.index, "n_train"] != table["n_train"]).any()
        ):
            raise FigureDataError(
                "n_train in comparison.csv differs from benchmark counts"
            )
    if not np.allclose(
        table["rep_delta"],
        table["rep_cliff"] - table["rep_rmse"],
        atol=1e-9,
        rtol=0,
    ):
        raise FigureDataError("Reproduced delta is not RMSE_cliff − RMSE")
    if table.isna().any().any():
        raise FigureDataError("Missing values in assembled figure table")
    _validate_numeric_columns(
        table,
        list(table.select_dtypes(include=np.number).columns),
        [
            "n_train",
            "n_test",
            "n_test_cliff",
            "pub_rmse",
            "pub_cliff",
            "rep_rmse",
            "rep_cliff",
            "rep_rmse_sd",
            "rep_cliff_sd",
            "rep_delta_sd",
        ],
        "Assembled figure table",
    )
    table["diff_rmse"] = table["rep_rmse"] - table["pub_rmse"]
    table["diff_cliff"] = table["rep_cliff"] - table["pub_cliff"]
    table["diff_delta"] = table["rep_delta"] - table["pub_delta"]
    return table


def dataset_order(table: pd.DataFrame, rule: str) -> list[str]:
    """Order the dataset rows by published values or inputs, never by reproduction."""
    if rule == "paper":
        return list(table.sort_values(["r_published_methods", "abbreviation"]).index)
    if rule == "n_train":
        return list(table.sort_values(["n_train", "abbreviation"]).index)
    if rule == "alphabetical":
        return list(
            table.sort_values("abbreviation", key=lambda s: s.str.lower()).index
        )
    raise ValueError(rule)


class FigureWriter:
    """Save each figure in every format and keep it for the captions and catalogue."""

    def __init__(self, output_dir: Path, is_demo: bool, formats: list[str]) -> None:
        self.output_dir = Path(output_dir)
        self.is_demo = is_demo
        self.formats = formats
        self.outputs = []
        self.figures = []

    def save(self, figure: Figure, name: str, title: str, caption: str) -> None:
        """Watermark demo figures, write every format and record the figure."""
        if self.is_demo:
            figure.text(
                0.5,
                0.5,
                "PREVIEW · SYNTHETIC REPRODUCTION VALUES",
                fontsize=17,
                color="#C0392B",
                alpha=0.16,
                ha="center",
                va="center",
                rotation=24,
                fontweight="bold",
                zorder=100,
            )
            figure.canvas.draw()
            renderer = figure.canvas.get_renderer()
            highest_content = max(
                panel.get_tightbbox(renderer)
                .transformed(figure.transFigure.inverted())
                .y1
                for panel in figure.axes
                if panel.get_visible()
            )
            figure.text(
                0.5,
                highest_content + 0.01,
                "Layout preview only: reproduction values are seeded synthetic data, "
                "not results.",
                fontsize=7,
                color="#C0392B",
                ha="center",
                va="bottom",
            )
        for suffix in self.formats:
            path = self.output_dir / f"{name}.{suffix}"
            figure.savefig(
                path,
                metadata={"Creator": Path(__file__).name} if suffix == "pdf" else None,
            )
            self.outputs.append(
                {"figure": name, "file": path.name, "sha256": sha256(path)}
            )
        self.figures.append(
            {"name": name, "title": title, "caption": caption, "figure": figure}
        )


def f01_fig4a_pair_cliff(writer: FigureWriter, table: pd.DataFrame) -> None:
    """F01 (cliff): published and reproduced GCN RMSE_cliff as paired boxes."""
    name = "paired_cliff_rmse"
    title = "Fig. 4a adaptation, paired GCN boxes (cliff)"
    caption = (
        "GCN-only adaptation of corrected Fig. 4a. RMSE_cliff of the published and the "
        "reproduced GCN across datasets. Boxes show the median and interquartile "
        "range; whiskers extend to 1.5 × IQR; thin grey lines join the two values of "
        "the same dataset."
    )
    figure_size = (2.5, 2.9)
    x_tick_labels = ["Published\n(corrected)", "Reproduction\n(this work)"]
    x_label = "GCN (graph)"
    y_label = r"RMSE$_{\mathrm{cliff}}$"

    published_values = table["pub_cliff"]
    reproduced_values = table["rep_cliff"]
    jitter = np.random.default_rng(4).uniform(-0.11, 0.11, len(table))
    all_values = np.concatenate([published_values, reproduced_values])
    y_lower = np.floor(all_values.min() / 0.25) * 0.25
    y_upper = np.ceil(all_values.max() / 0.25) * 0.25

    figure, axis = plt.subplots(figsize=figure_size)
    for index in range(len(table)):
        axis.plot(
            [jitter[index], 1 + jitter[index]],
            [published_values.iloc[index], reproduced_values.iloc[index]],
            color="#B5B5B5",
            lw=0.5,
            zorder=2,
        )
    axis.boxplot(
        [published_values],
        positions=[0],
        widths=0.5,
        whis=1.5,
        zorder=3,
        **PUBLISHED_BOX,
    )
    axis.boxplot(
        [reproduced_values],
        positions=[1],
        widths=0.5,
        whis=1.5,
        zorder=3,
        **REPRODUCED_BOX,
    )
    axis.scatter(jitter, published_values, s=15, **PUBLISHED_MARKER, zorder=4)
    axis.scatter(1 + jitter, reproduced_values, s=12, **REPRODUCED_MARKER, zorder=4)
    axis.set_xticks([0, 1])
    axis.set_xticklabels(x_tick_labels)
    axis.set_xlim(-0.55, 1.55)
    axis.set_ylim(y_lower, y_upper)
    axis.yaxis.set_major_locator(MultipleLocator(0.25))
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2f}"))
    axis.set_ylabel(y_label)
    axis.set_xlabel(x_label)
    writer.save(figure, name, title, caption)


def f01_fig4a_pair_rmse(writer: FigureWriter, table: pd.DataFrame) -> None:
    """F01 (RMSE): published and reproduced GCN whole-test RMSE as paired boxes."""
    name = "paired_rmse"
    title = "Fig. 4a adaptation, paired GCN boxes (rmse)"
    caption = "As paired_cliff_rmse for whole-test RMSE."
    figure_size = (2.5, 2.9)
    x_tick_labels = ["Published\n(corrected)", "Reproduction\n(this work)"]
    x_label = "GCN (graph)"
    y_label = "RMSE"

    published_values = table["pub_rmse"]
    reproduced_values = table["rep_rmse"]
    jitter = np.random.default_rng(4).uniform(-0.11, 0.11, len(table))
    all_values = np.concatenate([published_values, reproduced_values])
    y_lower = np.floor(all_values.min() / 0.25) * 0.25
    y_upper = np.ceil(all_values.max() / 0.25) * 0.25

    figure, axis = plt.subplots(figsize=figure_size)
    for index in range(len(table)):
        axis.plot(
            [jitter[index], 1 + jitter[index]],
            [published_values.iloc[index], reproduced_values.iloc[index]],
            color="#B5B5B5",
            lw=0.5,
            zorder=2,
        )
    axis.boxplot(
        [published_values],
        positions=[0],
        widths=0.5,
        whis=1.5,
        zorder=3,
        **PUBLISHED_BOX,
    )
    axis.boxplot(
        [reproduced_values],
        positions=[1],
        widths=0.5,
        whis=1.5,
        zorder=3,
        **REPRODUCED_BOX,
    )
    axis.scatter(jitter, published_values, s=15, **PUBLISHED_MARKER, zorder=4)
    axis.scatter(1 + jitter, reproduced_values, s=12, **REPRODUCED_MARKER, zorder=4)
    axis.set_xticks([0, 1])
    axis.set_xticklabels(x_tick_labels)
    axis.set_xlim(-0.55, 1.55)
    axis.set_ylim(y_lower, y_upper)
    axis.yaxis.set_major_locator(MultipleLocator(0.25))
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2f}"))
    axis.set_ylabel(y_label)
    axis.set_xlabel(x_label)
    writer.save(figure, name, title, caption)


def f02_fig4a_pair_three_metrics(writer: FigureWriter, table: pd.DataFrame) -> None:
    """F02: paired GCN boxes for RMSE, RMSE_cliff and RMSE_cliff − RMSE."""
    name = "paired_error_metrics"
    title = "Paired GCN boxes for RMSE, RMSE_cliff and their difference"
    caption = (
        "(a) RMSE, (b) RMSE_cliff and (c) RMSE_cliff − RMSE for the published and "
        "reproduced GCN; lines join datasets. The difference in (c) is computed within "
        "folds before averaging and therefore preserves the dependence between the "
        "cliff subset and the whole test."
    )
    figure_size = (MAX_FIGURE_WIDTH_CM / 2.54, 2.7)
    x_tick_labels = ["Published", "Reproduction"]
    panel_a_y_label = "RMSE"
    panel_b_y_label = r"RMSE$_{\mathrm{cliff}}$"
    panel_c_y_label = r"RMSE$_{\mathrm{cliff}}$ − RMSE"

    jitter = np.random.default_rng(4).uniform(-0.11, 0.11, len(table))
    figure, (axis_a, axis_b, axis_c) = plt.subplots(
        1, 3, figsize=figure_size, gridspec_kw={"wspace": 0.55}
    )

    for index in range(len(table)):
        axis_a.plot(
            [jitter[index], 1 + jitter[index]],
            [table["pub_rmse"].iloc[index], table["rep_rmse"].iloc[index]],
            color="#B5B5B5",
            lw=0.5,
            zorder=2,
        )
    axis_a.boxplot(
        [table["pub_rmse"]],
        positions=[0],
        widths=0.5,
        whis=1.5,
        zorder=3,
        **PUBLISHED_BOX,
    )
    axis_a.boxplot(
        [table["rep_rmse"]],
        positions=[1],
        widths=0.5,
        whis=1.5,
        zorder=3,
        **REPRODUCED_BOX,
    )
    axis_a.scatter(jitter, table["pub_rmse"], s=14, **PUBLISHED_MARKER, zorder=4)
    axis_a.scatter(1 + jitter, table["rep_rmse"], s=11, **REPRODUCED_MARKER, zorder=4)
    axis_a.set_xticks([0, 1])
    axis_a.set_xticklabels(x_tick_labels)
    axis_a.set_xlim(-0.55, 1.55)
    panel_a_values = np.concatenate([table["pub_rmse"], table["rep_rmse"]])
    axis_a.set_ylim(
        np.floor(panel_a_values.min() / 0.25) * 0.25,
        np.ceil(panel_a_values.max() / 0.25) * 0.25,
    )
    axis_a.yaxis.set_major_locator(MultipleLocator(0.25))
    axis_a.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2f}"))
    axis_a.set_ylabel(panel_a_y_label)
    axis_a.text(-0.36, 1.04, "a", transform=axis_a.transAxes, **PANEL_LETTER)

    for index in range(len(table)):
        axis_b.plot(
            [jitter[index], 1 + jitter[index]],
            [table["pub_cliff"].iloc[index], table["rep_cliff"].iloc[index]],
            color="#B5B5B5",
            lw=0.5,
            zorder=2,
        )
    axis_b.boxplot(
        [table["pub_cliff"]],
        positions=[0],
        widths=0.5,
        whis=1.5,
        zorder=3,
        **PUBLISHED_BOX,
    )
    axis_b.boxplot(
        [table["rep_cliff"]],
        positions=[1],
        widths=0.5,
        whis=1.5,
        zorder=3,
        **REPRODUCED_BOX,
    )
    axis_b.scatter(jitter, table["pub_cliff"], s=14, **PUBLISHED_MARKER, zorder=4)
    axis_b.scatter(1 + jitter, table["rep_cliff"], s=11, **REPRODUCED_MARKER, zorder=4)
    axis_b.set_xticks([0, 1])
    axis_b.set_xticklabels(x_tick_labels)
    axis_b.set_xlim(-0.55, 1.55)
    panel_b_values = np.concatenate([table["pub_cliff"], table["rep_cliff"]])
    axis_b.set_ylim(
        np.floor(panel_b_values.min() / 0.25) * 0.25,
        np.ceil(panel_b_values.max() / 0.25) * 0.25,
    )
    axis_b.yaxis.set_major_locator(MultipleLocator(0.25))
    axis_b.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2f}"))
    axis_b.set_ylabel(panel_b_y_label)
    axis_b.text(-0.36, 1.04, "b", transform=axis_b.transAxes, **PANEL_LETTER)

    for index in range(len(table)):
        axis_c.plot(
            [jitter[index], 1 + jitter[index]],
            [table["pub_delta"].iloc[index], table["rep_delta"].iloc[index]],
            color="#B5B5B5",
            lw=0.5,
            zorder=2,
        )
    axis_c.boxplot(
        [table["pub_delta"]],
        positions=[0],
        widths=0.5,
        whis=1.5,
        zorder=3,
        **PUBLISHED_BOX,
    )
    axis_c.boxplot(
        [table["rep_delta"]],
        positions=[1],
        widths=0.5,
        whis=1.5,
        zorder=3,
        **REPRODUCED_BOX,
    )
    axis_c.scatter(jitter, table["pub_delta"], s=14, **PUBLISHED_MARKER, zorder=4)
    axis_c.scatter(1 + jitter, table["rep_delta"], s=11, **REPRODUCED_MARKER, zorder=4)
    axis_c.set_xticks([0, 1])
    axis_c.set_xticklabels(x_tick_labels)
    axis_c.set_xlim(-0.55, 1.55)
    axis_c.axhline(0, color=GUIDE_COLOUR, lw=0.7, ls=(0, (3, 3)), zorder=1)
    axis_c.yaxis.set_major_locator(MultipleLocator(0.1))
    axis_c.set_ylabel(panel_c_y_label)
    axis_c.text(-0.36, 1.04, "c", transform=axis_c.transAxes, **PANEL_LETTER)

    legend_handles = [
        Line2D([], [], **PUBLISHED_LEGEND_MARKER),
        Line2D([], [], **REPRODUCED_LEGEND_MARKER),
    ]
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    lowest_content = min(
        panel.get_tightbbox(renderer).transformed(figure.transFigure.inverted()).y0
        for panel in figure.axes
        if panel.get_visible()
    )
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, lowest_content - 0.012),
        ncol=len(legend_handles),
        handletextpad=0.35,
        columnspacing=1.6,
    )
    writer.save(figure, name, title, caption)


def f03_fig4a_context(
    writer: FigureWriter, table: pd.DataFrame, published: pd.DataFrame
) -> None:
    """F03: corrected Fig. 4a with the reproduced GCN beside the published GCN."""
    name = "cliff_rmse_strategy_context"
    title = (
        "Fig. 4a with the reproduction inserted among published "
        "deep-learning strategies"
    )
    caption = (
        "Corrected Fig. 4a with the reproduced GCN inserted beside the published GCN "
        "(shaded). Grey boxes are the other published deep-learning strategies and the "
        "two ECFP reference models (MLP, SVM), in the order of the corrected figure; "
        "they were not retrained here. The equally spaced GCN categories are labelled "
        "publ. (published) and repr. (reproduced)."
    )
    figure_size = (MAX_FIGURE_WIDTH_CM / 2.54, 2.7)
    methods = [
        ("AFP", "GRAPH"),
        ("MPNN", "GRAPH"),
        ("GAT", "GRAPH"),
        ("GCN", "GRAPH"),
        ("Transformer", "TOKENS"),
        ("CNN", "SMILES"),
        ("LSTM", "SMILES"),
        ("MLP", "ECFP"),
        ("SVM", "ECFP"),
    ]
    method_tick_labels = {"Transformer": "Trans-\nformer"}
    published_gcn_tick_label = "GCN\npubl."
    reproduced_gcn_tick_label = "GCN\nrepr."
    x_label = "Algorithm"
    y_label = r"RMSE$_{\mathrm{cliff}}$"
    y_limits = (0.25, 1.75)

    figure, axis = plt.subplots(figsize=figure_size)
    tick_positions = []
    tick_labels = []
    position = 0
    for algorithm, descriptor in methods:
        is_method = (published["algorithm"] == algorithm) & (
            published["descriptor"] == descriptor
        )
        values = published[is_method].set_index("dataset")["cliff_rmse"]
        values = values.loc[table.index]
        jitter = np.random.default_rng(11 + position).uniform(-0.12, 0.12, len(values))
        if algorithm == "GCN":
            axis.boxplot(
                [values],
                positions=[position],
                widths=0.58,
                whis=1.5,
                zorder=3,
                **PUBLISHED_BOX,
            )
            axis.scatter(position + jitter, values, s=12, **PUBLISHED_MARKER, zorder=4)
            tick_positions.append(position)
            tick_labels.append(published_gcn_tick_label)
            position += 1
            axis.boxplot(
                [table["rep_cliff"]],
                positions=[position],
                widths=0.58,
                whis=1.5,
                zorder=3,
                **REPRODUCED_BOX,
            )
            axis.scatter(
                position + jitter,
                table["rep_cliff"],
                s=10,
                **REPRODUCED_MARKER,
                zorder=4,
            )
            tick_positions.append(position)
            tick_labels.append(reproduced_gcn_tick_label)
            axis.axvspan(
                position - 1.45, position + 0.45, color=BAND_COLOUR, zorder=0, lw=0
            )
        else:
            axis.boxplot(
                [values],
                positions=[position],
                widths=0.58,
                whis=1.5,
                zorder=3,
                **OTHER_STRATEGIES_BOX,
            )
            axis.scatter(
                position + jitter,
                values,
                s=9,
                color=OTHER_STRATEGIES["edge"],
                edgecolor="none",
                alpha=0.75,
                zorder=4,
            )
            tick_positions.append(position)
            tick_labels.append(method_tick_labels.get(algorithm, algorithm))
        position += 1
    axis.set_xticks(tick_positions)
    axis.set_xticklabels(tick_labels)
    axis.set_xlim(-0.7, position - 0.3)
    axis.set_ylim(*y_limits)
    axis.yaxis.set_major_locator(MultipleLocator(0.25))
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2f}"))
    axis.set_ylabel(y_label)
    axis.set_xlabel(x_label)

    legend_handles = [
        Line2D([], [], **PUBLISHED_LEGEND_MARKER),
        Line2D([], [], **REPRODUCED_LEGEND_MARKER),
        Line2D([], [], **OTHER_STRATEGIES_LEGEND_MARKER),
    ]
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    lowest_content = min(
        panel.get_tightbbox(renderer).transformed(figure.transFigure.inverted()).y0
        for panel in figure.axes
        if panel.get_visible()
    )
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, lowest_content - 0.012),
        ncol=len(legend_handles),
        handletextpad=0.35,
        columnspacing=1.6,
    )
    writer.save(figure, name, title, caption)


def f04_fig4c_side_by_side(
    writer: FigureWriter, table: pd.DataFrame, published: pd.DataFrame
) -> None:
    """F04: corrected Fig. 4c with published and reproduced GCN in separate panels."""
    name = "rmse_cliff_comparison"
    title = "Fig. 4c: published and reproduced GCN in separate panels"
    caption = (
        "Corrected Fig. 4c adaptation. RMSE_cliff against RMSE for (a) the published "
        "and (b) the reproduced GCN, on identical axes. Grey points are all other "
        "published strategy–dataset pairs. The black dashed line is RMSE_cliff = RMSE; "
        "grey dashed lines mark ±0.5 log units. Fold SD bars are omitted so that both "
        "panels show the same quantity."
    )
    figure_size = (5.6, 2.9)
    panel_a_title = "GCN · published"
    panel_a_x_label = "RMSE"
    panel_a_y_label = r"RMSE$_{\mathrm{cliff}}$"
    panel_b_title = "GCN · reproduction"
    panel_b_x_label = "RMSE"
    panel_b_y_label = r"RMSE$_{\mathrm{cliff}}$"

    all_values = np.concatenate(
        [
            published["rmse"],
            published["cliff_rmse"],
            table["rep_rmse"],
            table["rep_cliff"],
        ]
    )
    axis_lower = min(0.25, np.floor(all_values.min() / 0.25) * 0.25)
    axis_upper = max(1.75, np.ceil(all_values.max() / 0.25) * 0.25)
    other_strategies = published[~published["is_gcn"]]
    figure, (axis_a, axis_b) = plt.subplots(
        1, 2, figsize=figure_size, gridspec_kw={"wspace": 0.38}
    )

    axis_a.scatter(
        other_strategies["rmse"],
        other_strategies["cliff_rmse"],
        s=11,
        **OTHER_STRATEGIES_CLOUD,
        zorder=2,
    )
    axis_a.plot([axis_lower, axis_upper], [axis_lower, axis_upper], **IDENTITY_LINE)
    axis_a.plot(
        [axis_lower, axis_upper],
        [axis_lower - 0.5, axis_upper - 0.5],
        **HALF_LOG_UNIT_LINE,
    )
    axis_a.plot(
        [axis_lower, axis_upper],
        [axis_lower + 0.5, axis_upper + 0.5],
        **HALF_LOG_UNIT_LINE,
    )
    axis_a.set_xlim(axis_lower, axis_upper)
    axis_a.set_ylim(axis_lower, axis_upper)
    axis_a.set_aspect("equal", adjustable="box")
    axis_a.scatter(
        table["pub_rmse"], table["pub_cliff"], s=20, **PUBLISHED_MARKER, zorder=4
    )
    axis_a.text(
        0.05,
        0.95,
        panel_a_title,
        transform=axis_a.transAxes,
        fontweight="bold",
        fontsize=7.2,
        va="top",
    )
    axis_a.xaxis.set_major_locator(MultipleLocator(0.25))
    axis_a.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2f}"))
    axis_a.yaxis.set_major_locator(MultipleLocator(0.25))
    axis_a.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2f}"))
    axis_a.set_xlabel(panel_a_x_label)
    axis_a.set_ylabel(panel_a_y_label)

    axis_b.scatter(
        other_strategies["rmse"],
        other_strategies["cliff_rmse"],
        s=11,
        **OTHER_STRATEGIES_CLOUD,
        zorder=2,
    )
    axis_b.plot([axis_lower, axis_upper], [axis_lower, axis_upper], **IDENTITY_LINE)
    axis_b.plot(
        [axis_lower, axis_upper],
        [axis_lower - 0.5, axis_upper - 0.5],
        **HALF_LOG_UNIT_LINE,
    )
    axis_b.plot(
        [axis_lower, axis_upper],
        [axis_lower + 0.5, axis_upper + 0.5],
        **HALF_LOG_UNIT_LINE,
    )
    axis_b.set_xlim(axis_lower, axis_upper)
    axis_b.set_ylim(axis_lower, axis_upper)
    axis_b.set_aspect("equal", adjustable="box")
    axis_b.scatter(
        table["rep_rmse"], table["rep_cliff"], s=16, **REPRODUCED_MARKER, zorder=4
    )
    axis_b.text(
        0.05,
        0.95,
        panel_b_title,
        transform=axis_b.transAxes,
        fontweight="bold",
        fontsize=7.2,
        va="top",
    )
    axis_b.xaxis.set_major_locator(MultipleLocator(0.25))
    axis_b.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2f}"))
    axis_b.yaxis.set_major_locator(MultipleLocator(0.25))
    axis_b.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2f}"))
    axis_b.set_xlabel(panel_b_x_label)
    axis_b.set_ylabel(panel_b_y_label)

    axis_a.text(-0.28, 1.04, "a", transform=axis_a.transAxes, **PANEL_LETTER)
    axis_b.text(-0.28, 1.04, "b", transform=axis_b.transAxes, **PANEL_LETTER)
    axis_a.tick_params(axis="x", labelsize=7)
    axis_b.tick_params(axis="x", labelsize=7)

    legend_handles = [
        Line2D([], [], **PUBLISHED_LEGEND_MARKER),
        Line2D([], [], **REPRODUCED_LEGEND_MARKER),
        Line2D([], [], **OTHER_STRATEGIES_LEGEND_MARKER),
    ]
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    lowest_content = min(
        panel.get_tightbbox(renderer).transformed(figure.transFigure.inverted()).y0
        for panel in figure.axes
        if panel.get_visible()
    )
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, lowest_content - 0.012),
        ncol=len(legend_handles),
        handletextpad=0.35,
        columnspacing=1.6,
    )
    writer.save(figure, name, title, caption)


def f05_fig4c_side_by_side_ghost(
    writer: FigureWriter, table: pd.DataFrame, published: pd.DataFrame
) -> None:
    """F05: as F04, with the other source drawn hollow in each panel."""
    name = "rmse_cliff_comparison_overlay"
    title = "Fig. 4c: published and reproduced GCN in separate panels (with ghost)"
    caption = (
        "As rmse_cliff_comparison, with the other source drawn as hollow markers in each panel to make "
        "the shift visible."
    )
    figure_size = (5.6, 2.9)
    panel_a_title = "GCN · published"
    panel_a_x_label = "RMSE"
    panel_a_y_label = r"RMSE$_{\mathrm{cliff}}$"
    panel_b_title = "GCN · reproduction"
    panel_b_x_label = "RMSE"
    panel_b_y_label = r"RMSE$_{\mathrm{cliff}}$"

    all_values = np.concatenate(
        [
            published["rmse"],
            published["cliff_rmse"],
            table["rep_rmse"],
            table["rep_cliff"],
        ]
    )
    axis_lower = min(0.25, np.floor(all_values.min() / 0.25) * 0.25)
    axis_upper = max(1.75, np.ceil(all_values.max() / 0.25) * 0.25)
    other_strategies = published[~published["is_gcn"]]
    figure, (axis_a, axis_b) = plt.subplots(
        1, 2, figsize=figure_size, gridspec_kw={"wspace": 0.38}
    )

    axis_a.scatter(
        other_strategies["rmse"],
        other_strategies["cliff_rmse"],
        s=11,
        **OTHER_STRATEGIES_CLOUD,
        zorder=2,
    )
    axis_a.plot([axis_lower, axis_upper], [axis_lower, axis_upper], **IDENTITY_LINE)
    axis_a.plot(
        [axis_lower, axis_upper],
        [axis_lower - 0.5, axis_upper - 0.5],
        **HALF_LOG_UNIT_LINE,
    )
    axis_a.plot(
        [axis_lower, axis_upper],
        [axis_lower + 0.5, axis_upper + 0.5],
        **HALF_LOG_UNIT_LINE,
    )
    axis_a.set_xlim(axis_lower, axis_upper)
    axis_a.set_ylim(axis_lower, axis_upper)
    axis_a.set_aspect("equal", adjustable="box")
    axis_a.scatter(
        table["rep_rmse"],
        table["rep_cliff"],
        s=13,
        marker=REPRODUCED["marker"],
        facecolor="white",
        edgecolor=REPRODUCED["colour"],
        linewidth=0.9,
        zorder=3,
        alpha=0.8,
    )
    axis_a.scatter(
        table["pub_rmse"], table["pub_cliff"], s=20, **PUBLISHED_MARKER, zorder=4
    )
    axis_a.text(
        0.05,
        0.95,
        panel_a_title,
        transform=axis_a.transAxes,
        fontweight="bold",
        fontsize=7.2,
        va="top",
    )
    axis_a.xaxis.set_major_locator(MultipleLocator(0.25))
    axis_a.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2f}"))
    axis_a.yaxis.set_major_locator(MultipleLocator(0.25))
    axis_a.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2f}"))
    axis_a.set_xlabel(panel_a_x_label)
    axis_a.set_ylabel(panel_a_y_label)

    axis_b.scatter(
        other_strategies["rmse"],
        other_strategies["cliff_rmse"],
        s=11,
        **OTHER_STRATEGIES_CLOUD,
        zorder=2,
    )
    axis_b.plot([axis_lower, axis_upper], [axis_lower, axis_upper], **IDENTITY_LINE)
    axis_b.plot(
        [axis_lower, axis_upper],
        [axis_lower - 0.5, axis_upper - 0.5],
        **HALF_LOG_UNIT_LINE,
    )
    axis_b.plot(
        [axis_lower, axis_upper],
        [axis_lower + 0.5, axis_upper + 0.5],
        **HALF_LOG_UNIT_LINE,
    )
    axis_b.set_xlim(axis_lower, axis_upper)
    axis_b.set_ylim(axis_lower, axis_upper)
    axis_b.set_aspect("equal", adjustable="box")
    axis_b.scatter(
        table["pub_rmse"],
        table["pub_cliff"],
        s=13,
        marker=PUBLISHED["marker"],
        facecolor="white",
        edgecolor=PUBLISHED["colour"],
        linewidth=0.9,
        zorder=3,
        alpha=0.8,
    )
    axis_b.scatter(
        table["rep_rmse"], table["rep_cliff"], s=16, **REPRODUCED_MARKER, zorder=4
    )
    axis_b.text(
        0.05,
        0.95,
        panel_b_title,
        transform=axis_b.transAxes,
        fontweight="bold",
        fontsize=7.2,
        va="top",
    )
    axis_b.xaxis.set_major_locator(MultipleLocator(0.25))
    axis_b.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2f}"))
    axis_b.yaxis.set_major_locator(MultipleLocator(0.25))
    axis_b.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2f}"))
    axis_b.set_xlabel(panel_b_x_label)
    axis_b.set_ylabel(panel_b_y_label)

    axis_a.text(-0.28, 1.04, "a", transform=axis_a.transAxes, **PANEL_LETTER)
    axis_b.text(-0.28, 1.04, "b", transform=axis_b.transAxes, **PANEL_LETTER)
    axis_a.tick_params(axis="x", labelsize=7)
    axis_b.tick_params(axis="x", labelsize=7)

    legend_handles = [
        Line2D([], [], **PUBLISHED_LEGEND_MARKER),
        Line2D([], [], **REPRODUCED_LEGEND_MARKER),
        Line2D([], [], **OTHER_STRATEGIES_LEGEND_MARKER),
    ]
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    lowest_content = min(
        panel.get_tightbbox(renderer).transformed(figure.transFigure.inverted()).y0
        for panel in figure.axes
        if panel.get_visible()
    )
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, lowest_content - 0.012),
        ncol=len(legend_handles),
        handletextpad=0.35,
        columnspacing=1.6,
    )
    writer.save(figure, name, title, caption)


def draw_shift_connectors(
    axis: plt.Axes, table: pd.DataFrame, short_threshold_points: float
) -> None:
    """Connect true endpoints; omit arrowheads for shifts below the display threshold."""
    figure = axis.figure
    figure.canvas.draw()
    for _, row in table.iterrows():
        published = (row["pub_rmse"], row["pub_cliff"])
        reproduced = (row["rep_rmse"], row["rep_cliff"])
        endpoints = axis.transData.transform([published, reproduced])
        length_points = np.linalg.norm(endpoints[1] - endpoints[0]) * 72 / figure.dpi
        is_short = length_points < short_threshold_points
        axis.annotate(
            "",
            xy=reproduced,
            xytext=published,
            arrowprops={
                "arrowstyle": "-" if is_short else "-|>",
                "color": "#6E6E6E",
                "lw": 0.45 if is_short else 0.7,
                "mutation_scale": 7,
                "shrinkA": 0 if is_short else 2.5,
                "shrinkB": 0 if is_short else 2.5,
            },
            zorder=5 if is_short else 3,
        )


def place_shift_labels(
    axis: plt.Axes, table: pd.DataFrame, font_size: float, clearance_points: float
) -> None:
    """Place the four largest-shift labels clear of plotted marks and arrows.

    Candidate positions are measured by the renderer in display coordinates, so
    clearance is a physical distance rather than a change to any plotted value.
    """
    figure = axis.figure
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    pixels_per_point = figure.dpi / 72
    clearance = clearance_points * pixels_per_point
    bounds = axis.get_window_extent().padded(-clearance)
    occupied = []
    label_bounds = []
    for collection in axis.collections:
        offsets = collection.get_offset_transform().transform(collection.get_offsets())
        sizes = collection.get_sizes()
        if not len(sizes):
            continue
        # This circumscribed box also covers rotated diamond markers.
        radius = np.sqrt(np.max(sizes)) * pixels_per_point * 0.8 + clearance
        for x, y in offsets:
            box = Bbox.from_extents(x - radius, y - radius, x + radius, y + radius)
            if box.overlaps(axis.get_window_extent()):
                occupied.append(box)
    arrows = [
        item.arrow_patch.get_path().transformed(item.arrow_patch.get_transform())
        for item in axis.texts
        if getattr(item, "arrow_patch", None) is not None
    ]
    shifts = np.hypot(table["diff_rmse"], table["diff_cliff"]).sort_values(
        ascending=False
    )
    for dataset in shifts.index[:4]:
        row = table.loc[dataset]
        point = (row["rep_rmse"], row["rep_cliff"])
        annotation = axis.annotate(
            row["abbreviation"],
            point,
            xytext=(0, 0),
            textcoords="offset points",
            fontsize=font_size,
            color=INK,
            zorder=8,
            va="center",
            arrowprops={
                "arrowstyle": "-",
                "color": "#7A7A7A",
                "lw": 0.4,
                "shrinkA": clearance_points,
                "shrinkB": 3,
            },
        )
        placed = False
        for radius_points in range(12, 181, 4):
            for angle in np.linspace(0, 2 * np.pi, 32, endpoint=False):
                dx, dy = radius_points * np.cos(angle), radius_points * np.sin(angle)
                annotation.set_position((dx, dy))
                annotation.set_ha("left" if dx >= 0 else "right")
                annotation.update_positions(renderer)
                box = Text.get_window_extent(annotation, renderer).padded(clearance)
                if not (
                    bounds.contains(box.x0, box.y0) and bounds.contains(box.x1, box.y1)
                ):
                    continue
                if any(box.overlaps(other) for other in occupied):
                    continue
                if any(path.intersects_bbox(box, filled=False) for path in arrows):
                    continue
                leader = annotation.arrow_patch.get_path().transformed(
                    annotation.arrow_patch.get_transform()
                )
                if any(
                    leader.intersects_bbox(previous, filled=False)
                    for previous in label_bounds
                ):
                    continue
                occupied.append(box)
                label_bounds.append(box)
                arrows.append(leader)
                placed = True
                break
            if placed:
                break
        if not placed:
            raise FigureDataError(f"No clear position for shift label {dataset}")


def f06_fig4c_shift_arrows_zoom(writer: FigureWriter, table: pd.DataFrame) -> None:
    """F06 (zoom): arrow from each published to its reproduced GCN position."""
    name = "rmse_cliff_shifts_zoomed"
    title = "Published-to-reproduced shift per dataset in the RMSE plane"
    caption = (
        "Shift of each dataset from its published (circle) to its reproduced (diamond) "
        "position in the RMSE–RMSE_cliff plane; axes are zoomed to the GCN range. The "
        "four largest shifts are labelled descriptively; labels do not imply a cause. "
        "Shifts shorter than 5 pt at the exported size use a thin connector without "
        "an arrowhead, drawn above the marker fills at the true endpoints; circles "
        "and diamonds identify the published and reproduced positions."
    )
    figure_size = (3.7, 3.7)
    x_label = "RMSE"
    y_label = r"RMSE$_{\mathrm{cliff}}$"
    short_shift_threshold_points = 5.0
    label_font_size = 7
    label_clearance_points = 2.5

    gcn_values = np.concatenate(
        [table["pub_rmse"], table["rep_rmse"], table["pub_cliff"], table["rep_cliff"]]
    )
    axis_lower = np.floor((gcn_values.min() - 0.04) * 20) / 20
    axis_upper = np.ceil((gcn_values.max() + 0.04) * 20) / 20

    figure, axis = plt.subplots(figsize=figure_size)
    axis.plot([axis_lower, axis_upper], [axis_lower, axis_upper], **IDENTITY_LINE)
    axis.plot(
        [axis_lower, axis_upper],
        [axis_lower - 0.5, axis_upper - 0.5],
        **HALF_LOG_UNIT_LINE,
    )
    axis.plot(
        [axis_lower, axis_upper],
        [axis_lower + 0.5, axis_upper + 0.5],
        **HALF_LOG_UNIT_LINE,
    )
    axis.set_xlim(axis_lower, axis_upper)
    axis.set_ylim(axis_lower, axis_upper)
    axis.set_aspect("equal", adjustable="box")
    draw_shift_connectors(axis, table, short_shift_threshold_points)
    axis.scatter(
        table["pub_rmse"], table["pub_cliff"], s=20, **PUBLISHED_MARKER, zorder=4
    )
    axis.scatter(
        table["rep_rmse"], table["rep_cliff"], s=16, **REPRODUCED_MARKER, zorder=4
    )
    place_shift_labels(axis, table, label_font_size, label_clearance_points)
    axis.xaxis.set_major_locator(MultipleLocator(0.1))
    axis.yaxis.set_major_locator(MultipleLocator(0.1))
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)

    legend_handles = [
        Line2D([], [], **PUBLISHED_LEGEND_MARKER),
        Line2D([], [], **REPRODUCED_LEGEND_MARKER),
    ]
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    lowest_content = min(
        panel.get_tightbbox(renderer).transformed(figure.transFigure.inverted()).y0
        for panel in figure.axes
        if panel.get_visible()
    )
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, lowest_content - 0.012),
        ncol=1,
        handletextpad=0.35,
        columnspacing=1.6,
    )
    writer.save(figure, name, title, caption)


def f06_fig4c_shift_arrows_full(
    writer: FigureWriter, table: pd.DataFrame, published: pd.DataFrame
) -> None:
    """F06 (full): as the zoomed shift plot, on the full axes of corrected Fig. 4c."""
    name = "rmse_cliff_shifts_context"
    title = "Published-to-reproduced shift per dataset in the RMSE plane"
    caption = (
        "As rmse_cliff_shifts_zoomed on the full axes of corrected Fig. 4c, with other published strategies "
        "in grey. Shifts shorter than 5 pt at the exported size use a thin connector "
        "without an arrowhead at the true endpoints; circles and diamonds identify "
        "the published and reproduced positions."
    )
    figure_size = (3.7, 3.7)
    x_label = "RMSE"
    y_label = r"RMSE$_{\mathrm{cliff}}$"
    short_shift_threshold_points = 5.0
    label_font_size = 7
    label_clearance_points = 2.5

    published_values = np.concatenate([published["rmse"], published["cliff_rmse"]])
    axis_lower = min(0.25, np.floor(published_values.min() / 0.25) * 0.25)
    axis_upper = max(1.75, np.ceil(published_values.max() / 0.25) * 0.25)
    other_strategies = published[~published["is_gcn"]]

    figure, axis = plt.subplots(figsize=figure_size)
    axis.scatter(
        other_strategies["rmse"],
        other_strategies["cliff_rmse"],
        s=11,
        **OTHER_STRATEGIES_CLOUD,
        zorder=2,
    )
    axis.plot([axis_lower, axis_upper], [axis_lower, axis_upper], **IDENTITY_LINE)
    axis.plot(
        [axis_lower, axis_upper],
        [axis_lower - 0.5, axis_upper - 0.5],
        **HALF_LOG_UNIT_LINE,
    )
    axis.plot(
        [axis_lower, axis_upper],
        [axis_lower + 0.5, axis_upper + 0.5],
        **HALF_LOG_UNIT_LINE,
    )
    axis.set_xlim(axis_lower, axis_upper)
    axis.set_ylim(axis_lower, axis_upper)
    axis.set_aspect("equal", adjustable="box")
    draw_shift_connectors(axis, table, short_shift_threshold_points)
    axis.scatter(
        table["pub_rmse"], table["pub_cliff"], s=20, **PUBLISHED_MARKER, zorder=4
    )
    axis.scatter(
        table["rep_rmse"], table["rep_cliff"], s=16, **REPRODUCED_MARKER, zorder=4
    )
    place_shift_labels(axis, table, label_font_size, label_clearance_points)
    axis.xaxis.set_major_locator(MultipleLocator(0.25))
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2f}"))
    axis.yaxis.set_major_locator(MultipleLocator(0.25))
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2f}"))
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)

    legend_handles = [
        Line2D([], [], **PUBLISHED_LEGEND_MARKER),
        Line2D([], [], **REPRODUCED_LEGEND_MARKER),
    ]
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    lowest_content = min(
        panel.get_tightbbox(renderer).transformed(figure.transFigure.inverted()).y0
        for panel in figure.axes
        if panel.get_visible()
    )
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, lowest_content - 0.012),
        ncol=1,
        handletextpad=0.35,
        columnspacing=1.6,
    )
    writer.save(figure, name, title, caption)


def f07_fig5a_context(
    writer: FigureWriter, table: pd.DataFrame, published: pd.DataFrame, order: list[str]
) -> None:
    """F07: corrected Fig. 5a, RMSE_cliff − RMSE per dataset with context boxes."""
    name = "cliff_error_difference_strategy_context"
    title = "Fig. 5a adaptation: RMSE_cliff − RMSE per dataset"
    caption = (
        "Adaptation of corrected Fig. 5a. RMSE_cliff − RMSE per dataset. Grey boxes "
        "summarise the 23 other published strategies (box: IQR and median; whiskers: "
        "lowest and highest value, as in the paper). Circles: published GCN; diamonds: "
        "reproduced GCN ± fold SD of the within-fold difference. Rows and the r column "
        "follow the corrected figure: Pearson r between RMSE and RMSE_cliff across the "
        "24 published strategies, recomputed from the published table."
    )
    figure_size = (3.6, 5.6)
    x_label = r"RMSE$_{\mathrm{cliff}}$ − RMSE"
    y_label = "Dataset"
    r_column_title = r"$r_{\mathrm{RMSE,\ RMSE_{cliff}}}$"
    context_legend_label = "Other published strategies (box: IQR; whiskers: range)"
    marker_offset = 0.17

    figure = plt.figure(figsize=figure_size)
    grid = figure.add_gridspec(1, 2, width_ratios=[1, 0.13], wspace=0.04)
    axis = figure.add_subplot(grid[0])
    row_count = len(order)
    axis.set_ylim(row_count - 0.5, -0.5)
    axis.set_yticks(range(row_count))
    axis.set_yticklabels([table.loc[dataset, "abbreviation"] for dataset in order])
    axis.tick_params(axis="y", length=2.5)
    for row_index in range(0, row_count, 2):
        axis.axhspan(
            row_index - 0.5, row_index + 0.5, color=BAND_COLOUR, zorder=0, lw=0
        )
    axis.axvline(0, color=GUIDE_COLOUR, lw=0.7, ls=(0, (3, 3)), zorder=1)
    for row_index, dataset in enumerate(order):
        row = table.loc[dataset]
        other_deltas = published[
            (published["dataset"] == dataset) & ~published["is_gcn"]
        ]["delta"]
        axis.boxplot(
            [other_deltas],
            positions=[row_index],
            widths=0.62,
            orientation="horizontal",
            whis=(0, 100),
            zorder=2,
            **OTHER_STRATEGIES_BOX,
        )
        axis.plot(
            [row["pub_delta"], row["rep_delta"]],
            [row_index - marker_offset, row_index + marker_offset],
            color="#9A9A9A",
            lw=0.5,
            zorder=3,
        )
        axis.errorbar(
            row["rep_delta"],
            row_index + marker_offset,
            xerr=row["rep_delta_sd"],
            fmt="none",
            ecolor=REPRODUCED["colour"],
            elinewidth=0.8,
            capsize=1.4,
            capthick=0.7,
            zorder=4,
        )
        axis.scatter(
            row["pub_delta"],
            row_index - marker_offset,
            s=16,
            **PUBLISHED_MARKER,
            zorder=5,
        )
        axis.scatter(
            row["rep_delta"],
            row_index + marker_offset,
            s=12,
            **REPRODUCED_MARKER,
            zorder=5,
        )
    axis.set_xlabel(x_label)
    axis.xaxis.set_major_locator(MultipleLocator(0.2))
    axis.set_ylabel(y_label)

    r_axis = figure.add_subplot(grid[1])
    r_values = table.loc[order, "r_published_methods"].to_numpy()
    r_shades = 0.18 + 0.57 * (1 - np.clip((r_values - 0.3) / 0.7, 0, 1))
    r_colours = plt.get_cmap("Greys")(r_shades)
    for row_index, (r_value, r_colour) in enumerate(zip(r_values, r_colours)):
        r_axis.add_patch(
            plt.Rectangle(
                (0, row_index - 0.5),
                1,
                1,
                facecolor=r_colour,
                edgecolor="white",
                lw=0.6,
            )
        )
        r_text = (
            f"{r_value:.2f}".rstrip("0").rstrip(".")
            if r_value >= 0.995
            else f"{r_value:.2f}"
        )
        r_axis.text(
            0.5,
            row_index,
            r_text,
            ha="center",
            va="center",
            fontsize=7,
            color="white" if r_shades[row_index] > 0.58 else INK,
        )
    r_axis.set_xlim(0, 1)
    r_axis.set_ylim(row_count - 0.5, -0.5)
    r_axis.axis("off")
    r_axis.text(0.5, -0.9, r_column_title, ha="center", va="bottom", fontsize=7)

    legend_handles = [
        Line2D([], [], **PUBLISHED_LEGEND_MARKER),
        Line2D([], [], **REPRODUCED_LEGEND_MARKER),
        Patch(
            facecolor=OTHER_STRATEGIES["fill"],
            edgecolor=OTHER_STRATEGIES["edge"],
            lw=0.6,
            label=context_legend_label,
        ),
    ]
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    lowest_content = min(
        panel.get_tightbbox(renderer).transformed(figure.transFigure.inverted()).y0
        for panel in figure.axes
        if panel.get_visible()
    )
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, lowest_content - 0.012),
        ncol=1,
        handletextpad=0.35,
        columnspacing=1.6,
    )
    writer.save(figure, name, title, caption)


def f08_fig5a_gcn_only(
    writer: FigureWriter, table: pd.DataFrame, order: list[str]
) -> None:
    """F08: as F07 without the boxes of the other published strategies."""
    name = "cliff_error_difference_by_dataset"
    title = "Fig. 5a adaptation: RMSE_cliff − RMSE per dataset"
    caption = (
        "As cliff_error_difference_strategy_context without the across-strategy boxes."
    )
    figure_size = (3.6, 5.6)
    x_label = r"RMSE$_{\mathrm{cliff}}$ − RMSE"
    y_label = "Dataset"
    r_column_title = r"$r_{\mathrm{RMSE,\ RMSE_{cliff}}}$"
    marker_offset = 0.17

    figure = plt.figure(figsize=figure_size)
    grid = figure.add_gridspec(1, 2, width_ratios=[1, 0.13], wspace=0.04)
    axis = figure.add_subplot(grid[0])
    row_count = len(order)
    axis.set_ylim(row_count - 0.5, -0.5)
    axis.set_yticks(range(row_count))
    axis.set_yticklabels([table.loc[dataset, "abbreviation"] for dataset in order])
    axis.tick_params(axis="y", length=2.5)
    for row_index in range(0, row_count, 2):
        axis.axhspan(
            row_index - 0.5, row_index + 0.5, color=BAND_COLOUR, zorder=0, lw=0
        )
    axis.axvline(0, color=GUIDE_COLOUR, lw=0.7, ls=(0, (3, 3)), zorder=1)
    for row_index, dataset in enumerate(order):
        row = table.loc[dataset]
        axis.plot(
            [row["pub_delta"], row["rep_delta"]],
            [row_index - marker_offset, row_index + marker_offset],
            color="#9A9A9A",
            lw=0.5,
            zorder=3,
        )
        axis.errorbar(
            row["rep_delta"],
            row_index + marker_offset,
            xerr=row["rep_delta_sd"],
            fmt="none",
            ecolor=REPRODUCED["colour"],
            elinewidth=0.8,
            capsize=1.4,
            capthick=0.7,
            zorder=4,
        )
        axis.scatter(
            row["pub_delta"],
            row_index - marker_offset,
            s=16,
            **PUBLISHED_MARKER,
            zorder=5,
        )
        axis.scatter(
            row["rep_delta"],
            row_index + marker_offset,
            s=12,
            **REPRODUCED_MARKER,
            zorder=5,
        )
    axis.set_xlabel(x_label)
    axis.xaxis.set_major_locator(MultipleLocator(0.2))
    axis.set_ylabel(y_label)

    r_axis = figure.add_subplot(grid[1])
    r_values = table.loc[order, "r_published_methods"].to_numpy()
    r_shades = 0.18 + 0.57 * (1 - np.clip((r_values - 0.3) / 0.7, 0, 1))
    r_colours = plt.get_cmap("Greys")(r_shades)
    for row_index, (r_value, r_colour) in enumerate(zip(r_values, r_colours)):
        r_axis.add_patch(
            plt.Rectangle(
                (0, row_index - 0.5),
                1,
                1,
                facecolor=r_colour,
                edgecolor="white",
                lw=0.6,
            )
        )
        r_text = (
            f"{r_value:.2f}".rstrip("0").rstrip(".")
            if r_value >= 0.995
            else f"{r_value:.2f}"
        )
        r_axis.text(
            0.5,
            row_index,
            r_text,
            ha="center",
            va="center",
            fontsize=7,
            color="white" if r_shades[row_index] > 0.58 else INK,
        )
    r_axis.set_xlim(0, 1)
    r_axis.set_ylim(row_count - 0.5, -0.5)
    r_axis.axis("off")
    r_axis.text(0.5, -0.9, r_column_title, ha="center", va="bottom", fontsize=7)

    legend_handles = [
        Line2D([], [], **PUBLISHED_LEGEND_MARKER),
        Line2D([], [], **REPRODUCED_LEGEND_MARKER),
    ]
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    lowest_content = min(
        panel.get_tightbbox(renderer).transformed(figure.transFigure.inverted()).y0
        for panel in figure.axes
        if panel.get_visible()
    )
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, lowest_content - 0.012),
        ncol=1,
        handletextpad=0.35,
        columnspacing=1.6,
    )
    writer.save(figure, name, title, caption)


def f09_dumbbell_mean_sd(
    writer: FigureWriter, table: pd.DataFrame, order: list[str]
) -> None:
    """F09: published and reproduced RMSE and RMSE_cliff per dataset, with fold SD."""
    name = "dataset_errors_mean_sd"
    title = "Per-dataset published versus reproduced RMSE and RMSE_cliff"
    caption = (
        "Published and reproduced (a) RMSE and (b) RMSE_cliff per dataset, joined by a "
        "line; reproduced values show ± fold SD. Rows follow the corrected Fig. 5a "
        "order, which is fixed by published data and independent of the reproduction."
    )
    figure_size = (5.2, 5.6)
    panel_a_x_label = "RMSE"
    panel_b_x_label = r"RMSE$_{\mathrm{cliff}}$"
    y_label = "Dataset"

    dataset_labels = [table.loc[dataset, "abbreviation"] for dataset in order]
    row_count = len(order)
    figure, (axis_a, axis_b) = plt.subplots(
        1, 2, figsize=figure_size, sharey=True, gridspec_kw={"wspace": 0.12}
    )

    axis_a.set_ylim(row_count - 0.5, -0.5)
    axis_a.set_yticks(range(row_count))
    axis_a.set_yticklabels(dataset_labels)
    axis_a.tick_params(axis="y", length=2.5)
    for row_index in range(0, row_count, 2):
        axis_a.axhspan(
            row_index - 0.5, row_index + 0.5, color=BAND_COLOUR, zorder=0, lw=0
        )
    for row_index, dataset in enumerate(order):
        row = table.loc[dataset]
        axis_a.plot(
            [row["pub_rmse"], row["rep_rmse"]],
            [row_index, row_index],
            color="#A8A8A8",
            lw=0.9,
            zorder=2,
            solid_capstyle="butt",
        )
        axis_a.errorbar(
            row["rep_rmse"],
            row_index,
            xerr=row["rep_rmse_sd"],
            fmt="none",
            ecolor=REPRODUCED["colour"],
            elinewidth=0.8,
            capsize=1.4,
            capthick=0.7,
            zorder=3,
        )
        axis_a.scatter(row["pub_rmse"], row_index, s=17, **PUBLISHED_MARKER, zorder=5)
        axis_a.scatter(row["rep_rmse"], row_index, s=13, **REPRODUCED_MARKER, zorder=6)
    axis_a.set_xlabel(panel_a_x_label)

    axis_b.set_ylim(row_count - 0.5, -0.5)
    axis_b.set_yticks(range(row_count))
    axis_b.set_yticklabels(dataset_labels)
    axis_b.tick_params(axis="y", length=2.5)
    for row_index in range(0, row_count, 2):
        axis_b.axhspan(
            row_index - 0.5, row_index + 0.5, color=BAND_COLOUR, zorder=0, lw=0
        )
    for row_index, dataset in enumerate(order):
        row = table.loc[dataset]
        axis_b.plot(
            [row["pub_cliff"], row["rep_cliff"]],
            [row_index, row_index],
            color="#A8A8A8",
            lw=0.9,
            zorder=2,
            solid_capstyle="butt",
        )
        axis_b.errorbar(
            row["rep_cliff"],
            row_index,
            xerr=row["rep_cliff_sd"],
            fmt="none",
            ecolor=REPRODUCED["colour"],
            elinewidth=0.8,
            capsize=1.4,
            capthick=0.7,
            zorder=3,
        )
        axis_b.scatter(row["pub_cliff"], row_index, s=17, **PUBLISHED_MARKER, zorder=5)
        axis_b.scatter(row["rep_cliff"], row_index, s=13, **REPRODUCED_MARKER, zorder=6)
    axis_b.set_xlabel(panel_b_x_label)

    axis_b.tick_params(axis="y", length=0)
    axis_a.set_ylabel(y_label)
    axis_a.xaxis.set_major_locator(MultipleLocator(0.25))
    axis_a.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2f}"))
    axis_a.text(-0.26, 1.01, "a", transform=axis_a.transAxes, **PANEL_LETTER)
    axis_b.xaxis.set_major_locator(MultipleLocator(0.25))
    axis_b.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2f}"))
    axis_b.text(-0.06, 1.01, "b", transform=axis_b.transAxes, **PANEL_LETTER)

    legend_handles = [
        Line2D([], [], **PUBLISHED_LEGEND_MARKER),
        Line2D([], [], **REPRODUCED_LEGEND_MARKER),
    ]
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    lowest_content = min(
        panel.get_tightbbox(renderer).transformed(figure.transFigure.inverted()).y0
        for panel in figure.axes
        if panel.get_visible()
    )
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, lowest_content - 0.012),
        ncol=len(legend_handles),
        handletextpad=0.35,
        columnspacing=1.6,
    )
    writer.save(figure, name, title, caption)


def f10_dumbbell_folds(
    writer: FigureWriter, table: pd.DataFrame, order: list[str], folds: pd.DataFrame
) -> None:
    """F10: as F09, with the five reproduced fold scores instead of the SD."""
    name = "dataset_errors_by_fold"
    title = "Per-dataset published versus reproduced RMSE and RMSE_cliff"
    caption = (
        "As dataset_errors_mean_sd, with the five individual reproduced fold scores (hollow diamonds, "
        "offset below each row) instead of the SD."
    )
    figure_size = (5.2, 5.6)
    panel_a_x_label = "RMSE"
    panel_b_x_label = r"RMSE$_{\mathrm{cliff}}$"
    y_label = "Dataset"
    fold_legend_label = "Reproduced folds (n = 5)"
    fold_row_offset = 0.22

    dataset_labels = [table.loc[dataset, "abbreviation"] for dataset in order]
    row_count = len(order)
    figure, (axis_a, axis_b) = plt.subplots(
        1, 2, figsize=figure_size, sharey=True, gridspec_kw={"wspace": 0.12}
    )

    axis_a.set_ylim(row_count - 0.5, -0.5)
    axis_a.set_yticks(range(row_count))
    axis_a.set_yticklabels(dataset_labels)
    axis_a.tick_params(axis="y", length=2.5)
    for row_index in range(0, row_count, 2):
        axis_a.axhspan(
            row_index - 0.5, row_index + 0.5, color=BAND_COLOUR, zorder=0, lw=0
        )
    for row_index, dataset in enumerate(order):
        row = table.loc[dataset]
        axis_a.plot(
            [row["pub_rmse"], row["rep_rmse"]],
            [row_index, row_index],
            color="#A8A8A8",
            lw=0.9,
            zorder=2,
            solid_capstyle="butt",
        )
        fold_values = folds.loc[folds["dataset"] == dataset, "rmse"]
        axis_a.scatter(
            fold_values,
            np.full(len(fold_values), row_index + fold_row_offset),
            s=7,
            marker="D",
            facecolor="white",
            edgecolor=REPRODUCED["colour"],
            linewidth=0.6,
            zorder=3,
        )
        axis_a.scatter(row["pub_rmse"], row_index, s=17, **PUBLISHED_MARKER, zorder=5)
        axis_a.scatter(row["rep_rmse"], row_index, s=13, **REPRODUCED_MARKER, zorder=6)
    axis_a.set_xlabel(panel_a_x_label)

    axis_b.set_ylim(row_count - 0.5, -0.5)
    axis_b.set_yticks(range(row_count))
    axis_b.set_yticklabels(dataset_labels)
    axis_b.tick_params(axis="y", length=2.5)
    for row_index in range(0, row_count, 2):
        axis_b.axhspan(
            row_index - 0.5, row_index + 0.5, color=BAND_COLOUR, zorder=0, lw=0
        )
    for row_index, dataset in enumerate(order):
        row = table.loc[dataset]
        axis_b.plot(
            [row["pub_cliff"], row["rep_cliff"]],
            [row_index, row_index],
            color="#A8A8A8",
            lw=0.9,
            zorder=2,
            solid_capstyle="butt",
        )
        fold_values = folds.loc[folds["dataset"] == dataset, "rmse_cliff"]
        axis_b.scatter(
            fold_values,
            np.full(len(fold_values), row_index + fold_row_offset),
            s=7,
            marker="D",
            facecolor="white",
            edgecolor=REPRODUCED["colour"],
            linewidth=0.6,
            zorder=3,
        )
        axis_b.scatter(row["pub_cliff"], row_index, s=17, **PUBLISHED_MARKER, zorder=5)
        axis_b.scatter(row["rep_cliff"], row_index, s=13, **REPRODUCED_MARKER, zorder=6)
    axis_b.set_xlabel(panel_b_x_label)

    axis_b.tick_params(axis="y", length=0)
    axis_a.set_ylabel(y_label)
    axis_a.xaxis.set_major_locator(MultipleLocator(0.25))
    axis_a.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2f}"))
    axis_a.text(-0.26, 1.01, "a", transform=axis_a.transAxes, **PANEL_LETTER)
    axis_b.xaxis.set_major_locator(MultipleLocator(0.25))
    axis_b.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2f}"))
    axis_b.text(-0.06, 1.01, "b", transform=axis_b.transAxes, **PANEL_LETTER)

    legend_handles = [
        Line2D([], [], **PUBLISHED_LEGEND_MARKER),
        Line2D([], [], **REPRODUCED_LEGEND_MARKER),
        Line2D(
            [],
            [],
            ls="none",
            marker="D",
            markersize=3.6,
            markerfacecolor="white",
            markeredgecolor=REPRODUCED["colour"],
            label=fold_legend_label,
        ),
    ]
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    lowest_content = min(
        panel.get_tightbbox(renderer).transformed(figure.transFigure.inverted()).y0
        for panel in figure.axes
        if panel.get_visible()
    )
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, lowest_content - 0.012),
        ncol=len(legend_handles),
        handletextpad=0.35,
        columnspacing=1.6,
    )
    writer.save(figure, name, title, caption)


def f11_fig5d_overlay(writer: FigureWriter, table: pd.DataFrame) -> None:
    """F11: corrected Fig. 5d, RMSE_cliff − RMSE against training-set size."""
    name = "cliff_error_difference_training_size"
    title = "Fig. 5d adaptation: RMSE_cliff − RMSE versus training-set size"
    caption = (
        "Adaptation of corrected Fig. 5d. RMSE_cliff − RMSE against the number of "
        "molecules in the published training partition (before the internal fold "
        "division). Vertical grey segments join the two values of a dataset. The "
        "relation is descriptive, not causal."
    )
    figure_size = (3.4, 2.6)
    x_label = "Training molecules"
    y_label = r"RMSE$_{\mathrm{cliff}}$ − RMSE"
    x_limits = (300, 3100)

    reproduced_low = table["rep_delta"] - table["rep_delta_sd"]
    reproduced_high = table["rep_delta"] + table["rep_delta_sd"]
    y_lower = min(
        -0.2, np.floor(min(table["pub_delta"].min(), reproduced_low.min()) * 10) / 10
    )
    y_upper = max(
        0.6, np.ceil(max(table["pub_delta"].max(), reproduced_high.max()) * 10) / 10
    )

    figure, axis = plt.subplots(figsize=figure_size)
    axis.axhline(0, color=GUIDE_COLOUR, lw=0.7, ls=(0, (3, 3)), zorder=1)
    for _, row in table.iterrows():
        axis.plot(
            [row["n_train"]] * 2,
            [row["pub_delta"], row["rep_delta"]],
            color="#A8A8A8",
            lw=0.6,
            zorder=2,
        )
    axis.errorbar(
        table["n_train"],
        table["rep_delta"],
        yerr=table["rep_delta_sd"],
        fmt="none",
        ecolor=REPRODUCED["colour"],
        elinewidth=0.7,
        alpha=0.65,
        zorder=3,
    )
    axis.scatter(
        table["n_train"], table["pub_delta"], s=18, **PUBLISHED_MARKER, zorder=4
    )
    axis.scatter(
        table["n_train"], table["rep_delta"], s=14, **REPRODUCED_MARKER, zorder=4
    )
    axis.set_xlim(*x_limits)
    axis.set_ylim(y_lower, y_upper)
    axis.xaxis.set_major_locator(MultipleLocator(500))
    axis.yaxis.set_major_locator(MultipleLocator(0.2))
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)

    legend_handles = [
        Line2D([], [], **PUBLISHED_LEGEND_MARKER),
        Line2D([], [], **REPRODUCED_LEGEND_MARKER),
    ]
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    lowest_content = min(
        panel.get_tightbbox(renderer).transformed(figure.transFigure.inverted()).y0
        for panel in figure.axes
        if panel.get_visible()
    )
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, lowest_content - 0.012),
        ncol=len(legend_handles),
        handletextpad=0.35,
        columnspacing=1.6,
    )
    writer.save(figure, name, title, caption)


def f12_fig5d_side_by_side(writer: FigureWriter, table: pd.DataFrame) -> None:
    """F12: as F11, with published and reproduced GCN in separate panels."""
    name = "cliff_error_difference_training_size_comparison"
    title = "Fig. 5d adaptation: RMSE_cliff − RMSE versus training-set size"
    caption = "As cliff_error_difference_training_size with (a) published and (b) reproduced GCN in separate panels."
    figure_size = (5.6, 2.5)
    panel_a_title = "GCN · published"
    panel_a_x_label = "Training molecules"
    panel_b_title = "GCN · reproduction"
    panel_b_x_label = "Training molecules"
    y_label = r"RMSE$_{\mathrm{cliff}}$ − RMSE"
    x_limits = (300, 3100)

    reproduced_low = table["rep_delta"] - table["rep_delta_sd"]
    reproduced_high = table["rep_delta"] + table["rep_delta_sd"]
    y_lower = min(
        -0.2, np.floor(min(table["pub_delta"].min(), reproduced_low.min()) * 10) / 10
    )
    y_upper = max(
        0.6, np.ceil(max(table["pub_delta"].max(), reproduced_high.max()) * 10) / 10
    )

    figure, (axis_a, axis_b) = plt.subplots(
        1, 2, figsize=figure_size, sharey=True, gridspec_kw={"wspace": 0.12}
    )

    axis_a.axhline(0, color=GUIDE_COLOUR, lw=0.7, ls=(0, (3, 3)), zorder=1)
    axis_a.scatter(
        table["n_train"], table["pub_delta"], s=20, **PUBLISHED_MARKER, zorder=4
    )
    axis_a.text(
        0.04,
        0.95,
        panel_a_title,
        transform=axis_a.transAxes,
        fontweight="bold",
        va="top",
    )
    axis_a.set_xlim(*x_limits)
    axis_a.set_ylim(y_lower, y_upper)
    axis_a.xaxis.set_major_locator(MultipleLocator(500))
    axis_a.yaxis.set_major_locator(MultipleLocator(0.2))
    axis_a.set_xlabel(panel_a_x_label)

    axis_b.axhline(0, color=GUIDE_COLOUR, lw=0.7, ls=(0, (3, 3)), zorder=1)
    axis_b.errorbar(
        table["n_train"],
        table["rep_delta"],
        yerr=table["rep_delta_sd"],
        fmt="none",
        ecolor=REPRODUCED["colour"],
        elinewidth=0.8,
        alpha=0.8,
        zorder=3,
    )
    axis_b.scatter(
        table["n_train"], table["rep_delta"], s=16, **REPRODUCED_MARKER, zorder=4
    )
    axis_b.text(
        0.04,
        0.95,
        panel_b_title,
        transform=axis_b.transAxes,
        fontweight="bold",
        va="top",
    )
    axis_b.set_xlim(*x_limits)
    axis_b.set_ylim(y_lower, y_upper)
    axis_b.xaxis.set_major_locator(MultipleLocator(500))
    axis_b.yaxis.set_major_locator(MultipleLocator(0.2))
    axis_b.set_xlabel(panel_b_x_label)

    axis_a.set_ylabel(y_label)
    axis_a.text(-0.22, 1.04, "a", transform=axis_a.transAxes, **PANEL_LETTER)
    axis_b.text(-0.06, 1.04, "b", transform=axis_b.transAxes, **PANEL_LETTER)

    legend_handles = [
        Line2D([], [], **PUBLISHED_LEGEND_MARKER),
        Line2D([], [], **REPRODUCED_LEGEND_MARKER),
    ]
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    lowest_content = min(
        panel.get_tightbbox(renderer).transformed(figure.transFigure.inverted()).y0
        for panel in figure.axes
        if panel.get_visible()
    )
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, lowest_content - 0.012),
        ncol=len(legend_handles),
        handletextpad=0.35,
        columnspacing=1.6,
    )
    writer.save(figure, name, title, caption)


def f13_agreement_scatter(writer: FigureWriter, table: pd.DataFrame) -> None:
    """F13: reproduced against published dataset means for the three metrics."""
    name = "published_reproduced_agreement"
    title = "Reproduced versus published dataset means"
    caption = (
        "Reproduced against published dataset means for (a) RMSE, (b) RMSE_cliff and "
        "(c) RMSE_cliff − RMSE; dashed line: identity. Summary statistics are "
        "descriptive over the 30 datasets; no paired fold test or equivalence claim is "
        "made, because published fold scores are unavailable."
    )
    figure_size = (MAX_FIGURE_WIDTH_CM / 2.54, 2.9)
    panel_a_x_label = "Published RMSE"
    panel_a_y_label = "Reproduced RMSE"
    panel_b_x_label = r"Published RMSE$_{\mathrm{cliff}}$"
    panel_b_y_label = r"Reproduced RMSE$_{\mathrm{cliff}}$"
    panel_c_x_label = r"Published (RMSE$_{\mathrm{cliff}}$ − RMSE)"
    panel_c_y_label = r"Reproduced (RMSE$_{\mathrm{cliff}}$ − RMSE)"
    identity_legend_label = "Identity (equal scores)"

    figure, (axis_a, axis_b, axis_c) = plt.subplots(
        1, 3, figsize=figure_size, gridspec_kw={"wspace": 0.45}
    )
    figure.subplots_adjust(left=0.10, right=0.98)

    panel_a_values = np.concatenate(
        [
            table["pub_rmse"],
            table["rep_rmse"] - table["rep_rmse_sd"],
            table["rep_rmse"] + table["rep_rmse_sd"],
        ]
    )
    panel_a_lower = np.floor((panel_a_values.min() - 0.02) / 0.1) * 0.1
    panel_a_upper = np.ceil((panel_a_values.max() + 0.02) / 0.1) * 0.1
    axis_a.plot(
        [panel_a_lower, panel_a_upper], [panel_a_lower, panel_a_upper], **IDENTITY_LINE
    )
    axis_a.errorbar(
        table["pub_rmse"],
        table["rep_rmse"],
        yerr=table["rep_rmse_sd"],
        fmt="none",
        ecolor=REPRODUCED["colour"],
        elinewidth=0.7,
        alpha=0.6,
        zorder=2,
    )
    axis_a.scatter(
        table["pub_rmse"], table["rep_rmse"], s=15, **REPRODUCED_MARKER, zorder=4
    )
    axis_a.set_xlim(panel_a_lower, panel_a_upper)
    axis_a.set_ylim(panel_a_lower, panel_a_upper)
    axis_a.set_aspect("equal", adjustable="box")
    axis_a.xaxis.set_major_locator(MultipleLocator(0.2))
    axis_a.yaxis.set_major_locator(MultipleLocator(0.1))
    axis_a.set_xlabel(panel_a_x_label, color=PUBLISHED["colour"])
    axis_a.set_ylabel(panel_a_y_label, color=REPRODUCED["colour"])
    panel_a_differences = table["rep_rmse"] - table["pub_rmse"]
    panel_a_correlation = np.corrcoef(table["pub_rmse"], table["rep_rmse"])[0, 1]
    panel_a_statistics = figure.text(
        0,
        0,
        "Mean difference {}\nMean |difference| {:.3f}\nPearson r {:.2f}\n"
        "n = {} datasets".format(
            f"{panel_a_differences.mean():+.3f}".replace("-", "−"),
            panel_a_differences.abs().mean(),
            panel_a_correlation,
            len(table),
        ),
        va="top",
        ha="center",
        fontsize=7,
        linespacing=1.35,
        color="#404040",
    )

    panel_b_values = np.concatenate(
        [
            table["pub_cliff"],
            table["rep_cliff"] - table["rep_cliff_sd"],
            table["rep_cliff"] + table["rep_cliff_sd"],
        ]
    )
    panel_b_lower = np.floor((panel_b_values.min() - 0.02) / 0.1) * 0.1
    panel_b_upper = np.ceil((panel_b_values.max() + 0.02) / 0.1) * 0.1
    axis_b.plot(
        [panel_b_lower, panel_b_upper], [panel_b_lower, panel_b_upper], **IDENTITY_LINE
    )
    axis_b.errorbar(
        table["pub_cliff"],
        table["rep_cliff"],
        yerr=table["rep_cliff_sd"],
        fmt="none",
        ecolor=REPRODUCED["colour"],
        elinewidth=0.7,
        alpha=0.6,
        zorder=2,
    )
    axis_b.scatter(
        table["pub_cliff"], table["rep_cliff"], s=15, **REPRODUCED_MARKER, zorder=4
    )
    axis_b.set_xlim(panel_b_lower, panel_b_upper)
    axis_b.set_ylim(panel_b_lower, panel_b_upper)
    axis_b.set_aspect("equal", adjustable="box")
    axis_b.xaxis.set_major_locator(MultipleLocator(0.1))
    axis_b.yaxis.set_major_locator(MultipleLocator(0.1))
    axis_b.set_xlabel(panel_b_x_label, color=PUBLISHED["colour"])
    axis_b.set_ylabel(panel_b_y_label, color=REPRODUCED["colour"])
    panel_b_differences = table["rep_cliff"] - table["pub_cliff"]
    panel_b_correlation = np.corrcoef(table["pub_cliff"], table["rep_cliff"])[0, 1]
    panel_b_statistics = figure.text(
        0,
        0,
        "Mean difference {}\nMean |difference| {:.3f}\nPearson r {:.2f}\n"
        "n = {} datasets".format(
            f"{panel_b_differences.mean():+.3f}".replace("-", "−"),
            panel_b_differences.abs().mean(),
            panel_b_correlation,
            len(table),
        ),
        va="top",
        ha="center",
        fontsize=7,
        linespacing=1.35,
        color="#404040",
    )

    panel_c_values = np.concatenate(
        [
            table["pub_delta"],
            table["rep_delta"] - table["rep_delta_sd"],
            table["rep_delta"] + table["rep_delta_sd"],
        ]
    )
    panel_c_lower = np.floor((panel_c_values.min() - 0.02) / 0.1) * 0.1
    panel_c_upper = np.ceil((panel_c_values.max() + 0.02) / 0.1) * 0.1
    axis_c.plot(
        [panel_c_lower, panel_c_upper], [panel_c_lower, panel_c_upper], **IDENTITY_LINE
    )
    axis_c.errorbar(
        table["pub_delta"],
        table["rep_delta"],
        yerr=table["rep_delta_sd"],
        fmt="none",
        ecolor=REPRODUCED["colour"],
        elinewidth=0.7,
        alpha=0.6,
        zorder=2,
    )
    axis_c.scatter(
        table["pub_delta"], table["rep_delta"], s=15, **REPRODUCED_MARKER, zorder=4
    )
    axis_c.set_xlim(panel_c_lower, panel_c_upper)
    axis_c.set_ylim(panel_c_lower, panel_c_upper)
    axis_c.set_aspect("equal", adjustable="box")
    axis_c.xaxis.set_major_locator(MultipleLocator(0.2))
    axis_c.yaxis.set_major_locator(MultipleLocator(0.1))
    axis_c.set_xlabel(panel_c_x_label, color=PUBLISHED["colour"])
    axis_c.set_ylabel(panel_c_y_label, color=REPRODUCED["colour"])
    panel_c_differences = table["rep_delta"] - table["pub_delta"]
    panel_c_correlation = np.corrcoef(table["pub_delta"], table["rep_delta"])[0, 1]
    panel_c_statistics = figure.text(
        0,
        0,
        "Mean difference {}\nMean |difference| {:.3f}\nPearson r {:.2f}\n"
        "n = {} datasets".format(
            f"{panel_c_differences.mean():+.3f}".replace("-", "−"),
            panel_c_differences.abs().mean(),
            panel_c_correlation,
            len(table),
        ),
        va="top",
        ha="center",
        fontsize=7,
        linespacing=1.35,
        color="#404040",
    )

    axis_a.text(0, 1.08, "a", transform=axis_a.transAxes, **PANEL_LETTER)
    axis_b.text(0, 1.08, "b", transform=axis_b.transAxes, **PANEL_LETTER)
    axis_c.text(0, 1.08, "c", transform=axis_c.transAxes, **PANEL_LETTER)

    legend_handles = [
        Line2D([], [], **REPRODUCED_LEGEND_MARKER),
        Line2D([], [], color=INK, ls=(0, (4, 3)), lw=0.8, label=identity_legend_label),
    ]
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    lowest_content = min(
        panel.get_tightbbox(renderer).transformed(figure.transFigure.inverted()).y0
        for panel in figure.axes
        if panel.get_visible()
    )
    for axis, statistics_text in zip(
        (axis_a, axis_b, axis_c),
        (panel_a_statistics, panel_b_statistics, panel_c_statistics),
        strict=True,
    ):
        position = axis.get_position()
        statistics_text.set_position(
            (position.x0 + position.width / 2, lowest_content - 0.03)
        )
    figure.canvas.draw()
    lowest_statistics = min(
        text.get_window_extent(renderer).transformed(figure.transFigure.inverted()).y0
        for text in (panel_a_statistics, panel_b_statistics, panel_c_statistics)
    )
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, lowest_statistics - 0.015),
        ncol=len(legend_handles),
        handletextpad=0.35,
        columnspacing=1.6,
    )
    writer.save(figure, name, title, caption)


def f14_bland_altman(writer: FigureWriter, table: pd.DataFrame) -> None:
    """F14: Bland–Altman plots of reproduced against published RMSE and RMSE_cliff."""
    name = "bland_altman_agreement"
    title = "Bland–Altman agreement plots (descriptive)"
    caption = (
        "Bland–Altman plots for (a) RMSE and (b) RMSE_cliff: reproduced minus "
        "published against their mean. Solid line: mean difference; dashed lines: mean "
        "± 1.96 SD of the 30 dataset differences (descriptive limits of agreement "
        "across datasets, not fold-level uncertainty)."
    )
    figure_size = (MAX_FIGURE_WIDTH_CM / 2.54, 2.5)
    panel_a_x_label = "Mean of published and reproduced RMSE"
    panel_a_y_label = "Reproduced − published RMSE"
    panel_b_x_label = r"Mean of published and reproduced RMSE$_{\mathrm{cliff}}$"
    panel_b_y_label = r"Reproduced − published RMSE$_{\mathrm{cliff}}$"

    figure, (axis_a, axis_b) = plt.subplots(
        1, 2, figsize=figure_size, gridspec_kw={"wspace": 0.75}
    )

    panel_a_means = (table["pub_rmse"] + table["rep_rmse"]) / 2
    panel_a_differences = table["rep_rmse"] - table["pub_rmse"]
    panel_a_bias = panel_a_differences.mean()
    panel_a_spread = panel_a_differences.std(ddof=1)
    panel_a_lower_limit = panel_a_bias - 1.96 * panel_a_spread
    panel_a_upper_limit = panel_a_bias + 1.96 * panel_a_spread
    axis_a.axhline(0, color=GUIDE_COLOUR, lw=0.7, ls=(0, (1, 2)), zorder=1)
    axis_a.axhline(panel_a_bias, color=INK, lw=0.8, zorder=1)
    axis_a.axhline(panel_a_lower_limit, color=INK, lw=0.7, ls=(0, (4, 3)), zorder=1)
    axis_a.axhline(panel_a_upper_limit, color=INK, lw=0.7, ls=(0, (4, 3)), zorder=1)
    axis_a.scatter(
        panel_a_means, panel_a_differences, s=13, **DIFFERENCE_MARKER, zorder=4
    )
    axis_a.set_xlabel(panel_a_x_label)
    axis_a.set_ylabel(panel_a_y_label)
    panel_a_right_edge = axis_a.get_xlim()[1]
    for value, label in (
        (panel_a_bias, "Mean"),
        (panel_a_upper_limit, "+1.96 SD"),
        (panel_a_lower_limit, "−1.96 SD"),
    ):
        axis_a.text(
            panel_a_right_edge,
            value,
            " {} {}".format(label, f"{value:+.3f}".replace("-", "−")),
            va="center",
            ha="left",
            fontsize=7,
        )
    axis_a.set_xlim(right=panel_a_right_edge)
    axis_a.yaxis.set_major_locator(MultipleLocator(0.05))

    panel_b_means = (table["pub_cliff"] + table["rep_cliff"]) / 2
    panel_b_differences = table["rep_cliff"] - table["pub_cliff"]
    panel_b_bias = panel_b_differences.mean()
    panel_b_spread = panel_b_differences.std(ddof=1)
    panel_b_lower_limit = panel_b_bias - 1.96 * panel_b_spread
    panel_b_upper_limit = panel_b_bias + 1.96 * panel_b_spread
    axis_b.axhline(0, color=GUIDE_COLOUR, lw=0.7, ls=(0, (1, 2)), zorder=1)
    axis_b.axhline(panel_b_bias, color=INK, lw=0.8, zorder=1)
    axis_b.axhline(panel_b_lower_limit, color=INK, lw=0.7, ls=(0, (4, 3)), zorder=1)
    axis_b.axhline(panel_b_upper_limit, color=INK, lw=0.7, ls=(0, (4, 3)), zorder=1)
    axis_b.scatter(
        panel_b_means, panel_b_differences, s=13, **DIFFERENCE_MARKER, zorder=4
    )
    axis_b.set_xlabel(panel_b_x_label)
    axis_b.set_ylabel(panel_b_y_label)
    panel_b_right_edge = axis_b.get_xlim()[1]
    for value, label in (
        (panel_b_bias, "Mean"),
        (panel_b_upper_limit, "+1.96 SD"),
        (panel_b_lower_limit, "−1.96 SD"),
    ):
        axis_b.text(
            panel_b_right_edge,
            value,
            " {} {}".format(label, f"{value:+.3f}".replace("-", "−")),
            va="center",
            ha="left",
            fontsize=7,
        )
    axis_b.set_xlim(right=panel_b_right_edge)
    axis_b.yaxis.set_major_locator(MultipleLocator(0.05))

    y_limit = max(
        abs(np.array(axis_a.get_ylim())).max(), abs(np.array(axis_b.get_ylim())).max()
    )
    axis_a.set_ylim(-y_limit, y_limit)
    axis_a.text(-0.26, 1.04, "a", transform=axis_a.transAxes, **PANEL_LETTER)
    axis_b.set_ylim(-y_limit, y_limit)
    axis_b.text(-0.26, 1.04, "b", transform=axis_b.transAxes, **PANEL_LETTER)
    writer.save(figure, name, title, caption)


def f15_differences_by_dataset(
    writer: FigureWriter, table: pd.DataFrame, order: list[str]
) -> None:
    """F15: reproduced minus published for each dataset and the three metrics."""
    name = "published_reproduced_differences_by_dataset"
    title = "Reproduced minus published, per dataset"
    caption = (
        "Reproduced minus published (a) RMSE, (b) RMSE_cliff and (c) RMSE_cliff − RMSE "
        "per dataset. Positive values indicate larger reproduced errors in (a, b), "
        "and a larger cliff-versus-whole-test error gap in (c). Dark horizontal "
        "segments connect each difference estimate to zero as a visual guide. "
        "Light-grey capped bars show ± "
        "reproduced fold SD around the difference from the fixed published value, not "
        "the uncertainty of a difference between two estimated means."
    )
    figure_size = (6.2, 5.4)
    panel_a_x_label = "Reproduced − published\nRMSE"
    panel_b_x_label = "Reproduced − published\n" + r"RMSE$_{\mathrm{cliff}}$"
    panel_c_x_label = "Reproduced − published\n" + r"(RMSE$_{\mathrm{cliff}}$ − RMSE)"
    y_label = "Dataset"
    difference_legend_label = "Difference (line to zero)"
    standard_deviation_legend_label = "± reproduced fold SD"

    dataset_labels = [table.loc[dataset, "abbreviation"] for dataset in order]
    row_count = len(order)
    x_limit = max(
        (table["diff_rmse"].abs() + table["rep_rmse_sd"]).max(),
        (table["diff_cliff"].abs() + table["rep_cliff_sd"]).max(),
        (table["diff_delta"].abs() + table["rep_delta_sd"]).max(),
    )
    x_limit = np.ceil((x_limit + 0.01) * 20) / 20
    figure, (axis_a, axis_b, axis_c) = plt.subplots(
        1, 3, figsize=figure_size, sharey=True, gridspec_kw={"wspace": 0.12}
    )

    axis_a.set_ylim(row_count - 0.5, -0.5)
    axis_a.set_yticks(range(row_count))
    axis_a.set_yticklabels(dataset_labels)
    axis_a.tick_params(axis="y", length=2.5)
    for row_index in range(0, row_count, 2):
        axis_a.axhspan(
            row_index - 0.5, row_index + 0.5, color=BAND_COLOUR, zorder=0, lw=0
        )
    axis_a.axvline(0, color=INK, lw=0.7, zorder=1)
    for row_index, dataset in enumerate(order):
        difference = table.loc[dataset, "diff_rmse"]
        axis_a.plot(
            [0, difference],
            [row_index, row_index],
            color=DIFFERENCE["colour"],
            lw=0.9,
            zorder=2,
        )
        axis_a.errorbar(
            difference,
            row_index,
            xerr=table.loc[dataset, "rep_rmse_sd"],
            fmt="none",
            ecolor="#A0A0A0",
            elinewidth=0.7,
            capsize=1.3,
            capthick=0.6,
            zorder=2,
        )
    axis_a.scatter(
        table.loc[order, "diff_rmse"],
        range(row_count),
        s=12,
        **DIFFERENCE_MARKER,
        zorder=4,
    )
    axis_a.set_xlim(-x_limit, x_limit)
    axis_a.xaxis.set_major_locator(MultipleLocator(0.1))
    axis_a.set_xlabel(panel_a_x_label)
    axis_a.text(-0.34, 1.01, "a", transform=axis_a.transAxes, **PANEL_LETTER)

    axis_b.set_ylim(row_count - 0.5, -0.5)
    axis_b.set_yticks(range(row_count))
    axis_b.set_yticklabels(dataset_labels)
    axis_b.tick_params(axis="y", length=2.5)
    for row_index in range(0, row_count, 2):
        axis_b.axhspan(
            row_index - 0.5, row_index + 0.5, color=BAND_COLOUR, zorder=0, lw=0
        )
    axis_b.axvline(0, color=INK, lw=0.7, zorder=1)
    for row_index, dataset in enumerate(order):
        difference = table.loc[dataset, "diff_cliff"]
        axis_b.plot(
            [0, difference],
            [row_index, row_index],
            color=DIFFERENCE["colour"],
            lw=0.9,
            zorder=2,
        )
        axis_b.errorbar(
            difference,
            row_index,
            xerr=table.loc[dataset, "rep_cliff_sd"],
            fmt="none",
            ecolor="#A0A0A0",
            elinewidth=0.7,
            capsize=1.3,
            capthick=0.6,
            zorder=2,
        )
    axis_b.scatter(
        table.loc[order, "diff_cliff"],
        range(row_count),
        s=12,
        **DIFFERENCE_MARKER,
        zorder=4,
    )
    axis_b.set_xlim(-x_limit, x_limit)
    axis_b.xaxis.set_major_locator(MultipleLocator(0.1))
    axis_b.set_xlabel(panel_b_x_label)
    axis_b.text(-0.08, 1.01, "b", transform=axis_b.transAxes, **PANEL_LETTER)
    axis_b.tick_params(axis="y", length=0)

    axis_c.set_ylim(row_count - 0.5, -0.5)
    axis_c.set_yticks(range(row_count))
    axis_c.set_yticklabels(dataset_labels)
    axis_c.tick_params(axis="y", length=2.5)
    for row_index in range(0, row_count, 2):
        axis_c.axhspan(
            row_index - 0.5, row_index + 0.5, color=BAND_COLOUR, zorder=0, lw=0
        )
    axis_c.axvline(0, color=INK, lw=0.7, zorder=1)
    for row_index, dataset in enumerate(order):
        difference = table.loc[dataset, "diff_delta"]
        axis_c.plot(
            [0, difference],
            [row_index, row_index],
            color=DIFFERENCE["colour"],
            lw=0.9,
            zorder=2,
        )
        axis_c.errorbar(
            difference,
            row_index,
            xerr=table.loc[dataset, "rep_delta_sd"],
            fmt="none",
            ecolor="#A0A0A0",
            elinewidth=0.7,
            capsize=1.3,
            capthick=0.6,
            zorder=2,
        )
    axis_c.scatter(
        table.loc[order, "diff_delta"],
        range(row_count),
        s=12,
        **DIFFERENCE_MARKER,
        zorder=4,
    )
    axis_c.set_xlim(-x_limit, x_limit)
    axis_c.xaxis.set_major_locator(MultipleLocator(0.1))
    axis_c.set_xlabel(panel_c_x_label)
    axis_c.text(-0.08, 1.01, "c", transform=axis_c.transAxes, **PANEL_LETTER)
    axis_c.tick_params(axis="y", length=0)

    axis_a.set_ylabel(y_label)

    legend_handles = [
        Line2D(
            [],
            [],
            **{
                **DIFFERENCE_LEGEND_MARKER,
                "ls": "-",
                "color": DIFFERENCE["colour"],
                "lw": 0.9,
                "label": difference_legend_label,
            },
        ),
        Line2D([], [], color="#A0A0A0", lw=0.8, label=standard_deviation_legend_label),
    ]
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    lowest_content = min(
        panel.get_tightbbox(renderer).transformed(figure.transFigure.inverted()).y0
        for panel in figure.axes
        if panel.get_visible()
    )
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, lowest_content - 0.012),
        ncol=len(legend_handles),
        handletextpad=0.35,
        columnspacing=1.6,
    )
    writer.save(figure, name, title, caption)


def f16_difference_distribution(writer: FigureWriter, table: pd.DataFrame) -> None:
    """F16: distribution over datasets of reproduced minus published scores."""
    name = "published_reproduced_difference_distribution"
    title = "Distribution of reproduced minus published differences"
    caption = (
        "Distribution over the 30 datasets of reproduced minus published scores. "
        "Violins are kernel density estimates; the vertical bar marks the median."
    )
    figure_size = (3.6, 2.2)
    rows = [
        ("diff_rmse", "RMSE"),
        ("diff_cliff", r"RMSE$_{\mathrm{cliff}}$"),
        ("diff_delta", r"RMSE$_{\mathrm{cliff}}$ − RMSE"),
    ]
    x_label = "Reproduced − published (log units)"

    figure, axis = plt.subplots(figsize=figure_size)
    axis.axvline(0, color=INK, lw=0.7, zorder=1)
    for row_index, (column, _) in enumerate(rows):
        differences = table[column]
        violin = axis.violinplot(
            differences,
            positions=[row_index],
            orientation="horizontal",
            widths=0.75,
            showextrema=False,
        )
        for body in violin["bodies"]:
            body.set_facecolor("#DADADA")
            body.set_edgecolor("#8C8C8C")
            body.set_linewidth(0.6)
            body.set_alpha(1)
        jitter = np.random.default_rng(30 + row_index).uniform(
            -0.09, 0.09, len(differences)
        )
        axis.scatter(
            differences,
            row_index + jitter,
            s=9,
            **DIFFERENCE_MARKER,
            zorder=4,
            alpha=0.9,
        )
        axis.plot(
            [differences.median()] * 2,
            [row_index - 0.3, row_index + 0.3],
            color=INK,
            lw=1.1,
            zorder=6,
        )
    axis.set_yticks(range(len(rows)))
    axis.set_yticklabels([label for _, label in rows])
    axis.set_ylim(len(rows) - 0.4, -0.6)
    axis.set_xlabel(x_label)
    writer.save(figure, name, title, caption)


def f17_strategy_ranking_context(
    writer: FigureWriter, table: pd.DataFrame, published: pd.DataFrame
) -> None:
    """F17: where the reproduced GCN sits among the published strategies."""
    name = "published_strategy_ranking_context"
    title = "Where the reproduced GCN sits among published strategies"
    caption = (
        "(a) Mean RMSE_cliff over the 30 datasets for the 24 published strategies, "
        "with the reproduced GCN added. (b) Rank of GCN among the 24 strategies on "
        "each dataset when the published GCN is used and when it is replaced by the "
        "reproduction; lines join datasets. Other strategies are external published "
        "results and were not retrained."
    )
    figure_size = (MAX_FIGURE_WIDTH_CM / 2.54, 4.15)
    published_gcn_strategy = "GCN (graph)"
    reproduced_gcn_row_label = "GCN (graph) · reproduction"
    panel_a_x_label = r"Mean RMSE$_{\mathrm{cliff}}$ over 30 datasets"
    panel_b_x_tick_labels = ["Published", "Reproduction"]
    panel_b_y_label = (
        "Rank of GCN among 24 strategies\n" + r"(1 = lowest RMSE$_{\mathrm{cliff}}$)"
    )
    panel_b_y_ticks = list(range(1, 25))

    strategy_means = published.groupby("strategy")["cliff_rmse"].mean()
    strategy_rows = [
        (strategy, mean, "published" if strategy == published_gcn_strategy else "other")
        for strategy, mean in strategy_means.items()
    ]
    strategy_rows.append(
        (reproduced_gcn_row_label, table["rep_cliff"].mean(), "reproduced")
    )
    strategy_rows.sort(key=lambda strategy_row: strategy_row[1])
    figure, (axis_a, axis_b) = plt.subplots(
        1,
        2,
        figsize=figure_size,
        gridspec_kw={"width_ratios": [1.35, 1], "wspace": 0.55},
    )

    figure.subplots_adjust(left=0.20, right=0.95)

    row_count = len(strategy_rows)
    axis_a.set_ylim(row_count - 0.5, -0.5)
    axis_a.set_yticks(range(row_count))
    axis_a.set_yticklabels([strategy for strategy, _, _ in strategy_rows])
    axis_a.tick_params(axis="y", length=2.5)
    for row_index in range(0, row_count, 2):
        axis_a.axhspan(
            row_index - 0.5, row_index + 0.5, color=BAND_COLOUR, zorder=0, lw=0
        )
    for row_index, (_, mean, kind) in enumerate(strategy_rows):
        if kind == "other":
            axis_a.plot(
                [0, mean], [row_index, row_index], color="#D0D0D0", lw=0.9, zorder=2
            )
            axis_a.scatter(
                mean,
                row_index,
                s=15,
                color=OTHER_STRATEGIES["edge"],
                edgecolor="none",
                zorder=3,
            )
        else:
            style = PUBLISHED if kind == "published" else REPRODUCED
            marker_style = (
                PUBLISHED_MARKER if kind == "published" else REPRODUCED_MARKER
            )
            axis_a.plot(
                [0, mean],
                [row_index, row_index],
                color=style["colour"],
                lw=1.1,
                zorder=2,
            )
            axis_a.scatter(mean, row_index, s=22, **marker_style, zorder=4)
            axis_a.get_yticklabels()[row_index].set_color(style["colour"])
            axis_a.get_yticklabels()[row_index].set_fontweight("bold")
    axis_a.set_xlim(0.6, 1.25)
    axis_a.xaxis.set_major_locator(MultipleLocator(0.1))
    axis_a.set_xlabel(panel_a_x_label)
    axis_a.text(-0.62, 1.04, "a", transform=axis_a.transAxes, **PANEL_LETTER)

    other_strategies = published[~published["is_gcn"]]
    published_ranks = []
    reproduced_ranks = []
    for dataset in table.index:
        other_scores = other_strategies.loc[
            other_strategies["dataset"] == dataset, "cliff_rmse"
        ].to_numpy()
        published_ranks.append(
            1 + int((other_scores < table.loc[dataset, "pub_cliff"]).sum())
        )
        reproduced_ranks.append(
            1 + int((other_scores < table.loc[dataset, "rep_cliff"]).sum())
        )
    published_ranks = np.array(published_ranks)
    reproduced_ranks = np.array(reproduced_ranks)
    jitter = np.random.default_rng(51).uniform(-0.10, 0.10, len(table))
    for index in range(len(table)):
        axis_b.plot(
            [jitter[index], 1 + jitter[index]],
            [published_ranks[index], reproduced_ranks[index]],
            color="#B5B5B5",
            lw=0.5,
            zorder=1,
        )
    axis_b.boxplot(
        [published_ranks],
        positions=[0],
        widths=0.45,
        whis=1.5,
        zorder=3,
        **PUBLISHED_BOX,
    )
    axis_b.boxplot(
        [reproduced_ranks],
        positions=[1],
        widths=0.45,
        whis=1.5,
        zorder=3,
        **REPRODUCED_BOX,
    )
    axis_b.scatter(jitter, published_ranks, s=13, **PUBLISHED_MARKER, zorder=4)
    axis_b.scatter(1 + jitter, reproduced_ranks, s=10, **REPRODUCED_MARKER, zorder=4)
    axis_b.set_ylim(24.8, 0.2)
    axis_b.set_yticks(panel_b_y_ticks)
    axis_b.set_xticks([0, 1])
    axis_b.set_xticklabels(panel_b_x_tick_labels)
    axis_b.set_xlim(-0.55, 1.55)
    axis_b.set_ylabel(panel_b_y_label)
    axis_b.text(-0.42, 1.04, "b", transform=axis_b.transAxes, **PANEL_LETTER)

    legend_handles = [
        Line2D([], [], **PUBLISHED_LEGEND_MARKER),
        Line2D([], [], **REPRODUCED_LEGEND_MARKER),
        Line2D([], [], **OTHER_STRATEGIES_LEGEND_MARKER),
    ]
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    lowest_content = min(
        panel.get_tightbbox(renderer).transformed(figure.transFigure.inverted()).y0
        for panel in figure.axes
        if panel.get_visible()
    )
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, lowest_content - 0.012),
        ncol=len(legend_handles),
        handletextpad=0.35,
        columnspacing=1.6,
    )
    writer.save(figure, name, title, caption)


def f18_fig5bc_examples(
    writer: FigureWriter, table: pd.DataFrame, published: pd.DataFrame
) -> None:
    """F18: corrected Fig. 5b,c, all published strategies on 5-HT1A and CLK4."""
    name = "activity_cliff_example_datasets"
    title = "Fig. 5b,c adaptation: 5-HT1A and CLK4 with all published strategies"
    caption = (
        "Adaptation of corrected Fig. 5b,c: RMSE_cliff against RMSE of all published "
        "strategies (grey) on (a) 5-HT1A and (b) CLK4, the examples discussed in the "
        "paper, with the published and reproduced GCN highlighted. Reproduced bars: ± "
        "fold SD."
    )
    figure_size = (5.0, 2.6)
    panel_a_dataset = "CHEMBL214_Ki"
    panel_a_x_label = "RMSE"
    panel_a_y_label = r"RMSE$_{\mathrm{cliff}}$"
    panel_b_dataset = "CHEMBL4203_Ki"
    panel_b_x_label = "RMSE"
    panel_b_y_label = r"RMSE$_{\mathrm{cliff}}$"
    axis_lower = 0.25
    axis_upper = 1.75

    figure, (axis_a, axis_b) = plt.subplots(
        1, 2, figsize=figure_size, gridspec_kw={"wspace": 0.38}
    )

    panel_a_strategies = published[published["dataset"] == panel_a_dataset]
    panel_a_others = panel_a_strategies[~panel_a_strategies["is_gcn"]]
    panel_a_row = table.loc[panel_a_dataset]
    axis_a.plot([axis_lower, axis_upper], [axis_lower, axis_upper], **IDENTITY_LINE)
    axis_a.plot(
        [axis_lower, axis_upper],
        [axis_lower - 0.5, axis_upper - 0.5],
        **HALF_LOG_UNIT_LINE,
    )
    axis_a.plot(
        [axis_lower, axis_upper],
        [axis_lower + 0.5, axis_upper + 0.5],
        **HALF_LOG_UNIT_LINE,
    )
    axis_a.set_xlim(axis_lower, axis_upper)
    axis_a.set_ylim(axis_lower, axis_upper)
    axis_a.set_aspect("equal", adjustable="box")
    axis_a.scatter(
        panel_a_others["rmse"],
        panel_a_others["cliff_rmse"],
        s=16,
        color=OTHER_STRATEGIES["edge"],
        alpha=0.55,
        edgecolor="none",
        zorder=2,
    )
    axis_a.errorbar(
        panel_a_row["rep_rmse"],
        panel_a_row["rep_cliff"],
        xerr=panel_a_row["rep_rmse_sd"],
        yerr=panel_a_row["rep_cliff_sd"],
        fmt="none",
        ecolor=REPRODUCED["colour"],
        elinewidth=0.7,
        zorder=3,
    )
    axis_a.scatter(
        panel_a_row["pub_rmse"],
        panel_a_row["pub_cliff"],
        s=24,
        **PUBLISHED_MARKER,
        zorder=4,
    )
    axis_a.scatter(
        panel_a_row["rep_rmse"],
        panel_a_row["rep_cliff"],
        s=19,
        **REPRODUCED_MARKER,
        zorder=4,
    )
    axis_a.text(
        0.05,
        0.95,
        panel_a_row["abbreviation"],
        transform=axis_a.transAxes,
        fontweight="bold",
        va="top",
    )
    axis_a.xaxis.set_major_locator(MultipleLocator(0.25))
    axis_a.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2f}"))
    axis_a.yaxis.set_major_locator(MultipleLocator(0.25))
    axis_a.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2f}"))
    axis_a.tick_params(labelsize=7)
    axis_a.set_xlabel(panel_a_x_label)
    axis_a.set_ylabel(panel_a_y_label)
    axis_a.text(-0.3, 1.04, "a", transform=axis_a.transAxes, **PANEL_LETTER)

    panel_b_strategies = published[published["dataset"] == panel_b_dataset]
    panel_b_others = panel_b_strategies[~panel_b_strategies["is_gcn"]]
    panel_b_row = table.loc[panel_b_dataset]
    axis_b.plot([axis_lower, axis_upper], [axis_lower, axis_upper], **IDENTITY_LINE)
    axis_b.plot(
        [axis_lower, axis_upper],
        [axis_lower - 0.5, axis_upper - 0.5],
        **HALF_LOG_UNIT_LINE,
    )
    axis_b.plot(
        [axis_lower, axis_upper],
        [axis_lower + 0.5, axis_upper + 0.5],
        **HALF_LOG_UNIT_LINE,
    )
    axis_b.set_xlim(axis_lower, axis_upper)
    axis_b.set_ylim(axis_lower, axis_upper)
    axis_b.set_aspect("equal", adjustable="box")
    axis_b.scatter(
        panel_b_others["rmse"],
        panel_b_others["cliff_rmse"],
        s=16,
        color=OTHER_STRATEGIES["edge"],
        alpha=0.55,
        edgecolor="none",
        zorder=2,
    )
    axis_b.errorbar(
        panel_b_row["rep_rmse"],
        panel_b_row["rep_cliff"],
        xerr=panel_b_row["rep_rmse_sd"],
        yerr=panel_b_row["rep_cliff_sd"],
        fmt="none",
        ecolor=REPRODUCED["colour"],
        elinewidth=0.7,
        zorder=3,
    )
    axis_b.scatter(
        panel_b_row["pub_rmse"],
        panel_b_row["pub_cliff"],
        s=24,
        **PUBLISHED_MARKER,
        zorder=4,
    )
    axis_b.scatter(
        panel_b_row["rep_rmse"],
        panel_b_row["rep_cliff"],
        s=19,
        **REPRODUCED_MARKER,
        zorder=4,
    )
    axis_b.text(
        0.05,
        0.95,
        panel_b_row["abbreviation"],
        transform=axis_b.transAxes,
        fontweight="bold",
        va="top",
    )
    axis_b.xaxis.set_major_locator(MultipleLocator(0.25))
    axis_b.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2f}"))
    axis_b.yaxis.set_major_locator(MultipleLocator(0.25))
    axis_b.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2f}"))
    axis_b.tick_params(labelsize=7)
    axis_b.set_xlabel(panel_b_x_label)
    axis_b.set_ylabel(panel_b_y_label)
    axis_b.text(-0.3, 1.04, "b", transform=axis_b.transAxes, **PANEL_LETTER)

    legend_handles = [
        Line2D([], [], **PUBLISHED_LEGEND_MARKER),
        Line2D([], [], **REPRODUCED_LEGEND_MARKER),
        Line2D([], [], **OTHER_STRATEGIES_LEGEND_MARKER),
    ]
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    lowest_content = min(
        panel.get_tightbbox(renderer).transformed(figure.transFigure.inverted()).y0
        for panel in figure.axes
        if panel.get_visible()
    )
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, lowest_content - 0.012),
        ncol=len(legend_handles),
        handletextpad=0.35,
        columnspacing=1.6,
    )
    writer.save(figure, name, title, caption)


def f19_composite_figure4_layout(
    writer: FigureWriter, table: pd.DataFrame, published: pd.DataFrame
) -> None:
    """F19: full-width composite in the layout of corrected Figure 4."""
    name = "activity_cliff_overview"
    title = "Composite in the layout of corrected Figure 4"
    caption = (
        "Composite in the layout of corrected Fig. 4: (a) Fig. 4a with the "
        "reproduction inserted; (b) reproduced against published RMSE_cliff; (c, d) "
        "Fig. 4c for published and reproduced GCN; (e) RMSE_cliff − RMSE against "
        "training-set size for both sources. Error bars in (b, e) show ±1 SD "
        "across the five reproduced folds; in (e), the SD is calculated from "
        "each fold's RMSE_cliff − RMSE. In (a), publ. denotes the published "
        "GCN and repr. denotes the reproduced GCN."
    )
    figure_size = (MAX_FIGURE_WIDTH_CM / 2.54, 5.4)
    panel_a_methods = [
        ("AFP", "GRAPH"),
        ("MPNN", "GRAPH"),
        ("GAT", "GRAPH"),
        ("GCN", "GRAPH"),
        ("Transformer", "TOKENS"),
        ("CNN", "SMILES"),
        ("LSTM", "SMILES"),
        ("MLP", "ECFP"),
        ("SVM", "ECFP"),
    ]
    panel_a_method_tick_labels = {"Transformer": "Trans-\nformer"}
    panel_a_published_gcn_tick_label = "GCN\npubl."
    panel_a_reproduced_gcn_tick_label = "GCN\nrepr."
    panel_a_x_label = "Algorithm"
    panel_a_y_label = r"RMSE$_{\mathrm{cliff}}$"
    panel_b_x_label = r"Published RMSE$_{\mathrm{cliff}}$"
    panel_b_y_label = r"Reproduced RMSE$_{\mathrm{cliff}}$"
    panel_b_label_size = 7
    panel_c_title = "GCN · published"
    panel_c_x_label = "RMSE"
    panel_c_y_label = r"RMSE$_{\mathrm{cliff}}$"
    panel_d_title = "GCN · reproduction"
    panel_d_x_label = "RMSE"
    panel_d_y_label = r"RMSE$_{\mathrm{cliff}}$"
    panel_e_x_label = "Training molecules"
    panel_e_y_label = r"RMSE$_{\mathrm{cliff}}$ − RMSE"

    figure = plt.figure(figsize=figure_size)
    figure.subplots_adjust(left=0.10, right=0.98)
    outer_grid = figure.add_gridspec(2, 1, height_ratios=[1, 0.7], hspace=0.40)
    top_grid = outer_grid[0].subgridspec(1, 2, width_ratios=[2.1, 1], wspace=0.42)
    bottom_grid = outer_grid[1].subgridspec(1, 3, wspace=0.55)

    axis_a = figure.add_subplot(top_grid[0])
    tick_positions = []
    tick_labels = []
    position = 0
    for algorithm, descriptor in panel_a_methods:
        is_method = (published["algorithm"] == algorithm) & (
            published["descriptor"] == descriptor
        )
        values = (
            published[is_method].set_index("dataset")["cliff_rmse"].loc[table.index]
        )
        jitter = np.random.default_rng(11 + position).uniform(-0.12, 0.12, len(values))
        if algorithm == "GCN":
            axis_a.axvspan(
                position - 0.45, position + 1.45, color=BAND_COLOUR, zorder=0, lw=0
            )
            axis_a.boxplot(
                [values],
                positions=[position],
                widths=0.58,
                whis=1.5,
                zorder=3,
                **PUBLISHED_BOX,
            )
            axis_a.scatter(
                position + jitter, values, s=10, **PUBLISHED_MARKER, zorder=4
            )
            tick_positions.append(position)
            tick_labels.append(panel_a_published_gcn_tick_label)
            position += 1
            axis_a.boxplot(
                [table["rep_cliff"]],
                positions=[position],
                widths=0.58,
                whis=1.5,
                zorder=3,
                **REPRODUCED_BOX,
            )
            axis_a.scatter(
                position + jitter,
                table["rep_cliff"],
                s=8,
                **REPRODUCED_MARKER,
                zorder=4,
            )
            tick_positions.append(position)
            tick_labels.append(panel_a_reproduced_gcn_tick_label)
        else:
            axis_a.boxplot(
                [values],
                positions=[position],
                widths=0.58,
                whis=1.5,
                zorder=3,
                **OTHER_STRATEGIES_BOX,
            )
            axis_a.scatter(
                position + jitter,
                values,
                s=7,
                color=OTHER_STRATEGIES["edge"],
                edgecolor="none",
                alpha=0.75,
                zorder=4,
            )
            tick_positions.append(position)
            tick_labels.append(panel_a_method_tick_labels.get(algorithm, algorithm))
        position += 1
    axis_a.set_xticks(tick_positions)
    axis_a.set_xticklabels(tick_labels, fontsize=7)
    axis_a.set_xlim(-0.7, position - 0.3)
    axis_a.set_ylim(0.25, 1.8)
    axis_a.yaxis.set_major_locator(MultipleLocator(0.25))
    axis_a.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2f}"))
    axis_a.set_ylabel(panel_a_y_label)
    axis_a.set_xlabel(panel_a_x_label)
    axis_a.text(-0.09, 1.04, "a", transform=axis_a.transAxes, **PANEL_LETTER)

    axis_b = figure.add_subplot(top_grid[1])
    panel_b_values = np.concatenate(
        [
            table["pub_cliff"],
            table["rep_cliff"] - table["rep_cliff_sd"],
            table["rep_cliff"] + table["rep_cliff_sd"],
        ]
    )
    panel_b_lower = np.floor((panel_b_values.min() - 0.02) / 0.1) * 0.1
    panel_b_upper = np.ceil((panel_b_values.max() + 0.02) / 0.1) * 0.1
    axis_b.plot(
        [panel_b_lower, panel_b_upper], [panel_b_lower, panel_b_upper], **IDENTITY_LINE
    )
    axis_b.errorbar(
        table["pub_cliff"],
        table["rep_cliff"],
        yerr=table["rep_cliff_sd"],
        fmt="none",
        ecolor=REPRODUCED["colour"],
        elinewidth=0.7,
        alpha=0.6,
        zorder=2,
    )
    axis_b.scatter(
        table["pub_cliff"], table["rep_cliff"], s=15, **REPRODUCED_MARKER, zorder=4
    )
    axis_b.set_xlim(panel_b_lower, panel_b_upper)
    axis_b.set_ylim(panel_b_lower, panel_b_upper)
    axis_b.set_aspect("equal", adjustable="box")
    axis_b.xaxis.set_major_locator(MultipleLocator(0.1))
    axis_b.yaxis.set_major_locator(MultipleLocator(0.1))
    axis_b.set_xlabel(
        panel_b_x_label, color=PUBLISHED["colour"], fontsize=panel_b_label_size
    )
    axis_b.set_ylabel(
        panel_b_y_label, color=REPRODUCED["colour"], fontsize=panel_b_label_size
    )
    panel_b_differences = table["rep_cliff"] - table["pub_cliff"]
    panel_b_correlation = np.corrcoef(table["pub_cliff"], table["rep_cliff"])[0, 1]
    axis_b.text(
        0.5,
        -0.3,
        "Mean difference {}\nMean |difference| {:.3f}\nPearson r {:.2f}\n"
        "n = {} datasets".format(
            f"{panel_b_differences.mean():+.3f}".replace("-", "−"),
            panel_b_differences.abs().mean(),
            panel_b_correlation,
            len(table),
        ),
        transform=axis_b.transAxes,
        va="top",
        ha="center",
        fontsize=7,
        linespacing=1.15,
        color="#404040",
    )
    axis_b.text(-0.42, 1.04, "b", transform=axis_b.transAxes, **PANEL_LETTER)

    all_values = np.concatenate(
        [
            published["rmse"],
            published["cliff_rmse"],
            table["rep_rmse"],
            table["rep_cliff"],
        ]
    )
    scatter_lower = min(0.25, np.floor(all_values.min() / 0.25) * 0.25)
    scatter_upper = max(1.75, np.ceil(all_values.max() / 0.25) * 0.25)
    other_strategies = published[~published["is_gcn"]]

    axis_c = figure.add_subplot(bottom_grid[0])
    axis_c.scatter(
        other_strategies["rmse"],
        other_strategies["cliff_rmse"],
        s=11,
        **OTHER_STRATEGIES_CLOUD,
        zorder=2,
    )
    axis_c.plot(
        [scatter_lower, scatter_upper], [scatter_lower, scatter_upper], **IDENTITY_LINE
    )
    axis_c.plot(
        [scatter_lower, scatter_upper],
        [scatter_lower - 0.5, scatter_upper - 0.5],
        **HALF_LOG_UNIT_LINE,
    )
    axis_c.plot(
        [scatter_lower, scatter_upper],
        [scatter_lower + 0.5, scatter_upper + 0.5],
        **HALF_LOG_UNIT_LINE,
    )
    axis_c.set_xlim(scatter_lower, scatter_upper)
    axis_c.set_ylim(scatter_lower, scatter_upper)
    axis_c.set_aspect("equal", adjustable="box")
    axis_c.scatter(
        table["pub_rmse"], table["pub_cliff"], s=20, **PUBLISHED_MARKER, zorder=4
    )
    axis_c.text(
        0.05,
        0.95,
        panel_c_title,
        transform=axis_c.transAxes,
        fontweight="bold",
        fontsize=7.2,
        va="top",
    )
    axis_c.xaxis.set_major_locator(MultipleLocator(0.5))
    axis_c.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2f}"))
    axis_c.yaxis.set_major_locator(MultipleLocator(0.5))
    axis_c.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2f}"))
    axis_c.xaxis.set_minor_locator(MultipleLocator(0.25))
    axis_c.yaxis.set_minor_locator(MultipleLocator(0.25))
    axis_c.tick_params(which="minor", length=1.8, width=0.6)
    axis_c.set_xlabel(panel_c_x_label)
    axis_c.set_ylabel(panel_c_y_label)
    axis_c.text(-0.36, 1.04, "c", transform=axis_c.transAxes, **PANEL_LETTER)

    axis_d = figure.add_subplot(bottom_grid[1])
    axis_d.scatter(
        other_strategies["rmse"],
        other_strategies["cliff_rmse"],
        s=11,
        **OTHER_STRATEGIES_CLOUD,
        zorder=2,
    )
    axis_d.plot(
        [scatter_lower, scatter_upper], [scatter_lower, scatter_upper], **IDENTITY_LINE
    )
    axis_d.plot(
        [scatter_lower, scatter_upper],
        [scatter_lower - 0.5, scatter_upper - 0.5],
        **HALF_LOG_UNIT_LINE,
    )
    axis_d.plot(
        [scatter_lower, scatter_upper],
        [scatter_lower + 0.5, scatter_upper + 0.5],
        **HALF_LOG_UNIT_LINE,
    )
    axis_d.set_xlim(scatter_lower, scatter_upper)
    axis_d.set_ylim(scatter_lower, scatter_upper)
    axis_d.set_aspect("equal", adjustable="box")
    axis_d.scatter(
        table["rep_rmse"], table["rep_cliff"], s=16, **REPRODUCED_MARKER, zorder=4
    )
    axis_d.text(
        0.05,
        0.95,
        panel_d_title,
        transform=axis_d.transAxes,
        fontweight="bold",
        fontsize=7.2,
        va="top",
    )
    axis_d.xaxis.set_major_locator(MultipleLocator(0.5))
    axis_d.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2f}"))
    axis_d.yaxis.set_major_locator(MultipleLocator(0.5))
    axis_d.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2f}"))
    axis_d.xaxis.set_minor_locator(MultipleLocator(0.25))
    axis_d.yaxis.set_minor_locator(MultipleLocator(0.25))
    axis_d.tick_params(which="minor", length=1.8, width=0.6)
    axis_d.set_xlabel(panel_d_x_label)
    axis_d.set_ylabel(panel_d_y_label)
    axis_d.text(-0.36, 1.04, "d", transform=axis_d.transAxes, **PANEL_LETTER)

    axis_e = figure.add_subplot(bottom_grid[2])
    axis_e.axhline(0, color=GUIDE_COLOUR, lw=0.7, ls=(0, (3, 3)), zorder=1)
    for _, row in table.iterrows():
        axis_e.plot(
            [row["n_train"]] * 2,
            [row["pub_delta"], row["rep_delta"]],
            color="#A8A8A8",
            lw=0.6,
            zorder=2,
        )
    axis_e.errorbar(
        table["n_train"],
        table["rep_delta"],
        yerr=table["rep_delta_sd"],
        fmt="none",
        ecolor=REPRODUCED["colour"],
        elinewidth=0.6,
        alpha=0.6,
        zorder=3,
    )
    axis_e.scatter(
        table["n_train"], table["pub_delta"], s=13, **PUBLISHED_MARKER, zorder=4
    )
    axis_e.scatter(
        table["n_train"], table["rep_delta"], s=10, **REPRODUCED_MARKER, zorder=4
    )
    axis_e.set_xlim(300, 3100)
    axis_e.xaxis.set_major_locator(MultipleLocator(1000))
    axis_e.yaxis.set_major_locator(MultipleLocator(0.2))
    axis_e.set_xlabel(panel_e_x_label)
    axis_e.set_ylabel(panel_e_y_label)
    axis_e.set_box_aspect(1)
    axis_e.text(-0.42, 1.04, "e", transform=axis_e.transAxes, **PANEL_LETTER)

    legend_handles = [
        Line2D([], [], **PUBLISHED_LEGEND_MARKER),
        Line2D([], [], **REPRODUCED_LEGEND_MARKER),
        Line2D([], [], **OTHER_STRATEGIES_LEGEND_MARKER),
    ]
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    lowest_content = min(
        panel.get_tightbbox(renderer).transformed(figure.transFigure.inverted()).y0
        for panel in figure.axes
        if panel.get_visible()
    )
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, lowest_content - 0.012),
        ncol=len(legend_handles),
        handletextpad=0.35,
        columnspacing=1.6,
    )
    writer.save(figure, name, title, caption)


def f20_composite_figure5_layout(
    writer: FigureWriter, table: pd.DataFrame, published: pd.DataFrame, order: list[str]
) -> None:
    """F20: full-width composite in the layout of corrected Figure 5."""
    name = "dataset_comparison_overview"
    title = "Composite in the layout of corrected Figure 5"
    caption = (
        "Composite in the layout of corrected Fig. 5: (a) Fig. 5a with the published "
        "GCN and the reproduction; (b, c) 5-HT1A and CLK4; (d) Fig. 5d for both "
        "sources; (e) reproduced against published RMSE_cliff − RMSE. The r "
        "column reports the Pearson correlation between RMSE and RMSE_cliff "
        "across published strategies for each dataset. Error bars show ±1 SD "
        "across the five reproduced folds; those for RMSE_cliff − RMSE use "
        "the per-fold differences."
    )
    figure_size = (MAX_FIGURE_WIDTH_CM / 2.54, 8.0)
    panel_a_x_label = r"RMSE$_{\mathrm{cliff}}$ − RMSE"
    panel_a_y_label = "Dataset"
    panel_a_r_column_title = r"$r_{\mathrm{RMSE,\ RMSE_{cliff}}}$"
    panel_a_marker_offset = 0.17
    panel_b_dataset = "CHEMBL214_Ki"
    panel_b_x_label = "RMSE"
    panel_b_y_label = r"RMSE$_{\mathrm{cliff}}$"
    panel_c_dataset = "CHEMBL4203_Ki"
    panel_c_x_label = "RMSE"
    panel_c_y_label = r"RMSE$_{\mathrm{cliff}}$"
    panel_d_x_label = "Training molecules"
    panel_d_y_label = r"RMSE$_{\mathrm{cliff}}$ − RMSE"
    panel_e_x_label = r"Published (RMSE$_{\mathrm{cliff}}$ − RMSE)"
    panel_e_y_label = r"Reproduced (RMSE$_{\mathrm{cliff}}$ − RMSE)"
    panel_e_label_size = 7
    range_legend_label = "Published range (a)"
    example_lower = 0.25
    example_upper = 1.75

    figure = plt.figure(figsize=figure_size)
    figure.subplots_adjust(left=0.13, right=0.98, top=0.96, bottom=0.10)
    outer_grid = figure.add_gridspec(
        2, 2, width_ratios=[1.1, 1], height_ratios=[2, 1], wspace=0.4, hspace=0.24
    )
    left_grid = outer_grid[0, 0].subgridspec(1, 2, width_ratios=[1, 0.17], wspace=0.05)

    axis_a = figure.add_subplot(left_grid[0])
    row_count = len(order)
    axis_a.set_ylim(row_count - 0.5, -0.5)
    axis_a.set_yticks(range(row_count))
    axis_a.set_yticklabels([table.loc[dataset, "abbreviation"] for dataset in order])
    axis_a.tick_params(axis="y", length=2.5)
    for row_index in range(0, row_count, 2):
        axis_a.axhspan(
            row_index - 0.5, row_index + 0.5, color=BAND_COLOUR, zorder=0, lw=0
        )
    axis_a.axvline(0, color=GUIDE_COLOUR, lw=0.7, ls=(0, (3, 3)), zorder=1)
    for row_index, dataset in enumerate(order):
        row = table.loc[dataset]
        other_deltas = published[
            (published["dataset"] == dataset) & ~published["is_gcn"]
        ]["delta"]
        axis_a.boxplot(
            [other_deltas],
            positions=[row_index],
            widths=0.62,
            orientation="horizontal",
            whis=(0, 100),
            zorder=2,
            **OTHER_STRATEGIES_BOX,
        )
        axis_a.plot(
            [row["pub_delta"], row["rep_delta"]],
            [row_index - panel_a_marker_offset, row_index + panel_a_marker_offset],
            color="#9A9A9A",
            lw=0.5,
            zorder=3,
        )
        axis_a.errorbar(
            row["rep_delta"],
            row_index + panel_a_marker_offset,
            xerr=row["rep_delta_sd"],
            fmt="none",
            ecolor=REPRODUCED["colour"],
            elinewidth=0.8,
            capsize=1.4,
            capthick=0.7,
            zorder=4,
        )
        axis_a.scatter(
            row["pub_delta"],
            row_index - panel_a_marker_offset,
            s=16,
            **PUBLISHED_MARKER,
            zorder=5,
        )
        axis_a.scatter(
            row["rep_delta"],
            row_index + panel_a_marker_offset,
            s=12,
            **REPRODUCED_MARKER,
            zorder=5,
        )
    axis_a.set_xlabel(panel_a_x_label)
    axis_a.xaxis.set_major_locator(MultipleLocator(0.2))
    axis_a.set_ylabel(panel_a_y_label)
    axis_a.tick_params(axis="y", labelsize=7)
    axis_a.text(-0.3, 1.0, "a", transform=axis_a.transAxes, **PANEL_LETTER)

    r_axis = figure.add_subplot(left_grid[1])
    r_values = table.loc[order, "r_published_methods"].to_numpy()
    r_shades = 0.18 + 0.57 * (1 - np.clip((r_values - 0.3) / 0.7, 0, 1))
    r_colours = plt.get_cmap("Greys")(r_shades)
    for row_index, (r_value, r_colour) in enumerate(zip(r_values, r_colours)):
        r_axis.add_patch(
            plt.Rectangle(
                (0, row_index - 0.5),
                1,
                1,
                facecolor=r_colour,
                edgecolor="white",
                lw=0.6,
            )
        )
        r_text = (
            f"{r_value:.2f}".rstrip("0").rstrip(".")
            if r_value >= 0.995
            else f"{r_value:.2f}"
        )
        r_axis.text(
            0.5,
            row_index,
            r_text,
            ha="center",
            va="center",
            fontsize=7,
            color="white" if r_shades[row_index] > 0.58 else INK,
        )
    r_axis.set_xlim(0, 1)
    r_axis.set_ylim(row_count - 0.5, -0.5)
    r_axis.axis("off")
    r_axis.text(0.5, -0.9, panel_a_r_column_title, ha="center", va="bottom", fontsize=7)

    right_grid = outer_grid[0, 1].subgridspec(2, 1, hspace=0.58)

    axis_b = figure.add_subplot(right_grid[0])
    panel_b_strategies = published[published["dataset"] == panel_b_dataset]
    panel_b_others = panel_b_strategies[~panel_b_strategies["is_gcn"]]
    panel_b_row = table.loc[panel_b_dataset]
    axis_b.plot(
        [example_lower, example_upper], [example_lower, example_upper], **IDENTITY_LINE
    )
    axis_b.plot(
        [example_lower, example_upper],
        [example_lower - 0.5, example_upper - 0.5],
        **HALF_LOG_UNIT_LINE,
    )
    axis_b.plot(
        [example_lower, example_upper],
        [example_lower + 0.5, example_upper + 0.5],
        **HALF_LOG_UNIT_LINE,
    )
    axis_b.set_xlim(example_lower, example_upper)
    axis_b.set_ylim(example_lower, example_upper)
    axis_b.set_aspect("equal", adjustable="box")
    axis_b.scatter(
        panel_b_others["rmse"],
        panel_b_others["cliff_rmse"],
        s=16,
        color=OTHER_STRATEGIES["edge"],
        alpha=0.55,
        edgecolor="none",
        zorder=2,
    )
    axis_b.errorbar(
        panel_b_row["rep_rmse"],
        panel_b_row["rep_cliff"],
        xerr=panel_b_row["rep_rmse_sd"],
        yerr=panel_b_row["rep_cliff_sd"],
        fmt="none",
        ecolor=REPRODUCED["colour"],
        elinewidth=0.7,
        zorder=3,
    )
    axis_b.scatter(
        panel_b_row["pub_rmse"],
        panel_b_row["pub_cliff"],
        s=24,
        **PUBLISHED_MARKER,
        zorder=4,
    )
    axis_b.scatter(
        panel_b_row["rep_rmse"],
        panel_b_row["rep_cliff"],
        s=19,
        **REPRODUCED_MARKER,
        zorder=4,
    )
    axis_b.text(
        0.05,
        0.95,
        panel_b_row["abbreviation"],
        transform=axis_b.transAxes,
        fontweight="bold",
        va="top",
    )
    axis_b.xaxis.set_major_locator(MultipleLocator(0.25))
    axis_b.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2f}"))
    axis_b.yaxis.set_major_locator(MultipleLocator(0.25))
    axis_b.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2f}"))
    axis_b.tick_params(labelsize=7)
    axis_b.set_xlabel(panel_b_x_label)
    axis_b.set_ylabel(panel_b_y_label)
    axis_b.text(-0.36, 1.04, "b", transform=axis_b.transAxes, **PANEL_LETTER)

    axis_c = figure.add_subplot(right_grid[1])
    panel_c_strategies = published[published["dataset"] == panel_c_dataset]
    panel_c_others = panel_c_strategies[~panel_c_strategies["is_gcn"]]
    panel_c_row = table.loc[panel_c_dataset]
    axis_c.plot(
        [example_lower, example_upper], [example_lower, example_upper], **IDENTITY_LINE
    )
    axis_c.plot(
        [example_lower, example_upper],
        [example_lower - 0.5, example_upper - 0.5],
        **HALF_LOG_UNIT_LINE,
    )
    axis_c.plot(
        [example_lower, example_upper],
        [example_lower + 0.5, example_upper + 0.5],
        **HALF_LOG_UNIT_LINE,
    )
    axis_c.set_xlim(example_lower, example_upper)
    axis_c.set_ylim(example_lower, example_upper)
    axis_c.set_aspect("equal", adjustable="box")
    axis_c.scatter(
        panel_c_others["rmse"],
        panel_c_others["cliff_rmse"],
        s=16,
        color=OTHER_STRATEGIES["edge"],
        alpha=0.55,
        edgecolor="none",
        zorder=2,
    )
    axis_c.errorbar(
        panel_c_row["rep_rmse"],
        panel_c_row["rep_cliff"],
        xerr=panel_c_row["rep_rmse_sd"],
        yerr=panel_c_row["rep_cliff_sd"],
        fmt="none",
        ecolor=REPRODUCED["colour"],
        elinewidth=0.7,
        zorder=3,
    )
    axis_c.scatter(
        panel_c_row["pub_rmse"],
        panel_c_row["pub_cliff"],
        s=24,
        **PUBLISHED_MARKER,
        zorder=4,
    )
    axis_c.scatter(
        panel_c_row["rep_rmse"],
        panel_c_row["rep_cliff"],
        s=19,
        **REPRODUCED_MARKER,
        zorder=4,
    )
    axis_c.text(
        0.05,
        0.95,
        panel_c_row["abbreviation"],
        transform=axis_c.transAxes,
        fontweight="bold",
        va="top",
    )
    axis_c.xaxis.set_major_locator(MultipleLocator(0.25))
    axis_c.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2f}"))
    axis_c.yaxis.set_major_locator(MultipleLocator(0.25))
    axis_c.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2f}"))
    axis_c.tick_params(labelsize=7)
    axis_c.set_xlabel(panel_c_x_label)
    axis_c.set_ylabel(panel_c_y_label)
    axis_c.text(-0.36, 1.04, "c", transform=axis_c.transAxes, **PANEL_LETTER)

    axis_d = figure.add_subplot(outer_grid[1, 0])
    axis_d.axhline(0, color=GUIDE_COLOUR, lw=0.7, ls=(0, (3, 3)), zorder=1)
    for _, row in table.iterrows():
        axis_d.plot(
            [row["n_train"]] * 2,
            [row["pub_delta"], row["rep_delta"]],
            color="#A8A8A8",
            lw=0.6,
            zorder=2,
        )
    axis_d.errorbar(
        table["n_train"],
        table["rep_delta"],
        yerr=table["rep_delta_sd"],
        fmt="none",
        ecolor=REPRODUCED["colour"],
        elinewidth=0.6,
        alpha=0.6,
        zorder=3,
    )
    axis_d.scatter(
        table["n_train"], table["pub_delta"], s=15, **PUBLISHED_MARKER, zorder=4
    )
    axis_d.scatter(
        table["n_train"], table["rep_delta"], s=12, **REPRODUCED_MARKER, zorder=4
    )
    axis_d.set_xlim(300, 3100)
    axis_d.xaxis.set_major_locator(MultipleLocator(500))
    axis_d.yaxis.set_major_locator(MultipleLocator(0.2))
    axis_d.set_xlabel(panel_d_x_label)
    axis_d.set_ylabel(panel_d_y_label)
    axis_d.text(-0.2, 1.04, "d", transform=axis_d.transAxes, **PANEL_LETTER)

    axis_e = figure.add_subplot(outer_grid[1, 1])
    panel_e_values = np.concatenate(
        [
            table["pub_delta"],
            table["rep_delta"] - table["rep_delta_sd"],
            table["rep_delta"] + table["rep_delta_sd"],
        ]
    )
    panel_e_lower = np.floor((panel_e_values.min() - 0.02) / 0.1) * 0.1
    panel_e_upper = np.ceil((panel_e_values.max() + 0.02) / 0.1) * 0.1
    axis_e.plot(
        [panel_e_lower, panel_e_upper], [panel_e_lower, panel_e_upper], **IDENTITY_LINE
    )
    axis_e.errorbar(
        table["pub_delta"],
        table["rep_delta"],
        yerr=table["rep_delta_sd"],
        fmt="none",
        ecolor=REPRODUCED["colour"],
        elinewidth=0.7,
        alpha=0.6,
        zorder=2,
    )
    axis_e.scatter(
        table["pub_delta"], table["rep_delta"], s=15, **REPRODUCED_MARKER, zorder=4
    )
    axis_e.set_xlim(panel_e_lower, panel_e_upper)
    axis_e.set_ylim(panel_e_lower, panel_e_upper)
    axis_e.set_aspect("equal", adjustable="box")
    axis_e.xaxis.set_major_locator(MultipleLocator(0.1))
    axis_e.yaxis.set_major_locator(MultipleLocator(0.1))
    axis_e.set_xlabel(
        panel_e_x_label, color=PUBLISHED["colour"], fontsize=panel_e_label_size
    )
    axis_e.set_ylabel(
        panel_e_y_label, color=REPRODUCED["colour"], fontsize=panel_e_label_size
    )
    panel_e_differences = table["rep_delta"] - table["pub_delta"]
    panel_e_correlation = np.corrcoef(table["pub_delta"], table["rep_delta"])[0, 1]
    axis_e.text(
        0.97,
        0.04,
        "Mean difference {}\nMean |difference| {:.3f}\nPearson r {:.2f}\n"
        "n = {} datasets".format(
            f"{panel_e_differences.mean():+.3f}".replace("-", "−"),
            panel_e_differences.abs().mean(),
            panel_e_correlation,
            len(table),
        ),
        transform=axis_e.transAxes,
        va="bottom",
        ha="right",
        fontsize=7,
        linespacing=1.35,
        color="#404040",
    )
    axis_e.text(-0.42, 1.04, "e", transform=axis_e.transAxes, **PANEL_LETTER)

    legend_handles = [
        Line2D([], [], **PUBLISHED_LEGEND_MARKER),
        Line2D([], [], **REPRODUCED_LEGEND_MARKER),
        Line2D([], [], **OTHER_STRATEGIES_LEGEND_MARKER),
        Patch(
            facecolor=OTHER_STRATEGIES["fill"],
            edgecolor=OTHER_STRATEGIES["edge"],
            lw=0.6,
            label=range_legend_label,
        ),
    ]
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    lowest_content = min(
        panel.get_tightbbox(renderer).transformed(figure.transFigure.inverted()).y0
        for panel in figure.axes
        if panel.get_visible()
    )
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, lowest_content - 0.012),
        ncol=2,
        handletextpad=0.35,
        columnspacing=1.6,
    )
    writer.save(figure, name, title, caption)


def f21_readme_banner(
    writer: FigureWriter, table: pd.DataFrame, published: pd.DataFrame
) -> None:
    """F21: wide, sparse summary for the repository README."""
    name = "gcn_results_overview"
    title = "README summary banner"
    caption = (
        "Summary for the README: published and reproduced GCN in the Fig. 4c layout, "
        "and reproduced against published RMSE_cliff."
    )
    figure_size = (MAX_FIGURE_WIDTH_CM / 2.54, 2.5)
    panel_a_title = "Published GCN"
    panel_a_x_label = "RMSE"
    panel_a_y_label = r"RMSE$_{\mathrm{cliff}}$"
    panel_b_title = "Reproduced GCN"
    panel_b_x_label = "RMSE"
    panel_b_y_label = r"RMSE$_{\mathrm{cliff}}$"
    panel_c_title = "Reproduced vs published"
    panel_c_x_label = r"Published RMSE$_{\mathrm{cliff}}$"
    panel_c_y_label = r"Reproduced RMSE$_{\mathrm{cliff}}$"
    panel_c_label_size = 7

    figure, (axis_a, axis_b, axis_c) = plt.subplots(
        1,
        3,
        figsize=figure_size,
        gridspec_kw={"width_ratios": [1, 1, 1.1], "wspace": 0.42},
    )
    all_values = np.concatenate(
        [
            published["rmse"],
            published["cliff_rmse"],
            table["rep_rmse"],
            table["rep_cliff"],
        ]
    )
    scatter_lower = min(0.25, np.floor(all_values.min() / 0.25) * 0.25)
    scatter_upper = max(1.75, np.ceil(all_values.max() / 0.25) * 0.25)
    other_strategies = published[~published["is_gcn"]]

    axis_a.scatter(
        other_strategies["rmse"],
        other_strategies["cliff_rmse"],
        s=11,
        **OTHER_STRATEGIES_CLOUD,
        zorder=2,
    )
    axis_a.plot(
        [scatter_lower, scatter_upper], [scatter_lower, scatter_upper], **IDENTITY_LINE
    )
    axis_a.plot(
        [scatter_lower, scatter_upper],
        [scatter_lower - 0.5, scatter_upper - 0.5],
        **HALF_LOG_UNIT_LINE,
    )
    axis_a.plot(
        [scatter_lower, scatter_upper],
        [scatter_lower + 0.5, scatter_upper + 0.5],
        **HALF_LOG_UNIT_LINE,
    )
    axis_a.set_xlim(scatter_lower, scatter_upper)
    axis_a.set_ylim(scatter_lower, scatter_upper)
    axis_a.set_aspect("equal", adjustable="box")
    axis_a.scatter(
        table["pub_rmse"], table["pub_cliff"], s=20, **PUBLISHED_MARKER, zorder=4
    )
    axis_a.text(
        0.0,
        1.06,
        panel_a_title,
        transform=axis_a.transAxes,
        fontweight="bold",
        fontsize=7.2,
        va="bottom",
    )
    axis_a.xaxis.set_major_locator(MultipleLocator(0.5))
    axis_a.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2f}"))
    axis_a.yaxis.set_major_locator(MultipleLocator(0.5))
    axis_a.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2f}"))
    axis_a.xaxis.set_minor_locator(MultipleLocator(0.25))
    axis_a.yaxis.set_minor_locator(MultipleLocator(0.25))
    axis_a.tick_params(which="minor", length=1.8, width=0.6)
    axis_a.set_xlabel(panel_a_x_label)
    axis_a.set_ylabel(panel_a_y_label)

    axis_b.scatter(
        other_strategies["rmse"],
        other_strategies["cliff_rmse"],
        s=11,
        **OTHER_STRATEGIES_CLOUD,
        zorder=2,
    )
    axis_b.plot(
        [scatter_lower, scatter_upper], [scatter_lower, scatter_upper], **IDENTITY_LINE
    )
    axis_b.plot(
        [scatter_lower, scatter_upper],
        [scatter_lower - 0.5, scatter_upper - 0.5],
        **HALF_LOG_UNIT_LINE,
    )
    axis_b.plot(
        [scatter_lower, scatter_upper],
        [scatter_lower + 0.5, scatter_upper + 0.5],
        **HALF_LOG_UNIT_LINE,
    )
    axis_b.set_xlim(scatter_lower, scatter_upper)
    axis_b.set_ylim(scatter_lower, scatter_upper)
    axis_b.set_aspect("equal", adjustable="box")
    axis_b.scatter(
        table["rep_rmse"], table["rep_cliff"], s=16, **REPRODUCED_MARKER, zorder=4
    )
    axis_b.text(
        0.0,
        1.06,
        panel_b_title,
        transform=axis_b.transAxes,
        fontweight="bold",
        fontsize=7.2,
        va="bottom",
    )
    axis_b.xaxis.set_major_locator(MultipleLocator(0.5))
    axis_b.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2f}"))
    axis_b.yaxis.set_major_locator(MultipleLocator(0.5))
    axis_b.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2f}"))
    axis_b.xaxis.set_minor_locator(MultipleLocator(0.25))
    axis_b.yaxis.set_minor_locator(MultipleLocator(0.25))
    axis_b.tick_params(which="minor", length=1.8, width=0.6)
    axis_b.set_xlabel(panel_b_x_label)
    axis_b.set_ylabel(panel_b_y_label)

    panel_c_values = np.concatenate(
        [
            table["pub_cliff"],
            table["rep_cliff"] - table["rep_cliff_sd"],
            table["rep_cliff"] + table["rep_cliff_sd"],
        ]
    )
    panel_c_lower = np.floor((panel_c_values.min() - 0.02) / 0.1) * 0.1
    panel_c_upper = np.ceil((panel_c_values.max() + 0.02) / 0.1) * 0.1
    axis_c.plot(
        [panel_c_lower, panel_c_upper], [panel_c_lower, panel_c_upper], **IDENTITY_LINE
    )
    axis_c.errorbar(
        table["pub_cliff"],
        table["rep_cliff"],
        yerr=table["rep_cliff_sd"],
        fmt="none",
        ecolor=REPRODUCED["colour"],
        elinewidth=0.7,
        alpha=0.6,
        zorder=2,
    )
    axis_c.scatter(
        table["pub_cliff"], table["rep_cliff"], s=15, **REPRODUCED_MARKER, zorder=4
    )
    axis_c.set_xlim(panel_c_lower, panel_c_upper)
    axis_c.set_ylim(panel_c_lower, panel_c_upper)
    axis_c.set_aspect("equal", adjustable="box")
    axis_c.xaxis.set_major_locator(MultipleLocator(0.1))
    axis_c.yaxis.set_major_locator(MultipleLocator(0.1))
    axis_c.set_xlabel(
        panel_c_x_label, color=PUBLISHED["colour"], fontsize=panel_c_label_size
    )
    axis_c.set_ylabel(
        panel_c_y_label, color=REPRODUCED["colour"], fontsize=panel_c_label_size
    )
    axis_c.text(
        0.0,
        1.06,
        panel_c_title,
        transform=axis_c.transAxes,
        fontweight="bold",
        fontsize=7.2,
        va="bottom",
    )
    panel_c_differences = table["rep_cliff"] - table["pub_cliff"]
    panel_c_correlation = np.corrcoef(table["pub_cliff"], table["rep_cliff"])[0, 1]
    agreement_statistics = figure.text(
        0,
        0,
        "Mean difference {}\nMean |difference| {:.3f}\nPearson r {:.2f}\n"
        "n = {} datasets".format(
            f"{panel_c_differences.mean():+.3f}".replace("-", "−"),
            panel_c_differences.abs().mean(),
            panel_c_correlation,
            len(table),
        ),
        va="top",
        ha="center",
        fontsize=7,
        linespacing=1.35,
        color="#404040",
    )

    legend_handles = [
        Line2D([], [], **PUBLISHED_LEGEND_MARKER),
        Line2D([], [], **REPRODUCED_LEGEND_MARKER),
        Line2D([], [], **OTHER_STRATEGIES_LEGEND_MARKER),
    ]
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    lowest_content = min(
        panel.get_tightbbox(renderer).transformed(figure.transFigure.inverted()).y0
        for panel in figure.axes
        if panel.get_visible()
    )
    statistics_panel = axis_c.get_position()
    agreement_statistics.set_position(
        (statistics_panel.x0 + statistics_panel.width / 2, lowest_content - 0.03)
    )
    figure.canvas.draw()
    lowest_statistics = (
        agreement_statistics.get_window_extent(renderer)
        .transformed(figure.transFigure.inverted())
        .y0
    )
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, lowest_statistics - 0.015),
        ncol=len(legend_handles),
        handletextpad=0.35,
        columnspacing=1.6,
    )
    writer.save(figure, name, title, caption)


COMMON_CAPTION = (
    "Each dataset is a target–response combination from the corrected MoleculeACE "
    "benchmark (n = 30). Reproduced values are arithmetic means of five official fold "
    "RMSEs, each evaluated on the same published test set; they are not RMSEs of "
    "averaged predictions. Where shown, reproduced error bars are the sample SD across "
    "those folds (ddof = 1): folds share the test set and overlap in their training "
    "data, so the SD is neither a confidence interval nor seed-to-seed variation. "
    "Published values are corrected dataset-level scores (upstream commit 45cf46d; DOI "
    + SOURCE_DOI
    + ") with no fold dispersion available. Errors are in log10 units of the response "
    "y = −log10(concentration in nM). Indigo circles: published GCN; crimson diamonds: "
    "reproduced GCN; grey: other published strategies (external data, not reproduced)."
)


def f22_dataset_error_comparison_table(
    writer: FigureWriter, table: pd.DataFrame
) -> None:
    """Draw dataset identifiers, published molecule counts and two error comparisons.

    Parameters
    ----------
    writer : FigureWriter
        Export destination and shared figure metadata.
    table : pd.DataFrame
        The 30 dataset rows from ``build_table``. Error metrics are in log10
        units of the response, -log10(concentration in nM).

    Returns
    -------
    None
        Save the figure and its caption through ``writer``.
    """
    name = "dataset_error_comparison_table"
    title = "Dataset counts and GCN error comparisons"
    caption = (
        "All 30 target–response datasets, ordered by the size of the published "
        "training partition (n_train, before internal fold subdivision), then by "
        "dataset identifier. The table gives the full ChEMBL dataset identifier, "
        "target abbreviation and n_cliff: the number of published "
        "test molecules labelled as activity-cliff molecules, with each molecule "
        "counted once rather than counting cliff pairs. The response type (Ki or "
        "EC50) remains in the full dataset identifier. (a) RMSE_cliff − RMSE for "
        "the corrected published GCN (indigo circles) and the reproduction "
        "(crimson diamonds). Reproduced bars are ± the sample SD of the five "
        "paired within-fold differences, retaining the covariance of the two "
        "errors; no published fold dispersion is available. (b) Reproduced minus "
        "published RMSE_cliff (grey squares); positive values indicate a larger "
        "reproduced cliff error. Dark horizontal segments connect each difference "
        "estimate to zero as a visual guide. Light-grey capped bars show ± "
        "reproduced RMSE_cliff fold SD "
        "around the difference from the fixed published value, not uncertainty "
        "in the difference between two estimated means. These SDs describe the "
        "five overlapping training folds evaluated on the same published test "
        "set, not confidence intervals or seed-to-seed variation. Errors and "
        "differences are in log10 units of the response "
        "y = −log10(concentration in nM)."
    )
    figure_size = (MAX_FIGURE_WIDTH_CM / 2.54, 8.4)
    table_headers = (
        "ChEMBL dataset",
        "Target",
        r"$n_{\mathrm{train}}$",
        r"$n_{\mathrm{cliff}}$",
    )
    table_columns = (0.015, 0.52, 0.835, 0.995)
    table_alignments = ("left", "left", "right", "right")
    panel_a_x_label = r"RMSE$_{\mathrm{cliff}}$ − RMSE"
    panel_b_x_label = "Reproduced − published\n" + r"RMSE$_{\mathrm{cliff}}$"
    difference_legend_label = "Difference (line to zero)"
    standard_deviation_legend_label = "± reproduced fold SD"
    text_size = 7.2
    label_size = 7.5
    row_offset = 0.16
    header_position = -1.0
    header_rule_position = -0.55
    first_row_limit = -1.65
    width_ratios = (4, 2.5, 2.5)
    column_spacing = 0.12
    horizontal_tick_spacing = 0.1
    axis_padding = 0.025
    error_bar_colour = "#A0A0A0"
    difference_marker_size = 12
    legend_columns = 2

    order = sorted(
        table.index, key=lambda dataset: (table.loc[dataset, "n_train"], dataset)
    )
    ordered = table.loc[order]
    row_count = len(ordered)
    row_positions = np.arange(row_count)
    published_positions = row_positions - row_offset
    reproduced_positions = row_positions + row_offset
    figure = plt.figure(figsize=figure_size)
    grid = figure.add_gridspec(
        1,
        3,
        width_ratios=width_ratios,
        wspace=column_spacing,
        left=0.02,
        right=0.985,
        top=0.955,
        bottom=0.105,
    )
    table_axis = figure.add_subplot(grid[0, 0])
    axis_a = figure.add_subplot(grid[0, 1])
    axis_b = figure.add_subplot(grid[0, 2])
    for axis in (table_axis, axis_a, axis_b):
        axis.set_ylim(row_count - 0.5, first_row_limit)
        axis.set_yticks([])
        for row_index in range(0, row_count, 2):
            axis.axhspan(
                row_index - 0.5,
                row_index + 0.5,
                color=BAND_COLOUR,
                zorder=0,
                lw=0,
            )

    table_axis.set_xlim(0, 1)
    table_axis.axis("off")
    table_axis.axhline(header_rule_position, color=GUIDE_COLOUR, lw=0.6)
    for position, alignment, header in zip(
        table_columns, table_alignments, table_headers, strict=True
    ):
        table_axis.text(
            position,
            header_position,
            header,
            ha=alignment,
            va="center",
            fontsize=label_size,
            fontweight="bold",
            color=INK,
        )
    for row_index, dataset in enumerate(order):
        row = ordered.loc[dataset]
        cells = (
            dataset,
            row["abbreviation"],
            str(int(row["n_train"])),
            str(int(row["n_test_cliff"])),
        )
        for position, alignment, value in zip(
            table_columns, table_alignments, cells, strict=True
        ):
            table_axis.text(
                position,
                row_index,
                value,
                ha=alignment,
                va="center",
                fontsize=text_size,
                color=INK,
            )

    for axis in (axis_a, axis_b):
        axis.spines["left"].set_visible(False)
        axis.axvline(0, color=INK, lw=0.7, zorder=1)
        axis.xaxis.set_major_locator(MultipleLocator(horizontal_tick_spacing))
        axis.tick_params(axis="x", labelsize=text_size)
    axis_a.errorbar(
        ordered["rep_delta"],
        reproduced_positions,
        xerr=ordered["rep_delta_sd"],
        fmt="none",
        ecolor=REPRODUCED["colour"],
        elinewidth=0.7,
        capsize=1.3,
        capthick=0.6,
        zorder=2,
    )
    axis_a.scatter(
        ordered["pub_delta"],
        published_positions,
        s=PUBLISHED["size"],
        **PUBLISHED_MARKER,
        zorder=4,
    )
    axis_a.scatter(
        ordered["rep_delta"],
        reproduced_positions,
        s=REPRODUCED["size"],
        **REPRODUCED_MARKER,
        zorder=4,
    )
    delta_minimum = min(
        0,
        ordered["pub_delta"].min(),
        (ordered["rep_delta"] - ordered["rep_delta_sd"]).min(),
    )
    delta_maximum = max(
        0,
        ordered["pub_delta"].max(),
        (ordered["rep_delta"] + ordered["rep_delta_sd"]).max(),
    )
    axis_a.set_xlim(delta_minimum - axis_padding, delta_maximum + axis_padding)
    axis_a.set_xlabel(panel_a_x_label, fontsize=label_size)
    axis_a.text(-0.04, 1.01, "a", transform=axis_a.transAxes, **PANEL_LETTER)

    for row_index, dataset in enumerate(order):
        difference = ordered.loc[dataset, "diff_cliff"]
        axis_b.plot(
            [0, difference],
            [row_index, row_index],
            color=DIFFERENCE["colour"],
            lw=0.9,
            zorder=2,
        )
    axis_b.errorbar(
        ordered["diff_cliff"],
        row_positions,
        xerr=ordered["rep_cliff_sd"],
        fmt="none",
        ecolor=error_bar_colour,
        elinewidth=0.7,
        capsize=1.3,
        capthick=0.6,
        zorder=2,
    )
    axis_b.scatter(
        ordered["diff_cliff"],
        row_positions,
        s=difference_marker_size,
        **DIFFERENCE_MARKER,
        zorder=4,
    )
    difference_limit = (ordered["diff_cliff"].abs() + ordered["rep_cliff_sd"]).max()
    difference_limit = np.ceil((difference_limit + axis_padding) * 20) / 20
    axis_b.set_xlim(-difference_limit, difference_limit)
    axis_b.set_xlabel(panel_b_x_label, fontsize=label_size)
    axis_b.text(-0.04, 1.01, "b", transform=axis_b.transAxes, **PANEL_LETTER)

    legend_handles = [
        Line2D([], [], **PUBLISHED_LEGEND_MARKER),
        Line2D([], [], **REPRODUCED_LEGEND_MARKER),
        Line2D(
            [],
            [],
            **{
                **DIFFERENCE_LEGEND_MARKER,
                "ls": "-",
                "color": DIFFERENCE["colour"],
                "lw": 0.9,
                "label": difference_legend_label,
            },
        ),
        Line2D(
            [],
            [],
            color=error_bar_colour,
            lw=0.8,
            label=standard_deviation_legend_label,
        ),
    ]
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    lowest_content = min(
        panel.get_tightbbox(renderer).transformed(figure.transFigure.inverted()).y0
        for panel in figure.axes
        if panel.get_visible()
    )
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, lowest_content - 0.012),
        ncol=legend_columns,
        handletextpad=0.35,
        columnspacing=1.6,
        fontsize=text_size,
    )
    writer.save(figure, name, title, caption)


def write_captions(path: Path, figures: list[dict], is_demo: bool) -> None:
    """Write the caption of every figure, after the text common to all."""
    lines = ["# Figure captions", ""]
    if is_demo:
        preview_warning = (
            "> **PREVIEW ONLY.** Reproduction values in these files are seeded "
            "synthetic data."
        )
        lines += [preview_warning, ""]
    lines += ["Common text for every caption:", "", COMMON_CAPTION, ""]
    for figure in figures:
        lines += [
            "## {} — {}".format(figure["name"], figure["title"]),
            "",
            figure["caption"],
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_catalogue(path: Path, figures: list[dict]) -> None:
    """Write a multi-page PDF with one titled figure per page, for choosing figures."""
    with PdfPages(path) as pdf:
        for figure in figures:
            figure["figure"].suptitle(
                "{} — {}".format(figure["name"], figure["title"]),
                x=0.01,
                ha="left",
                fontsize=7,
                color="#666666",
                y=1.04,
            )
            pdf.savefig(figure["figure"], bbox_inches="tight")


def build_argument_parser() -> argparse.ArgumentParser:
    """Describe the command-line options."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--published",
        type=Path,
        required=True,
        help="Corrected MoleculeACE_results.csv (not MoleculeACE_results_old.csv)",
    )
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        required=True,
        help="Corrected benchmark CSV folder",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--comparison", type=Path, help="_analysis/comparison.csv of the campaign"
    )
    source.add_argument(
        "--demo",
        action="store_true",
        help="Synthetic reproduction for layout review only",
    )
    parser.add_argument(
        "--per-fold",
        type=Path,
        help="_summary/per_fold.csv (enables fold-level figures)",
    )
    parser.add_argument(
        "--fold-columns",
        default="",
        help="Override per-fold names: dataset=...,fold=...,rmse=...,rmse_cliff=...",
    )
    parser.add_argument(
        "--order",
        choices=("paper", "n_train", "alphabetical"),
        default="paper",
        help="Row order of dataset-level figures (default: corrected Fig. 5a order)",
    )
    parser.add_argument("--formats", default="pdf,svg,png")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Write figures directly to this new directory; it must not exist",
    )
    return parser


def load_figure_inputs(
    arguments: argparse.Namespace, output_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """Check and load the published, benchmark and reproduction tables."""
    if output_dir.exists():
        raise FigureDataError(
            f"Output directory exists; choose a new one: {output_dir}"
        )
    if arguments.published.name == "MoleculeACE_results_old.csv":
        raise FigureDataError(
            "Use the corrected published table, not MoleculeACE_results_old.csv"
        )
    published = load_published(arguments.published)
    counts = load_benchmark_counts(arguments.benchmark_dir)
    comparison = None
    if arguments.demo:
        folds = synthetic_reproduction(published)
        reproduction = summarise_folds(folds)
    else:
        comparison = load_comparison(arguments.comparison)
        reproduction = comparison
        folds = None
        if arguments.per_fold:
            overrides = dict(
                item.split("=", 1) for item in arguments.fold_columns.split(",") if item
            )
            folds = load_per_fold(arguments.per_fold, overrides)
            fold_summary = summarise_folds(folds)
            for column in (
                "rmse_mean",
                "rmse_cliff_mean",
                "rmse_sd",
                "rmse_cliff_sd",
                "delta_mean",
                "delta_sd",
            ):
                gap = (
                    (fold_summary[column] - comparison.loc[fold_summary.index, column])
                    .abs()
                    .max()
                )
                if gap > 1e-6:
                    raise FigureDataError(
                        f"Per-fold {column} disagrees with comparison.csv by {gap:.2e}"
                    )
    table = build_table(published, counts, reproduction, comparison)
    return published, table, folds


def write_manifest(
    arguments: argparse.Namespace,
    output_dir: Path,
    order: list[str],
    writer: FigureWriter,
) -> None:
    """Record inputs, script, environment and written files of this export."""
    inputs = {
        "published": arguments.published,
        "benchmark_dir": arguments.benchmark_dir,
        "comparison": arguments.comparison,
        "per_fold": arguments.per_fold,
    }
    input_identities = {
        key: {"path": str(value), "sha256": sha256(value)}
        for key, value in inputs.items()
        if value and key != "benchmark_dir"
    }
    input_identities["benchmark_dir"] = {
        "path": str(arguments.benchmark_dir),
        "files": [
            {
                "dataset": dataset,
                "file": f"{dataset}.csv",
                "sha256": sha256(arguments.benchmark_dir / f"{dataset}.csv"),
            }
            for dataset in sorted(ABBREVIATIONS)
        ],
    }
    supporting_outputs = [
        {"file": name, "sha256": sha256(output_dir / name)}
        for name in ("figure_data.csv", "figure_captions.md", "figure_catalogue.pdf")
    ]
    manifest = {
        "schema_version": 2,
        "hash_policy": (
            "SHA-256 covers the input files, generator and generated products. "
            "figure_manifest.json excludes itself to avoid a circular self-hash."
        ),
        "status": "PREVIEW_SYNTHETIC_REPRODUCTION"
        if arguments.demo
        else "saved_reproduction_results",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "script": {"path": Path(__file__).name, "sha256": sha256(Path(__file__))},
        "inputs": input_identities,
        "order_rule": arguments.order,
        "dataset_order": order,
        "identity": {
            "published": PUBLISHED,
            "reproduction": REPRODUCED,
            "context": OTHER_STRATEGIES,
            "difference": DIFFERENCE,
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "matplotlib": matplotlib.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "outputs": writer.outputs,
        "supporting_outputs": supporting_outputs,
    }
    (output_dir / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> None:
    """Load the inputs, draw every figure and write captions, catalogue and manifest."""
    parser = build_argument_parser()
    arguments = parser.parse_args(argv)
    output_dir = arguments.output_dir
    try:
        published, table, folds = load_figure_inputs(arguments, output_dir)
    except (FigureDataError, FileNotFoundError, KeyError, ValueError) as error:
        parser.exit(1, f"Figure export failed: {error}\n")

    output_dir.mkdir(parents=True)
    formats = [
        suffix.strip() for suffix in arguments.formats.split(",") if suffix.strip()
    ]
    writer = FigureWriter(output_dir, arguments.demo, formats)
    order = dataset_order(table, arguments.order)
    with plt.rc_context(STYLE):
        f01_fig4a_pair_cliff(writer, table)
        f01_fig4a_pair_rmse(writer, table)
        f02_fig4a_pair_three_metrics(writer, table)
        f03_fig4a_context(writer, table, published)
        f04_fig4c_side_by_side(writer, table, published)
        f05_fig4c_side_by_side_ghost(writer, table, published)
        f06_fig4c_shift_arrows_zoom(writer, table)
        f06_fig4c_shift_arrows_full(writer, table, published)
        f07_fig5a_context(writer, table, published, order)
        f08_fig5a_gcn_only(writer, table, order)
        f09_dumbbell_mean_sd(writer, table, order)
        if folds is not None:
            f10_dumbbell_folds(writer, table, order, folds)
        f11_fig5d_overlay(writer, table)
        f12_fig5d_side_by_side(writer, table)
        f13_agreement_scatter(writer, table)
        f14_bland_altman(writer, table)
        f15_differences_by_dataset(writer, table, order)
        f16_difference_distribution(writer, table)
        f17_strategy_ranking_context(writer, table, published)
        f18_fig5bc_examples(writer, table, published)
        f19_composite_figure4_layout(writer, table, published)
        f20_composite_figure5_layout(writer, table, published, order)
        f21_readme_banner(writer, table, published)
        f22_dataset_error_comparison_table(writer, table)
        write_captions(
            output_dir / "figure_captions.md", writer.figures, arguments.demo
        )
        write_catalogue(output_dir / "figure_catalogue.pdf", writer.figures)
    for figure in writer.figures:
        plt.close(figure["figure"])
    table.to_csv(output_dir / "figure_data.csv", float_format="%.10g")
    write_manifest(arguments, output_dir, order, writer)
    print(
        f"Wrote {len(writer.figures)} figures ({len(writer.outputs)} files) "
        f"to {output_dir}"
    )


if __name__ == "__main__":
    main()

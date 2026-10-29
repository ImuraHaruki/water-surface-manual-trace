"""Interactive result graphs for the manual trace application."""

from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path
from tkinter import DoubleVar, Toplevel, filedialog, messagebox
from tkinter import ttk
from typing import Any

import numpy as np
from PIL import Image as PILImage


# Keep Matplotlib's cache out of a possibly restricted home directory.
cache_directory = Path(tempfile.gettempdir()) / "manual_trace_matplotlib"
cache_directory.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_directory))

from matplotlib import rcParams
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from core import (
    ImageGeometry,
    calculate_manual_velocity_rows,
    calculate_rows,
    export_manual_velocity_results_csv,
)


rcParams["font.sans-serif"] = ["Yu Gothic", "Meiryo", "Noto Sans CJK JP", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False


def _values(rows: list[dict[str, Any]], key: str, multiplier: float = 1.0) -> list[float]:
    return [math.nan if row.get(key) is None else float(row[key]) * multiplier for row in rows]


def _has_values(values: list[float]) -> bool:
    return any(not math.isnan(value) for value in values)


def _combined_legend(axis: Any, second_axis: Any | None = None) -> None:
    handles, labels = axis.get_legend_handles_labels()
    if second_axis is not None:
        second_handles, second_labels = second_axis.get_legend_handles_labels()
        handles += second_handles
        labels += second_labels
    if handles:
        axis.legend(handles, labels, loc="best", fontsize=9)


def _draw_spacetime_image(
    axis: Any,
    project: dict[str, Any],
    rows: list[dict[str, Any]],
    time: list[float],
    water_height: list[float],
    ground_height: list[float],
) -> None:
    geometry = ImageGeometry.from_dict(project["geometry"])
    calibration = project["calibration"]
    pixels_per_cm = float(calibration["pixels_per_cm"])
    fps = float(calibration["fps"])
    duration = len(rows) / fps
    physical_height_cm = max(0.01, (geometry.source_height - 1) / pixels_per_cm)
    image_path = Path(project.get("image_path", ""))
    try:
        with PILImage.open(image_path) as source:
            left = max(0, int(round(geometry.plot_left)))
            top = max(0, int(round(geometry.plot_top)))
            right = min(source.width, int(round(geometry.plot_left + geometry.plot_width)))
            bottom = min(source.height, int(round(geometry.plot_top + geometry.plot_height)))
            cropped = source.crop((left, top, right, bottom)).convert("RGB")
            cropped.thumbnail((4000, 1000), PILImage.Resampling.LANCZOS)
            display_array = np.asarray(cropped.transpose(PILImage.Transpose.FLIP_TOP_BOTTOM))
        axis.imshow(
            display_array,
            extent=(0, duration, 0, physical_height_cm),
            origin="lower",
            aspect="auto",
            interpolation="nearest",
        )
    except Exception as error:
        axis.set_facecolor("#eeeeee")
        axis.text(0.5, 0.5, f"時系列画像を表示できません\n{error}", ha="center", va="center", transform=axis.transAxes)
    for values, color, label in (
        (water_height, "#008de5", "水面トレース"),
        (ground_height, "#d100a7", "地表面トレース"),
    ):
        axis.plot(time, values, color="white", linewidth=3.2, alpha=0.9, zorder=3)
        axis.plot(time, values, color=color, linewidth=1.5, label=label, zorder=4)
    axis.set_ylim(0, physical_height_cm)
    axis.set_ylabel("高さ (cm)")
    axis.set_title("時系列画像（トレース重ね表示）", loc="left", fontsize=10)
    axis.legend(loc="upper right", fontsize=8, ncol=2)


def build_results_figure(
    project: dict[str, Any],
    froude_limits: tuple[float, float] = (0.0, 2.0),
    shields_limits: tuple[float, float] = (0.0, 2.0),
) -> tuple[Figure, int]:
    rows = calculate_rows(project)
    if not rows:
        raise ValueError("グラフ化できるデータがありません")
    if froude_limits[0] >= froude_limits[1] or shields_limits[0] >= shields_limits[1]:
        raise ValueError("縦軸の最小値は最大値より小さくしてください")

    time = _values(rows, "time_s")
    manual_rows = calculate_manual_velocity_rows(project)
    manual_time = _values(manual_rows, "time_s")
    figure = Figure(figsize=(12, 12), dpi=100, constrained_layout=True)
    axes = figure.subplots(6, 1, sharex=True, gridspec_kw={"height_ratios": (1.6, 1, 1, 1, 1, 1)})
    title = Path(project.get("image_path", "")).stem
    figure.suptitle(f"水面・地表面トレース解析 — {title}", fontsize=14)

    water_height = _values(rows, "water_surface_height_cm")
    ground_height = _values(rows, "ground_surface_height_cm")
    _draw_spacetime_image(axes[0], project, rows, time, water_height, ground_height)
    axes[1].plot(time, water_height, color="#008de5", linewidth=1.3, label="水面高度")
    axes[1].plot(time, ground_height, color="#d100a7", linewidth=1.3, label="地表面高度")
    axes[1].set_ylabel("高度 (cm)")
    axes[1].set_title("水面・地表面高度", loc="left", fontsize=10)
    _combined_legend(axes[1])

    depth = _values(rows, "water_depth_cm")
    depth_lower = _values(rows, "water_depth_lower_bound_cm")
    axes[2].plot(time, depth, color="#007f7f", linewidth=1.3, label="水深")
    if _has_values(depth_lower):
        axes[2].scatter(time, depth_lower, color="#00a8a8", marker="^", s=10, label="水深下限値", zorder=3)
    axes[2].set_ylabel("水深 (cm)")
    axes[2].set_title("水深", loc="left", fontsize=10)
    _combined_legend(axes[2])

    velocity = _values(rows, "velocity_m_s", 100.0)
    axes[3].plot(time, velocity, color="#e56b00", linewidth=1.1, label="流速")
    axes[3].axhline(0, color="#777777", linewidth=0.7)
    axes[3].set_ylabel("流速 (cm/s)")
    axes[3].set_title("オプティカルフロー流速", loc="left", fontsize=10)
    if not _has_values(velocity):
        axes[3].text(0.5, 0.5, "流速CSVが未設定、または時刻範囲外です", ha="center", va="center", transform=axes[3].transAxes)
    manual_velocity = _values(manual_rows, "manual_velocity_m_s", 100.0)
    if _has_values(manual_velocity):
        axes[3].scatter(manual_time, manual_velocity, color="#d62728", edgecolors="white", linewidths=0.55, s=38, zorder=4, label="手動計測流速")
    _combined_legend(axes[3])

    froude = _values(rows, "froude_number")
    shields = _values(rows, "shields_number")
    reference_froude = _values(rows, "reference_froude_number")
    reference_shields = _values(rows, "reference_shields_number")
    manual_froude = _values(manual_rows, "manual_froude_number")
    manual_shields = _values(manual_rows, "manual_shields_number")
    axes[4].plot(time, froude, color="#15803d", linewidth=1.2, label="フルード数")
    shields_axis = axes[4].twinx()
    shields_axis.plot(time, shields, color="#7c3aed", linewidth=1.2, label="シールズ数")
    if _has_values(reference_froude):
        axes[4].plot(
            time,
            reference_froude,
            color="#15803d",
            linewidth=1.3,
            linestyle="--",
            label="フルード数（画像端を水面とした参考値）",
        )
    if _has_values(reference_shields):
        shields_axis.plot(
            time,
            reference_shields,
            color="#7c3aed",
            linewidth=1.3,
            linestyle="--",
            label="シールズ数（画像端を水面とした参考値）",
        )
    if _has_values(manual_froude):
        axes[4].scatter(manual_time, manual_froude, color="#43a047", edgecolors="white", linewidths=0.55, s=38, zorder=5, label="手動計測のフルード数")
    if _has_values(manual_shields):
        shields_axis.scatter(manual_time, manual_shields, color="#a855f7", marker="D", edgecolors="white", linewidths=0.55, s=36, zorder=5, label="手動計測のシールズ数")
    axes[4].axhline(1, color="#15803d", linestyle="--", linewidth=0.8, alpha=0.65, label="Fr = 1")
    axes[4].set_ylim(*froude_limits)
    shields_axis.set_ylim(*shields_limits)
    if froude_limits == (0.0, 2.0):
        axes[4].set_yticks((0, 1, 2))
    if shields_limits == (0.0, 2.0):
        shields_axis.set_yticks((0, 1, 2))
    axes[4].set_ylabel("フルード数")
    shields_axis.set_ylabel("シールズ数")
    axes[4].set_title("無次元量", loc="left", fontsize=10)
    _combined_legend(axes[4], shields_axis)

    ground_change = _values(rows, "ground_elevation_change_cm")
    axes[5].plot(time, ground_change, color="#9a5a00", linewidth=1.3, label="地表面高度変化")
    axes[5].axhline(0, color="#777777", linewidth=0.8)
    axes[5].set_ylabel("変化量 (cm)")
    axes[5].set_title("地表面高度変化", loc="left", fontsize=10)
    axes[5].set_xlabel("経過時間 (s)")
    _combined_legend(axes[5])

    for axis in axes[1:]:
        axis.grid(True, color="#d8d8d8", linewidth=0.6, alpha=0.8)
        axis.margins(x=0)
    axes[0].margins(x=0)
    return figure, len(rows)


def show_results_graph(parent: Any, project: dict[str, Any]) -> None:
    figure, row_count = build_results_figure(project)
    window = Toplevel(parent)
    window.title("解析結果グラフ")
    window.geometry("1200x900")
    window.minsize(850, 650)

    controls = ttk.Frame(window, padding=(8, 6))
    controls.pack(fill="x")
    ttk.Label(controls, text=f"{row_count}フレーム分").pack(side="left")
    froude_min_var = DoubleVar(value=0.0)
    froude_max_var = DoubleVar(value=2.0)
    shields_min_var = DoubleVar(value=0.0)
    shields_max_var = DoubleVar(value=2.0)
    axis_controls = ttk.Frame(controls)
    axis_controls.pack(side="right", padx=(0, 12))
    ttk.Label(axis_controls, text="Fr 軸").grid(row=0, column=0, padx=(0, 3))
    ttk.Entry(axis_controls, textvariable=froude_min_var, width=6).grid(row=0, column=1)
    ttk.Label(axis_controls, text="～").grid(row=0, column=2)
    ttk.Entry(axis_controls, textvariable=froude_max_var, width=6).grid(row=0, column=3)
    ttk.Label(axis_controls, text="θ 軸").grid(row=0, column=4, padx=(10, 3))
    ttk.Entry(axis_controls, textvariable=shields_min_var, width=6).grid(row=0, column=5)
    ttk.Label(axis_controls, text="～").grid(row=0, column=6)
    ttk.Entry(axis_controls, textvariable=shields_max_var, width=6).grid(row=0, column=7)

    graph_container = ttk.Frame(window)
    graph_container.pack(fill="both", expand=True)
    canvas: FigureCanvasTkAgg
    navigation: NavigationToolbar2Tk

    def draw_figure() -> None:
        nonlocal canvas, figure, navigation
        for child in graph_container.winfo_children():
            child.destroy()
        canvas = FigureCanvasTkAgg(figure, master=graph_container)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        navigation = NavigationToolbar2Tk(canvas, graph_container, pack_toolbar=False)
        navigation.update()
        navigation.pack(fill="x")

    def apply_vertical_limits() -> None:
        nonlocal figure
        try:
            froude_limits = (float(froude_min_var.get()), float(froude_max_var.get()))
            shields_limits = (float(shields_min_var.get()), float(shields_max_var.get()))
            figure, _count = build_results_figure(project, froude_limits, shields_limits)
            draw_figure()
            window._graph_canvas = canvas
            window._graph_figure = figure
        except Exception as error:
            messagebox.showerror("縦軸を設定できません", str(error), parent=window)

    def save_png() -> None:
        initial = Path(project.get("image_path", "results")).stem + "_analysis_graph.png"
        path_string = filedialog.asksaveasfilename(
            parent=window,
            defaultextension=".png",
            initialfile=initial,
            filetypes=[("PNG画像", "*.png"), ("PDF", "*.pdf")],
        )
        if not path_string:
            return
        try:
            figure.savefig(path_string, dpi=200, bbox_inches="tight")
            messagebox.showinfo("保存完了", f"グラフを保存しました。\n{path_string}", parent=window)
        except Exception as error:
            messagebox.showerror("保存できません", str(error), parent=window)

    def export_manual_summary() -> None:
        if not calculate_manual_velocity_rows(project):
            messagebox.showinfo(
                "手動測定値CSV",
                "手動計測流速CSVが選択されていません。",
                parent=window,
            )
            return
        initial = Path(project.get("image_path", "results")).stem + "_manual_measurement_summary.csv"
        path_string = filedialog.asksaveasfilename(
            parent=window,
            defaultextension=".csv",
            initialfile=initial,
            filetypes=[("CSVファイル", "*.csv")],
        )
        if not path_string:
            return
        try:
            count = export_manual_velocity_results_csv(project, Path(path_string))
            messagebox.showinfo(
                "保存完了",
                f"手動測定値の要約CSVを保存しました。\n{count}行\n{path_string}",
                parent=window,
            )
        except Exception as error:
            messagebox.showerror("保存できません", str(error), parent=window)

    ttk.Button(controls, text="グラフを画像保存", command=save_png).pack(side="right")
    ttk.Button(controls, text="手動測定値CSVを書き出す", command=export_manual_summary).pack(side="right", padx=(0, 8))
    ttk.Button(axis_controls, text="軸を反映", command=apply_vertical_limits).grid(row=0, column=8, padx=(8, 0))
    draw_figure()
    window._graph_canvas = canvas
    window._graph_figure = figure

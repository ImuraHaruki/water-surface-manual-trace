"""Ohata et al. (2017) Figure 8 phase diagrams for trace results."""

from __future__ import annotations

import csv
from io import BytesIO
import json
import math
import os
from queue import Empty, Queue
import tempfile
from threading import Event, Thread
from pathlib import Path
from tkinter import DoubleVar, PhotoImage, Scale, StringVar, Toplevel, filedialog, messagebox
from tkinter import ttk
from typing import Any


cache_directory = Path(tempfile.gettempdir()) / "manual_trace_matplotlib"
cache_directory.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_directory))

import numpy as np
from PIL import Image
from matplotlib import rcParams
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.figure import Figure

from core import calculate_manual_velocity_rows, calculate_rows


rcParams["font.sans-serif"] = ["Yu Gothic", "Meiryo", "Noto Sans CJK JP", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

PLOT_MODE_LABEL_TO_KEY = {
    "点": "points",
    "線（時刻順）": "line",
    "点＋線": "points_and_line",
}

GRAIN_CLASSES = (
    ("fine", "Fine sand", "細粒砂", 0.088, 0.177),
    ("medium", "Medium sand", "中粒砂", 0.177, 0.354),
    ("coarse", "Coarse sand", "粗粒砂", 0.354, 0.707),
)

# Digitized coordinates from the left-hand panels of Ohata et al. (2017), Fig. 8.
FIGURE8_LEFT_PANELS = {
    "fine": {
        "critical_log10_shields": -1.41,
        "boundaries": (
            ((-1.00, 0.10), (-0.78, 0.06), (-0.58, 0.025), (-0.34, -0.005), (-0.08, -0.015), (0.12, -0.005), (0.30, -0.01)),
            ((0.05, 0.32), (0.06, 0.26), (0.12, 0.16), (0.23, 0.03), (0.38, -0.10), (0.55, -0.22), (0.68, -0.31)),
            ((-0.62, -0.59), (-0.48, -0.51), (-0.30, -0.45), (-0.08, -0.40), (0.18, -0.36), (0.45, -0.33), (0.68, -0.31), (1.00, -0.28)),
            ((-0.55, -1.05), (-0.59, -0.87), (-0.61, -0.70), (-0.61, -0.54), (-0.58, -0.30)),
            ((-0.60, -0.88), (-0.52, -0.90), (-0.44, -0.88), (-0.34, -0.82), (-0.25, -0.73), (-0.16, -0.61), (-0.09, -0.48), (-0.04, -0.38)),
        ),
        "labels": ((-0.55, 0.23, "Antidunes"), (0.62, 0.19, "Cyclic steps"), (-0.08, -0.18, "Transition & U-pb"), (-1.14, -0.68, "L-pb"), (-0.42, -0.66, "R"), (0.18, -0.84, "Dunes")),
    },
    "medium": {
        "critical_log10_shields": -1.66,
        "boundaries": (
            ((-1.35, 0.29), (-1.12, 0.25), (-0.88, 0.19), (-0.62, 0.13), (-0.36, 0.06), (-0.10, -0.03), (0.32, -0.12)),
            ((0.25, 0.10), (0.23, 0.05), (0.24, -0.02), (0.28, -0.08), (0.34, -0.14), (0.44, -0.24)),
            ((-0.90, -0.45), (-0.72, -0.38), (-0.50, -0.33), (-0.25, -0.29), (0.02, -0.26), (0.30, -0.24), (0.65, -0.22)),
            ((-0.75, -0.95), (-0.82, -0.79), (-0.87, -0.62), (-0.90, -0.45), (-0.89, -0.23)),
            ((-0.83, -0.85), (-0.72, -0.78), (-0.61, -0.68), (-0.50, -0.55), (-0.41, -0.42), (-0.34, -0.30)),
        ),
        "labels": ((-0.48, 0.25, "Antidunes"), (0.56, -0.06, "CS"), (-0.13, -0.16, "Transition & U-pb"), (-1.28, -0.76, "L-pb"), (-0.68, -0.58, "R"), (0.06, -0.64, "Dunes")),
    },
    "coarse": {
        "critical_log10_shields": -1.79,
        "boundaries": (
            ((-1.00, -0.30), (-0.80, -0.26), (-0.55, -0.23), (-0.28, -0.21), (0.00, -0.19), (0.30, -0.17), (0.65, -0.15)),
            ((-0.52, 0.25), (-0.52, 0.19), (-0.47, 0.11), (-0.37, 0.03), (-0.25, -0.07), (-0.12, -0.15), (0.00, -0.20)),
            ((0.27, -0.17), (0.23, -0.08), (0.24, 0.00), (0.31, 0.05), (0.42, 0.07), (0.54, 0.065), (0.65, 0.04)),
            ((-1.04, 0.00), (-1.07, -0.16), (-1.07, -0.36), (-1.03, -0.52), (-0.97, -0.64), (-0.84, -0.73)),
            ((-0.98, -0.64), (-0.89, -0.56), (-0.82, -0.45), (-0.77, -0.34), (-0.76, -0.25)),
        ),
        "labels": ((-0.20, 0.20, "Antidunes"), (0.48, -0.08, "CS"), (-0.70, -0.10, "T & U-pb"), (-1.32, -0.52, "L-pb"), (-0.98, -0.41, "R"), (0.12, -0.54, "Dunes")),
    },
}


def prepare_phase_rows(project: dict[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    def append_row(sequence: int, source: dict[str, Any], froude: Any, shields: Any, image_name: str, source_name: str) -> None:
        status = "描画対象"
        log_froude: float | str = ""
        log_shields: float | str = ""
        if froude is None:
            status = "除外: フルード数が空欄"
        elif shields is None:
            status = "除外: シールズ数が空欄"
        else:
            try:
                froude_value = float(froude)
                shields_value = float(shields)
                if not math.isfinite(froude_value) or not math.isfinite(shields_value):
                    raise ValueError("非有限値")
                if froude_value <= 0 or shields_value <= 0:
                    raise ValueError("Frまたはθが0以下")
                log_froude = math.log10(froude_value)
                log_shields = math.log10(shields_value)
            except (TypeError, ValueError) as error:
                status = f"除外: {error}"
        output.append({
            "sequence": sequence,
            "time_s": source.get("time_s", ""),
            "image_name": image_name,
            "source": source_name,
            "froude_number": "" if froude is None else froude,
            "shields_number": "" if shields is None else shields,
            "log10_froude": log_froude,
            "log10_shields": log_shields,
            "plot_status": status,
        })

    primary_rows = calculate_rows(project)
    for sequence, source in enumerate(primary_rows, start=1):
        append_row(
            sequence,
            source,
            source.get("froude_number"),
            source.get("shields_number"),
            f"frame_{int(source.get('frame_index', sequence - 1)):08d}",
            "optical_flow",
        )
    for offset, source in enumerate(calculate_manual_velocity_rows(project), start=1):
        sequence = len(primary_rows) + offset
        append_row(
            sequence,
            source,
            source.get("manual_froude_number"),
            source.get("manual_shields_number"),
            f"manual_velocity_{sequence:04d}",
            "manual",
        )
    return output


def valid_phase_points(rows: list[dict[str, Any]]) -> list[tuple[float, float, int, float, str]]:
    points: list[tuple[float, float, int, float, str]] = []
    for row in rows:
        if row.get("plot_status") != "描画対象":
            continue
        try:
            point = (float(row["log10_shields"]), float(row["log10_froude"]), int(row["sequence"]), float(row["time_s"]), str(row.get("source", "optical_flow")))
        except (KeyError, TypeError, ValueError):
            continue
        if all(math.isfinite(value) for value in (point[0], point[1], point[3])):
            points.append(point)
    return points


def build_phase_figure(
    rows: list[dict[str, Any]],
    start_time_s: float | None = None,
    end_time_s: float | None = None,
    plot_mode: str = "points",
    frame_rate_fps: float = 240.0,
    x_limits: tuple[float, float] | None = None,
    y_limits: tuple[float, float] | None = None,
) -> tuple[Figure, dict[str, Any]]:
    if plot_mode not in PLOT_MODE_LABEL_TO_KEY.values():
        raise ValueError("表示方法が不正です")
    all_points = valid_phase_points(rows)
    if not all_points:
        raise ValueError("相図に描画できるフルード数・シールズ数データがありません")
    available_start = min(point[3] for point in all_points)
    available_end = max(point[3] for point in all_points)
    selected_start = available_start if start_time_s is None else max(available_start, float(start_time_s))
    selected_end = available_end if end_time_s is None else min(available_end, float(end_time_s))
    if selected_start > selected_end:
        raise ValueError("開始時間は終了時間以下にしてください")
    points = sorted(
        (point for point in all_points if selected_start <= point[3] <= selected_end),
        key=lambda point: (point[3], point[2]),
    )
    if not points:
        raise ValueError("指定した時間範囲に描画対象データがありません")

    valid_x = np.asarray([point[0] for point in points], dtype=float)
    valid_y = np.asarray([point[1] for point in points], dtype=float)
    valid_time = np.asarray([point[3] for point in points], dtype=float)
    optical_points = [point for point in points if point[4] == "optical_flow"]
    manual_points = [point for point in points if point[4] == "manual"]
    optical_x = np.asarray([point[0] for point in optical_points], dtype=float)
    optical_y = np.asarray([point[1] for point in optical_points], dtype=float)
    optical_time = np.asarray([point[3] for point in optical_points], dtype=float)
    manual_x = np.asarray([point[0] for point in manual_points], dtype=float)
    manual_y = np.asarray([point[1] for point in manual_points], dtype=float)
    manual_time = np.asarray([point[3] for point in manual_points], dtype=float)
    x_min = min(-2.5, float(valid_x.min()) - 0.12)
    x_max = max(1.0, float(valid_x.max()) + 0.12)
    y_min = min(-1.5, float(valid_y.min()) - 0.12)
    y_max = max(0.7, float(valid_y.max()) + 0.12)
    if x_limits is not None:
        x_min, x_max = map(float, x_limits)
        if x_min >= x_max:
            raise ValueError("横軸の最小値は最大値より小さくしてください")
    if y_limits is not None:
        y_min, y_max = map(float, y_limits)
        if y_min >= y_max:
            raise ValueError("縦軸の最小値は最大値より小さくしてください")
    time_min, time_max = float(valid_time.min()), float(valid_time.max())
    if time_max <= time_min:
        time_max = time_min + max(1.0 / frame_rate_fps, 1e-9)
    time_norm = Normalize(vmin=time_min, vmax=time_max)
    color_mappable = ScalarMappable(norm=time_norm, cmap="viridis")
    color_mappable.set_array(valid_time)

    figure = Figure(figsize=(17.2, 6.8), facecolor="white")
    axes = np.atleast_1d(figure.subplots(1, 3, sharex=True, sharey=True))
    panel_metadata = []
    for axis, (code, english, japanese, diameter_min, diameter_max) in zip(axes, GRAIN_CLASSES):
        panel = FIGURE8_LEFT_PANELS[code]
        for boundary in panel["boundaries"]:
            boundary_points = np.asarray(boundary, dtype=float)
            axis.plot(boundary_points[:, 0], boundary_points[:, 1], color="#263238", linewidth=1.45, solid_capstyle="round", zorder=2)
        for label_x, label_y, label in panel["labels"]:
            axis.text(label_x, label_y, label, fontsize=10.2, ha="center", va="center", color="#20272A", zorder=3)
        critical = float(panel["critical_log10_shields"])
        axis.axvline(critical, color="#5D4037", linewidth=1.15, linestyle=(0, (4, 3)), zorder=1)
        axis.text(critical + 0.06, -1.42, r"$\theta=\theta_c$", fontsize=8.8, ha="left", va="bottom", color="#5D4037", zorder=3)
        if plot_mode in {"line", "points_and_line"} and len(optical_x) >= 2:
            path_points = np.column_stack((optical_x, optical_y))
            segments = np.stack((path_points[:-1], path_points[1:]), axis=1)
            line = LineCollection(segments, cmap="viridis", norm=time_norm, linewidths=2.4, alpha=0.92, zorder=4, capstyle="round", joinstyle="round")
            line.set_array((optical_time[:-1] + optical_time[1:]) / 2.0)
            axis.add_collection(line)
        if (plot_mode in {"points", "points_and_line"} or len(optical_x) == 1) and len(optical_x):
            axis.scatter(optical_x, optical_y, c=optical_time, cmap="viridis", norm=time_norm, s=26, edgecolors="white", linewidths=0.5, alpha=0.94, zorder=5)
        if len(manual_x):
            axis.scatter(manual_x, manual_y, c=manual_time, cmap="viridis", norm=time_norm, marker="D", s=30, edgecolors="#263238", linewidths=0.55, alpha=0.98, zorder=6, label="手動計測流速（点のみ）")
        axis.set_title(f"{english} / {japanese}\nD50 = {diameter_min:.3f}–{diameter_max:.3f} mm", fontsize=12.2, pad=9)
        axis.set_xlabel("log10(シールズ数 θ)")
        axis.set_xlim(x_min, x_max)
        axis.set_ylim(y_min, y_max)
        axis.grid(True, color="#DCE3E6", linewidth=0.65, alpha=0.72)
        axis.set_axisbelow(True)
        panel_metadata.append({
            "class": code,
            "name": japanese,
            "grain_diameter_min_mm": diameter_min,
            "grain_diameter_max_mm": diameter_max,
            "critical_log10_shields": critical,
            "boundary_source": "digitized from Figure 8 left panel",
        })
    if len(manual_x):
        axes[0].legend(loc="lower left", fontsize=8.2, frameon=True, framealpha=0.86)
    axes[0].set_ylabel("log10(フルード数 Fr)")
    figure.subplots_adjust(left=0.055, right=0.91, top=0.82, bottom=0.12, wspace=0.13)
    colorbar_axis = figure.add_axes([0.925, 0.18, 0.012, 0.60])
    colorbar = figure.colorbar(color_mappable, cax=colorbar_axis)
    colorbar.set_label("経過時間 (s)")
    figure.suptitle(
        f"Ohata et al. (2017) Figure 8 左列 相図\n表示時間: {selected_start:.3f}–{selected_end:.3f} s（{len(points)}/{len(all_points)}点）",
        fontsize=15,
        y=0.975,
    )
    figure.text(0.5, 0.035, "境界線はFigure 8左列の読み取り座標。Table 2判別式は使用していません。", ha="center", va="center", fontsize=8.2, color="#46565C")
    metadata = {
        "reference": "Ohata et al. (2017), JGR Earth Surface, doi:10.1002/2017JF004290",
        "phase_boundary_method": "digitized_figure8_left_panels",
        "table2_discriminants_used": False,
        "plot_mode": plot_mode,
        "frame_rate_fps": float(frame_rate_fps),
        "time_range": {
            "available_start_s": available_start,
            "available_end_s": available_end,
            "selected_start_s": selected_start,
            "selected_end_s": selected_end,
            "available_point_count": len(all_points),
            "selected_point_count": len(points),
        },
        "source_point_count": {
            "optical_flow": len(optical_points),
            "manual": len(manual_points),
        },
        "axis_limits": {"x": [x_min, x_max], "y": [y_min, y_max]},
        "grain_size_panels": panel_metadata,
    }
    return figure, metadata


def write_phase_package(
    project: dict[str, Any],
    rows: list[dict[str, Any]],
    output_csv: Path,
    start_time_s: float,
    end_time_s: float,
    plot_mode: str,
) -> tuple[Path, Path, Path]:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fps = float(project["calibration"]["fps"])
    figure, phase_metadata = build_phase_figure(rows, start_time_s, end_time_s, plot_mode, fps)
    image_path = output_csv.with_suffix(".png")
    figure.savefig(image_path, dpi=180, bbox_inches="tight", facecolor="white")
    metadata_path = output_csv.with_suffix(".json")
    metadata = {
        "source_image": project.get("image_path", ""),
        "row_count": len(rows),
        "plotted_count": sum(row["plot_status"] == "描画対象" for row in rows),
        "excluded_count": sum(row["plot_status"] != "描画対象" for row in rows),
        "x_axis": "log10_shields",
        "y_axis": "log10_froude",
        "graph_file": image_path.name,
        "frame_rate_fps": fps,
        "ohata_figure8_phase_diagrams": phase_metadata,
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_csv, image_path, metadata_path


class PhasePlotDialog:
    def __init__(self, parent: Any, project: dict[str, Any]) -> None:
        self.parent = parent
        self.project = project
        self.rows = prepare_phase_rows(project)
        points = valid_phase_points(self.rows)
        if not points:
            raise ValueError("相図に描画できるデータがありません。水面・地表面・流速CSVと計算方式を確認してください。")
        self.available_start = min(point[3] for point in points)
        self.available_end = max(point[3] for point in points)
        self.window = Toplevel(parent)
        self.window.title("フルード数・シールズ数相図")
        self.window.geometry("1400x850")
        self.window.minsize(950, 650)
        self.mode_var = StringVar(value="点＋線")
        self.start_var = DoubleVar(value=self.available_start)
        self.end_var = DoubleVar(value=self.available_end)
        self.x_min_var = DoubleVar(value=min(-2.5, min(point[0] for point in points) - 0.12))
        self.x_max_var = DoubleVar(value=max(1.0, max(point[0] for point in points) + 0.12))
        self.y_min_var = DoubleVar(value=min(-1.5, min(point[1] for point in points) - 0.12))
        self.y_max_var = DoubleVar(value=max(0.7, max(point[1] for point in points) + 0.12))
        self.animation_position_var = DoubleVar(value=self.available_start)
        self.animation_duration_var = DoubleVar(value=6.0)
        self.animation_status_var = StringVar(value="動画は未作成です")
        self.animation_after_id: str | None = None
        self.animation_running = False
        self.animation_cancel = Event()
        self.animation_thread: Thread | None = None
        self.animation_events: Queue[tuple[str, Any]] = Queue()
        self.figure: Figure | None = None
        self.metadata: dict[str, Any] | None = None
        self.plot_container = ttk.Frame(self.window)
        self._build_controls()
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        self.plot_container.pack(fill="both", expand=True)
        self.refresh()

    def _build_controls(self) -> None:
        controls = ttk.Frame(self.window, padding=8)
        controls.pack(fill="x")
        ttk.Label(controls, text="表示方法").grid(row=0, column=0, sticky="w")
        ttk.Combobox(controls, textvariable=self.mode_var, values=tuple(PLOT_MODE_LABEL_TO_KEY), state="readonly", width=15).grid(row=0, column=1, sticky="w", padx=(4, 12))
        ttk.Label(controls, text="開始時間 (s)").grid(row=0, column=2, sticky="w")
        ttk.Entry(controls, textvariable=self.start_var, width=10).grid(row=0, column=3, sticky="w", padx=(4, 12))
        ttk.Label(controls, text="終了時間 (s)").grid(row=0, column=4, sticky="w")
        ttk.Entry(controls, textvariable=self.end_var, width=10).grid(row=0, column=5, sticky="w", padx=(4, 12))
        ttk.Button(controls, text="表示を更新", command=self.refresh).grid(row=0, column=6, padx=3)
        ttk.Button(controls, text="表示中の3相図を画像出力", command=self.save_image).grid(row=0, column=7, padx=3)
        ttk.Button(controls, text="CSV・画像・設定JSONを出力", command=self.save_package).grid(row=0, column=8, padx=3)
        ttk.Label(controls, text=f"利用可能: {self.available_start:.3f}～{self.available_end:.3f} s").grid(row=1, column=0, columnspan=2, sticky="w", pady=(5, 0))
        ttk.Scale(controls, from_=self.available_start, to=self.available_end, variable=self.start_var).grid(row=1, column=2, columnspan=3, sticky="ew", padx=(4, 12), pady=(5, 0))
        ttk.Scale(controls, from_=self.available_start, to=self.available_end, variable=self.end_var).grid(row=1, column=5, columnspan=4, sticky="ew", padx=(4, 0), pady=(5, 0))
        ttk.Label(controls, text="アニメーション").grid(row=2, column=0, sticky="w", pady=(6, 0))
        ttk.Button(controls, text="動画を作成・表示", command=self.start_animation).grid(row=2, column=1, columnspan=2, sticky="w", padx=(4, 3), pady=(6, 0))
        ttk.Button(controls, text="作成を中止", command=self.stop_animation).grid(row=2, column=3, sticky="w", padx=3, pady=(6, 0))
        ttk.Label(controls, text="動画時間 (s)").grid(row=2, column=4, sticky="e", pady=(6, 0))
        ttk.Entry(controls, textvariable=self.animation_duration_var, width=7).grid(row=2, column=5, sticky="w", padx=(4, 10), pady=(6, 0))
        ttk.Label(controls, textvariable=self.animation_status_var).grid(row=2, column=6, columnspan=3, sticky="w", pady=(6, 0))
        ttk.Label(controls, text="再生位置 (s)").grid(row=3, column=0, sticky="w", pady=(5, 0))
        ttk.Scale(controls, from_=self.available_start, to=self.available_end, variable=self.animation_position_var, command=self.show_animation_position).grid(row=3, column=1, columnspan=8, sticky="ew", padx=(4, 0), pady=(5, 0))
        ttk.Label(controls, text="横軸 log10(θ): 最小 / 最大").grid(row=4, column=0, columnspan=2, sticky="w", pady=(7, 0))
        ttk.Entry(controls, textvariable=self.x_min_var, width=7).grid(row=4, column=2, sticky="w", padx=(4, 2), pady=(7, 0))
        ttk.Entry(controls, textvariable=self.x_max_var, width=7).grid(row=4, column=3, sticky="w", padx=2, pady=(7, 0))
        Scale(controls, from_=-5.0, to=1.5, resolution=0.05, showvalue=False, variable=self.x_min_var, orient="horizontal", command=self.refresh_from_axis_control).grid(row=4, column=4, columnspan=2, sticky="ew", padx=(4, 2), pady=(4, 0))
        Scale(controls, from_=-5.0, to=1.5, resolution=0.05, showvalue=False, variable=self.x_max_var, orient="horizontal", command=self.refresh_from_axis_control).grid(row=4, column=6, columnspan=3, sticky="ew", padx=2, pady=(4, 0))
        ttk.Label(controls, text="縦軸 log10(Fr): 最小 / 最大").grid(row=5, column=0, columnspan=2, sticky="w", pady=(4, 0))
        ttk.Entry(controls, textvariable=self.y_min_var, width=7).grid(row=5, column=2, sticky="w", padx=(4, 2), pady=(4, 0))
        ttk.Entry(controls, textvariable=self.y_max_var, width=7).grid(row=5, column=3, sticky="w", padx=2, pady=(4, 0))
        Scale(controls, from_=-3.0, to=2.0, resolution=0.05, showvalue=False, variable=self.y_min_var, orient="horizontal", command=self.refresh_from_axis_control).grid(row=5, column=4, columnspan=2, sticky="ew", padx=(4, 2), pady=(2, 0))
        Scale(controls, from_=-3.0, to=2.0, resolution=0.05, showvalue=False, variable=self.y_max_var, orient="horizontal", command=self.refresh_from_axis_control).grid(row=5, column=6, columnspan=3, sticky="ew", padx=2, pady=(2, 0))
        for column in range(2, 9):
            controls.columnconfigure(column, weight=1 if column in (3, 5, 8) else 0)

    def _settings(self) -> tuple[float, float, str, tuple[float, float], tuple[float, float]]:
        start = float(self.start_var.get())
        end = float(self.end_var.get())
        if start > end:
            raise ValueError("開始時間は終了時間以下にしてください")
        x_limits = (float(self.x_min_var.get()), float(self.x_max_var.get()))
        y_limits = (float(self.y_min_var.get()), float(self.y_max_var.get()))
        if x_limits[0] >= x_limits[1] or y_limits[0] >= y_limits[1]:
            raise ValueError("軸の最小値は最大値より小さくしてください")
        return start, end, PLOT_MODE_LABEL_TO_KEY[self.mode_var.get()], x_limits, y_limits

    def refresh(self, visible_end: float | None = None) -> None:
        try:
            start, end, mode, x_limits, y_limits = self._settings()
            current_end = end if visible_end is None else min(end, max(start, visible_end))
            figure, metadata = build_phase_figure(self.rows, start, current_end, mode, float(self.project["calibration"]["fps"]), x_limits, y_limits)
            for child in self.plot_container.winfo_children():
                child.destroy()
            canvas = FigureCanvasTkAgg(figure, master=self.plot_container)
            toolbar = NavigationToolbar2Tk(canvas, self.plot_container, pack_toolbar=False)
            toolbar.update()
            toolbar.pack(fill="x")
            canvas.get_tk_widget().pack(fill="both", expand=True)
            canvas.draw()
            self.figure = figure
            self.metadata = metadata
            self.canvas = canvas
        except Exception as error:
            messagebox.showerror("相図を更新できません", str(error), parent=self.window)

    def refresh_from_axis_control(self, _value: str = "") -> None:
        if not self.animation_running:
            if self.x_min_var.get() >= self.x_max_var.get() or self.y_min_var.get() >= self.y_max_var.get():
                return
            self.refresh(self.animation_position_var.get() if self.animation_position_var.get() > self.start_var.get() else None)

    def show_animation_position(self, _value: str = "") -> None:
        if self.animation_running:
            return
        position = max(float(self.start_var.get()), min(float(self.end_var.get()), float(self.animation_position_var.get())))
        self.animation_position_var.set(position)
        self.animation_status_var.set(f"アニメーション位置: {position:.3f} s")
        self.refresh(position)

    def start_animation(self) -> None:
        try:
            if self.animation_thread is not None and self.animation_thread.is_alive():
                return
            start, end, mode, x_limits, y_limits = self._settings()
            duration = float(self.animation_duration_var.get())
            if duration <= 0:
                raise ValueError("動画時間は0より大きくしてください")
            frame_count = max(24, min(120, round(duration * 15)))
            self.animation_cancel.clear()
            self.animation_running = True
            self.animation_status_var.set("動画をバックグラウンドで作成中… 0%")
            self.animation_thread = Thread(
                target=self._render_animation,
                args=(start, end, mode, x_limits, y_limits, duration, frame_count),
                daemon=True,
            )
            self.animation_thread.start()
            self.window.after(100, self._poll_animation_events)
        except Exception as error:
            messagebox.showerror("アニメーションを開始できません", str(error), parent=self.window)

    def _render_animation(
        self,
        start: float,
        end: float,
        mode: str,
        x_limits: tuple[float, float],
        y_limits: tuple[float, float],
        duration: float,
        frame_count: int,
    ) -> None:
        frames: list[Image.Image] = []
        try:
            times = np.linspace(start, end, frame_count)
            fps = float(self.project["calibration"]["fps"])
            for index, current in enumerate(times, start=1):
                if self.animation_cancel.is_set():
                    return
                figure, _metadata = build_phase_figure(self.rows, start, float(current), mode, fps, x_limits, y_limits)
                buffer = BytesIO()
                figure.savefig(buffer, format="png", dpi=72, facecolor="white")
                figure.clear()
                buffer.seek(0)
                with Image.open(buffer) as opened:
                    frames.append(opened.convert("P", palette=Image.Palette.ADAPTIVE))
                self.animation_events.put(("progress", (index, frame_count)))
            if not frames or self.animation_cancel.is_set():
                return
            path = Path(tempfile.gettempdir()) / f"manual_trace_phase_{os.getpid()}_{id(self)}.gif"
            frames[0].save(path, save_all=True, append_images=frames[1:], duration=max(20, round(duration * 1000 / frame_count)), loop=0, optimize=False)
            self.animation_events.put(("ready", (path, frame_count, max(20, round(duration * 1000 / frame_count)))))
        except Exception as error:
            self.animation_events.put(("failed", str(error)))

    def _poll_animation_events(self) -> None:
        try:
            while True:
                event, payload = self.animation_events.get_nowait()
                if event == "progress":
                    self._set_animation_progress(*payload)
                elif event == "ready":
                    self._show_animation(*payload)
                elif event == "failed":
                    self._animation_failed(str(payload))
        except Empty:
            pass
        if self.animation_thread is not None and self.animation_thread.is_alive():
            self.window.after(100, self._poll_animation_events)

    def _set_animation_progress(self, index: int, count: int) -> None:
        if not self.window.winfo_exists():
            return
        self.animation_status_var.set(f"動画をバックグラウンドで作成中… {index * 100 // count}%")

    def _show_animation(self, path: Path, frame_count: int, interval_ms: int) -> None:
        if not self.window.winfo_exists() or self.animation_cancel.is_set():
            return
        self.animation_running = False
        self.animation_status_var.set("動画を別ウインドウで再生中")
        viewer = Toplevel(self.window)
        viewer.title("Fr–θ相図 アニメーション")
        viewer.geometry("1120x620")
        label = ttk.Label(viewer)
        label.pack(fill="both", expand=True, padx=6, pady=6)
        ttk.Label(viewer, text="指定時間範囲内で点・線を順に追加して表示しています。閉じると再生を停止します。").pack(pady=(0, 6))
        state = {"index": 0, "photo": None, "running": True}

        def advance() -> None:
            if not state["running"] or not viewer.winfo_exists():
                return
            state["photo"] = PhotoImage(file=str(path), format=f"gif -index {state['index']}")
            label.configure(image=state["photo"])
            state["index"] = (state["index"] + 1) % frame_count
            viewer.after(interval_ms, advance)

        def close_viewer() -> None:
            state["running"] = False
            viewer.destroy()

        viewer.protocol("WM_DELETE_WINDOW", close_viewer)
        advance()

    def _animation_failed(self, error: str) -> None:
        self.animation_running = False
        self.animation_status_var.set("動画を作成できませんでした")
        messagebox.showerror("アニメーションを作成できません", error, parent=self.window)

    def stop_animation(self) -> None:
        self.animation_running = False
        self.animation_cancel.set()
        if self.animation_after_id is not None:
            self.window.after_cancel(self.animation_after_id)
            self.animation_after_id = None

    def close(self) -> None:
        self.stop_animation()
        self.window.destroy()

    def save_image(self) -> None:
        if self.figure is None:
            return
        initial = Path(self.project.get("image_path", "results")).stem + "_froude_shields_phase.png"
        path_string = filedialog.asksaveasfilename(parent=self.window, defaultextension=".png", initialfile=initial, filetypes=[("PNG画像", "*.png"), ("JPEG画像", "*.jpg *.jpeg")])
        if not path_string:
            return
        try:
            self.figure.savefig(path_string, dpi=180, bbox_inches="tight", facecolor="white")
            messagebox.showinfo("保存完了", "表示中の3相図を保存しました。", parent=self.window)
        except Exception as error:
            messagebox.showerror("保存できません", str(error), parent=self.window)

    def save_package(self) -> None:
        initial = Path(self.project.get("image_path", "results")).stem + "_froude_shields_log10.csv"
        path_string = filedialog.asksaveasfilename(parent=self.window, defaultextension=".csv", initialfile=initial, filetypes=[("CSV", "*.csv")])
        if not path_string:
            return
        try:
            start, end, mode, _x_limits, _y_limits = self._settings()
            csv_path, image_path, json_path = write_phase_package(self.project, self.rows, Path(path_string), start, end, mode)
            messagebox.showinfo("保存完了", f"3ファイルを保存しました。\n{csv_path.name}\n{image_path.name}\n{json_path.name}", parent=self.window)
        except Exception as error:
            messagebox.showerror("保存できません", str(error), parent=self.window)


def show_phase_plot(parent: Any, project: dict[str, Any]) -> PhasePlotDialog:
    return PhasePlotDialog(parent, project)

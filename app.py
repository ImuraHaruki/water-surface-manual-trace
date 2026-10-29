"""Tkinter desktop UI for manually tracing water and ground surfaces."""

from __future__ import annotations

import csv
import copy
import math
import sys
from pathlib import Path
from tkinter import BooleanVar, DoubleVar, StringVar, Tk, Canvas, filedialog, messagebox
from tkinter import ttk
from typing import Any

from PIL import Image, ImageTk

from core import (
    ImageGeometry,
    default_project,
    detect_geometry,
    export_manual_velocity_results_csv,
    export_results_csv,
    load_project,
    save_project,
)


Image.MAX_IMAGE_PIXELS = None
SURFACE_COLORS = {"water": "#0096ff", "ground": "#e600aa"}
SURFACE_LABELS = {"water": "水面", "ground": "地表面"}
MODE_LABEL_TO_KEY = {
    "点をクリックして結ぶ": "anchor",
    "ドラッグで描く": "freehand",
    "点・線を修正": "edit",
    "表示を移動": "pan",
    "水面が画像上方": "outside_top",
    "水面が画像下方": "outside_bottom",
}
SHIELDS_LABEL_TO_KEY = {
    "エネルギー勾配から計算": "slope",
    "Darcy摩擦係数から計算": "darcy",
    "Manning粗度から計算": "manning",
}
SHIELDS_KEY_TO_LABEL = {value: key for key, value in SHIELDS_LABEL_TO_KEY.items()}


class TraceApp:
    def __init__(self, root: Tk) -> None:
        self.root = root
        self.root.title("水面・地表面 手動トレース")
        self.root.geometry("1500x900")
        self.root.minsize(1100, 700)

        self.image: Image.Image | None = None
        self.photo: ImageTk.PhotoImage | None = None
        self.project: dict[str, Any] | None = None
        self.project_path: Path | None = None
        self.geometry: ImageGeometry | None = None
        self.zoom = 1.0
        self.origin_x = 0.0
        self.origin_y = 0.0
        self.drag_start: tuple[float, float] | None = None
        self.pan_start: tuple[float, float, float, float] | None = None
        self.current_segment = {"water": None, "ground": None}
        self.undo_history: list[dict[str, Any]] = []
        self.active_undo_state: dict[str, Any] | None = None
        self.active_edit_point: tuple[str, int, int] | None = None
        self.dirty = False

        self.surface_var = StringVar(value="water")
        self.mode_var = StringVar(value="点をクリックして結ぶ")
        self.show_water = BooleanVar(value=True)
        self.show_ground = BooleanVar(value=True)
        self.status_var = StringVar(value="画像またはプロジェクトを開いてください")

        self._build_ui()
        self._bind_shortcuts()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_ui(self) -> None:
        toolbar = ttk.Frame(self.root, padding=(8, 6))
        toolbar.pack(fill="x")
        ttk.Button(toolbar, text="画像を開く", command=self.open_image).pack(side="left", padx=2)
        ttk.Button(toolbar, text="プロジェクトを開く", command=self.open_project).pack(side="left", padx=2)
        ttk.Button(toolbar, text="保存", command=self.save_project).pack(side="left", padx=2)
        ttk.Button(toolbar, text="名前を付けて保存", command=self.save_project_as).pack(side="left", padx=2)
        ttk.Button(toolbar, text="結果CSVを書き出す", command=self.export_csv).pack(side="left", padx=(12, 2))
        ttk.Button(toolbar, text="グラフを表示", command=self.show_graph).pack(side="left", padx=2)
        ttk.Button(toolbar, text="Fr–θ相図", command=self.show_phase_plot).pack(side="left", padx=2)
        ttk.Separator(toolbar, orient="vertical").pack(side="left", fill="y", padx=10)
        ttk.Button(toolbar, text="横幅に合わせる", command=self.fit_width).pack(side="left", padx=2)
        ttk.Button(toolbar, text="高さに合わせる", command=self.fit_height).pack(side="left", padx=2)
        ttk.Button(toolbar, text="100%", command=self.actual_size).pack(side="left", padx=2)

        paned = ttk.Panedwindow(self.root, orient="horizontal")
        paned.pack(fill="both", expand=True)
        controls_container = ttk.Frame(paned, width=340)
        viewer = ttk.Frame(paned)
        paned.add(controls_container, weight=0)
        paned.add(viewer, weight=1)

        control_canvas = Canvas(controls_container, width=335, highlightthickness=0)
        control_scroll = ttk.Scrollbar(controls_container, orient="vertical", command=control_canvas.yview)
        self.controls = ttk.Frame(control_canvas, padding=8)
        self.controls.bind(
            "<Configure>", lambda event: control_canvas.configure(scrollregion=control_canvas.bbox("all"))
        )
        control_canvas.create_window((0, 0), window=self.controls, anchor="nw", width=320)
        control_canvas.configure(yscrollcommand=control_scroll.set)
        control_canvas.pack(side="left", fill="both", expand=True)
        control_scroll.pack(side="right", fill="y")

        self._build_trace_controls()
        self._build_calibration_controls()
        self._build_velocity_controls()
        self._build_manual_velocity_controls()
        self._build_physics_controls()

        self.canvas = Canvas(viewer, background="#303030", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda event: self.render())
        self.canvas.bind("<Button-1>", self.on_left_press)
        self.canvas.bind("<B1-Motion>", self.on_left_motion)
        self.canvas.bind("<ButtonRelease-1>", self.on_left_release)
        self.canvas.bind("<Button-3>", self.on_pan_press)
        self.canvas.bind("<B3-Motion>", self.on_pan_motion)
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)

        status = ttk.Label(self.root, textvariable=self.status_var, padding=(8, 4), relief="sunken", anchor="w")
        status.pack(fill="x")

    def _section(self, title: str) -> ttk.LabelFrame:
        frame = ttk.LabelFrame(self.controls, text=title, padding=8)
        frame.pack(fill="x", pady=(0, 8))
        return frame

    def _build_trace_controls(self) -> None:
        frame = self._section("トレース")
        ttk.Radiobutton(frame, text="水面", value="water", variable=self.surface_var).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(frame, text="地表面", value="ground", variable=self.surface_var).grid(row=0, column=1, sticky="w")
        ttk.Label(frame, text="操作").grid(row=1, column=0, sticky="w", pady=(6, 0))
        mode = ttk.Combobox(
            frame,
            textvariable=self.mode_var,
            state="readonly",
            values=tuple(MODE_LABEL_TO_KEY),
            width=19,
        )
        mode.grid(row=1, column=1, sticky="ew", pady=(6, 0))
        self.mode_var.trace_add("write", lambda *_: self.status_var.set(self.mode_var.get()))
        ttk.Button(frame, text="新しい区間（線を切る）", command=self.new_segment).grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 2))
        ttk.Button(frame, text="一つ戻す", command=self.undo_last).grid(row=3, column=0, sticky="ew", padx=(0, 2))
        ttk.Button(frame, text="現在区間を削除", command=self.delete_segment).grid(row=3, column=1, sticky="ew", padx=(2, 0))
        ttk.Button(frame, text="選択面をすべて消去", command=self.clear_surface).grid(row=4, column=0, columnspan=2, sticky="ew", pady=(2, 0))
        ttk.Button(frame, text="最後の画像外区間を戻す", command=self.undo_outside_interval).grid(row=5, column=0, columnspan=2, sticky="ew", pady=(2, 0))
        ttk.Checkbutton(frame, text="水面線を表示", variable=self.show_water, command=self.render).grid(row=6, column=0, sticky="w", pady=(6, 0))
        ttk.Checkbutton(frame, text="地表面線を表示", variable=self.show_ground, command=self.render).grid(row=6, column=1, sticky="w", pady=(6, 0))
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)

    def _entry_row(self, parent: ttk.Widget, row: int, label: str, variable: StringVar) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=2)
        ttk.Entry(parent, textvariable=variable, width=14).grid(row=row, column=1, sticky="ew", pady=2)

    def _build_calibration_controls(self) -> None:
        frame = self._section("画像・時刻・長さ")
        self.fps_var = StringVar(value="240")
        self.strip_var = StringVar(value="3")
        self.ppcm_var = StringVar(value="93.4")
        self.reference_time_var = StringVar(value="")
        self.plot_left_var = StringVar(value="0")
        self.plot_top_var = StringVar(value="0")
        self.plot_width_var = StringVar(value="0")
        self.plot_height_var = StringVar(value="0")
        self.vertical_scale_var = StringVar(value="1")
        fields = (
            ("fps", self.fps_var),
            ("1フレームの横幅 px", self.strip_var),
            ("1 cmの画素数", self.ppcm_var),
            ("地表基準時刻 s（空欄=最初）", self.reference_time_var),
            ("画像領域 左 px", self.plot_left_var),
            ("画像領域 上 px", self.plot_top_var),
            ("画像領域 幅 px", self.plot_width_var),
            ("画像領域 高さ px", self.plot_height_var),
            ("縦拡大率", self.vertical_scale_var),
        )
        for row, (label, variable) in enumerate(fields):
            self._entry_row(frame, row, label, variable)
        ttk.Button(frame, text="設定を反映", command=self.apply_settings).grid(row=len(fields), column=0, columnspan=2, sticky="ew", pady=(6, 0))
        frame.columnconfigure(1, weight=1)

    def _build_velocity_controls(self) -> None:
        frame = self._section("オプティカルフロー流速CSV")
        self.velocity_path_var = StringVar(value="")
        self.velocity_time_var = StringVar(value="time_s")
        self.velocity_value_var = StringVar(value="velocity")
        self.velocity_unit_var = StringVar(value="cm/s")
        ttk.Button(frame, text="流速CSVを選択", command=self.choose_velocity_csv).grid(row=0, column=0, sticky="ew")
        ttk.Button(frame, text="選択を解除", command=self.clear_velocity_csv).grid(row=0, column=1, sticky="ew", padx=(4, 0))
        ttk.Label(frame, textvariable=self.velocity_path_var, wraplength=285).grid(row=1, column=0, columnspan=2, sticky="w", pady=4)
        ttk.Label(frame, text="時刻列").grid(row=2, column=0, sticky="w")
        self.time_combo = ttk.Combobox(frame, textvariable=self.velocity_time_var, state="readonly")
        self.time_combo.grid(row=2, column=1, sticky="ew")
        ttk.Label(frame, text="流速列").grid(row=3, column=0, sticky="w")
        self.velocity_combo = ttk.Combobox(frame, textvariable=self.velocity_value_var, state="readonly")
        self.velocity_combo.grid(row=3, column=1, sticky="ew")
        ttk.Label(frame, text="単位").grid(row=4, column=0, sticky="w")
        ttk.Combobox(frame, textvariable=self.velocity_unit_var, values=("m/s", "cm/s", "mm/s"), state="readonly").grid(row=4, column=1, sticky="ew")
        frame.columnconfigure(1, weight=1)

    def _build_manual_velocity_controls(self) -> None:
        frame = self._section("手動計測流速CSV（点表示）")
        self.manual_velocity_path_var = StringVar(value="")
        self.manual_velocity_time_var = StringVar(value="time_s")
        self.manual_velocity_value_var = StringVar(value="velocity")
        self.manual_velocity_unit_var = StringVar(value="cm/s")
        ttk.Button(frame, text="手動計測CSVを選択", command=self.choose_manual_velocity_csv).grid(row=0, column=0, sticky="ew")
        ttk.Button(frame, text="選択を解除", command=self.clear_manual_velocity_csv).grid(row=0, column=1, sticky="ew", padx=(4, 0))
        ttk.Label(frame, textvariable=self.manual_velocity_path_var, wraplength=285).grid(row=1, column=0, columnspan=2, sticky="w", pady=4)
        ttk.Label(frame, text="時刻列").grid(row=2, column=0, sticky="w")
        self.manual_time_combo = ttk.Combobox(frame, textvariable=self.manual_velocity_time_var, state="readonly")
        self.manual_time_combo.grid(row=2, column=1, sticky="ew")
        ttk.Label(frame, text="流速列").grid(row=3, column=0, sticky="w")
        self.manual_velocity_combo = ttk.Combobox(frame, textvariable=self.manual_velocity_value_var, state="readonly")
        self.manual_velocity_combo.grid(row=3, column=1, sticky="ew")
        ttk.Label(frame, text="単位").grid(row=4, column=0, sticky="w")
        ttk.Combobox(frame, textvariable=self.manual_velocity_unit_var, values=("m/s", "cm/s", "mm/s"), state="readonly").grid(row=4, column=1, sticky="ew")
        ttk.Label(frame, text="時系列グラフでは赤い点として表示され、Fr・θも同じ点時刻で計算します。", wraplength=285).grid(row=5, column=0, columnspan=2, sticky="w", pady=(4, 0))
        frame.columnconfigure(1, weight=1)

    def _build_physics_controls(self) -> None:
        frame = self._section("フルード数・シールズ数")
        self.gravity_var = StringVar(value="9.80665")
        self.rho_var = StringVar(value="1000")
        self.rho_s_var = StringVar(value="2650")
        self.d50_var = StringVar(value="0.2")
        self.shields_method_var = StringVar(value="Darcy摩擦係数から計算")
        self.slope_var = StringVar(value="0.01")
        self.darcy_var = StringVar(value="0.03")
        self.manning_var = StringVar(value="0.012")
        fields = (
            ("重力加速度 m/s²", self.gravity_var),
            ("水密度 kg/m³", self.rho_var),
            ("土粒子密度 kg/m³", self.rho_s_var),
            ("d50 mm", self.d50_var),
        )
        for row, (label, variable) in enumerate(fields):
            self._entry_row(frame, row, label, variable)
        ttk.Label(frame, text="シールズ数のせん断応力").grid(row=4, column=0, sticky="w", pady=(6, 2))
        ttk.Combobox(frame, textvariable=self.shields_method_var, values=tuple(SHIELDS_LABEL_TO_KEY), state="readonly").grid(row=4, column=1, sticky="ew")
        self._entry_row(frame, 5, "エネルギー勾配 S", self.slope_var)
        self._entry_row(frame, 6, "Darcy摩擦係数 f", self.darcy_var)
        self._entry_row(frame, 7, "Manning粗度 n", self.manning_var)
        ttk.Label(frame, text="選択した方式に対応する係数だけを計算に使います。", wraplength=285).grid(row=8, column=0, columnspan=2, sticky="w", pady=(4, 0))
        frame.columnconfigure(1, weight=1)

    def _bind_shortcuts(self) -> None:
        self.root.bind("<Control-s>", lambda event: self.save_project())
        self.root.bind("<Control-z>", lambda event: self.undo_last())
        self.root.bind("<Escape>", lambda event: self.cancel_action())

    def open_image(self) -> None:
        if not self._confirm_save_changes():
            return
        path_string = filedialog.askopenfilename(filetypes=[("画像", "*.png *.jpg *.jpeg *.tif *.tiff"), ("すべて", "*.*")])
        if not path_string:
            return
        try:
            path = Path(path_string)
            image = Image.open(path).convert("RGB")
            geometry = detect_geometry(path, image.width, image.height)
            self.image = image
            self.geometry = geometry
            self.project = default_project(path, geometry)
            self.project_path = None
            self.current_segment = {"water": None, "ground": None}
            self.undo_history.clear()
            self._load_controls_from_project()
            self.dirty = False
            self.root.after(50, self.fit_height)
            self.status_var.set(f"読み込み: {path.name}  {image.width}×{image.height} px")
        except Exception as error:
            messagebox.showerror("画像を開けません", str(error))

    def open_project(self) -> None:
        if not self._confirm_save_changes():
            return
        path_string = filedialog.askopenfilename(filetypes=[("トレースプロジェクト", "*.trace.json"), ("JSON", "*.json")])
        if not path_string:
            return
        try:
            path = Path(path_string)
            project = load_project(path)
            image_path = Path(project["image_path"])
            image = Image.open(image_path).convert("RGB")
            self.project = project
            self.project_path = path
            self.image = image
            self.geometry = ImageGeometry.from_dict(project["geometry"])
            self.current_segment = {"water": None, "ground": None}
            self.undo_history.clear()
            self._load_controls_from_project()
            self.dirty = False
            self.root.after(50, self.fit_height)
            self.status_var.set(f"プロジェクト: {path.name}")
        except Exception as error:
            messagebox.showerror("プロジェクトを開けません", str(error))

    def _load_controls_from_project(self) -> None:
        assert self.project and self.geometry
        calibration = self.project["calibration"]
        self.fps_var.set(str(calibration["fps"]))
        self.strip_var.set(str(calibration["strip_width_px"]))
        self.ppcm_var.set(str(calibration["pixels_per_cm"]))
        self.reference_time_var.set("" if calibration.get("ground_reference_time_s") in (None, "") else str(calibration["ground_reference_time_s"]))
        self.plot_left_var.set(str(self.geometry.plot_left))
        self.plot_top_var.set(str(self.geometry.plot_top))
        self.plot_width_var.set(str(self.geometry.plot_width))
        self.plot_height_var.set(str(self.geometry.plot_height))
        self.vertical_scale_var.set(str(self.geometry.vertical_scale))
        velocity = self.project["velocity"]
        self.velocity_path_var.set(velocity.get("csv_path", ""))
        self.velocity_time_var.set(velocity.get("time_column", "time_s"))
        self.velocity_value_var.set(velocity.get("velocity_column", "velocity"))
        self.velocity_unit_var.set(velocity.get("unit", "cm/s"))
        if velocity.get("csv_path") and Path(velocity["csv_path"]).exists():
            self._populate_velocity_columns(Path(velocity["csv_path"]))
        manual_velocity = self.project.setdefault("manual_velocity", {"csv_path": "", "time_column": "time_s", "velocity_column": "velocity", "unit": "cm/s"})
        self.manual_velocity_path_var.set(manual_velocity.get("csv_path", ""))
        self.manual_velocity_time_var.set(manual_velocity.get("time_column", "time_s"))
        self.manual_velocity_value_var.set(manual_velocity.get("velocity_column", "velocity"))
        self.manual_velocity_unit_var.set(manual_velocity.get("unit", "cm/s"))
        if manual_velocity.get("csv_path") and Path(manual_velocity["csv_path"]).exists():
            self._populate_manual_velocity_columns(Path(manual_velocity["csv_path"]))
        physics = self.project["physics"]
        self.gravity_var.set(str(physics["gravity_m_s2"]))
        self.rho_var.set(str(physics["water_density_kg_m3"]))
        self.rho_s_var.set(str(physics["sediment_density_kg_m3"]))
        self.d50_var.set(str(physics["d50_mm"]))
        self.shields_method_var.set(SHIELDS_KEY_TO_LABEL.get(str(physics["shields_method"]), "Darcy摩擦係数から計算"))
        self.slope_var.set(str(physics["energy_slope"]))
        self.darcy_var.set(str(physics["darcy_friction_factor"]))
        self.manning_var.set(str(physics["manning_n"]))

    def apply_settings(self) -> bool:
        if not self.project or not self.image:
            return False
        try:
            geometry = ImageGeometry(
                self.image.width,
                self.image.height,
                float(self.plot_left_var.get()),
                float(self.plot_top_var.get()),
                float(self.plot_width_var.get()),
                float(self.plot_height_var.get()),
                float(self.vertical_scale_var.get()),
            )
            if geometry.plot_width <= 0 or geometry.plot_height <= 0 or geometry.vertical_scale <= 0:
                raise ValueError("画像領域と縦拡大率は正の値にしてください")
            calibration = self.project["calibration"]
            calibration["fps"] = float(self.fps_var.get())
            calibration["strip_width_px"] = float(self.strip_var.get())
            calibration["pixels_per_cm"] = float(self.ppcm_var.get())
            calibration["ground_reference_time_s"] = None if not self.reference_time_var.get().strip() else float(self.reference_time_var.get())
            self.geometry = geometry
            self.project["geometry"] = geometry.as_dict()
            velocity = self.project["velocity"]
            velocity.update({
                "csv_path": self.velocity_path_var.get(),
                "time_column": self.velocity_time_var.get(),
                "velocity_column": self.velocity_value_var.get(),
                "unit": self.velocity_unit_var.get(),
            })
            manual_velocity = self.project.setdefault("manual_velocity", {})
            manual_velocity.update({
                "csv_path": self.manual_velocity_path_var.get(),
                "time_column": self.manual_velocity_time_var.get(),
                "velocity_column": self.manual_velocity_value_var.get(),
                "unit": self.manual_velocity_unit_var.get(),
            })
            physics = self.project["physics"]
            physics.update({
                "gravity_m_s2": float(self.gravity_var.get()),
                "water_density_kg_m3": float(self.rho_var.get()),
                "sediment_density_kg_m3": float(self.rho_s_var.get()),
                "d50_mm": float(self.d50_var.get()),
                "shields_method": SHIELDS_LABEL_TO_KEY[self.shields_method_var.get()],
                "energy_slope": float(self.slope_var.get()),
                "darcy_friction_factor": float(self.darcy_var.get()),
                "manning_n": float(self.manning_var.get()),
            })
            self.dirty = True
            self.render()
            return True
        except Exception as error:
            messagebox.showerror("設定を確認してください", str(error))
            return False

    def save_project(self) -> None:
        if not self.project or not self.apply_settings():
            return
        if self.project_path is None:
            self.save_project_as()
            return
        try:
            save_project(self.project, self.project_path)
            self.dirty = False
            self.status_var.set(f"保存しました: {self.project_path.name}")
        except Exception as error:
            messagebox.showerror("保存できません", str(error))

    def save_project_as(self) -> None:
        if not self.project or not self.apply_settings():
            return
        initial = Path(self.project["image_path"]).stem + ".trace.json"
        path_string = filedialog.asksaveasfilename(defaultextension=".trace.json", initialfile=initial, filetypes=[("トレースプロジェクト", "*.trace.json")])
        if not path_string:
            return
        self.project_path = Path(path_string)
        self.save_project()

    def export_csv(self) -> None:
        if not self.project or not self.apply_settings():
            return
        initial = Path(self.project["image_path"]).stem + "_manual_results.csv"
        path_string = filedialog.asksaveasfilename(defaultextension=".csv", initialfile=initial, filetypes=[("CSV", "*.csv")])
        if not path_string:
            return
        try:
            output_path = Path(path_string)
            count = export_results_csv(self.project, output_path)
            manual_count = export_manual_velocity_results_csv(
                self.project,
                output_path.with_name(f"{output_path.stem}_manual_velocity.csv"),
            )
            self.status_var.set(f"結果CSVを書き出しました: {count}行")
            extra = "" if not manual_count else f"\n手動計測流速の計算結果: {manual_count}点"
            messagebox.showinfo("完了", f"{count}フレーム分を書き出しました。{extra}")
        except Exception as error:
            messagebox.showerror("CSVを書き出せません", str(error))

    def show_graph(self) -> None:
        if not self.project or not self.apply_settings():
            return
        try:
            from graph_window import show_results_graph

            show_results_graph(self.root, self.project)
            self.status_var.set("解析結果グラフを表示しました")
        except Exception as error:
            messagebox.showerror("グラフを表示できません", str(error))

    def show_phase_plot(self) -> None:
        if not self.project or not self.apply_settings():
            return
        try:
            from phase_plot import show_phase_plot

            show_phase_plot(self.root, self.project)
            self.status_var.set("フルード数・シールズ数相図を表示しました")
        except Exception as error:
            messagebox.showerror("相図を表示できません", str(error))

    def choose_velocity_csv(self) -> None:
        path_string = filedialog.askopenfilename(filetypes=[("CSV", "*.csv"), ("すべて", "*.*")])
        if not path_string:
            return
        path = Path(path_string)
        try:
            columns = self._populate_velocity_columns(path)
            self.velocity_path_var.set(str(path.resolve()))
            if "time_s" in columns:
                self.velocity_time_var.set("time_s")
            for candidate in ("velocity_used_m_s", "velocity_m_s", "velocity"):
                if candidate in columns:
                    self.velocity_value_var.set(candidate)
                    self.velocity_unit_var.set("m/s" if candidate.endswith("_m_s") else "cm/s")
                    break
            self.dirty = True
        except Exception as error:
            messagebox.showerror("流速CSVを読めません", str(error))

    def clear_velocity_csv(self) -> None:
        self.velocity_path_var.set("")
        self.velocity_time_var.set("time_s")
        self.velocity_value_var.set("velocity")
        self.velocity_unit_var.set("cm/s")
        self.time_combo.configure(values=())
        self.velocity_combo.configure(values=())
        if self.project:
            self.project["velocity"].update({"csv_path": "", "time_column": "time_s", "velocity_column": "velocity", "unit": "cm/s"})
        self.dirty = True
        self.status_var.set("オプティカルフロー流速CSVの選択を解除しました")

    def _populate_velocity_columns(self, path: Path) -> list[str]:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream)
            columns = next(reader)
        self.time_combo.configure(values=columns)
        self.velocity_combo.configure(values=columns)
        return columns

    def _populate_manual_velocity_columns(self, path: Path) -> list[str]:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            columns = next(csv.reader(stream))
        self.manual_time_combo.configure(values=columns)
        self.manual_velocity_combo.configure(values=columns)
        return columns

    def choose_manual_velocity_csv(self) -> None:
        path_string = filedialog.askopenfilename(filetypes=[("CSV", "*.csv"), ("すべて", "*.*")])
        if not path_string:
            return
        path = Path(path_string)
        try:
            columns = self._populate_manual_velocity_columns(path)
            self.manual_velocity_path_var.set(str(path.resolve()))
            if "time_s" in columns:
                self.manual_velocity_time_var.set("time_s")
            for candidate in ("velocity_used_m_s", "velocity_m_s", "velocity"):
                if candidate in columns:
                    self.manual_velocity_value_var.set(candidate)
                    self.manual_velocity_unit_var.set("m/s" if candidate.endswith("_m_s") else "cm/s")
                    break
            self.dirty = True
        except Exception as error:
            messagebox.showerror("手動計測CSVを読めません", str(error))

    def clear_manual_velocity_csv(self) -> None:
        self.manual_velocity_path_var.set("")
        self.manual_velocity_time_var.set("time_s")
        self.manual_velocity_value_var.set("velocity")
        self.manual_velocity_unit_var.set("cm/s")
        self.manual_time_combo.configure(values=())
        self.manual_velocity_combo.configure(values=())
        if self.project:
            self.project.setdefault("manual_velocity", {}).update({"csv_path": "", "time_column": "time_s", "velocity_column": "velocity", "unit": "cm/s"})
        self.dirty = True
        self.status_var.set("手動計測流速CSVの選択を解除しました")

    def new_segment(self) -> None:
        self.current_segment[self.surface_var.get()] = None
        self.status_var.set("次のクリックから新しい区間を開始します")

    def _capture_edit_state(self) -> dict[str, Any]:
        assert self.project is not None
        return {
            "traces": copy.deepcopy(self.project["traces"]),
            "water_outside": copy.deepcopy(self.project.get("water_outside", [])),
            "current_segment": copy.deepcopy(self.current_segment),
        }

    def _commit_undo_state(self, previous: dict[str, Any] | None) -> None:
        if previous is None or not self.project:
            return
        current = self._capture_edit_state()
        if current != previous:
            self.undo_history.append(previous)
            if len(self.undo_history) > 100:
                self.undo_history.pop(0)
            self.dirty = True

    def undo_last(self) -> None:
        if not self.project or not self.undo_history:
            self.status_var.set("戻せるトレース操作はありません")
            return
        previous = self.undo_history.pop()
        self.project["traces"] = previous["traces"]
        self.project["water_outside"] = previous["water_outside"]
        self.current_segment = previous["current_segment"]
        self.active_edit_point = None
        self.dirty = True
        self.status_var.set("直前のトレース操作を一つ戻しました")
        self.render()

    def delete_segment(self) -> None:
        if not self.project:
            return
        surface = self.surface_var.get()
        segments = self.project["traces"][surface]
        index = self.current_segment[surface]
        if index is None and segments:
            index = len(segments) - 1
        if index is not None and 0 <= index < len(segments):
            previous = self._capture_edit_state()
            segments.pop(index)
            self.current_segment[surface] = None
            self._commit_undo_state(previous)
            self.render()

    def undo_outside_interval(self) -> None:
        if self.project and self.project.get("water_outside"):
            previous = self._capture_edit_state()
            self.project["water_outside"].pop()
            self._commit_undo_state(previous)
            self.status_var.set("最後に指定した水面の画像外区間を戻しました")
            self.render()

    def clear_surface(self) -> None:
        if not self.project:
            return
        surface = self.surface_var.get()
        if not messagebox.askyesno("確認", f"{SURFACE_LABELS[surface]}の全トレースを消去しますか？"):
            return
        previous = self._capture_edit_state()
        self.project["traces"][surface] = []
        if surface == "water":
            self.project["water_outside"] = []
        self.current_segment[surface] = None
        self._commit_undo_state(previous)
        self.render()

    def cancel_action(self) -> None:
        if self.project and self.active_undo_state is not None:
            self.project["traces"] = self.active_undo_state["traces"]
            self.project["water_outside"] = self.active_undo_state["water_outside"]
            self.current_segment = self.active_undo_state["current_segment"]
        self.drag_start = None
        self.pan_start = None
        self.active_undo_state = None
        self.active_edit_point = None
        self.status_var.set("操作をキャンセルしました")
        self.render()

    def _confirm_save_changes(self) -> bool:
        if not self.dirty:
            return True
        answer = messagebox.askyesnocancel(
            "未保存の変更",
            "変更内容を保存しますか？",
        )
        if answer is None:
            return False
        if answer:
            self.save_project()
            return not self.dirty
        return True

    def on_close(self) -> None:
        if self._confirm_save_changes():
            self.root.destroy()

    def screen_to_image(self, screen_x: float, screen_y: float) -> tuple[float, float]:
        return self.origin_x + screen_x / self.zoom, self.origin_y + screen_y / self.zoom

    def image_to_screen(self, image_x: float, image_y: float) -> tuple[float, float]:
        return (image_x - self.origin_x) * self.zoom, (image_y - self.origin_y) * self.zoom

    def _add_data_point(self, image_x: float, image_y: float, start_new: bool = False) -> bool:
        if not self.project or not self.geometry or not self.geometry.contains_image_point(image_x, image_y):
            return False
        surface = self.surface_var.get()
        segments = self.project["traces"][surface]
        index = self.current_segment[surface]
        if start_new or index is None or not (0 <= index < len(segments)):
            segments.append([])
            index = len(segments) - 1
            self.current_segment[surface] = index
        data_x, data_y = self.geometry.image_to_data(image_x, image_y)
        if segments[index] and data_x < segments[index][-1][0]:
            self.status_var.set("時間は左から右へ進むように点を置いてください。新しい区間なら区間ボタンを押してください。")
            return False
        segments[index].append([data_x, data_y])
        self.dirty = True
        return True

    def _begin_edit(self, screen_x: float, screen_y: float) -> bool:
        if not self.project or not self.geometry:
            return False
        surface = self.surface_var.get()
        segments = self.project["traces"][surface]
        best_point: tuple[float, int, int] | None = None
        for segment_index, segment in enumerate(segments):
            for point_index, (data_x, data_y) in enumerate(segment):
                image_x, image_y = self.geometry.data_to_image(data_x, data_y)
                point_x, point_y = self.image_to_screen(image_x, image_y)
                distance = math.hypot(screen_x - point_x, screen_y - point_y)
                if best_point is None or distance < best_point[0]:
                    best_point = (distance, segment_index, point_index)
        if best_point and best_point[0] <= 12:
            self.active_edit_point = (surface, best_point[1], best_point[2])
            return True

        best_line: tuple[float, int, int, float] | None = None
        for segment_index, segment in enumerate(segments):
            for point_index in range(len(segment) - 1):
                image_x0, image_y0 = self.geometry.data_to_image(*segment[point_index])
                image_x1, image_y1 = self.geometry.data_to_image(*segment[point_index + 1])
                x0, y0 = self.image_to_screen(image_x0, image_y0)
                x1, y1 = self.image_to_screen(image_x1, image_y1)
                dx, dy = x1 - x0, y1 - y0
                length_squared = dx * dx + dy * dy
                ratio = 0.0 if length_squared == 0 else ((screen_x - x0) * dx + (screen_y - y0) * dy) / length_squared
                ratio = min(1.0, max(0.0, ratio))
                projected_x, projected_y = x0 + ratio * dx, y0 + ratio * dy
                distance = math.hypot(screen_x - projected_x, screen_y - projected_y)
                if best_line is None or distance < best_line[0]:
                    best_line = (distance, segment_index, point_index, ratio)
        if best_line and best_line[0] <= 12:
            _, segment_index, point_index, ratio = best_line
            first = segments[segment_index][point_index]
            second = segments[segment_index][point_index + 1]
            inserted = [
                first[0] + ratio * (second[0] - first[0]),
                first[1] + ratio * (second[1] - first[1]),
            ]
            segments[segment_index].insert(point_index + 1, inserted)
            self.active_edit_point = (surface, segment_index, point_index + 1)
            return True
        self.status_var.set("修正する点または線の近くをドラッグしてください")
        return False

    def _move_active_edit_point(self, screen_x: float, screen_y: float) -> None:
        if not self.project or not self.geometry or not self.active_edit_point:
            return
        surface, segment_index, point_index = self.active_edit_point
        segment = self.project["traces"][surface][segment_index]
        image_x, image_y = self.screen_to_image(screen_x, screen_y)
        image_x = min(self.geometry.plot_left + self.geometry.plot_width, max(self.geometry.plot_left, image_x))
        image_y = min(self.geometry.plot_top + self.geometry.plot_height, max(self.geometry.plot_top, image_y))
        data_x, data_y = self.geometry.image_to_data(image_x, image_y)
        if point_index > 0:
            data_x = max(segment[point_index - 1][0] + 1e-6, data_x)
        if point_index + 1 < len(segment):
            data_x = min(segment[point_index + 1][0] - 1e-6, data_x)
        segment[point_index] = [data_x, data_y]
        self.dirty = True

    def on_left_press(self, event: Any) -> None:
        if not self.image:
            return
        mode = MODE_LABEL_TO_KEY[self.mode_var.get()]
        if mode == "pan":
            self.on_pan_press(event)
            return
        image_x, image_y = self.screen_to_image(event.x, event.y)
        if mode == "anchor":
            previous = self._capture_edit_state() if self.project else None
            self._add_data_point(image_x, image_y)
            self._commit_undo_state(previous)
            self.render()
        elif mode == "freehand":
            self.active_undo_state = self._capture_edit_state() if self.project else None
            self._add_data_point(image_x, image_y, start_new=True)
            self.drag_start = (event.x, event.y)
            self.render()
        elif mode == "edit":
            self.active_undo_state = self._capture_edit_state() if self.project else None
            if self._begin_edit(event.x, event.y):
                self._move_active_edit_point(event.x, event.y)
                self.render()
            else:
                self.active_undo_state = None
        elif mode in ("outside_top", "outside_bottom"):
            if self.geometry:
                self.active_undo_state = self._capture_edit_state() if self.project else None
                data_x, _ = self.geometry.image_to_data(image_x, image_y)
                data_x = min(self.geometry.plot_width, max(0.0, data_x))
                self.drag_start = (data_x, data_x)

    def on_left_motion(self, event: Any) -> None:
        mode = MODE_LABEL_TO_KEY[self.mode_var.get()]
        if mode == "pan" and self.pan_start:
            self.on_pan_motion(event)
            return
        if mode == "freehand" and self.drag_start:
            if math.hypot(event.x - self.drag_start[0], event.y - self.drag_start[1]) >= 3:
                image_x, image_y = self.screen_to_image(event.x, event.y)
                self._add_data_point(image_x, image_y)
                self.drag_start = (event.x, event.y)
                self.render()
        elif mode == "edit" and self.active_edit_point:
            self._move_active_edit_point(event.x, event.y)
            self.render()
        elif mode in ("outside_top", "outside_bottom") and self.drag_start and self.geometry:
            image_x, _ = self.screen_to_image(event.x, event.y)
            data_x, _ = self.geometry.image_to_data(image_x, self.geometry.plot_top)
            self.drag_start = (self.drag_start[0], data_x)
            self.render()

    def on_left_release(self, event: Any) -> None:
        mode = MODE_LABEL_TO_KEY[self.mode_var.get()]
        if mode in ("outside_top", "outside_bottom") and self.drag_start and self.project:
            x0, x1 = self.drag_start
            self.project["water_outside"].append({"x0": min(x0, x1), "x1": max(x0, x1), "direction": "top" if mode == "outside_top" else "bottom"})
        if mode in ("freehand", "edit", "outside_top", "outside_bottom"):
            self._commit_undo_state(self.active_undo_state)
        self.active_undo_state = None
        self.active_edit_point = None
        self.drag_start = None
        self.pan_start = None
        self.render()

    def on_pan_press(self, event: Any) -> None:
        self.pan_start = (event.x, event.y, self.origin_x, self.origin_y)

    def on_pan_motion(self, event: Any) -> None:
        if not self.pan_start:
            return
        sx, sy, ox, oy = self.pan_start
        self.origin_x = ox - (event.x - sx) / self.zoom
        self.origin_y = oy - (event.y - sy) / self.zoom
        self._clamp_origin()
        self.render()

    def on_mouse_wheel(self, event: Any) -> None:
        if not self.image:
            return
        old_zoom = self.zoom
        factor = 1.2 if event.delta > 0 else 1 / 1.2
        new_zoom = min(8.0, max(0.02, old_zoom * factor))
        image_x, image_y = self.screen_to_image(event.x, event.y)
        self.zoom = new_zoom
        self.origin_x = image_x - event.x / new_zoom
        self.origin_y = image_y - event.y / new_zoom
        self._clamp_origin()
        self.render()

    def _clamp_origin(self) -> None:
        if not self.image:
            return
        width = max(1, self.canvas.winfo_width()) / self.zoom
        height = max(1, self.canvas.winfo_height()) / self.zoom
        self.origin_x = min(max(0.0, self.origin_x), max(0.0, self.image.width - width))
        self.origin_y = min(max(0.0, self.origin_y), max(0.0, self.image.height - height))

    def fit_width(self) -> None:
        if not self.image:
            return
        self.zoom = max(0.02, self.canvas.winfo_width() / self.image.width)
        self.origin_x = self.origin_y = 0.0
        self.render()

    def fit_height(self) -> None:
        if not self.image:
            return
        self.zoom = max(0.02, self.canvas.winfo_height() / self.image.height)
        self.origin_x = self.origin_y = 0.0
        self.render()

    def actual_size(self) -> None:
        self.zoom = 1.0
        self.origin_x = self.origin_y = 0.0
        self.render()

    def render(self) -> None:
        self.canvas.delete("all")
        if not self.image:
            return
        canvas_width = max(1, self.canvas.winfo_width())
        canvas_height = max(1, self.canvas.winfo_height())
        self._clamp_origin()
        crop_width = min(self.image.width - self.origin_x, canvas_width / self.zoom)
        crop_height = min(self.image.height - self.origin_y, canvas_height / self.zoom)
        if crop_width <= 0 or crop_height <= 0:
            return
        box = (
            int(self.origin_x),
            int(self.origin_y),
            int(math.ceil(self.origin_x + crop_width)),
            int(math.ceil(self.origin_y + crop_height)),
        )
        crop = self.image.crop(box)
        display_size = (max(1, round(crop.width * self.zoom)), max(1, round(crop.height * self.zoom)))
        crop = crop.resize(display_size, Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(crop)
        self.canvas.create_image(0, 0, image=self.photo, anchor="nw")
        self._render_plot_boundary()
        self._render_traces()
        self._render_outside_intervals()

    def _render_plot_boundary(self) -> None:
        if not self.geometry:
            return
        x0, y0 = self.image_to_screen(self.geometry.plot_left, self.geometry.plot_top)
        x1, y1 = self.image_to_screen(
            self.geometry.plot_left + self.geometry.plot_width,
            self.geometry.plot_top + self.geometry.plot_height,
        )
        self.canvas.create_rectangle(x0, y0, x1, y1, outline="#ffffff", dash=(4, 4))

    def _render_traces(self) -> None:
        if not self.project or not self.geometry:
            return
        for surface in ("ground", "water"):
            if surface == "water" and not self.show_water.get():
                continue
            if surface == "ground" and not self.show_ground.get():
                continue
            for segment in self.project["traces"][surface]:
                coordinates: list[float] = []
                last_screen_x: float | None = None
                for data_x, data_y in segment:
                    image_x, image_y = self.geometry.data_to_image(data_x, data_y)
                    screen_x, screen_y = self.image_to_screen(image_x, image_y)
                    if last_screen_x is not None and abs(screen_x - last_screen_x) < 0.8:
                        coordinates[-2:] = [screen_x, screen_y]
                    else:
                        coordinates.extend((screen_x, screen_y))
                    last_screen_x = screen_x
                if len(coordinates) >= 4:
                    self.canvas.create_line(*coordinates, fill="white", width=6, smooth=False)
                    self.canvas.create_line(*coordinates, fill=SURFACE_COLORS[surface], width=3, smooth=False)
                elif len(coordinates) == 2:
                    self.canvas.create_oval(coordinates[0]-4, coordinates[1]-4, coordinates[0]+4, coordinates[1]+4, fill=SURFACE_COLORS[surface], outline="white")
                if MODE_LABEL_TO_KEY[self.mode_var.get()] == "edit" and surface == self.surface_var.get():
                    for data_x, data_y in segment:
                        image_x, image_y = self.geometry.data_to_image(data_x, data_y)
                        screen_x, screen_y = self.image_to_screen(image_x, image_y)
                        self.canvas.create_oval(
                            screen_x - 5,
                            screen_y - 5,
                            screen_x + 5,
                            screen_y + 5,
                            fill=SURFACE_COLORS[surface],
                            outline="white",
                            width=2,
                        )

    def _render_outside_intervals(self) -> None:
        if not self.project or not self.geometry:
            return
        intervals = list(self.project.get("water_outside", []))
        mode = MODE_LABEL_TO_KEY[self.mode_var.get()]
        if self.drag_start and mode in ("outside_top", "outside_bottom"):
            intervals.append({"x0": self.drag_start[0], "x1": self.drag_start[1], "direction": "top" if mode == "outside_top" else "bottom"})
        for interval in intervals:
            x0, x1 = sorted((float(interval["x0"]), float(interval["x1"])))
            y_data = 0 if interval.get("direction") == "top" else self.geometry.source_height
            image_x0, image_y = self.geometry.data_to_image(x0, y_data)
            image_x1, _ = self.geometry.data_to_image(x1, y_data)
            sx0, sy = self.image_to_screen(image_x0, image_y)
            sx1, _ = self.image_to_screen(image_x1, image_y)
            color = "#00c8ff" if interval.get("direction") == "top" else "#0066cc"
            self.canvas.create_line(sx0, sy, sx1, sy, fill="white", width=9)
            self.canvas.create_line(sx0, sy, sx1, sy, fill=color, width=5)


def main() -> None:
    root = Tk()
    style = ttk.Style(root)
    if "vista" in style.theme_names():
        style.theme_use("vista")
    TraceApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

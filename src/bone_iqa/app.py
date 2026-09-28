from __future__ import annotations

import sys
import tkinter as tk
from tkinter import font as tkfont
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk

import numpy as np
from PIL import Image, ImageTk

from .evaluator import evaluate_project, export_csv_bundle
from .images import load_grayscale
from .metrics import EPSILON, roi_statistics
from .models import ROI
from .project_store import ProjectError, ProjectStore, nearest_surrounding_roi_id
from .roi_recommender import ROIRecommendation, remove_recommendation
from .roi_templates import ROI_TEMPLATE_SIZE, ROI_TEMPLATE_VERSION, build_roi_template
from .validation import validate_project


IMAGE_TYPES = [("灰度图像", "*.bmp *.png *.tif *.tiff"), ("所有文件", "*.*")]
ROI_LABELS = {
    "weak_bone": "弱骨骼",
    "strong_bone": "强骨骼",
    "surrounding": "邻域",
    "background": "背景",
}
ROI_TYPES_BY_LABEL = {label: roi_type for roi_type, label in ROI_LABELS.items()}
ROI_COLORS = {
    "weak_bone": "#ffd54f",
    "strong_bone": "#ef5350",
    "surrounding": "#26c6da",
    "background": "#42a5f5",
}
MAX_COMPARISON_SCHEMES = 10
SUMMARY_THUMBNAIL_SIZE = (60, 96)


def comparison_grid_columns(count: int) -> int:
    """Choose a readable comparison grid without shrinking ten images into one row."""
    return count if count <= 4 else 3


def format_number(value, digits: int = 4) -> str:
    if value is None or value == "":
        return "—"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


METRIC_TOOLTIPS = {
    "Local CNR": "衡量弱骨骼与其附近区域是否容易区分。通常越高表示局部辨识度越好，但它不直接代表图像是否锐利。",
    "Background CNR": "衡量弱骨骼相对于体外背景是否突出。通常越高越好。背景被抬亮或背景噪声增大时，该指标可能下降。",
    "Weak AG": "平均梯度，用于反映弱骨骼区域的边缘和细节变化。通常越高表示边缘更明显，但噪声也可能使 AG 虚高，因此必须结合 Noise 一起判断。",
    "Noise": "表示背景区域灰度波动程度。通常越低表示背景越干净。但图像被过度平滑或整体压暗时 Noise 也可能下降，因此不能单独判断图像质量。",
    "SSIM": "衡量处理结果与原图结构的相似程度，越接近 1 越相似。高 SSIM 只说明结构变化较小，不代表增强效果一定更好。",
    "Strong Sat.%": "强骨骼 ROI 中接近灰度饱和的像素比例。过高可能意味着强骨骼发白、内部灰度细节丢失。",
    "Δ%": "相对于原图的百分比变化。正负只表示变化方向，是否改善需要结合具体指标含义判断。",
    "Sat. Δpp": "强骨骼饱和率相对于原图变化的百分点，不是普通百分比变化。",
}


class HeaderTooltip:
    """Small delayed tooltip for metric table headings."""

    def __init__(self, widget: ttk.Treeview, descriptions: dict[str, str]):
        self.widget = widget
        self.descriptions = descriptions
        self.tip: tk.Toplevel | None = None
        self.after_id: str | None = None
        widget.bind("<Motion>", self._motion, add="+")
        widget.bind("<Leave>", self.hide, add="+")

    def _motion(self, event):
        self.hide()
        if self.widget.identify_region(event.x, event.y) != "heading":
            return
        column = self.widget.identify_column(event.x)
        heading = self.widget.heading(column, "text")
        text = next((value for key, value in self.descriptions.items() if key in heading), None)
        if text:
            self.after_id = self.widget.after(450, lambda: self.show(text, event.x_root + 12, event.y_root + 12))

    def show(self, text: str, x: int, y: int):
        self.after_id = None
        if self.tip is not None:
            return
        tip = tk.Toplevel(self.widget)
        tip.wm_overrideredirect(True)
        tip.attributes("-topmost", True)
        tip.geometry(f"+{x}+{y}")
        ttk.Label(tip, text=text, justify="left", wraplength=380, padding=8, relief="solid", borderwidth=1).pack()
        self.tip = tip

    def hide(self, _event=None):
        if self.after_id is not None:
            self.widget.after_cancel(self.after_id)
            self.after_id = None
        if self.tip is not None:
            self.tip.destroy()
            self.tip = None


class HoverTooltip:
    """Tooltip for a normal label, used by the scrollable summary header."""

    def __init__(self, widget: tk.Misc, text: str):
        self.widget = widget
        self.text = text
        self.tip: tk.Toplevel | None = None
        self.after_id: str | None = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self.hide, add="+")

    def _schedule(self, event):
        self.hide()
        self.after_id = self.widget.after(450, lambda: self.show(event.x_root + 12, event.y_root + 12))

    def show(self, x: int, y: int):
        self.after_id = None
        if self.tip is not None:
            return
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.attributes("-topmost", True)
        self.tip.geometry(f"+{x}+{y}")
        ttk.Label(self.tip, text=self.text, justify="left", wraplength=380, padding=8, relief="solid", borderwidth=1).pack()

    def hide(self, _event=None):
        if self.after_id is not None:
            self.widget.after_cancel(self.after_id)
            self.after_id = None
        if self.tip is not None:
            self.tip.destroy()
            self.tip = None


class MetricSummaryTable:
    """Scrollable widget table with real thumbnail cells and a shared header."""

    COLUMNS = (
        ("结果图", 88), ("比较", 60), ("方案", 150), ("Local CNR", 112), ("Local Δ%", 105),
        ("Background CNR", 132), ("Background Δ%", 126), ("Weak AG", 105), ("AG Δ%", 95),
        ("Noise", 100), ("Noise Δ%", 108), ("SSIM", 95), ("Strong Sat.%", 118), ("Sat. Δpp", 108),
        ("All CNR-bg", 118), ("All CNR-local", 128), ("Strong CNR-bg", 128), ("Strong CNR-local", 138),
        ("Global AG", 110), ("Strong AG", 110), ("状态", 230),
    )

    def __init__(self, parent, on_row_click, on_compare_click, on_preview):
        self.on_row_click = on_row_click
        self.on_compare_click = on_compare_click
        self.on_preview = on_preview
        self.frame = ttk.Frame(parent)
        self.frame.rowconfigure(1, weight=1)
        self.frame.columnconfigure(0, weight=1)
        self.header_canvas = tk.Canvas(self.frame, height=32, highlightthickness=0, background="#e8e8e8")
        self.body_canvas = tk.Canvas(self.frame, highlightthickness=0, background="#ffffff")
        self.xscroll = ttk.Scrollbar(self.frame, orient="horizontal", command=self._xview)
        self.yscroll = ttk.Scrollbar(self.frame, orient="vertical", command=self.body_canvas.yview)
        self.header_canvas.configure(xscrollcommand=self._header_xscroll, yscrollcommand=lambda *_: None)
        self.body_canvas.configure(xscrollcommand=self._body_xscroll, yscrollcommand=self.yscroll.set)
        self.header_canvas.grid(row=0, column=0, sticky="ew")
        self.body_canvas.grid(row=1, column=0, sticky="nsew")
        self.yscroll.grid(row=1, column=1, sticky="ns")
        self.xscroll.grid(row=2, column=0, sticky="ew")
        self.header_inner = ttk.Frame(self.header_canvas)
        self.body_inner = ttk.Frame(self.body_canvas)
        self.total_width = sum(width for _heading, width in self.COLUMNS)
        self.header_inner.configure(width=self.total_width)
        self.body_inner.configure(width=self.total_width)
        self.body_inner.columnconfigure(0, minsize=self.total_width, weight=0)
        self.header_window = self.header_canvas.create_window((0, 0), window=self.header_inner, anchor="nw")
        self.body_window = self.body_canvas.create_window((0, 0), window=self.body_inner, anchor="nw")
        self.header_canvas.itemconfigure(self.header_window, width=self.total_width)
        self.body_canvas.itemconfigure(self.body_window, width=self.total_width)
        self.header_inner.bind("<Configure>", self._update_scrollregions, add="+")
        self.body_inner.bind("<Configure>", self._update_scrollregions, add="+")
        self.body_canvas.bind("<Configure>", self._body_configure, add="+")
        self.header_canvas.bind("<Configure>", self._header_configure, add="+")
        self.row_widgets: dict[str, dict[str, tk.Misc]] = {}
        self._build_header()

    def _build_header(self):
        for index, (heading, width) in enumerate(self.COLUMNS):
            self.header_inner.columnconfigure(index, minsize=width, weight=0)
            label = tk.Label(self.header_inner, text=heading, width=max(1, width // 8), anchor="center", background="#e8e8e8", relief="groove", borderwidth=1, font=("Microsoft YaHei UI", 9, "bold"))
            label.grid(row=0, column=index, sticky="nsew")
            tooltip = METRIC_TOOLTIPS.get(heading)
            if heading == "Sat. Δpp":
                tooltip = METRIC_TOOLTIPS["Sat. Δpp"]
            elif tooltip is None and "Δ" in heading:
                tooltip = METRIC_TOOLTIPS["Δ%"]
            if tooltip:
                HoverTooltip(label, tooltip)

    def _body_configure(self, event):
        self.body_canvas.itemconfigure(self.body_window, height=max(event.height, self.body_inner.winfo_reqheight()))
        self._update_scrollregions()

    def _header_configure(self, event):
        self.header_canvas.itemconfigure(self.header_window, height=max(event.height, 32))
        self._update_scrollregions()

    def _update_scrollregions(self, _event=None):
        self.header_canvas.configure(scrollregion=self.header_canvas.bbox("all"))
        self.body_canvas.configure(scrollregion=self.body_canvas.bbox("all"))

    def _xview(self, *args):
        self.header_canvas.xview(*args)
        self.body_canvas.xview(*args)

    def _header_xscroll(self, first, last):
        self.xscroll.set(first, last)

    def _body_xscroll(self, first, last):
        self.xscroll.set(first, last)
        self.header_canvas.xview_moveto(first)

    def clear(self):
        for widgets in self.row_widgets.values():
            widgets["frame"].destroy()
        self.row_widgets.clear()
        self._update_scrollregions()

    def add_row(self, row_id: str, thumbnail, values: tuple[str, ...], compared: bool):
        row = ttk.Frame(self.body_inner, width=self.total_width, height=110)
        row.grid(row=len(self.row_widgets), column=0, sticky="ew")
        row.grid_propagate(False)
        for index, (_heading, width) in enumerate(self.COLUMNS):
            row.columnconfigure(index, minsize=width, weight=0)
        image_label = ttk.Label(row, image=thumbnail, anchor="center")
        image_label.image = thumbnail
        image_label.grid(row=0, column=0, sticky="nsew", padx=2, pady=5)
        image_label.bind("<Button-1>", lambda _event, rid=row_id: self.on_row_click(rid))
        image_label.bind("<Double-1>", lambda _event, rid=row_id: self.on_preview(rid))
        compare_label = ttk.Label(row, text="☑" if compared else "☐", anchor="center", font=("Segoe UI Symbol", 15))
        compare_label.grid(row=0, column=1, sticky="nsew")
        compare_label.bind("<Button-1>", lambda _event, rid=row_id: self.on_compare_click(rid))
        for index, value in enumerate(values, start=2):
            label = ttk.Label(row, text=value, anchor="center" if index != 2 and index != len(self.COLUMNS) - 1 else "w", justify="left", wraplength=self.COLUMNS[index][1] - 8)
            label.grid(row=0, column=index, sticky="nsew", padx=2)
            label.bind("<Button-1>", lambda _event, rid=row_id: self.on_row_click(rid))
        self.row_widgets[row_id] = {"frame": row, "compare": compare_label}
        self._update_scrollregions()

    def set_compared(self, row_id: str, compared: bool):
        widgets = self.row_widgets.get(row_id)
        if widgets:
            widgets["compare"].configure(text="☑" if compared else "☐")


class ImagePreviewWindow(tk.Toplevel):
    def __init__(self, parent, image: Image.Image, title: str):
        super().__init__(parent)
        self.title(f"图像预览：{title}")
        self.geometry("900x700")
        self.image = image.convert("L")
        self.zoom = 1.0
        self.center = (0.5, 0.5)
        self.photo = None
        self.canvas = tk.Canvas(self, background="#111", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=6, pady=6)
        toolbar = ttk.Frame(self)
        toolbar.pack(fill="x", padx=6, pady=(0, 6))
        ttk.Button(toolbar, text="放大", command=lambda: self._change_zoom(1.25)).pack(side="left", padx=2)
        ttk.Button(toolbar, text="缩小", command=lambda: self._change_zoom(0.8)).pack(side="left", padx=2)
        ttk.Button(toolbar, text="恢复原始视图", command=self.reset_view).pack(side="left", padx=2)
        self.canvas.bind("<Configure>", lambda _event: self.render())
        self.canvas.bind("<MouseWheel>", self._wheel)
        self.canvas.bind("<ButtonPress-1>", self._press)
        self.canvas.bind("<B1-Motion>", self._drag)
        self.drag_start = None
        self.after(30, self.render)

    def reset_view(self):
        self.zoom = 1.0
        self.center = (0.5, 0.5)
        self.render()

    def _change_zoom(self, factor: float):
        self.zoom = min(8.0, max(0.25, self.zoom * factor))
        self.render()

    def _wheel(self, event):
        self._change_zoom(1.2 if event.delta > 0 else 0.833333)

    def _press(self, event):
        self.drag_start = (event.x, event.y, self.center)

    def _drag(self, event):
        if not self.drag_start:
            return
        x0, y0, (cx, cy) = self.drag_start
        width = max(self.canvas.winfo_width(), 1)
        height = max(self.canvas.winfo_height(), 1)
        self.center = (min(1.0, max(0.0, cx - (event.x - x0) / width / self.zoom)), min(1.0, max(0.0, cy - (event.y - y0) / height / self.zoom)))
        self.render()

    def render(self):
        width = max(self.canvas.winfo_width(), 100)
        height = max(self.canvas.winfo_height(), 100)
        aspect = width / height
        source_aspect = self.image.width / max(self.image.height, 1)
        crop_w = self.image.width / self.zoom
        crop_h = self.image.height / self.zoom
        if crop_w / crop_h > aspect:
            crop_w = crop_h * aspect
        else:
            crop_h = crop_w / aspect
        cx, cy = self.center[0] * self.image.width, self.center[1] * self.image.height
        left = min(max(0, cx - crop_w / 2), self.image.width - crop_w)
        top = min(max(0, cy - crop_h / 2), self.image.height - crop_h)
        crop = self.image.crop((int(left), int(top), int(left + crop_w), int(top + crop_h))).resize((width, height), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(crop)
        self.canvas.delete("all")
        self.canvas.create_image(width / 2, height / 2, image=self.photo)


class ComparisonWindow(tk.Toplevel):
    METRIC_ROWS = (
        ("Local CNR", "weak_bone_mean_cnr_local"), ("Local Δ%", "local_cnr_change_percent"),
        ("Background CNR", "weak_bone_mean_cnr_background"), ("Background Δ%", "background_cnr_change_percent"),
        ("Weak AG", "weak_bone_mean_average_gradient"), ("AG Δ%", "weak_ag_change_percent"),
        ("Noise", "background_noise_pooled"), ("Noise Δ%", "background_noise_change_percent"),
        ("SSIM", "ssim"), ("Strong Sat.%", "strong_bone_saturation_mean"), ("Sat. Δpp", "saturation_change_pp"),
    )

    def __init__(self, parent, store: ProjectStore, result: dict, scheme_ids: list[str]):
        super().__init__(parent)
        self.title("方案比较")
        self.geometry("1280x820")
        self.minsize(900, 620)
        self.store = store
        self.result = result
        self.scheme_ids = scheme_ids
        self.images: list[Image.Image] = []
        self.names: list[str] = []
        self.zoom = 1.0
        self.center = (0.5, 0.5)
        self.canvases: list[tk.Canvas] = []
        self.drag_start = None
        records = ([store.project.original] if store.project.original else []) + store.project.algorithms
        by_id = {record.id: record for record in records}
        for scheme_id in scheme_ids:
            record = by_id.get(scheme_id)
            if record is None:
                continue
            self.names.append(record.display_name)
            try:
                self.images.append(Image.open(store.resolve(record.relative_path)).convert("L"))
            except Exception:
                self.images.append(Image.new("L", (500, 800), 0))

        toolbar = ttk.Frame(self, padding=6)
        toolbar.pack(fill="x")
        ttk.Button(toolbar, text="恢复全部视图", command=self.reset_view).pack(side="left", padx=2)
        roi_names = [roi.name for roi in store.project.rois if roi.type in {"weak_bone", "strong_bone"}]
        self.roi_var = tk.StringVar()
        self.roi_combo = ttk.Combobox(toolbar, textvariable=self.roi_var, values=roi_names, state="readonly", width=16)
        self.roi_combo.pack(side="left", padx=8)
        ttk.Button(toolbar, text="定位 ROI", command=self.focus_roi).pack(side="left", padx=2)
        ttk.Button(toolbar, text="查看两方案差异图", command=self.show_difference).pack(side="left", padx=2)

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=8, pady=4)
        image_tab = ttk.Frame(self.notebook)
        image_tab.rowconfigure(0, weight=1)
        image_tab.columnconfigure(0, weight=1)
        self.notebook.add(image_tab, text="图像比较")
        self._build_image_grid(image_tab)
        data_tab = ttk.Frame(self.notebook)
        data_tab.rowconfigure(0, weight=1)
        data_tab.columnconfigure(0, weight=1)
        self.notebook.add(data_tab, text="数据比较")
        self._build_data_tree(data_tab)
        self.after(80, self.render)

    def _build_image_grid(self, parent):
        count = len(self.images)
        columns = comparison_grid_columns(count)
        cell_width = 300 if columns <= 3 else 270
        cell_height = 315
        frame = ttk.Frame(parent)
        frame.grid(row=0, column=0, sticky="nsew")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        canvas = tk.Canvas(frame, background="#d5d5d5", highlightthickness=0)
        xbar = ttk.Scrollbar(frame, orient="horizontal", command=canvas.xview)
        ybar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        canvas.configure(xscrollcommand=xbar.set, yscrollcommand=ybar.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        ybar.grid(row=0, column=1, sticky="ns")
        xbar.grid(row=1, column=0, sticky="ew")
        inner = ttk.Frame(canvas)
        window = canvas.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>", lambda _event: canvas.configure(scrollregion=canvas.bbox("all")), add="+")
        canvas.bind("<Configure>", lambda event: canvas.itemconfigure(window, height=max(event.height, inner.winfo_reqheight())), add="+")
        for index, (name, image) in enumerate(zip(self.names, self.images)):
            row, column = divmod(index, columns)
            cell = ttk.Frame(inner, width=cell_width, height=cell_height, padding=4)
            cell.grid(row=row, column=column, sticky="nw", padx=5, pady=5)
            cell.grid_propagate(False)
            cell.rowconfigure(0, weight=1)
            cell.columnconfigure(0, weight=1)
            view = tk.Canvas(cell, width=cell_width - 12, height=cell_height - 42, background="#111", highlightthickness=0)
            view.grid(row=0, column=0, sticky="nsew")
            ttk.Label(cell, text=name, anchor="center", wraplength=cell_width - 10).grid(row=1, column=0, sticky="ew", pady=(3, 0))
            view.bind("<Configure>", lambda _event: self.render(), add="+")
            view.bind("<MouseWheel>", self._wheel, add="+")
            view.bind("<ButtonPress-1>", self._press, add="+")
            view.bind("<B1-Motion>", self._drag, add="+")
            self.canvases.append(view)

    def _build_data_tree(self, parent):
        columns = [str(index) for index in range(len(self.names))]
        tree_frame = ttk.Frame(parent)
        tree_frame.grid(row=0, column=0, sticky="nsew")
        tree_frame.rowconfigure(0, weight=1)
        tree_frame.columnconfigure(0, weight=1)
        tree = ttk.Treeview(tree_frame, columns=columns, show="tree headings")
        tree.heading("#0", text="指标")
        tree.column("#0", width=190, minwidth=190, anchor="w", stretch=False)
        for index, name in enumerate(self.names):
            key = str(index)
            tree.heading(key, text=name)
            tree.column(key, width=155, minwidth=125, anchor="center", stretch=False)
        ybar = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
        xbar = ttk.Scrollbar(tree_frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=ybar.set, xscrollcommand=xbar.set)
        tree.grid(row=0, column=0, sticky="nsew")
        ybar.grid(row=0, column=1, sticky="ns")
        xbar.grid(row=1, column=0, sticky="ew")
        by_id = {row.get("scheme_id"): row for row in self.result.get("metrics", [])}
        for label, key in self.METRIC_ROWS:
            values = []
            for scheme_id in self.scheme_ids:
                value = by_id.get(scheme_id, {}).get(key)
                if key == "strong_bone_saturation_mean" and value is not None:
                    value *= 100
                values.append(format_number(value, 2 if "Δ" in label or "Sat" in label else 4))
            tree.insert("", "end", text=label, values=values)
        HeaderTooltip(tree, METRIC_TOOLTIPS)
        self.data_tree = tree

    def reset_view(self):
        self.zoom, self.center = 1.0, (0.5, 0.5)
        self.render()

    def _wheel(self, event):
        self.zoom = min(8.0, max(0.25, self.zoom * (1.2 if event.delta > 0 else 0.833333)))
        self.render()

    def _press(self, event):
        self.drag_start = (event.x, event.y, self.center)

    def _drag(self, event):
        if not self.drag_start or not self.canvases:
            return
        x0, y0, (cx, cy) = self.drag_start
        canvas = self.canvases[0]
        self.center = (
            min(1.0, max(0.0, cx - (event.x - x0) / max(canvas.winfo_width(), 1) / self.zoom)),
            min(1.0, max(0.0, cy - (event.y - y0) / max(canvas.winfo_height(), 1) / self.zoom)),
        )
        self.render()

    def focus_roi(self):
        name = self.roi_var.get()
        roi = next((roi for roi in self.store.project.rois if roi.name == name), None)
        if roi is None or not self.images:
            return
        width, height = self.images[0].size
        self.center = ((roi.x + roi.width / 2) / width, (roi.y + roi.height / 2) / height)
        self.zoom = 3.0
        self.render()

    def render(self):
        if not self.canvases or not self.images:
            return
        for canvas, image in zip(self.canvases, self.images):
            width = max(canvas.winfo_width(), 100)
            height = max(canvas.winfo_height(), 100)
            aspect = width / height
            crop_w = image.width / self.zoom
            crop_h = image.height / self.zoom
            if crop_w / crop_h > aspect:
                crop_w = crop_h * aspect
            else:
                crop_h = crop_w / aspect
            cx, cy = self.center[0] * image.width, self.center[1] * image.height
            left = min(max(0, cx - crop_w / 2), image.width - crop_w)
            top = min(max(0, cy - crop_h / 2), image.height - crop_h)
            crop = image.crop((int(left), int(top), int(left + crop_w), int(top + crop_h))).resize((width, height), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(crop)
            canvas.delete("all")
            canvas.create_image(width / 2, height / 2, image=photo)
            canvas._comparison_photo = photo

    def show_difference(self):
        if len(self.images) != 2:
            messagebox.showinfo("差异图", "差异图需要恰好选择两个方案。", parent=self)
            return
        first = np.asarray(self.images[0], dtype=np.float32)
        second = np.asarray(self.images[1], dtype=np.float32)
        if first.shape != second.shape:
            messagebox.showwarning("差异图", "两个方案尺寸不一致，不能生成差异图。", parent=self)
            return
        difference = np.abs(first - second)
        peak = float(difference.max())
        if peak > 0:
            difference = difference / peak * 255.0
        ImagePreviewWindow(self, Image.fromarray(difference.astype("uint8")), f"{self.names[0]} - {self.names[1]}（差异图）")


class BoneIQAApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("骨骼图像增强质量评价平台")
        self.geometry("1280x800")
        self.minsize(1050, 680)
        self.store: ProjectStore | None = None
        self._roi_photo = None
        self._roi_scale = 1.0
        self._roi_offset = (0.0, 0.0)
        self._drag_start = None
        self._drag_preview = None
        self._selected_roi_id: str | None = None
        self._selected_recommendation_id: str | None = None
        self.roi_recommendations: list[ROIRecommendation] = []
        self._summary_thumbnail_refs: dict[str, ImageTk.PhotoImage] = {}
        self._comparison_selected_ids: set[str] = set()
        self._selected_summary_scheme_id: str | None = None
        self._last_evaluation: dict | None = None
        self._build_menu()
        self._build_ui()
        self._set_status("请新建或打开评价项目")

    def _build_menu(self):
        menu = tk.Menu(self)
        project_menu = tk.Menu(menu, tearoff=False)
        project_menu.add_command(label="新建项目…", command=self.new_project)
        project_menu.add_command(label="打开项目…", command=self.open_project)
        project_menu.add_command(label="保存项目", command=self.save_project)
        project_menu.add_separator()
        project_menu.add_command(label="退出", command=self.destroy)
        menu.add_cascade(label="项目", menu=project_menu)
        self.config(menu=menu)

    def _build_ui(self):
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=8, pady=(8, 2))
        self.overview_tab = ttk.Frame(self.notebook)
        self.images_tab = ttk.Frame(self.notebook)
        self.roi_tab = ttk.Frame(self.notebook)
        self.evaluate_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.overview_tab, text="项目概览")
        self.notebook.add(self.images_tab, text="图像 / 算法")
        self.notebook.add(self.roi_tab, text="ROI 标注")
        self.notebook.add(self.evaluate_tab, text="评价结果")
        self._build_overview()
        self._build_images()
        self._build_roi_editor()
        self._build_evaluation()
        self.status_var = tk.StringVar()
        ttk.Label(self, textvariable=self.status_var, anchor="w", relief="sunken").pack(fill="x", padx=8, pady=(2, 8))

    def _build_overview(self):
        frame = ttk.Frame(self.overview_tab, padding=24)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="Bone Image Enhancement Quality Assessment Tool", font=("Microsoft YaHei UI", 18, "bold")).pack(anchor="w")
        ttk.Label(frame, text="统一 ROI、统一灰度范围、统一指标定义", font=("Microsoft YaHei UI", 11)).pack(anchor="w", pady=(4, 24))
        self.overview_text = tk.StringVar(value="尚未打开项目")
        ttk.Label(frame, textvariable=self.overview_text, justify="left", font=("Microsoft YaHei UI", 12)).pack(anchor="w")
        actions = ttk.Frame(frame)
        actions.pack(anchor="w", pady=24)
        ttk.Button(actions, text="导入 / 管理图像", command=lambda: self.notebook.select(self.images_tab)).pack(side="left", padx=(0, 8))
        ttk.Button(actions, text="标注 ROI", command=lambda: self.notebook.select(self.roi_tab)).pack(side="left", padx=8)
        ttk.Button(actions, text="开始评价", command=self.run_evaluation).pack(side="left", padx=8)

    def _build_images(self):
        toolbar = ttk.Frame(self.images_tab, padding=8)
        toolbar.pack(fill="x")
        ttk.Button(toolbar, text="导入 / 替换原图", command=self.import_original).pack(side="left", padx=3)
        ttk.Button(toolbar, text="添加方案", command=self.add_algorithm).pack(side="left", padx=3)
        ttk.Button(toolbar, text="重命名方案", command=self.rename_algorithm).pack(side="left", padx=3)
        ttk.Button(toolbar, text="替换方案图片", command=self.replace_algorithm).pack(side="left", padx=3)
        ttk.Button(toolbar, text="删除方案", command=self.remove_algorithm).pack(side="left", padx=3)
        ttk.Button(toolbar, text="重新检查", command=self.show_validation).pack(side="right", padx=3)
        columns = ("role", "name", "size", "dtype", "status", "file")
        self.image_tree = ttk.Treeview(self.images_tab, columns=columns, show="headings", selectmode="browse")
        headings = {"role": "角色", "name": "方案名称", "size": "尺寸", "dtype": "位深", "status": "状态", "file": "项目文件"}
        widths = {"role": 85, "name": 210, "size": 110, "dtype": 80, "status": 160, "file": 430}
        for key in columns:
            self.image_tree.heading(key, text=headings[key])
            self.image_tree.column(key, width=widths[key], anchor="w")
        self.image_tree.pack(fill="both", expand=True, padx=8, pady=(0, 8))

    def _build_roi_editor(self):
        self.roi_tab.columnconfigure(0, weight=1)
        self.roi_tab.rowconfigure(0, weight=1)
        left = ttk.Frame(self.roi_tab, padding=8)
        left.grid(row=0, column=0, sticky="nsew")
        left.columnconfigure(0, weight=1)
        left.rowconfigure(0, weight=1)
        self.roi_canvas = tk.Canvas(left, background="#222", highlightthickness=0, cursor="crosshair")
        self.roi_canvas.grid(row=0, column=0, sticky="nsew")
        self.roi_canvas.bind("<Configure>", lambda event: self.refresh_roi_canvas())
        self.roi_canvas.bind("<ButtonPress-1>", self._roi_press)
        self.roi_canvas.bind("<B1-Motion>", self._roi_drag)
        self.roi_canvas.bind("<ButtonRelease-1>", self._roi_release)
        ttk.Label(left, text="在原图上拖动可填入矩形坐标；确认名称、类型和配对后保存。", foreground="#555").grid(row=1, column=0, sticky="w", pady=(6, 0))

        right = ttk.Frame(self.roi_tab, padding=8, width=340)
        right.grid(row=0, column=1, sticky="ns")
        recommend_bar = ttk.Frame(right)
        recommend_bar.pack(fill="x", pady=(0, 6))
        ttk.Button(recommend_bar, text="自动推荐 ROI", command=self.auto_recommend_rois).pack(side="left", padx=2)
        ttk.Button(recommend_bar, text="重新推荐", command=self.auto_recommend_rois).pack(side="left", padx=2)
        ttk.Label(right, text="ROI 列表", font=("Microsoft YaHei UI", 11, "bold")).pack(anchor="w")
        self.roi_tree = ttk.Treeview(right, columns=("type", "name", "pair"), show="headings", height=7, selectmode="browse")
        self.roi_tree.heading("type", text="类型")
        self.roi_tree.heading("name", text="名称")
        self.roi_tree.heading("pair", text="配对")
        self.roi_tree.column("type", width=90)
        self.roi_tree.column("name", width=145)
        self.roi_tree.column("pair", width=145)
        self.roi_tree.pack(fill="x", pady=(5, 8))
        self.roi_tree.bind("<<TreeviewSelect>>", self._roi_selected)

        ttk.Label(right, text="推荐候选（虚线）", font=("Microsoft YaHei UI", 10, "bold")).pack(anchor="w")
        self.recommendation_tree = ttk.Treeview(right, columns=("type", "name", "quality"), show="headings", height=6, selectmode="browse")
        for key, label, width in (("type", "类型", 90), ("name", "名称", 160), ("quality", "质量", 100)):
            self.recommendation_tree.heading(key, text=label)
            self.recommendation_tree.column(key, width=width)
        self.recommendation_tree.pack(fill="x", pady=(4, 5))
        self.recommendation_tree.bind("<<TreeviewSelect>>", self._recommendation_selected)
        candidate_buttons = ttk.Frame(right)
        candidate_buttons.pack(fill="x", pady=(0, 7))
        ttk.Button(candidate_buttons, text="接受选中", command=self.accept_selected_recommendation).pack(side="left", padx=2)
        ttk.Button(candidate_buttons, text="全部接受", command=self.accept_all_recommendations).pack(side="left", padx=2)
        ttk.Button(candidate_buttons, text="删除推荐", command=self.delete_recommendations).pack(side="left", padx=2)

        form = ttk.LabelFrame(right, text="ROI 属性", padding=8)
        form.pack(fill="x")
        self.roi_name_var = tk.StringVar()
        self.roi_type_var = tk.StringVar(value="weak_bone")
        self.roi_type_display_var = tk.StringVar(value=ROI_LABELS["weak_bone"])
        self.roi_pair_var = tk.StringVar()
        self.roi_coord_vars = {key: tk.StringVar(value="0") for key in ("x", "y", "width", "height")}
        ttk.Label(form, text="名称").grid(row=0, column=0, sticky="w", pady=3)
        ttk.Entry(form, textvariable=self.roi_name_var, width=28).grid(row=0, column=1, columnspan=3, sticky="ew", pady=3)
        ttk.Label(form, text="类型").grid(row=1, column=0, sticky="w", pady=3)
        self.roi_type_combo = ttk.Combobox(form, textvariable=self.roi_type_display_var, values=list(ROI_LABELS.values()), state="readonly", width=25)
        self.roi_type_combo.grid(row=1, column=1, columnspan=3, sticky="ew", pady=3)
        self.roi_type_combo.bind("<<ComboboxSelected>>", self._roi_type_display_changed)
        labels = ("x", "y", "width", "height")
        for index, key in enumerate(labels):
            ttk.Label(form, text=key).grid(row=2 + index // 2, column=(index % 2) * 2, sticky="w", pady=3)
            ttk.Entry(form, textvariable=self.roi_coord_vars[key], width=9).grid(row=2 + index // 2, column=(index % 2) * 2 + 1, sticky="ew", padx=(2, 8), pady=3)
        ttk.Label(form, text="邻域配对").grid(row=4, column=0, sticky="w", pady=3)
        self.roi_pair_combo = ttk.Combobox(form, textvariable=self.roi_pair_var, state="readonly", width=25)
        self.roi_pair_combo.grid(row=4, column=1, columnspan=3, sticky="ew", pady=3)
        form.columnconfigure(1, weight=1)
        buttons = ttk.Frame(right)
        buttons.pack(fill="x", pady=8)
        ttk.Button(buttons, text="新建 / 清空", command=self.clear_roi_form).pack(side="left", padx=2)
        ttk.Button(buttons, text="保存 ROI", command=self.save_roi).pack(side="left", padx=2)
        ttk.Button(buttons, text="删除", command=self.delete_roi).pack(side="left", padx=2)
        ttk.Button(buttons, text="显示/隐藏", command=self.toggle_roi).pack(side="left", padx=2)
        ttk.Button(right, text="清空已保存 ROI", command=self.clear_saved_rois).pack(fill="x", pady=(0, 8))

    def _build_evaluation(self):
        toolbar = ttk.Frame(self.evaluate_tab, padding=8)
        toolbar.pack(fill="x")
        ttk.Button(toolbar, text="开始评价", command=self.run_evaluation).pack(side="left", padx=3)
        ttk.Button(toolbar, text="比较所选方案", command=self.compare_selected_schemes).pack(side="left", padx=3)
        ttk.Button(toolbar, text="导出 CSV", command=self.export_results).pack(side="left", padx=3)

        result_book = ttk.Notebook(self.evaluate_tab)
        result_book.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        summary_frame = ttk.Frame(result_book)
        roi_frame = ttk.Frame(result_book)
        result_book.add(summary_frame, text="总体表")
        result_book.add(roi_frame, text="ROI 表")

        summary_frame.rowconfigure(0, weight=1)
        summary_frame.columnconfigure(0, weight=1)
        self.summary_table = MetricSummaryTable(summary_frame, self._summary_row_click, self._summary_compare_click, self._summary_preview)
        self.summary_table.frame.grid(row=0, column=0, sticky="nsew")
        diagnostic_frame = ttk.LabelFrame(summary_frame, text="指标解释 / 诊断", padding=6)
        diagnostic_frame.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        self.diagnostic_var = tk.StringVar(value="请完成评价后点击总体表中的某个方案，查看该方案的指标诊断。")
        ttk.Label(diagnostic_frame, textvariable=self.diagnostic_var, justify="left", wraplength=1150).pack(fill="x")

        roi_columns = ("scheme", "roi", "type", "mean", "std", "ag", "cnr_bg", "cnr_local", "sat", "status", "message")
        roi_frame.rowconfigure(0, weight=1)
        roi_frame.columnconfigure(0, weight=1)
        roi_table_frame = ttk.Frame(roi_frame)
        roi_table_frame.grid(row=0, column=0, sticky="nsew")
        roi_table_frame.rowconfigure(0, weight=1)
        roi_table_frame.columnconfigure(0, weight=1)
        self.roi_result_tree = ttk.Treeview(roi_table_frame, columns=roi_columns, show="headings")
        for key, label in zip(roi_columns, ("方案", "ROI", "类型", "均值", "标准差", "AG", "CNR-bg", "CNR-local", "饱和率", "状态", "原因")):
            self.roi_result_tree.heading(key, text=label)
            self.roi_result_tree.column(key, width=300 if key == "message" else 125 if key in {"scheme", "roi"} else 92, minwidth=70, anchor="w" if key == "message" else "center", stretch=False)
        roi_ybar = ttk.Scrollbar(roi_table_frame, orient="vertical", command=self.roi_result_tree.yview)
        roi_xbar = ttk.Scrollbar(roi_table_frame, orient="horizontal", command=self.roi_result_tree.xview)
        self.roi_result_tree.configure(yscrollcommand=roi_ybar.set, xscrollcommand=roi_xbar.set)
        self.roi_result_tree.grid(row=0, column=0, sticky="nsew")
        roi_ybar.grid(row=0, column=1, sticky="ns")
        roi_xbar.grid(row=1, column=0, sticky="ew")

    def _require_store(self) -> ProjectStore | None:
        if self.store is None:
            messagebox.showwarning("尚未打开项目", "请先新建或打开项目。")
        return self.store

    def _set_status(self, message: str):
        self.status_var.set(message)
        self.update_idletasks()

    def new_project(self):
        directory = filedialog.askdirectory(title="选择新项目目录")
        if not directory:
            return
        name = simpledialog.askstring("项目名称", "请输入项目名称：", initialvalue=Path(directory).name)
        if name is None:
            return
        try:
            self.store = ProjectStore.create(Path(directory), name)
            self.roi_recommendations = []
            self.refresh_all()
            self._set_status(f"已创建项目：{self.store.project.name}")
        except Exception as exc:
            messagebox.showerror("创建失败", str(exc))

    def open_project(self):
        directory = filedialog.askdirectory(title="选择包含 project.json 的项目目录")
        if not directory:
            return
        try:
            self.store = ProjectStore.open(Path(directory))
            self.roi_recommendations = []
            self.refresh_all()
            self._set_status(f"已打开项目：{self.store.project.name}")
        except Exception as exc:
            messagebox.showerror("打开失败", str(exc))

    def save_project(self):
        store = self._require_store()
        if store:
            store.save()
            self._set_status("项目已保存")

    def import_original(self):
        store = self._require_store()
        if not store:
            return
        path = filedialog.askopenfilename(title="选择原始灰度图", filetypes=IMAGE_TYPES)
        if not path:
            return
        if store.project.original and not messagebox.askyesno("替换原图", "替换原图会使现有评价结果失效，并可能使 ROI 越界。是否继续？"):
            return
        try:
            store.set_original(Path(path))
            self.roi_recommendations = []
            self.refresh_all()
            self._set_status("原图已导入并复制到项目目录")
        except Exception as exc:
            messagebox.showerror("导入失败", str(exc))

    def add_algorithm(self):
        store = self._require_store()
        if not store:
            return
        path = filedialog.askopenfilename(title="选择增强结果", filetypes=IMAGE_TYPES)
        if not path:
            return
        name = simpledialog.askstring("方案名称", "请输入算法方案名称：", initialvalue=Path(path).stem)
        if not name:
            return
        try:
            store.add_algorithm(Path(path), name)
            self.refresh_all()
            self._set_status(f"已添加方案：{name}")
        except Exception as exc:
            messagebox.showerror("添加失败", str(exc))

    def _selected_algorithm_id(self) -> str | None:
        selection = self.image_tree.selection()
        if not selection:
            messagebox.showwarning("未选择方案", "请先选择一个算法方案。")
            return None
        item_id = selection[0]
        if item_id == "original":
            messagebox.showwarning("选择无效", "该操作只适用于算法方案。")
            return None
        return item_id

    def rename_algorithm(self):
        store = self._require_store()
        algorithm_id = self._selected_algorithm_id()
        if not store or not algorithm_id:
            return
        record = store.get_algorithm(algorithm_id)
        name = simpledialog.askstring("重命名方案", "新方案名称：", initialvalue=record.display_name)
        if not name:
            return
        try:
            store.rename_algorithm(algorithm_id, name)
            self.refresh_all()
        except Exception as exc:
            messagebox.showerror("重命名失败", str(exc))

    def replace_algorithm(self):
        store = self._require_store()
        algorithm_id = self._selected_algorithm_id()
        if not store or not algorithm_id:
            return
        path = filedialog.askopenfilename(title="选择新的增强结果", filetypes=IMAGE_TYPES)
        if path:
            try:
                store.replace_algorithm(algorithm_id, Path(path))
                self.refresh_all()
            except Exception as exc:
                messagebox.showerror("替换失败", str(exc))

    def remove_algorithm(self):
        store = self._require_store()
        algorithm_id = self._selected_algorithm_id()
        if not store or not algorithm_id:
            return
        if messagebox.askyesno("删除方案", "从项目清单中删除该方案？项目内导入文件暂时保留。"):
            store.remove_algorithm(algorithm_id)
            self.refresh_all()

    def clear_roi_form(self):
        self._selected_roi_id = None
        self._selected_recommendation_id = None
        self.roi_name_var.set("")
        self.roi_type_var.set("weak_bone")
        self.roi_type_display_var.set(ROI_LABELS["weak_bone"])
        self.roi_pair_var.set("")
        for variable in self.roi_coord_vars.values():
            variable.set("0")
        self.roi_tree.selection_remove(self.roi_tree.selection())
        self.recommendation_tree.selection_remove(self.recommendation_tree.selection())

    def _roi_selected(self, _event=None):
        store = self.store
        selection = self.roi_tree.selection()
        if not store or not selection:
            return
        roi_id = selection[0]
        roi = next((item for item in store.project.rois if item.id == roi_id), None)
        if roi is None:
            return
        self._selected_roi_id = roi.id
        self._selected_recommendation_id = None
        self.recommendation_tree.selection_remove(self.recommendation_tree.selection())
        self._populate_roi_form(roi)
        self.refresh_roi_canvas()

    def _populate_roi_form(self, roi: ROI):
        self.roi_name_var.set(roi.name)
        self.roi_type_var.set(roi.type)
        self.roi_type_display_var.set(ROI_LABELS[roi.type])
        for key, value in zip(("x", "y", "width", "height"), roi.geometry()):
            self.roi_coord_vars[key].set(str(value))
        all_rois = list(self.store.project.rois) if self.store else []
        all_rois.extend(item.roi for item in self.roi_recommendations)
        pair = next((item for item in all_rois if item.id == roi.paired_surrounding_roi_id), None)
        self.roi_pair_var.set(f"{pair.name} | {pair.id}" if pair else "")

    def _roi_type_display_changed(self, _event=None):
        self.roi_type_var.set(ROI_TYPES_BY_LABEL[self.roi_type_display_var.get()])

    def _recommendation_selected(self, _event=None):
        selection = self.recommendation_tree.selection()
        if not selection:
            return
        recommendation = next((item for item in self.roi_recommendations if item.roi.id == selection[0]), None)
        if recommendation is None:
            return
        self._selected_recommendation_id = recommendation.roi.id
        self._selected_roi_id = None
        self.roi_tree.selection_remove(self.roi_tree.selection())
        self._populate_roi_form(recommendation.roi)
        self._set_status(recommendation.explanation)
        self.refresh_roi_canvas()

    def save_roi(self):
        store = self._require_store()
        if not store or store.project.original is None:
            messagebox.showwarning("缺少原图", "请先导入原图。")
            return
        try:
            values = {key: int(variable.get()) for key, variable in self.roi_coord_vars.items()}
            pair_text = self.roi_pair_var.get()
            pair_id = pair_text.rsplit(" | ", 1)[-1] if " | " in pair_text else None
            if self._selected_roi_id:
                roi = next(item for item in store.project.rois if item.id == self._selected_roi_id)
                roi.name = self.roi_name_var.get().strip()
                roi.type = self.roi_type_var.get()
                roi.x, roi.y, roi.width, roi.height = values["x"], values["y"], values["width"], values["height"]
                roi.paired_surrounding_roi_id = pair_id if roi.type in {"weak_bone", "strong_bone"} else None
            else:
                roi = ROI.create(
                    self.roi_name_var.get().strip(), self.roi_type_var.get(),
                    values["x"], values["y"], values["width"], values["height"],
                    pair_id if self.roi_type_var.get() in {"weak_bone", "strong_bone"} else None,
                )
            if roi.x < 0 or roi.y < 0 or roi.width <= 0 or roi.height <= 0 or roi.x + roi.width > store.project.original.width or roi.y + roi.height > store.project.original.height:
                raise ProjectError("ROI 必须具有正尺寸并完全位于原图范围内")
            if roi.type in {"weak_bone", "strong_bone"} and not roi.paired_surrounding_roi_id:
                roi.paired_surrounding_roi_id = nearest_surrounding_roi_id(store.project.rois, roi)
            if roi.type in {"weak_bone", "strong_bone"} and not roi.paired_surrounding_roi_id:
                messagebox.showwarning("CNR 无法计算", "该骨骼 ROI 尚未设置邻域 ROI，因此 CNR 无法计算。允许保存，但请随后创建或选择邻域 ROI。")
            store.upsert_roi(roi)
            if roi.type == "background":
                image = load_grayscale(store.resolve(store.project.original.relative_path))
                stats = roi_statistics(image[roi.y : roi.y + roi.height, roi.x : roi.x + roi.width])
                if stats["std_intensity"] is None or float(stats["std_intensity"]) <= EPSILON or int(stats["unique_pixel_count"] or 0) < 2:
                    messagebox.showwarning(
                        "背景 ROI 波动不足",
                        "该背景 ROI 几乎为常量区域，无法有效估计背景噪声。\n\n"
                        f"像素数：{stats['pixel_count']}\n均值：{stats['mean_intensity']}\n标准差：{stats['std_intensity']}\n"
                        f"最小值：{stats['min_intensity']}\n最大值：{stats['max_intensity']}\n唯一灰度数：{stats['unique_pixel_count']}\n\n"
                        "建议重新选择包含真实背景波动但不含人体结构的区域。",
                    )
            self._selected_roi_id = roi.id
            self.refresh_all()
            self._set_status(f"ROI 已保存：{roi.name}")
        except Exception as exc:
            messagebox.showerror("保存 ROI 失败", str(exc))

    def auto_recommend_rois(self):
        store = self._require_store()
        if not store or store.project.original is None:
            messagebox.showwarning("缺少原图", "请先导入原图。")
            return
        if (store.project.original.width, store.project.original.height) != ROI_TEMPLATE_SIZE:
            messagebox.showwarning(
                "ROI 模板尺寸不一致",
                "当前图像尺寸与 ROI Template v1 不一致，无法加载固定 ROI 模板。",
            )
            return
        if store.project.rois and not messagebox.askyesno(
            "加载固定 ROI 模板",
            "当前已有正式 ROI。加载 ROI Template v1 将替换当前正式 ROI，是否继续？",
        ):
            return
        try:
            store.replace_rois(build_roi_template())
            self.roi_recommendations = []
            self.clear_roi_form()
            self.refresh_all()
            self._set_status(f"已加载 {ROI_TEMPLATE_VERSION}，共保存 12 个固定 ROI。")
        except Exception as exc:
            messagebox.showerror("加载 ROI 模板失败", str(exc))

    def _update_recommendation_from_form(self, recommendation: ROIRecommendation) -> None:
        roi = recommendation.roi
        values = {key: int(variable.get()) for key, variable in self.roi_coord_vars.items()}
        roi.name = self.roi_name_var.get().strip() or roi.name
        roi.type = self.roi_type_var.get()
        roi.x, roi.y, roi.width, roi.height = values["x"], values["y"], values["width"], values["height"]
        pair_text = self.roi_pair_var.get()
        roi.paired_surrounding_roi_id = pair_text.rsplit(" | ", 1)[-1] if roi.type in {"weak_bone", "strong_bone"} and " | " in pair_text else None

    def accept_selected_recommendation(self):
        store = self._require_store()
        if not store or not self._selected_recommendation_id:
            messagebox.showwarning("未选择推荐", "请先选择一个推荐候选。")
            return
        recommendation = next(item for item in self.roi_recommendations if item.roi.id == self._selected_recommendation_id)
        self._update_recommendation_from_form(recommendation)
        roi = recommendation.roi
        paired = next((item for item in self.roi_recommendations if item.roi.id == roi.paired_surrounding_roi_id), None)
        if paired is not None:
            store.upsert_roi(paired.roi)
            self.roi_recommendations.remove(paired)
        store.upsert_roi(roi)
        self.roi_recommendations.remove(recommendation)
        self.clear_roi_form()
        self.refresh_all()
        self._set_status(f"已接受推荐 ROI：{roi.name}")

    def accept_all_recommendations(self):
        store = self._require_store()
        if not store or not self.roi_recommendations:
            return
        for recommendation in sorted(self.roi_recommendations, key=lambda item: item.roi.type != "surrounding"):
            store.upsert_roi(recommendation.roi)
        count = len(self.roi_recommendations)
        self.roi_recommendations = []
        self.clear_roi_form()
        self.refresh_all()
        self._set_status(f"已接受并保存 {count} 个推荐 ROI；请检查并按需微调。")

    def delete_recommendations(self):
        if self._selected_recommendation_id:
            removed_id = self._selected_recommendation_id
            removed = next((item for item in self.roi_recommendations if item.roi.id == removed_id), None)
            remove_neighbor = False
            if removed and removed.roi.type in {"weak_bone", "strong_bone"} and removed.roi.paired_surrounding_roi_id:
                pair_id = removed.roi.paired_surrounding_roi_id
                used_elsewhere = any(item.roi.id != removed_id and item.roi.paired_surrounding_roi_id == pair_id for item in self.roi_recommendations)
                pair_exists = any(item.roi.id == pair_id for item in self.roi_recommendations)
                remove_neighbor = pair_exists and not used_elsewhere and messagebox.askyesno("删除配对候选", "该骨骼 ROI 的邻域未被其他候选使用，是否一并删除？")
            self.roi_recommendations = remove_recommendation(self.roi_recommendations, removed_id, remove_neighbor)
        else:
            self.roi_recommendations = []
        self.clear_roi_form()
        self.refresh_roi_list()
        self.refresh_roi_canvas()

    def delete_roi(self):
        store = self._require_store()
        if not store or not self._selected_roi_id:
            return
        if messagebox.askyesno("删除 ROI", "删除后，引用它的 Bone ROI 配对将被清空。是否继续？"):
            store.remove_roi(self._selected_roi_id)
            self.clear_roi_form()
            self.refresh_all()

    def clear_saved_rois(self):
        store = self._require_store()
        if not store:
            return
        count = len(store.project.rois)
        if count == 0:
            messagebox.showinfo("清空已保存 ROI", "当前没有已保存的 ROI。")
            return
        confirmed = messagebox.askokcancel(
            "清空已保存 ROI",
            "确定清空当前项目中的全部已保存 ROI 吗？\n\n"
            "将删除：\n弱骨骼、强骨骼、邻域和背景 ROI。\n\n"
            "已有评价结果将失效。\n"
            "此操作不会删除尚未接受的推荐候选。",
        )
        if not confirmed:
            return
        store.clear_rois()
        self.clear_roi_form()
        self.refresh_all()
        self._set_status(f"已清空全部已保存 ROI，共删除 {count} 个。")

    def toggle_roi(self):
        store = self._require_store()
        if not store or not self._selected_roi_id:
            return
        roi = next(item for item in store.project.rois if item.id == self._selected_roi_id)
        roi.visible = not roi.visible
        store.upsert_roi(roi)
        self.refresh_all()

    def _roi_press(self, event):
        if not self.store or not self.store.project.original:
            return
        self._drag_start = (event.x, event.y)
        if self._drag_preview:
            self.roi_canvas.delete(self._drag_preview)
        self._drag_preview = self.roi_canvas.create_rectangle(event.x, event.y, event.x, event.y, outline="white", dash=(4, 2), width=2)

    def _roi_drag(self, event):
        if self._drag_start and self._drag_preview:
            self.roi_canvas.coords(self._drag_preview, self._drag_start[0], self._drag_start[1], event.x, event.y)

    def _roi_release(self, event):
        if not self._drag_start or not self.store or not self.store.project.original:
            return
        x1, y1 = self._drag_start
        x2, y2 = event.x, event.y
        ox, oy = self._roi_offset
        scale = self._roi_scale
        ix1 = round((min(x1, x2) - ox) / scale)
        iy1 = round((min(y1, y2) - oy) / scale)
        ix2 = round((max(x1, x2) - ox) / scale)
        iy2 = round((max(y1, y2) - oy) / scale)
        original = self.store.project.original
        ix1, iy1 = max(0, ix1), max(0, iy1)
        ix2, iy2 = min(original.width, ix2), min(original.height, iy2)
        self.roi_coord_vars["x"].set(str(ix1))
        self.roi_coord_vars["y"].set(str(iy1))
        self.roi_coord_vars["width"].set(str(max(0, ix2 - ix1)))
        self.roi_coord_vars["height"].set(str(max(0, iy2 - iy1)))
        if self.roi_type_var.get() in {"weak_bone", "strong_bone"} and not self.roi_pair_var.get():
            provisional = ROI.create("provisional", self.roi_type_var.get(), ix1, iy1, max(0, ix2 - ix1), max(0, iy2 - iy1))
            pair_id = nearest_surrounding_roi_id(self.store.project.rois, provisional)
            pair = next((item for item in self.store.project.rois if item.id == pair_id), None)
            if pair:
                self.roi_pair_var.set(f"{pair.name} | {pair.id}")
        self._drag_start = None

    def show_validation(self):
        store = self._require_store()
        if not store:
            return
        issues = validate_project(store)
        if not issues:
            self._set_status("检查通过：未发现图像、ROI 或配对问题")
            messagebox.showinfo("检查完成", "未发现问题。")
            return
        severity_labels = {"error": "错误", "warning": "警告", "info": "提示"}
        summary = "\n".join(f"[{severity_labels.get(issue.severity, issue.severity)}] {issue.message}" for issue in issues)
        self._set_status(f"检查发现 {len(issues)} 项问题：{issues[0].message}")
        messagebox.showwarning("检查发现问题", summary)

    def run_evaluation(self):
        store = self._require_store()
        if not store:
            return
        self._set_status("正在计算评价指标…")
        self.config(cursor="watch")
        try:
            result = evaluate_project(store)
            self._fill_results(result)
            self.notebook.select(self.evaluate_tab)
            warning_count = sum(1 for row in result.get("metrics", []) if row.get("status") in {"partial", "valid_with_warnings", "failed"})
            if warning_count:
                self._set_status(f"评价完成，但有 {warning_count} 个方案需要检查状态列")
            else:
                self._set_status(f"评价完成：{result['evaluation_id']}")
        except Exception as exc:
            self._set_status("评价失败")
            messagebox.showerror("评价失败", str(exc))
        finally:
            self.config(cursor="")

    def export_results(self):
        store = self._require_store()
        if not store:
            return
        directory = filedialog.askdirectory(title="选择 CSV 导出目录", initialdir=str(store.root / "exports"))
        if not directory:
            return
        try:
            paths = export_csv_bundle(store, Path(directory))
            messagebox.showinfo("导出完成", "已导出：\n" + "\n".join(str(path) for path in paths))
        except Exception as exc:
            messagebox.showerror("导出失败", str(exc))

    def refresh_all(self):
        self.refresh_overview()
        self.refresh_images()
        self.refresh_roi_list()
        self.refresh_roi_canvas()
        if self.store:
            result = self.store.load_latest_evaluation()
            if result:
                self._fill_results(result)
            else:
                self._show_expired_evaluation(
                    "检测到图像文件或 ROI 配置已发生变化，请重新评价。"
                    if self.store.evaluation_status() == "expired"
                    else "当前没有可用的评价结果，请点击“开始评价”。"
                )

    def refresh_overview(self):
        if not self.store:
            self.overview_text.set("尚未打开项目")
            return
        project = self.store.project
        counts = {key: sum(1 for roi in project.rois if roi.type == key) for key in ROI_LABELS}
        evaluation_state = self.store.evaluation_status()
        evaluation_label = {
            "current": "已有当前评价结果",
            "legacy": "已有旧格式评价结果（建议重新评价）",
            "expired": "结果已过期",
            "none": "尚未评价或结果已失效",
        }.get(evaluation_state, "尚未评价或结果已失效")
        self.overview_text.set(
            f"项目：{project.name}\n"
            f"原图：{project.original.display_name if project.original else '未导入'}\n"
            f"增强方案：{len(project.algorithms)} 个\n"
            f"ROI：弱骨骼 {counts['weak_bone']} / 强骨骼 {counts['strong_bone']} / 邻域 {counts['surrounding']} / 背景 {counts['background']}\n"
            f"评价状态：{evaluation_label}"
        )

    def refresh_images(self):
        self.image_tree.delete(*self.image_tree.get_children())
        if not self.store:
            return
        issues = validate_project(self.store)
        error_ids = {issue.object_id for issue in issues if issue.severity == "error"}
        evaluation_expired = self.store.evaluation_status() == "expired"
        records = ([self.store.project.original] if self.store.project.original else []) + self.store.project.algorithms
        for record in records:
            record_status = "结果已过期" if evaluation_expired else "通过"
            self.image_tree.insert("", "end", iid=record.id, values=(
                "原图" if record.role == "original" else "算法",
                record.display_name,
                f"{record.width} × {record.height}",
                record.dtype,
                "错误" if record.id in error_ids else record_status,
                record.relative_path,
            ))

    def refresh_roi_list(self):
        self.roi_tree.delete(*self.roi_tree.get_children())
        self.recommendation_tree.delete(*self.recommendation_tree.get_children())
        if not self.store:
            self.roi_pair_combo["values"] = []
            return
        for roi in self.store.project.rois:
            suffix = "" if roi.visible else "（隐藏）"
            pair = next((item for item in self.store.project.rois if item.id == roi.paired_surrounding_roi_id), None)
            pair_text = f"→ {pair.name}" if pair else ""
            self.roi_tree.insert("", "end", iid=roi.id, values=(ROI_LABELS[roi.type], roi.name + suffix, pair_text))
        for recommendation in self.roi_recommendations:
            roi = recommendation.roi
            self.recommendation_tree.insert("", "end", iid=roi.id, values=(ROI_LABELS[roi.type], roi.name, recommendation.quality))
        all_rois = list(self.store.project.rois) + [item.roi for item in self.roi_recommendations]
        surroundings = [f"{roi.name} | {roi.id}" for roi in all_rois if roi.type == "surrounding"]
        self.roi_pair_combo["values"] = [""] + surroundings
        if self._selected_roi_id and self.roi_tree.exists(self._selected_roi_id):
            self.roi_tree.selection_set(self._selected_roi_id)
        if self._selected_recommendation_id and self.recommendation_tree.exists(self._selected_recommendation_id):
            self.recommendation_tree.selection_set(self._selected_recommendation_id)

    def refresh_roi_canvas(self):
        canvas = getattr(self, "roi_canvas", None)
        if canvas is None:
            return
        canvas.delete("all")
        if not self.store or not self.store.project.original:
            canvas.create_text(20, 20, text="请先导入原图", fill="white", anchor="nw", font=("Microsoft YaHei UI", 14))
            return
        path = self.store.resolve(self.store.project.original.relative_path)
        try:
            image = Image.open(path).convert("L")
            width = max(canvas.winfo_width(), 400)
            height = max(canvas.winfo_height(), 400)
            scale = min(width / image.width, height / image.height)
            shown_size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
            shown = image.resize(shown_size, Image.Resampling.LANCZOS)
            self._roi_photo = ImageTk.PhotoImage(shown)
            ox = (width - shown_size[0]) / 2
            oy = (height - shown_size[1]) / 2
            self._roi_scale = scale
            self._roi_offset = (ox, oy)
            canvas.create_image(ox, oy, image=self._roi_photo, anchor="nw")
            formal = [roi for roi in self.store.project.rois if roi.visible]
            recommended = [item.roi for item in self.roi_recommendations]
            all_rois = formal + recommended
            labels = self._short_roi_labels(all_rois)
            highlighted = self._linked_highlight_ids(all_rois)
            label_positions, label_fonts = self._layout_roi_labels(canvas, all_rois, labels, ox, oy, scale, highlighted)
            for roi in formal:
                self._draw_roi(canvas, roi, labels[roi.id], ox, oy, scale, roi.id in highlighted, recommended=False, label_position=label_positions[roi.id], label_font=label_fonts[roi.id])
            for roi in recommended:
                self._draw_roi(canvas, roi, labels[roi.id], ox, oy, scale, roi.id in highlighted, recommended=True, label_position=label_positions[roi.id], label_font=label_fonts[roi.id])
            legend = "弱骨骼 ｜ 强骨骼 ｜ 邻域 ｜ 背景"
            canvas.create_rectangle(8, 8, 315, 32, fill="#111", outline="#888")
            canvas.create_text(16, 20, text=legend, fill="white", anchor="w", font=("Microsoft YaHei UI", 9))
        except Exception as exc:
            canvas.create_text(20, 20, text=f"无法显示原图：{exc}", fill="white", anchor="nw")

    def _short_roi_labels(self, rois: list[ROI]) -> dict[str, str]:
        prefixes = {"weak_bone": "弱骨骼", "strong_bone": "强骨骼", "surrounding": "邻域", "background": "背景"}
        counters = {key: 0 for key in prefixes}
        labels: dict[str, str] = {}
        for roi in rois:
            counters[roi.type] += 1
            labels[roi.id] = f"{prefixes[roi.type]}{counters[roi.type]}"
        return labels

    def _linked_highlight_ids(self, rois: list[ROI]) -> set[str]:
        selected_id = self._selected_roi_id or self._selected_recommendation_id
        if not selected_id:
            return set()
        selected = next((roi for roi in rois if roi.id == selected_id), None)
        if selected is None:
            return set()
        highlighted = {selected.id}
        if selected.type in {"weak_bone", "strong_bone"} and selected.paired_surrounding_roi_id:
            highlighted.add(selected.paired_surrounding_roi_id)
        if selected.type == "surrounding":
            highlighted.update(roi.id for roi in rois if roi.paired_surrounding_roi_id == selected.id)
        return highlighted

    def _layout_roi_labels(self, canvas, rois: list[ROI], labels: dict[str, str], ox: float, oy: float, scale: float, highlighted: set[str]):
        """Place labels around rectangles without changing any ROI geometry."""
        compact = len(rois) > 8
        fonts = {
            roi.id: tkfont.Font(family="Microsoft YaHei UI", size=8 if compact else 9, weight="bold" if roi.id in highlighted else "normal")
            for roi in rois
        }
        canvas_width = max(canvas.winfo_width(), 400)
        canvas_height = max(canvas.winfo_height(), 400)
        occupied: list[tuple[float, float, float, float]] = []
        positions: dict[str, tuple[float, float]] = {}

        def overlaps(candidate, other):
            return not (candidate[2] <= other[0] or candidate[0] >= other[2] or candidate[3] <= other[1] or candidate[1] >= other[3])

        for roi in sorted(rois, key=lambda item: (item.id not in highlighted, item.y, item.x)):
            x1, y1 = ox + roi.x * scale, oy + roi.y * scale
            x2, y2 = ox + (roi.x + roi.width) * scale, oy + (roi.y + roi.height) * scale
            font = fonts[roi.id]
            text_width = max(20, font.measure(labels[roi.id]))
            text_height = max(14, int(font.metrics("linespace")))
            candidates = (
                (x1, y1 - text_height - 4),
                (x2 + 4, y1),
                (x1 - text_width - 4, y1),
                (x1, y2 + 4),
                (x1 + 3, y1 + 3),
            )
            chosen = None
            for candidate_x, candidate_y in candidates:
                candidate_x = min(max(4, candidate_x), max(4, canvas_width - text_width - 4))
                candidate_y = min(max(4, candidate_y), max(4, canvas_height - text_height - 4))
                candidate = (candidate_x, candidate_y, candidate_x + text_width + 4, candidate_y + text_height + 3)
                if not any(overlaps(candidate, previous) for previous in occupied):
                    chosen = candidate
                    break
            if chosen is None:
                chosen = (min(max(4, x1), max(4, canvas_width - text_width - 4)), min(max(4, y1), max(4, canvas_height - text_height - 4)), 0, 0)
                chosen = (chosen[0], chosen[1], chosen[0] + text_width + 4, chosen[1] + text_height + 3)
            occupied.append(chosen)
            positions[roi.id] = (chosen[0] + 2, chosen[1] + 1)
        return positions, fonts

    def _draw_roi(self, canvas, roi: ROI, label: str, ox: float, oy: float, scale: float, highlighted: bool, recommended: bool, label_position=None, label_font=None):
        x1, y1 = ox + roi.x * scale, oy + roi.y * scale
        x2, y2 = ox + (roi.x + roi.width) * scale, oy + (roi.y + roi.height) * scale
        any_selection = bool(self._selected_roi_id or self._selected_recommendation_id)
        color = ROI_COLORS[roi.type] if highlighted or not any_selection else "#707070"
        width = 4 if highlighted else 2 if not any_selection else 1
        options = {"outline": color, "width": width}
        if recommended:
            options["dash"] = (6, 4)
        canvas.create_rectangle(x1, y1, x2, y2, **options)
        if label_position is not None:
            font = label_font or ("Microsoft YaHei UI", 9, "bold" if highlighted else "normal")
            label_x, label_y = label_position
            if isinstance(font, tkfont.Font):
                text_width = font.measure(label)
                text_height = int(font.metrics("linespace"))
            else:
                text_width, text_height = 64, 16
            canvas.create_rectangle(label_x - 2, label_y - 1, label_x + text_width + 2, label_y + text_height + 2, fill="#111", outline="")
            canvas.create_text(label_x, label_y, text=label, fill=color, anchor="nw", font=font)

    def _record_for_id(self, scheme_id: str):
        if not self.store:
            return None
        records = ([self.store.project.original] if self.store.project.original else []) + self.store.project.algorithms
        return next((record for record in records if record.id == scheme_id), None)

    def _thumbnail_for_record(self, record):
        thumbnail_size = SUMMARY_THUMBNAIL_SIZE
        if record is None:
            thumbnail = ImageTk.PhotoImage(Image.new("L", thumbnail_size, 80))
            self._summary_thumbnail_refs["missing"] = thumbnail
            return thumbnail
        try:
            image = Image.open(self.store.resolve(record.relative_path)).convert("L")
            image.thumbnail(thumbnail_size, Image.Resampling.LANCZOS)
            canvas = Image.new("L", thumbnail_size, 32)
            canvas.paste(image, ((thumbnail_size[0] - image.width) // 2, (thumbnail_size[1] - image.height) // 2))
        except Exception:
            canvas = Image.new("L", thumbnail_size, 90)
        thumbnail = ImageTk.PhotoImage(canvas)
        self._summary_thumbnail_refs[record.id if record else "missing"] = thumbnail
        return thumbnail

    def _summary_row(self, scheme_id: str):
        result = self._last_evaluation or {}
        return next((row for row in result.get("metrics", []) if row.get("scheme_id") == scheme_id), None)

    def _summary_row_click(self, scheme_id: str):
        self._selected_summary_scheme_id = scheme_id
        row = self._summary_row(scheme_id)
        if row is not None:
            self.diagnostic_var.set(self._diagnostic_for_row(row))
            self._set_status(f"已选择方案：{row.get('scheme_name', scheme_id)}")

    def _summary_compare_click(self, scheme_id: str):
        if scheme_id in self._comparison_selected_ids:
            self._comparison_selected_ids.remove(scheme_id)
        else:
            self._comparison_selected_ids.add(scheme_id)
        self.summary_table.set_compared(scheme_id, scheme_id in self._comparison_selected_ids)
        self._set_status(f"已选择 {len(self._comparison_selected_ids)} 个比较方案（最多 10 个）")

    def _summary_preview(self, scheme_id: str):
        record = self._record_for_id(scheme_id)
        if record is None or self.store is None:
            return
        try:
            image = Image.open(self.store.resolve(record.relative_path)).convert("L")
        except Exception:
            messagebox.showwarning("图像缺失", f"无法打开结果图：{record.display_name}", parent=self)
            return
        ImagePreviewWindow(self, image, record.display_name)

    def compare_selected_schemes(self):
        if not self.store:
            return
        if self.store.evaluation_status() == "expired":
            self._show_expired_evaluation("检测到图像文件或 ROI 配置已发生变化，请重新评价。")
            messagebox.showwarning("结果已过期", "图像或 ROI 已改变，当前评价结果不能继续用于比较。", parent=self)
            return
        result = self._last_evaluation or self.store.load_latest_evaluation()
        if not result:
            messagebox.showwarning("尚无评价结果", "请先完成一次评价。", parent=self)
            return
        selected = [row.get("scheme_id") for row in (result or {}).get("metrics", []) if row.get("scheme_id") in self._comparison_selected_ids]
        if len(selected) < 2:
            messagebox.showinfo("方案比较", "请至少选择两个方案。", parent=self)
            return
        if len(selected) > MAX_COMPARISON_SCHEMES:
            messagebox.showinfo("方案比较", f"最多同时比较 {MAX_COMPARISON_SCHEMES} 个方案。", parent=self)
            return
        ComparisonWindow(self, self.store, result, selected)

    def _show_expired_evaluation(self, message: str):
        self._last_evaluation = None
        self._comparison_selected_ids.clear()
        self._selected_summary_scheme_id = None
        self.summary_table.clear()
        self.roi_result_tree.delete(*self.roi_result_tree.get_children())
        self.diagnostic_var.set(message)
        self._set_status(message)

    @staticmethod
    def _status_label(row: dict) -> str:
        status = row.get("status", "")
        if status == "valid":
            return "计算正常"
        if status == "valid_with_warnings":
            return "计算正常（有警告）"
        if status == "partial":
            return "指标缺失"
        if status == "failed":
            return "计算失败"
        return status or "未知状态"

    def _diagnostic_for_row(self, row: dict) -> str:
        if row.get("role") == "original":
            return "原图作为变化基准：Local CNR、Background CNR、Weak AG 和 Background Noise 的变化量显示为“—”；SSIM 为 1。"
        local_change = row.get("local_cnr_change_percent")
        background_change = row.get("background_cnr_change_percent")
        ag_change = row.get("weak_ag_change_percent")
        noise_change = row.get("background_noise_change_percent")
        ssim = row.get("ssim")
        sat_change = row.get("saturation_change_pp")
        notes: list[str] = []
        if local_change is not None and local_change > 10:
            notes.append("Local CNR 上升，说明弱骨骼与对应邻域的局部区分度提高")
        elif local_change is not None and local_change < -10:
            notes.append("Local CNR 下降，说明弱骨骼与对应参考区域的区分度下降")
        if background_change is not None and background_change > 10:
            notes.append("Background CNR 上升，弱骨骼相对体外背景更突出")
        elif background_change is not None and background_change < -10:
            notes.append("Background CNR 下降，弱骨骼相对体外背景的突出程度减弱")
        if ag_change is not None and noise_change is not None and ag_change > 10 and noise_change > 10:
            notes.append("AG 明显上升且 Noise 也上升，边缘响应增强可能混入噪声贡献")
        elif ag_change is not None and noise_change is not None and ag_change > 10 and noise_change < -10:
            notes.append("AG 上升且 Noise 下降，弱骨骼细节增强同时背景更稳定")
        elif ag_change is not None and ag_change < -10 and ssim is not None and ssim > 0.95:
            notes.append("AG 下降而 SSIM 较高，结构变化较小但可能存在过度平滑")
        if sat_change is not None and sat_change > 5:
            notes.append("Strong Sat.% 相比原图上升，强骨骼区域的饱和风险增加")
        elif sat_change is not None and sat_change < -5:
            notes.append("Strong Sat.% 相比原图下降，强骨骼饱和风险减弱")
        values = (
            f"Local CNR={format_number(row.get('weak_bone_mean_cnr_local'))}",
            f"Background CNR={format_number(row.get('weak_bone_mean_cnr_background'))}",
            f"Weak AG={format_number(row.get('weak_bone_mean_average_gradient'))}",
            f"Noise={format_number(row.get('background_noise_pooled'))}",
            f"SSIM={format_number(ssim)}",
            f"Strong Sat.={format_number(None if row.get('strong_bone_saturation_mean') is None else row['strong_bone_saturation_mean'] * 100, 2)}%",
        )
        prefix = f"{row.get('scheme_name', '方案')}：" + "；".join(values) + "。"
        return prefix + (" " + "；".join(notes) + "。" if notes else " 当前组合未触发预设提示，请结合图像和 ROI 明细判断。")

    def _fill_results(self, result):
        self._last_evaluation = result
        self.summary_table.clear()
        self.roi_result_tree.delete(*self.roi_result_tree.get_children())
        self._summary_thumbnail_refs.clear()
        valid_ids = {row.get("scheme_id") for row in result.get("metrics", [])}
        self._comparison_selected_ids.intersection_update(valid_ids)
        for row in result.get("metrics", []):
            record = self._record_for_id(row.get("scheme_id", ""))
            thumbnail = self._thumbnail_for_record(record)
            status = self._status_label(row)
            if row.get("message"):
                status += "：" + str(row["message"])
            values = (
                row.get("scheme_name", ""), format_number(row.get("weak_bone_mean_cnr_local")),
                format_number(row.get("local_cnr_change_percent"), 2), format_number(row.get("weak_bone_mean_cnr_background")),
                format_number(row.get("background_cnr_change_percent"), 2), format_number(row.get("weak_bone_mean_average_gradient")),
                format_number(row.get("weak_ag_change_percent"), 2), format_number(row.get("background_noise_pooled")),
                format_number(row.get("background_noise_change_percent"), 2), format_number(row.get("ssim")),
                format_number(None if row.get("strong_bone_saturation_mean") is None else row["strong_bone_saturation_mean"] * 100, 2),
                format_number(row.get("saturation_change_pp"), 2),
                format_number(row.get("all_bone_mean_cnr_background")),
                format_number(row.get("all_bone_mean_cnr_local")),
                format_number(row.get("strong_bone_mean_cnr_background")),
                format_number(row.get("strong_bone_mean_cnr_local")),
                format_number(row.get("average_gradient_global")),
                format_number(row.get("strong_bone_mean_average_gradient")),
                status,
            )
            self.summary_table.add_row(row.get("scheme_id"), thumbnail, values, row.get("scheme_id") in self._comparison_selected_ids)
        for row in result.get("roi_metrics", []):
            self.roi_result_tree.insert("", "end", values=(
                row.get("scheme_name", ""), row.get("roi_name", ""), ROI_LABELS.get(row.get("roi_type"), row.get("roi_type", "")),
                format_number(row.get("mean_intensity")), format_number(row.get("std_intensity")), format_number(row.get("average_gradient_roi")),
                format_number(row.get("cnr_background")), format_number(row.get("cnr_local")),
                format_number(None if row.get("saturation_ratio") is None else row["saturation_ratio"] * 100, 2), self._status_label(row),
                row.get("message") or row.get("cnr_background_reason") or row.get("cnr_local_reason") or row.get("saturation_reason", ""),
            ))
        self._selected_summary_scheme_id = None
        self.diagnostic_var.set("请点击总体表中的某个方案，查看该方案的指标诊断。")
        warnings = [issue.get("message", "") for issue in result.get("validation", []) if issue.get("severity") in {"warning", "error"}]
        if warnings:
            self._set_status("评价完成：" + warnings[0])


def main() -> int:
    try:
        app = BoneIQAApp()
        app.mainloop()
        return 0
    except Exception as exc:
        print(f"启动失败：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

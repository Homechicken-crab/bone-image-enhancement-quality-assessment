from __future__ import annotations

import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk

from PIL import Image, ImageTk

from .evaluator import evaluate_project, export_csv_bundle
from .models import ROI
from .project_store import ProjectError, ProjectStore
from .validation import validate_project


IMAGE_TYPES = [("灰度图像", "*.bmp *.png *.tif *.tiff"), ("所有文件", "*.*")]
ROI_LABELS = {
    "weak_bone": "Weak Bone",
    "strong_bone": "Strong Bone",
    "surrounding": "Surrounding",
    "background": "Background",
}
ROI_COLORS = {
    "weak_bone": "#ffd54f",
    "strong_bone": "#ef5350",
    "surrounding": "#26c6da",
    "background": "#42a5f5",
}


def format_number(value, digits: int = 4) -> str:
    if value is None or value == "":
        return "—"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


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
        ttk.Label(right, text="ROI 列表", font=("Microsoft YaHei UI", 11, "bold")).pack(anchor="w")
        self.roi_tree = ttk.Treeview(right, columns=("type", "name"), show="headings", height=11, selectmode="browse")
        self.roi_tree.heading("type", text="类型")
        self.roi_tree.heading("name", text="名称")
        self.roi_tree.column("type", width=110)
        self.roi_tree.column("name", width=170)
        self.roi_tree.pack(fill="x", pady=(5, 8))
        self.roi_tree.bind("<<TreeviewSelect>>", self._roi_selected)

        form = ttk.LabelFrame(right, text="ROI 属性", padding=8)
        form.pack(fill="x")
        self.roi_name_var = tk.StringVar()
        self.roi_type_var = tk.StringVar(value="weak_bone")
        self.roi_pair_var = tk.StringVar()
        self.roi_coord_vars = {key: tk.StringVar(value="0") for key in ("x", "y", "width", "height")}
        ttk.Label(form, text="名称").grid(row=0, column=0, sticky="w", pady=3)
        ttk.Entry(form, textvariable=self.roi_name_var, width=28).grid(row=0, column=1, columnspan=3, sticky="ew", pady=3)
        ttk.Label(form, text="类型").grid(row=1, column=0, sticky="w", pady=3)
        self.roi_type_combo = ttk.Combobox(form, textvariable=self.roi_type_var, values=list(ROI_LABELS), state="readonly", width=25)
        self.roi_type_combo.grid(row=1, column=1, columnspan=3, sticky="ew", pady=3)
        labels = ("x", "y", "width", "height")
        for index, key in enumerate(labels):
            ttk.Label(form, text=key).grid(row=2 + index // 2, column=(index % 2) * 2, sticky="w", pady=3)
            ttk.Entry(form, textvariable=self.roi_coord_vars[key], width=9).grid(row=2 + index // 2, column=(index % 2) * 2 + 1, sticky="ew", padx=(2, 8), pady=3)
        ttk.Label(form, text="Surrounding 配对").grid(row=4, column=0, sticky="w", pady=3)
        self.roi_pair_combo = ttk.Combobox(form, textvariable=self.roi_pair_var, state="readonly", width=25)
        self.roi_pair_combo.grid(row=4, column=1, columnspan=3, sticky="ew", pady=3)
        form.columnconfigure(1, weight=1)
        buttons = ttk.Frame(right)
        buttons.pack(fill="x", pady=8)
        ttk.Button(buttons, text="新建 / 清空", command=self.clear_roi_form).pack(side="left", padx=2)
        ttk.Button(buttons, text="保存 ROI", command=self.save_roi).pack(side="left", padx=2)
        ttk.Button(buttons, text="删除", command=self.delete_roi).pack(side="left", padx=2)
        ttk.Button(buttons, text="显示/隐藏", command=self.toggle_roi).pack(side="left", padx=2)

    def _build_evaluation(self):
        toolbar = ttk.Frame(self.evaluate_tab, padding=8)
        toolbar.pack(fill="x")
        ttk.Button(toolbar, text="开始评价", command=self.run_evaluation).pack(side="left", padx=3)
        ttk.Button(toolbar, text="导出 CSV", command=self.export_results).pack(side="left", padx=3)
        self.primary_label = tk.StringVar(value="主 CNR：Background-based / Weak Bone Mean")
        ttk.Label(toolbar, textvariable=self.primary_label).pack(side="right", padx=8)

        result_book = ttk.Notebook(self.evaluate_tab)
        result_book.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        summary_frame = ttk.Frame(result_book)
        auxiliary_frame = ttk.Frame(result_book)
        roi_frame = ttk.Frame(result_book)
        validation_frame = ttk.Frame(result_book)
        result_book.add(summary_frame, text="总体表")
        result_book.add(auxiliary_frame, text="辅助指标")
        result_book.add(roi_frame, text="ROI 表")
        result_book.add(validation_frame, text="检查结果")

        summary_columns = ("name", "cnr", "cnr_change", "ag", "ag_change", "noise", "noise_change", "ssim", "sat", "sat_change", "status")
        self.summary_tree = ttk.Treeview(summary_frame, columns=summary_columns, show="headings")
        summary_headers = {
            "name": "方案", "cnr": "Weak CNR", "cnr_change": "CNR Δ%", "ag": "Weak AG",
            "ag_change": "AG Δ%", "noise": "Noise", "noise_change": "Noise Δ%",
            "ssim": "SSIM", "sat": "Strong Sat.%", "sat_change": "Sat. Δpp", "status": "状态",
        }
        for key in summary_columns:
            self.summary_tree.heading(key, text=summary_headers[key])
            self.summary_tree.column(key, width=115 if key == "name" else 88, anchor="center")
        self.summary_tree.pack(fill="both", expand=True)

        auxiliary_columns = ("name", "all_bg", "all_local", "strong_bg", "strong_local", "global_ag", "strong_ag")
        self.auxiliary_tree = ttk.Treeview(auxiliary_frame, columns=auxiliary_columns, show="headings")
        auxiliary_headers = {
            "name": "方案", "all_bg": "All Bone CNR-bg", "all_local": "All Bone CNR-local",
            "strong_bg": "Strong CNR-bg", "strong_local": "Strong CNR-local",
            "global_ag": "Global AG", "strong_ag": "Strong Mean AG",
        }
        for key in auxiliary_columns:
            self.auxiliary_tree.heading(key, text=auxiliary_headers[key])
            self.auxiliary_tree.column(key, width=150 if key == "name" else 125, anchor="center")
        self.auxiliary_tree.pack(fill="both", expand=True)

        roi_columns = ("scheme", "roi", "type", "mean", "std", "ag", "cnr_bg", "cnr_local", "sat", "status")
        self.roi_result_tree = ttk.Treeview(roi_frame, columns=roi_columns, show="headings")
        for key, label in zip(roi_columns, ("方案", "ROI", "类型", "均值", "标准差", "AG", "CNR-bg", "CNR-local", "饱和率", "状态")):
            self.roi_result_tree.heading(key, text=label)
            self.roi_result_tree.column(key, width=115 if key in {"scheme", "roi"} else 88, anchor="center")
        self.roi_result_tree.pack(fill="both", expand=True)

        self.validation_tree = ttk.Treeview(validation_frame, columns=("severity", "object", "message"), show="headings")
        for key, label, width in (("severity", "级别", 80), ("object", "对象", 160), ("message", "说明", 700)):
            self.validation_tree.heading(key, text=label)
            self.validation_tree.column(key, width=width, anchor="w")
        self.validation_tree.pack(fill="both", expand=True)

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
        self.roi_name_var.set("")
        self.roi_type_var.set("weak_bone")
        self.roi_pair_var.set("")
        for variable in self.roi_coord_vars.values():
            variable.set("0")
        self.roi_tree.selection_remove(self.roi_tree.selection())

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
        self.roi_name_var.set(roi.name)
        self.roi_type_var.set(roi.type)
        for key, value in zip(("x", "y", "width", "height"), roi.geometry()):
            self.roi_coord_vars[key].set(str(value))
        pair = next((item for item in store.project.rois if item.id == roi.paired_surrounding_roi_id), None)
        self.roi_pair_var.set(f"{pair.name} | {pair.id}" if pair else "")

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
            store.upsert_roi(roi)
            self._selected_roi_id = roi.id
            self.refresh_all()
            self._set_status(f"ROI 已保存：{roi.name}")
        except Exception as exc:
            messagebox.showerror("保存 ROI 失败", str(exc))

    def delete_roi(self):
        store = self._require_store()
        if not store or not self._selected_roi_id:
            return
        if messagebox.askyesno("删除 ROI", "删除后，引用它的 Bone ROI 配对将被清空。是否继续？"):
            store.remove_roi(self._selected_roi_id)
            self.clear_roi_form()
            self.refresh_all()

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
        self._drag_start = None

    def show_validation(self):
        store = self._require_store()
        if not store:
            return
        issues = validate_project(store)
        self._fill_validation([issue.to_dict() for issue in issues])
        self.notebook.select(self.evaluate_tab)
        if not issues:
            messagebox.showinfo("检查完成", "未发现问题。")

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

    def refresh_overview(self):
        if not self.store:
            self.overview_text.set("尚未打开项目")
            return
        project = self.store.project
        counts = {key: sum(1 for roi in project.rois if roi.type == key) for key in ROI_LABELS}
        self.overview_text.set(
            f"项目：{project.name}\n"
            f"原图：{project.original.display_name if project.original else '未导入'}\n"
            f"增强方案：{len(project.algorithms)} 个\n"
            f"ROI：Weak {counts['weak_bone']} / Strong {counts['strong_bone']} / Surrounding {counts['surrounding']} / Background {counts['background']}\n"
            f"评价状态：{'已有结果 ' + project.latest_evaluation_id if project.latest_evaluation_id else '尚未评价或结果已失效'}"
        )

    def refresh_images(self):
        self.image_tree.delete(*self.image_tree.get_children())
        if not self.store:
            return
        issues = validate_project(self.store)
        error_ids = {issue.object_id for issue in issues if issue.severity == "error"}
        records = ([self.store.project.original] if self.store.project.original else []) + self.store.project.algorithms
        for record in records:
            self.image_tree.insert("", "end", iid=record.id, values=(
                "原图" if record.role == "original" else "算法",
                record.display_name,
                f"{record.width} × {record.height}",
                record.dtype,
                "错误" if record.id in error_ids else "通过",
                record.relative_path,
            ))

    def refresh_roi_list(self):
        self.roi_tree.delete(*self.roi_tree.get_children())
        if not self.store:
            self.roi_pair_combo["values"] = []
            return
        for roi in self.store.project.rois:
            suffix = "" if roi.visible else "（隐藏）"
            self.roi_tree.insert("", "end", iid=roi.id, values=(ROI_LABELS[roi.type], roi.name + suffix))
        surroundings = [f"{roi.name} | {roi.id}" for roi in self.store.project.rois if roi.type == "surrounding"]
        self.roi_pair_combo["values"] = [""] + surroundings
        if self._selected_roi_id and self.roi_tree.exists(self._selected_roi_id):
            self.roi_tree.selection_set(self._selected_roi_id)

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
            for roi in self.store.project.rois:
                if not roi.visible:
                    continue
                x1 = ox + roi.x * scale
                y1 = oy + roi.y * scale
                x2 = ox + (roi.x + roi.width) * scale
                y2 = oy + (roi.y + roi.height) * scale
                color = ROI_COLORS[roi.type]
                canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=2)
                canvas.create_text(x1 + 3, y1 + 3, text=roi.name, fill=color, anchor="nw", font=("Microsoft YaHei UI", 9, "bold"))
        except Exception as exc:
            canvas.create_text(20, 20, text=f"无法显示原图：{exc}", fill="white", anchor="nw")

    def _fill_results(self, result):
        self.summary_tree.delete(*self.summary_tree.get_children())
        self.auxiliary_tree.delete(*self.auxiliary_tree.get_children())
        self.roi_result_tree.delete(*self.roi_result_tree.get_children())
        self._fill_validation(result.get("validation", []))
        primary = result.get("config", {}).get("primary_cnr", "background")
        self.primary_label.set(f"主 CNR：{primary.title()} / Weak Bone Mean")
        for row in result.get("metrics", []):
            self.summary_tree.insert("", "end", values=(
                row.get("scheme_name", ""), format_number(row.get("primary_cnr")),
                format_number(row.get("primary_cnr_change_percent"), 2), format_number(row.get("weak_bone_mean_average_gradient")),
                format_number(row.get("weak_bone_mean_average_gradient_change_percent"), 2), format_number(row.get("background_noise_pooled")),
                format_number(row.get("background_noise_change_percent"), 2), format_number(row.get("ssim")),
                format_number(None if row.get("strong_bone_saturation_mean") is None else row["strong_bone_saturation_mean"] * 100, 2),
                format_number(row.get("saturation_change_pp"), 2), row.get("status", ""),
            ))
            self.auxiliary_tree.insert("", "end", values=(
                row.get("scheme_name", ""),
                format_number(row.get("all_bone_mean_cnr_background")),
                format_number(row.get("all_bone_mean_cnr_local")),
                format_number(row.get("strong_bone_mean_cnr_background")),
                format_number(row.get("strong_bone_mean_cnr_local")),
                format_number(row.get("average_gradient_global")),
                format_number(row.get("strong_bone_mean_average_gradient")),
            ))
        for row in result.get("roi_metrics", []):
            self.roi_result_tree.insert("", "end", values=(
                row.get("scheme_name", ""), row.get("roi_name", ""), ROI_LABELS.get(row.get("roi_type"), row.get("roi_type", "")),
                format_number(row.get("mean_intensity")), format_number(row.get("std_intensity")), format_number(row.get("average_gradient_roi")),
                format_number(row.get("cnr_background")), format_number(row.get("cnr_local")),
                format_number(None if row.get("saturation_ratio") is None else row["saturation_ratio"] * 100, 2), row.get("status", ""),
            ))

    def _fill_validation(self, rows):
        self.validation_tree.delete(*self.validation_tree.get_children())
        for row in rows:
            self.validation_tree.insert("", "end", values=(row.get("severity", ""), row.get("object_id", ""), row.get("message", "")))


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

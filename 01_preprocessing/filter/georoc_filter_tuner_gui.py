import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

from georoc_filter_analyzer import (
    DEFAULT_ALLOWED_SETTINGS,
    DEFAULT_FILE_PATH,
    DEFAULT_REFERENCES_PATH,
    FilterParams,
    TECTONIC_REPLACEMENTS,
    analyze_filters,
)


def parse_allowed_settings(text):
    return [item.strip() for item in text.split(",") if item.strip()]


class FilterTunerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("GEOROC Filter Tuner")
        self.root.geometry("1280x780")

        self.file_path_var = tk.StringVar(value=DEFAULT_FILE_PATH)
        self.references_path_var = tk.StringVar(value=DEFAULT_REFERENCES_PATH)

        self.use_sio2_var = tk.BooleanVar(value=True)
        self.sio2_min_var = tk.StringVar(value="44")
        self.sio2_max_var = tk.StringVar(value="53")

        self.use_mgo_al2o3_var = tk.BooleanVar(value=True)
        self.mgo_min_var = tk.StringVar(value="4.5")
        self.mgo_max_var = tk.StringVar(value="12")
        self.al2o3_min_var = tk.StringVar(value="12")
        self.al2o3_max_var = tk.StringVar(value="19")

        self.use_loi_var = tk.BooleanVar(value=True)
        self.loi_max_var = tk.StringVar(value="5")
        self.keep_loi_nan_var = tk.BooleanVar(value=True)

        self.use_non_nan_var = tk.BooleanVar(value=True)
        self.min_non_nan_var = tk.StringVar(value="20")

        self.exclude_archean_var = tk.BooleanVar(value=True)

        self.drop_duplicates_var = tk.BooleanVar(value=True)
        self.apply_setting_filter_var = tk.BooleanVar(value=True)
        self.rename_settings_var = tk.BooleanVar(value=True)
        self.add_publication_year_var = tk.BooleanVar(value=True)
        self.keep_all_columns_var = tk.BooleanVar(value=False)
        self.final_count_var = tk.StringVar(value="20000")
        self.allowed_settings_var = tk.StringVar(value=", ".join(DEFAULT_ALLOWED_SETTINGS))

        self.summary_var = tk.StringVar(value="等待运行")
        self.latest_output_df = None

        self._build_layout()

    def _build_layout(self):
        container = ttk.Frame(self.root, padding=12)
        container.pack(fill=tk.BOTH, expand=True)

        # 左侧筛选参数很多，放进可滚动区域，避免窗口高度不够时控件被截断。
        left_canvas = tk.Canvas(container, width=620, highlightthickness=0)
        left_scrollbar = ttk.Scrollbar(container, orient=tk.VERTICAL, command=left_canvas.yview)
        left = ttk.Frame(left_canvas)
        left_window = left_canvas.create_window((0, 0), window=left, anchor="nw")

        left.bind(
            "<Configure>",
            lambda event: left_canvas.configure(scrollregion=left_canvas.bbox("all")),
        )
        left_canvas.bind(
            "<Configure>",
            lambda event: left_canvas.itemconfigure(left_window, width=event.width),
        )
        left_canvas.configure(yscrollcommand=left_scrollbar.set)
        left_canvas.pack(side=tk.LEFT, fill=tk.Y)
        left_scrollbar.pack(side=tk.LEFT, fill=tk.Y)
        self._bind_left_mousewheel(left_canvas, left)

        right = ttk.Frame(container)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(12, 0))

        self._build_controls(left)
        self._build_results(right)

    def _bind_left_mousewheel(self, canvas, scroll_area):
        # 鼠标进入左侧参数区后，滚轮控制左侧滚动条。
        scroll_area.bind("<Enter>", lambda event: canvas.bind_all("<MouseWheel>", self._on_left_mousewheel))
        scroll_area.bind("<Leave>", lambda event: canvas.unbind_all("<MouseWheel>"))
        self.left_canvas = canvas

    def _on_left_mousewheel(self, event):
        self.left_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _build_controls(self, parent):
        file_frame = ttk.LabelFrame(parent, text="Data")
        file_frame.pack(fill=tk.X, pady=(0, 10))

        self._add_path_row(file_frame, "Data CSV", self.file_path_var, self._browse_data_file)
        self._add_path_row(file_frame, "References", self.references_path_var, self._browse_references_file)

        sio2_frame = ttk.LabelFrame(parent, text="SiO2 Filter")
        sio2_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Checkbutton(sio2_frame, text="Enable", variable=self.use_sio2_var).grid(row=0, column=0, sticky="w")
        self._add_labeled_entry(sio2_frame, "Min", self.sio2_min_var, 1, 0)
        self._add_labeled_entry(sio2_frame, "Max", self.sio2_max_var, 1, 2)

        mgo_frame = ttk.LabelFrame(parent, text="MgO / Al2O3 Filter")
        mgo_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Checkbutton(mgo_frame, text="Enable", variable=self.use_mgo_al2o3_var).grid(
            row=0, column=0, sticky="w"
        )
        self._add_labeled_entry(mgo_frame, "MgO Min", self.mgo_min_var, 1, 0)
        self._add_labeled_entry(mgo_frame, "MgO Max", self.mgo_max_var, 1, 2)
        self._add_labeled_entry(mgo_frame, "Al2O3 Min", self.al2o3_min_var, 2, 0)
        self._add_labeled_entry(mgo_frame, "Al2O3 Max", self.al2o3_max_var, 2, 2)

        loi_frame = ttk.LabelFrame(parent, text="LOI Filter")
        loi_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Checkbutton(loi_frame, text="Enable", variable=self.use_loi_var).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(loi_frame, text="Keep NaN", variable=self.keep_loi_nan_var).grid(
            row=0, column=1, sticky="w"
        )
        self._add_labeled_entry(loi_frame, "LOI Max", self.loi_max_var, 1, 0)

        nan_frame = ttk.LabelFrame(parent, text="Completeness Filter")
        nan_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Checkbutton(nan_frame, text="Enable", variable=self.use_non_nan_var).grid(
            row=0, column=0, sticky="w"
        )
        self._add_labeled_entry(nan_frame, "Min non-NaN", self.min_non_nan_var, 1, 0)

        age_frame = ttk.LabelFrame(parent, text="Age Filter")
        age_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Checkbutton(
            age_frame, text="Exclude Archean age", variable=self.exclude_archean_var
        ).grid(row=0, column=0, sticky="w")

        setting_frame = ttk.LabelFrame(parent, text="Tectonic Setting")
        setting_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Checkbutton(
            setting_frame, text="Apply allowed settings", variable=self.apply_setting_filter_var
        ).grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Checkbutton(setting_frame, text="Drop duplicates", variable=self.drop_duplicates_var).grid(
            row=1, column=0, sticky="w"
        )
        ttk.Checkbutton(
            setting_frame,
            text="Rename RIFT VOLCANICS / Back-arc basin",
            variable=self.rename_settings_var,
        ).grid(
            row=1, column=1, sticky="w"
        )
        ttk.Checkbutton(
            setting_frame, text="Add publication year", variable=self.add_publication_year_var
        ).grid(row=2, column=0, columnspan=2, sticky="w")
        ttk.Checkbutton(
            setting_frame, text="Keep all original columns", variable=self.keep_all_columns_var
        ).grid(row=2, column=1, columnspan=2, sticky="w")
        self._add_labeled_entry(setting_frame, "Max count per class", self.final_count_var, 3, 0)

        ttk.Label(setting_frame, text="Allowed settings").grid(row=4, column=0, sticky="w", pady=(8, 4))
        ttk.Entry(setting_frame, textvariable=self.allowed_settings_var, width=48).grid(
            row=5, column=0, columnspan=4, sticky="ew"
        )

        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, pady=(8, 0))
        ttk.Button(button_frame, text="Run Analysis", command=self.run_analysis).pack(side=tk.LEFT)
        ttk.Button(button_frame, text="Reset Defaults", command=self.reset_defaults).pack(side=tk.LEFT, padx=(8, 0))
        self.save_button = ttk.Button(button_frame, text="Save Results", command=self.save_results, state=tk.DISABLED)
        self.save_button.pack(side=tk.LEFT, padx=(8, 0))

    def _build_results(self, parent):
        summary_frame = ttk.LabelFrame(parent, text="Summary")
        summary_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(summary_frame, textvariable=self.summary_var).pack(anchor="w", padx=8, pady=8)

        counts_frame = ttk.LabelFrame(parent, text="Final Counts")
        counts_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        self.counts_tree = ttk.Treeview(
            counts_frame,
            columns=("tectonic_setting", "count"),
            show="headings",
            height=12,
        )
        self.counts_tree.heading("tectonic_setting", text="TECTONIC SETTING")
        self.counts_tree.heading("count", text="Count")
        self.counts_tree.column("tectonic_setting", width=260, anchor="w")
        self.counts_tree.column("count", width=120, anchor="center")
        counts_scrollbar = ttk.Scrollbar(counts_frame, orient=tk.VERTICAL, command=self.counts_tree.yview)
        self.counts_tree.configure(yscrollcommand=counts_scrollbar.set)
        self.counts_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        counts_scrollbar.pack(side=tk.LEFT, fill=tk.Y)

        step_frame = ttk.LabelFrame(parent, text="Step-by-Step Counts")
        step_frame.pack(fill=tk.BOTH, expand=True)
        self.step_text = scrolledtext.ScrolledText(step_frame, wrap=tk.WORD, font=("Consolas", 10))
        self.step_text.pack(fill=tk.BOTH, expand=True)

    def _add_path_row(self, parent, label, variable, command):
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, padx=8, pady=4)
        ttk.Label(row, text=label, width=10).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=variable, width=52).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        ttk.Button(row, text="Browse", command=command).pack(side=tk.LEFT)

    def _add_labeled_entry(self, parent, label, variable, row, column):
        ttk.Label(parent, text=label).grid(row=row, column=column, sticky="w", padx=(0, 6), pady=4)
        ttk.Entry(parent, textvariable=variable, width=10).grid(row=row, column=column + 1, sticky="w", pady=4)

    def _browse_data_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if file_path:
            self.file_path_var.set(file_path)

    def _browse_references_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if file_path:
            self.references_path_var.set(file_path)

    def _get_display_settings(self, params, counts):
        # 按允许构造环境逐项显示数量，筛选后为 0 的类别也保留在表格里。
        display_settings = []
        seen_settings = set()
        for setting in params.allowed_settings:
            display_setting = TECTONIC_REPLACEMENTS.get(setting, setting) if params.rename_tectonic_settings else setting
            if display_setting not in seen_settings:
                display_settings.append(display_setting)
                seen_settings.add(display_setting)

        for setting in counts:
            if setting not in seen_settings:
                display_settings.append(setting)
                seen_settings.add(setting)
        return display_settings

    def build_params(self):
        params = FilterParams()
        params.use_sio2_filter = self.use_sio2_var.get()
        params.sio2_min = float(self.sio2_min_var.get())
        params.sio2_max = float(self.sio2_max_var.get())

        params.use_mgo_al2o3_filter = self.use_mgo_al2o3_var.get()
        params.mgo_min = float(self.mgo_min_var.get())
        params.mgo_max = float(self.mgo_max_var.get())
        params.al2o3_min = float(self.al2o3_min_var.get())
        params.al2o3_max = float(self.al2o3_max_var.get())

        params.use_loi_filter = self.use_loi_var.get()
        params.loi_max = float(self.loi_max_var.get())
        params.keep_loi_nan = self.keep_loi_nan_var.get()

        params.use_min_non_nan_filter = self.use_non_nan_var.get()
        params.min_non_nan = int(self.min_non_nan_var.get())
        params.exclude_archean_age = self.exclude_archean_var.get()

        params.drop_duplicates = self.drop_duplicates_var.get()
        params.apply_setting_filter = self.apply_setting_filter_var.get()
        params.rename_tectonic_settings = self.rename_settings_var.get()
        params.add_publication_year = self.add_publication_year_var.get()
        params.keep_all_columns = self.keep_all_columns_var.get()
        params.final_count_per_setting = int(self.final_count_var.get())
        params.allowed_settings = parse_allowed_settings(self.allowed_settings_var.get())
        return params

    def _get_save_dialog_defaults(self):
        source_path = Path(self.file_path_var.get())
        initial_dir = str(source_path.parent) if source_path.parent.exists() else ""
        suffix = source_path.suffix or ".csv"
        stem = source_path.stem or "georoc_filtered"
        return initial_dir, f"{stem}_filtered{suffix}"

    def run_analysis(self):
        try:
            params = self.build_params()
            output_df, stats = analyze_filters(
                file_path=self.file_path_var.get(),
                params=params,
                references_path=self.references_path_var.get(),
            )
        except Exception as exc:
            messagebox.showerror("Run failed", str(exc))
            return

        self.latest_output_df = output_df
        self.save_button.config(state=tk.NORMAL)
        final_step = stats[-1]
        final_counts = final_step["counts"]
        display_settings = self._get_display_settings(params, final_counts)
        arc_count = final_counts.get("Island arc", 0) + final_counts.get("Intra-oceanic arc", 0)
        self.summary_var.set(
            f"Final rows: {final_step['total_rows']} | Arc total: {arc_count} | "
            f"Classes: {len([setting for setting in final_counts if final_counts[setting] > 0])}"
        )

        for item in self.counts_tree.get_children():
            self.counts_tree.delete(item)
        for tectonic_setting in display_settings:
            self.counts_tree.insert("", tk.END, values=(tectonic_setting, final_counts.get(tectonic_setting, 0)))

        lines = []
        for step in stats:
            lines.append(f"{step['step']} | total_rows={step['total_rows']}")
            if step["counts"]:
                for tectonic_setting, count in step["counts"].items():
                    lines.append(f"  {tectonic_setting}: {count}")
            else:
                lines.append("  No rows")
            lines.append("")

        self.step_text.delete("1.0", tk.END)
        self.step_text.insert(tk.END, "\n".join(lines))

    def save_results(self):
        if self.latest_output_df is None:
            messagebox.showinfo("No results", "Run Analysis before saving results.")
            return

        initial_dir, initial_file = self._get_save_dialog_defaults()
        file_path = filedialog.asksaveasfilename(
            title="Save filtered GEOROC results",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialdir=initial_dir,
            initialfile=initial_file,
        )
        if not file_path:
            return

        try:
            self.latest_output_df.to_csv(file_path, index=False)
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))
            return

        messagebox.showinfo("Save complete", f"Saved results to:\n{file_path}")

    def reset_defaults(self):
        self.file_path_var.set(DEFAULT_FILE_PATH)
        self.references_path_var.set(DEFAULT_REFERENCES_PATH)
        self.use_sio2_var.set(True)
        self.sio2_min_var.set("44")
        self.sio2_max_var.set("53")
        self.use_mgo_al2o3_var.set(True)
        self.mgo_min_var.set("4.5")
        self.mgo_max_var.set("12")
        self.al2o3_min_var.set("12")
        self.al2o3_max_var.set("19")
        self.use_loi_var.set(True)
        self.loi_max_var.set("5")
        self.keep_loi_nan_var.set(True)
        self.use_non_nan_var.set(True)
        self.min_non_nan_var.set("20")
        self.exclude_archean_var.set(True)
        self.drop_duplicates_var.set(True)
        self.apply_setting_filter_var.set(True)
        self.rename_settings_var.set(True)
        self.add_publication_year_var.set(True)
        self.keep_all_columns_var.set(False)
        self.final_count_var.set("20000")
        self.allowed_settings_var.set(", ".join(DEFAULT_ALLOWED_SETTINGS))
        self.summary_var.set("等待运行")
        self.latest_output_df = None
        self.save_button.config(state=tk.DISABLED)
        self.step_text.delete("1.0", tk.END)
        for item in self.counts_tree.get_children():
            self.counts_tree.delete(item)


def main():
    root = tk.Tk()
    style = ttk.Style(root)
    if "vista" in style.theme_names():
        style.theme_use("vista")
    app = FilterTunerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

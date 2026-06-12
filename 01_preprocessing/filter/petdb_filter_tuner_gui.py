import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

from petdb_filter_analyzer import DEFAULT_ALLOWED_SETTINGS, DEFAULT_FILE_PATH, FilterParams, analyze_filters


def parse_allowed_settings(text):
    return [item.strip() for item in text.split(",") if item.strip()]


class PetdbFilterTunerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PETDB Filter Tuner")
        self.root.geometry("1280x780")

        self.file_path_var = tk.StringVar(value=DEFAULT_FILE_PATH)
        self.use_non_nan_var = tk.BooleanVar(value=True)
        self.min_non_nan_var = tk.StringVar(value="18")
        self.require_major_oxides_var = tk.BooleanVar(value=True)

        self.use_sio2_var = tk.BooleanVar(value=True)
        self.sio2_min_var = tk.StringVar(value="45")
        self.sio2_max_var = tk.StringVar(value="53")

        self.use_mgo_al2o3_var = tk.BooleanVar(value=True)
        self.mgo_min_var = tk.StringVar(value="4.5")
        self.mgo_max_var = tk.StringVar(value="12")
        self.al2o3_min_var = tk.StringVar(value="12")
        self.al2o3_max_var = tk.StringVar(value="19")
        self.exclude_archean_var = tk.BooleanVar(value=True)

        self.normalize_settings_var = tk.BooleanVar(value=True)
        self.extract_year_var = tk.BooleanVar(value=True)
        self.drop_duplicates_var = tk.BooleanVar(value=True)
        self.apply_setting_filter_var = tk.BooleanVar(value=True)
        self.final_count_var = tk.StringVar(value="20000")
        self.allowed_settings_var = tk.StringVar(value=", ".join(DEFAULT_ALLOWED_SETTINGS))

        self.summary_var = tk.StringVar(value="等待运行")
        self.latest_output_df = None
        self._build_layout()

    def _build_layout(self):
        container = ttk.Frame(self.root, padding=12)
        container.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(container)
        left.pack(side=tk.LEFT, fill=tk.Y)

        right = ttk.Frame(container)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(12, 0))

        self._build_controls(left)
        self._build_results(right)

    def _build_controls(self, parent):
        file_frame = ttk.LabelFrame(parent, text="Data")
        file_frame.pack(fill=tk.X, pady=(0, 10))
        self._add_path_row(file_frame, "Data CSV", self.file_path_var, self._browse_data_file)

        non_nan_frame = ttk.LabelFrame(parent, text="Completeness Filter")
        non_nan_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Checkbutton(non_nan_frame, text="Enable", variable=self.use_non_nan_var).grid(
            row=0, column=0, sticky="w"
        )
        self._add_labeled_entry(non_nan_frame, "Min non-NaN", self.min_non_nan_var, 1, 0)
        ttk.Checkbutton(
            non_nan_frame,
            text="Require SiO2/Al2O3/FeOT/MgO/CaO (no missing)",
            variable=self.require_major_oxides_var,
        ).grid(row=2, column=0, columnspan=4, sticky="w")

        sio2_frame = ttk.LabelFrame(parent, text="SiO2 Filter (anhydrous 100%)")
        sio2_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Checkbutton(sio2_frame, text="Enable", variable=self.use_sio2_var).grid(row=0, column=0, sticky="w")
        self._add_labeled_entry(sio2_frame, "Min", self.sio2_min_var, 1, 0)
        self._add_labeled_entry(sio2_frame, "Max", self.sio2_max_var, 1, 2)

        mgo_frame = ttk.LabelFrame(parent, text="MgO / Al2O3 Filter (anhydrous 100%)")
        mgo_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Checkbutton(mgo_frame, text="Enable", variable=self.use_mgo_al2o3_var).grid(
            row=0, column=0, sticky="w"
        )
        self._add_labeled_entry(mgo_frame, "MgO Min", self.mgo_min_var, 1, 0)
        self._add_labeled_entry(mgo_frame, "MgO Max", self.mgo_max_var, 1, 2)
        self._add_labeled_entry(mgo_frame, "Al2O3 Min", self.al2o3_min_var, 2, 0)
        self._add_labeled_entry(mgo_frame, "Al2O3 Max", self.al2o3_max_var, 2, 2)

        age_frame = ttk.LabelFrame(parent, text="Age Filter")
        age_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Checkbutton(
            age_frame,
            text="Exclude Archean age",
            variable=self.exclude_archean_var,
        ).grid(row=0, column=0, sticky="w")

        setting_frame = ttk.LabelFrame(parent, text="Tectonic Setting")
        setting_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Checkbutton(
            setting_frame, text="Normalize setting names", variable=self.normalize_settings_var
        ).grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Checkbutton(setting_frame, text="Extract publication year", variable=self.extract_year_var).grid(
            row=1, column=0, columnspan=2, sticky="w"
        )
        ttk.Checkbutton(setting_frame, text="Drop duplicates", variable=self.drop_duplicates_var).grid(
            row=2, column=0, sticky="w"
        )
        ttk.Checkbutton(
            setting_frame, text="Apply allowed settings", variable=self.apply_setting_filter_var
        ).grid(row=2, column=1, sticky="w")
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
        self.counts_tree.pack(fill=tk.BOTH, expand=True)

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

    def build_params(self):
        params = FilterParams()
        params.use_min_non_nan_filter = self.use_non_nan_var.get()
        params.min_non_nan = int(self.min_non_nan_var.get())
        params.require_major_oxides = self.require_major_oxides_var.get()
        params.use_sio2_filter = self.use_sio2_var.get()
        params.sio2_min = float(self.sio2_min_var.get())
        params.sio2_max = float(self.sio2_max_var.get())
        params.use_mgo_al2o3_filter = self.use_mgo_al2o3_var.get()
        params.mgo_min = float(self.mgo_min_var.get())
        params.mgo_max = float(self.mgo_max_var.get())
        params.al2o3_min = float(self.al2o3_min_var.get())
        params.al2o3_max = float(self.al2o3_max_var.get())
        params.exclude_archean_age = self.exclude_archean_var.get()
        params.normalize_setting_names = self.normalize_settings_var.get()
        params.extract_publication_year = self.extract_year_var.get()
        params.drop_duplicates = self.drop_duplicates_var.get()
        params.apply_setting_filter = self.apply_setting_filter_var.get()
        params.final_count_per_setting = int(self.final_count_var.get())
        params.allowed_settings = parse_allowed_settings(self.allowed_settings_var.get())
        return params

    def _get_save_dialog_defaults(self):
        source_path = Path(self.file_path_var.get())
        initial_dir = str(source_path.parent) if source_path.parent.exists() else ""
        suffix = source_path.suffix or ".csv"
        stem = source_path.stem or "petdb_filtered"
        return initial_dir, f"{stem}_filtered{suffix}"

    def run_analysis(self):
        try:
            params = self.build_params()
            output_df, stats = analyze_filters(file_path=self.file_path_var.get(), params=params)
        except Exception as exc:
            messagebox.showerror("Run failed", str(exc))
            return

        self.latest_output_df = output_df
        self.save_button.config(state=tk.NORMAL)
        final_step = stats[-1]
        self.summary_var.set(
            f"Final rows: {final_step['total_rows']} | "
            f"Classes: {len(final_step['counts'])}"
        )

        for item in self.counts_tree.get_children():
            self.counts_tree.delete(item)
        for tectonic_setting, count in final_step["counts"].items():
            self.counts_tree.insert("", tk.END, values=(tectonic_setting, count))

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
            title="Save filtered PETDB results",
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
        self.use_non_nan_var.set(True)
        self.min_non_nan_var.set("20")
        self.require_major_oxides_var.set(True)
        self.use_sio2_var.set(True)
        self.sio2_min_var.set("45")
        self.sio2_max_var.set("53")
        self.use_mgo_al2o3_var.set(True)
        self.mgo_min_var.set("4.5")
        self.mgo_max_var.set("12")
        self.al2o3_min_var.set("12")
        self.al2o3_max_var.set("19")
        self.exclude_archean_var.set(True)
        self.normalize_settings_var.set(True)
        self.extract_year_var.set(True)
        self.drop_duplicates_var.set(True)
        self.apply_setting_filter_var.set(True)
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
    PetdbFilterTunerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

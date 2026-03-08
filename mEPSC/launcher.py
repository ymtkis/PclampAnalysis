import sys
import threading
import subprocess
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox
import tkinter.font as tkfont


SCRIPTS = [
    ("1) Event annotation", "Event_annotation.py"),
    ("2) Create dataset", "Create_dataset.py"),
    ("3) Training", "Training.py"),
    ("4) Prediction", "Prediction.py"),
    ("5) Edit predicted", "Edit_predicted.py"),
]


class Launcher(tk.Tk):
    def __init__(self):
        super().__init__()
        default_font = tkfont.nametofont("TkDefaultFont")
        default_font.configure(family="Arial", size=12)
        self.option_add("*Font", default_font)

        self.title("Pclamp pipeline launcher")
        self.geometry("980x620")

        self.workdir = Path(__file__).resolve().parent
        self.pyexe = sys.executable

        self.var_auto = tk.BooleanVar(value=False)
        self.var_config = tk.StringVar(value="config.yaml")

        self._proc = None
        self._reader_thread = None

        self._build_ui()

    def _build_ui(self):
        top = ttk.Frame(self, padding=10)
        top.pack(fill="x")

        ttk.Label(top, text="Working dir:").grid(row=0, column=0, sticky="w")
        ttk.Label(top, text=str(self.workdir)).grid(row=0, column=1, sticky="w", padx=8)

        ttk.Label(top, text="Python:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Label(top, text=self.pyexe).grid(row=1, column=1, sticky="w", padx=8, pady=(6, 0))

        opts = ttk.Frame(top)
        opts.grid(row=2, column=0, columnspan=2, sticky="w", pady=(10, 0))

        ttk.Checkbutton(
            opts,
            text="--auto (used by scripts that support it)",
            variable=self.var_auto,
        ).grid(row=0, column=0, sticky="w")

        ttk.Label(opts, text="Config:").grid(row=0, column=1, sticky="w", padx=(18, 6))
        cfg_entry = ttk.Entry(opts, textvariable=self.var_config, width=28)
        cfg_entry.grid(row=0, column=2, sticky="w")

        btns = ttk.Frame(self, padding=(10, 6))
        btns.pack(fill="x")

        ttk.Label(btns, text="Run stage:").pack(anchor="w")

        grid = ttk.Frame(btns)
        grid.pack(fill="x", pady=(6, 0))

        for i, (label, script) in enumerate(SCRIPTS):
            b = ttk.Button(
                grid,
                text=label,
                command=lambda s=script: self.run_script(s),
                width=26,
            )
            b.grid(row=i // 2, column=i % 2, padx=6, pady=6, sticky="w")

        ctrl = ttk.Frame(btns)
        ctrl.pack(fill="x", pady=(8, 0))

        ttk.Button(ctrl, text="Stop", command=self.stop_proc).pack(side="left")
        ttk.Button(ctrl, text="Clear log", command=self.clear_log).pack(side="left", padx=(8, 0))

        self.status = tk.StringVar(value="Idle.")
        ttk.Label(ctrl, textvariable=self.status).pack(side="left", padx=(16, 0))

        # Log
        logframe = ttk.Frame(self, padding=10)
        logframe.pack(fill="both", expand=True)

        self.text = tk.Text(logframe, wrap="word")
        self.text.pack(side="left", fill="both", expand=True)

        sb = ttk.Scrollbar(logframe, orient="vertical", command=self.text.yview)
        sb.pack(side="right", fill="y")
        self.text.configure(yscrollcommand=sb.set)

        self._append(f"[Launcher] workdir = {self.workdir}\n")

    def _append(self, s: str):
        self.text.insert("end", s)
        self.text.see("end")
        self.text.update_idletasks()

    def clear_log(self):
        self.text.delete("1.0", "end")
        self.status.set("Idle.")

    def _script_path(self, script_name: str) -> Path:
        p = (self.workdir / script_name).resolve()
        return p

    def run_script(self, script_name: str):
        if self._proc is not None and self._proc.poll() is None:
            messagebox.showwarning("Running", "A process is already running. Stop it first.")
            return

        script = self._script_path(script_name)
        if not script.exists():
            messagebox.showerror("Not found", f"Script not found:\n{script}")
            return

        args = [self.pyexe, str(script)]
        # Add --auto if requested AND the script supports it (safe to pass only to those)
        if self.var_auto.get() and script_name in ("4_Prediction.py", "5_Edit_predicted.py"):
            args.append("--auto")

        # If user wants a non-default config, pass it when supported (Prediction/Edit commonly read config.yaml implicitly)
        # Here we just set cwd so "config.yaml" resolves; you can extend per-script if needed.
        cfg = self.var_config.get().strip()
        if cfg and cfg != "config.yaml":
            # Many of your scripts call load_config("config.yaml") without CLI parsing.
            # Best practice: run from a directory where that file name exists.
            # If you want CLI-based config selection, I can add argparse to each stage.
            self._append(f"[Note] custom config is set to '{cfg}', but scripts may ignore it unless they accept CLI.\n")

        self._append("\n" + "=" * 80 + "\n")
        self._append("[RUN] " + " ".join(args) + "\n\n")
        self.status.set(f"Running: {script_name}")

        self._proc = subprocess.Popen(
            args,
            cwd=str(self.workdir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        self._reader_thread = threading.Thread(target=self._read_output, daemon=True)
        self._reader_thread.start()

    def _read_output(self):
        try:
            assert self._proc is not None and self._proc.stdout is not None
            for line in self._proc.stdout:
                self._append(line)
        finally:
            rc = None
            try:
                rc = self._proc.poll() if self._proc is not None else None
            except Exception:
                pass
            self._proc = None
            if rc is None:
                self.status.set("Stopped.")
            elif rc == 0:
                self.status.set("Done (exit=0).")
            else:
                self.status.set(f"Done (exit={rc}).")

    def stop_proc(self):
        if self._proc is None or self._proc.poll() is not None:
            self.status.set("Idle.")
            return
        try:
            self._append("\n[STOP] terminating...\n")
            self._proc.terminate()
        except Exception as e:
            self._append(f"[STOP] error: {e}\n")


def main():
    app = Launcher()
    app.mainloop()

if __name__ == "__main__":
    main()
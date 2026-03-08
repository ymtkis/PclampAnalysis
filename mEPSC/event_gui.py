import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

from event_gui_base import BaseSweepViewer, WindowInfo


class EventAnnotator(BaseSweepViewer):
    """
    Manual interval annotator for ABF sweeps.

    Mouse:
      - Left click: set interval start / select interval edge for dragging
      - Right click: set interval end
      - Drag inside an interval: adjust start or end

    Keys:
      - Up/Down: change sweep
      - Left/Right: pan time window (t0)
      - Wheel: pan/zoom (handled by BaseSweepViewer)
      - Shift: delete selected (or last) interval
      - s: save to *_edited.json (if json_path provided)
      - q: quit
    """

    def __init__(
        self,
        fig,
        ax,
        signals,
        fs,
        sweep_lengths,
        window_sec,
        y_range,
        initial_events=None,
        lowpass_hz=None,
        ax_ref=None,
        signals_ref=None,
        json_path: str | Path | None = None,
    ):
        self.lowpass_hz = lowpass_hz
        self.json_path = str(json_path) if json_path is not None else None

        super().__init__(
            fig=fig,
            ax=ax,
            signals=signals,
            fs=fs,
            sweep_lengths=sweep_lengths,
            window_sec=window_sec,
            y_range=y_range,
            ax_ref=ax_ref,
            signals_ref=signals_ref,
        )

        self.events_per_sweep = (
            [list(ev) for ev in initial_events]
            if initial_events is not None
            else [[] for _ in range(self.n_sweeps)]
        )
        self.current_event_idx_per_sweep = [None for _ in range(self.n_sweeps)]

        self.current_start = None
        self.dragging = None  # None / "start" / "end"

        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        self.update_plot()

    @property
    def events(self):
        return self.events_per_sweep[self.current_sweep]

    @events.setter
    def events(self, value):
        self.events_per_sweep[self.current_sweep] = value

    @property
    def current_event_index(self):
        return self.current_event_idx_per_sweep[self.current_sweep]

    @current_event_index.setter
    def current_event_index(self, value):
        self.current_event_idx_per_sweep[self.current_sweep] = value

    def _main_title(self) -> str:
        if self.lowpass_hz is None:
            return "Main"
        return f"Main (50 Hz notch + {self.lowpass_hz} Hz LP)"

    def _ref_title(self) -> str:
        if self.lowpass_hz is None:
            return "Reference"
        return f"Reference (LP only, {self.lowpass_hz} Hz)"

    def _draw_overlays_main(self, ax_main, w: WindowInfo):
        for i, (s, e) in enumerate(self.events):
            if e < w.t0 or s > w.t1:
                continue
            is_sel = (i == self.current_event_index)
            ax_main.axvspan(
                s,
                e,
                alpha=0.4 if is_sel else 0.2,
                edgecolor="red" if is_sel else None,
            )

        if self.current_start is not None and w.t0 <= self.current_start <= w.t1:
            ax_main.axvline(self.current_start, color="orange", linestyle="--")

        title = (
            f"Sweep {self.current_sweep}/{self.n_sweeps - 1} : [{w.t0:.2f}, {w.t1:.2f}] s\n"
            "Mouse: L=start, R=end, drag adjust | "
            "Keys: ↑/↓ sweep, ←/→ pan, wheel pan/zoom, Shift delete, s save, q quit"
        )
        if self.current_event_index is not None:
            title += f" | Selected event #{self.current_event_index}"
        ax_main.set_title(title, fontsize=10)

    def _sort_events_and_keep_selection(self):
        if not self.events:
            return

        selected_event = None
        if self.current_event_index is not None and 0 <= self.current_event_index < len(self.events):
            selected_event = self.events[self.current_event_index]

        self.events = sorted(self.events, key=lambda ev: ev[0])

        if selected_event is not None:
            try:
                self.current_event_index = self.events.index(selected_event)
            except ValueError:
                self.current_event_index = None

    def set_end(self, x):
        if self.current_start is None:
            return

        s = min(self.current_start, x)
        e = max(self.current_start, x)
        self.events.append((s, e))

        self.current_start = None
        self.current_event_index = len(self.events) - 1
        self._sort_events_and_keep_selection()
        self.update_plot()

    def on_click(self, event):
        if (not self._in_trace_axes(event)) or (event.xdata is None):
            return

        x = float(event.xdata)
        total_sec = float(self.sweep_lengths[self.current_sweep])
        x = max(0.0, min(total_sec, x))

        if event.button == 1:
            hit = None
            for i, (s, e) in enumerate(self.events):
                if s <= x <= e:
                    hit = i
                    break

            if hit is not None:
                self.current_event_index = hit
                s, e = self.events[hit]
                self.dragging = "start" if abs(x - s) <= abs(x - e) else "end"
                self.update_plot()
                return

            self.current_start = x
            self.update_plot()

        elif event.button == 3:
            self.set_end(x)

    def on_motion(self, event):
        if self.dragging is None:
            return
        if (not self._in_trace_axes(event)) or (event.xdata is None):
            return

        x = float(event.xdata)
        total_sec = float(self.sweep_lengths[self.current_sweep])
        x = max(0.0, min(total_sec, x))

        if self.current_event_index is None:
            return

        s, e = self.events[self.current_event_index]
        if self.dragging == "start":
            s = min(x, e - 1e-5)
        else:
            e = max(x, s + 1e-5)

        self.events[self.current_event_index] = (s, e)
        self.update_plot()

    def on_release(self, event):
        if self.dragging is not None:
            self._sort_events_and_keep_selection()
            self.update_plot()
        self.dragging = None

    def on_key(self, event):
        key = event.key

        if key in ["up", "down"]:
            self.change_sweep(+1 if key == "up" else -1)

        elif key in ["left", "right"]:
            delta = self.window_sec * 0.25
            self.t0 += (-delta if key == "left" else +delta)
            if self.slider_t0 is not None:
                center = self.t0 + self.window_sec / 2.0
                self.slider_t0.set_val(center)
            self.update_plot()

        elif key == "shift":
            if not self.events:
                return

            idx = (len(self.events) - 1) if (self.current_event_index is None) else self.current_event_index
            removed = self.events.pop(idx)
            print(f"Removed event (sweep {self.current_sweep}, idx {idx}): {removed}")

            self.current_event_index = min(idx, len(self.events) - 1) if self.events else None
            self.update_plot()

        elif key == "s":
            self.save_events()

        elif key == "q":
            plt.close(self.fig)

    def change_sweep(self, delta):
        self.current_sweep = (self.current_sweep + delta) % self.n_sweeps
        self.dragging = None
        self.current_start = None
        self.update_plot()

    def save_events(self):
        if not self.json_path:
            print("[EventAnnotator] json_path is not set; skipping save.")
            return

        if self.json_path.lower().endswith(".json"):
            out_path = self.json_path.replace(".json", "_edited.json")
        else:
            out_path = self.json_path + "_edited.json"

        data = {"events": self.events}
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        print(f"[EventAnnotator] Saved to {out_path}")


class PredictionViewer(BaseSweepViewer):
    """
    Viewer/editor for CNN intervals and refined peak events.

    Keys:
      - Up/Down: change sweep
      - Left/Right: pan time window (t0)
      - i: toggle interval display
      - Enter: add peak at cursor
      - Shift: delete selected peak
      - s: save to *_edited.json and optionally recompute
      - q: quit

    Mouse:
      - Left click: select nearest peak
      - Drag: move selected peak
    """

    def __init__(
        self,
        fig,
        ax,
        signals,
        fs,
        sweep_lengths,
        intervals_by_sweep,
        peaks_by_sweep,
        window_sec,
        y_range,
        json_data,
        json_path,
        default_valid_start,
        default_valid_end,
        show_intervals=True,
        enable_peak_edit=True,
        enable_save=True,
        recompute_cmd_builder=None,
        baseline_before_sec=None,
        baseline_win_sec=None,
        local_sd_win_sec=None,
        highpass_hz=None,
        highpass_order=2,
        show_baseline=False,
        ax_ref=None,
        signals_ref=None,
        on_close_callback=None,
    ):
        self.intervals_by_sweep = intervals_by_sweep
        self.peaks_by_sweep = peaks_by_sweep
        self.json_data = json_data

        self.json_path = Path(json_path)
        self.original_json_path = self.json_path
        if self.json_path.name.endswith("_edited.json"):
            candidate = self.json_path.with_name(self.json_path.name.replace("_edited.json", ".json"))
            if candidate.exists():
                self.original_json_path = candidate

        self.cursor_x = None
        self.show_intervals = show_intervals
        self.enable_peak_edit = enable_peak_edit
        self.enable_save = enable_save
        self.recompute_cmd_builder = recompute_cmd_builder

        self.selected_peak_index = None
        self.dragging_peak = False

        self.baseline_before_sec = baseline_before_sec
        self.baseline_win_sec = baseline_win_sec
        self.local_sd_win_sec = local_sd_win_sec
        self.highpass_hz = highpass_hz
        self.highpass_order = highpass_order

        self.show_baseline = (
            show_baseline
            and baseline_before_sec is not None
            and baseline_win_sec is not None
            and highpass_hz is not None
        )

        self._hp_cache = {}

        t0_per_sweep = []
        for L in sweep_lengths:
            L = float(L)
            max_t0 = max(0.0, L - window_sec)
            t0 = min(max(default_valid_start, 0.0), max_t0)
            t0_per_sweep.append(t0)

        super().__init__(
            fig=fig,
            ax=ax,
            signals=signals,
            fs=fs,
            sweep_lengths=sweep_lengths,
            window_sec=window_sec,
            y_range=y_range,
            ax_ref=ax_ref,
            signals_ref=signals_ref,
            t0_per_sweep=t0_per_sweep,
        )

        self._undo_stack = []
        self._redo_stack = []
        self._max_history = 10
        self._is_restoring = False
        self.on_close_callback = on_close_callback

        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.fig.canvas.mpl_connect("close_event", self.on_close)

        # Debounce for key-repeat peak nudging
        self._nudge_burst_active = False
        self._nudge_burst_timer = None
        self._nudge_burst_clear_ms = 200

        self.update_plot()

    def _main_title(self) -> str:
        return "Main"

    def _ref_title(self) -> str:
        return "Reference"

    def _get_hp_signal(self, sweep_idx):
        """High-pass filtered signal (cached per sweep)."""
        if sweep_idx in self._hp_cache:
            return self._hp_cache[sweep_idx]

        sig = self.signals[sweep_idx]
        fs = self.fs
        hp_cut = self.highpass_hz
        hp_order = self.highpass_order

        if hp_cut is None:
            self._hp_cache[sweep_idx] = sig
            return sig

        nyq = fs * 0.5
        Wn = hp_cut / nyq
        b, a = butter(hp_order, Wn, btype="high", analog=False)
        sig_hp = filtfilt(b, a, sig).astype(np.float32)
        self._hp_cache[sweep_idx] = sig_hp
        return sig_hp

    def _draw_overlays_main(self, ax_main, w: WindowInfo):
        if self.show_intervals:
            intervals = self.intervals_by_sweep.get(self.current_sweep, [])
            for (s, e, p) in intervals:
                if e < w.t0 or s > w.t1:
                    continue
                ax_main.axvspan(s, e, color="orange", alpha=0.25)

                center_itv = 0.5 * (s + e)
                if w.t0 <= center_itv <= w.t1:
                    y_top = ax_main.get_ylim()[1]
                    y_bot = ax_main.get_ylim()[0]
                    ax_main.text(
                        center_itv,
                        y_top - (y_top - y_bot) * 0.10,
                        f"{p:.2f}",
                        ha="center",
                        va="top",
                        fontsize=8,
                        color="red",
                    )

        peaks = self.peaks_by_sweep.get(self.current_sweep, [])
        for i, d in enumerate(peaks):
            pk = float(d["peak_sec"])
            if pk < w.t0 or pk > w.t1:
                continue
            lw = 2.5 if (self.selected_peak_index == i) else 1.2
            ax_main.axvline(pk, lw=lw, alpha=0.9)

        y_top = ax_main.get_ylim()[1]
        y_bot = ax_main.get_ylim()[0]
        y_base = y_top - (y_top - y_bot) * 0.18

        stack_dx_sec = 0.03
        stack_dy = (y_top - y_bot) * 0.06
        max_levels = 6

        peaks_in_view = []
        for d in peaks:
            t = float(d["peak_sec"])
            if w.t0 <= t <= w.t1:
                peaks_in_view.append(d)
        peaks_in_view.sort(key=lambda dd: float(dd["peak_sec"]))

        placed = []  # (t, level)

        for d in peaks_in_view:
            peak_t = float(d["peak_sec"])
            snr = float(d.get("snr", np.nan))
            text = f"SNR={snr:.2f}" if np.isfinite(snr) else "SNR=NA"

            used_levels = set()
            for t_prev, lvl_prev in placed:
                if abs(peak_t - t_prev) < stack_dx_sec:
                    used_levels.add(lvl_prev)

            level = 0
            while level in used_levels and level < max_levels:
                level += 1
            if level >= max_levels:
                level = max_levels - 1

            placed.append((peak_t, level))
            y_txt = y_base - level * stack_dy

            ax_main.text(
                peak_t,
                y_txt,
                text,
                ha="center",
                va="top",
                fontsize=8,
                color="blue",
            )

        if self.show_baseline and peaks:
            sig_hp = self._get_hp_signal(self.current_sweep)
            fs = self.fs
            bl_before = self.baseline_before_sec
            bl_win = self.baseline_win_sec

            for d in peaks:
                peak_t = float(d["peak_sec"])
                peak_idx = int(peak_t * fs)

                bl_end = peak_idx - int(bl_before * fs)
                bl_start = bl_end - int(bl_win * fs)
                if bl_start < 0 or bl_end > len(sig_hp):
                    continue

                baseline_val = sig_hp[bl_start:bl_end].mean()

                t_bl0 = bl_start / fs
                t_bl1 = bl_end / fs
                if t_bl1 < w.t0 or t_bl0 > w.t1:
                    continue

                ax_main.axhline(baseline_val, color="cyan", linewidth=0.8, alpha=0.7)
                ax_main.axvspan(t_bl0, t_bl1, color="cyan", alpha=0.15)

        ax_main.set_title(
            f"Sweep {self.current_sweep} / {self.n_sweeps - 1}  "
            f"(t0={w.t0:.3f} s, window={self.window_sec:.3f} s)",
            fontsize=10,
        )

    def _begin_nudge_burst(self):
        # Push undo once per key-repeat burst
        if not self._nudge_burst_active:
            self._push_undo()
            self._nudge_burst_active = True

        # Clear burst flag after a short idle time
        if self._nudge_burst_timer is not None:
            try:
                self._nudge_burst_timer.stop()
            except Exception:
                pass

        self._nudge_burst_timer = self.fig.canvas.new_timer(interval=self._nudge_burst_clear_ms)
        self._nudge_burst_timer.single_shot = True
        self._nudge_burst_timer.add_callback(self._end_nudge_burst)
        self._nudge_burst_timer.start()


    def _end_nudge_burst(self):
        self._nudge_burst_active = False


    def _nudge_selected_peak(self, step_sec: float):
        if self.selected_peak_index is None:
            return

        peaks = self.peaks_by_sweep.get(self.current_sweep, [])
        if not (0 <= self.selected_peak_index < len(peaks)):
            return

        L = float(self.sweep_lengths[self.current_sweep])
        x = float(peaks[self.selected_peak_index]["peak_sec"]) + float(step_sec)
        x = max(0.0, min(L, x))

        peaks[self.selected_peak_index]["peak_sec"] = x
        peaks.sort(key=lambda d: d["peak_sec"])

        # Keep selection pointing to the moved peak (nearest index)
        self.selected_peak_index = int(np.argmin([abs(p["peak_sec"] - x) for p in peaks]))


    def on_key_press(self, event):
        if event.key in ["up", "down"]:
            self.current_sweep = (self.current_sweep + (1 if event.key == "up" else -1)) % self.n_sweeps

            L = float(self.sweep_lengths[self.current_sweep])
            max_t0 = max(0.0, L - self.window_sec)
            if self.t0 > max_t0:
                self.t0 = max_t0

            if self.slider_t0 is not None:
                center = self.t0 + self.window_sec / 2.0
                self.slider_t0.eventson = False
                self.slider_t0.set_val(center)
                self.slider_t0.eventson = True
            self.update_plot()

        elif event.key in ("ctrl+z", "control+z"):
            self.undo()
            return

        elif event.key in ("ctrl+y", "control+y", "ctrl+shift+z", "control+shift+z"):
            self.redo()
            return

        elif event.key in ["left", "right"]:
            if getattr(self, "enable_peak_edit", True) and (self.selected_peak_index is not None):
                base = 1.0 / float(self.fs)  # 1 sample    
                step = 5.0 * base
                step *= (-1.0 if "left" in event.key else +1.0)

                self._begin_nudge_burst()
                self._nudge_selected_peak(step)

                # Important: debounce heavy redraw (avoid freezing on key repeat)
                self._schedule_update()
                return

            # Otherwise pan the view window
            delta = self.window_sec * 0.25
            self.t0 += (-delta if "left" in event.key else +delta)

            L = float(self.sweep_lengths[self.current_sweep])
            max_t0 = max(0.0, L - self.window_sec)
            self.t0 = max(0.0, min(max_t0, self.t0))

            if self.slider_t0 is not None:
                center = self.t0 + self.window_sec / 2.0
                self.slider_t0.eventson = False
                self.slider_t0.set_val(center)
                self.slider_t0.eventson = True

            self._update_slider_window_patch()
            self._schedule_update()
            return

        elif event.key == "i":
            self.show_intervals = not self.show_intervals
            self.update_plot()

        elif event.key in ("ctrl+s", "control+s") and self.enable_save:
            self.save_edited_peaks()

        elif event.key == "q":
            plt.close(self.fig)

    def _hit_test_peak(self, x, tol_sec=0.003):
        peaks = self.peaks_by_sweep.get(self.current_sweep, [])
        if not peaks:
            return None
        best_i, best_dt = None, 1e9
        for i, d in enumerate(peaks):
            dt = abs(d["peak_sec"] - x)
            if dt < best_dt:
                best_dt = dt
                best_i = i
        return best_i if best_dt <= tol_sec else None

    def _snapshot_peaks(self):
        """Deep-copy peaks_by_sweep (only the peak dicts)."""
        snap = {}
        for sw, plist in self.peaks_by_sweep.items():
            snap[int(sw)] = [dict(p) for p in plist]
        return snap

    def _restore_peaks(self, snap):
        """Restore peaks_by_sweep from snapshot."""
        self._is_restoring = True
        try:
            self.peaks_by_sweep = {}
            for sw, plist in snap.items():
                self.peaks_by_sweep[int(sw)] = [dict(p) for p in plist]
                self.peaks_by_sweep[int(sw)].sort(key=lambda d: d["peak_sec"])
            self.selected_peak_index = None
            self.dragging_peak = False
        finally:
            self._is_restoring = False

    def _push_undo(self):
        """Push current state to undo stack and clear redo stack."""
        if self._is_restoring:
            return
        self._undo_stack.append(self._snapshot_peaks())
        if len(self._undo_stack) > self._max_history:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    def undo(self):
        if not self._undo_stack:
            return
        self._redo_stack.append(self._snapshot_peaks())
        snap = self._undo_stack.pop()
        self._restore_peaks(snap)
        self.update_plot()

    def redo(self):
        if not self._redo_stack:
            return
        self._undo_stack.append(self._snapshot_peaks())
        snap = self._redo_stack.pop()
        self._restore_peaks(snap)
        self.update_plot()

    def on_click(self, event):
        self.cursor_x = float(event.xdata) if event.xdata is not None else None

        if (not self._in_trace_axes(event)) or (event.xdata is None):
            return
        if event.button != 1 or (not self.enable_peak_edit):
            return

        x = float(event.xdata)

        # Double click: add/delete
        if getattr(event, "dblclick", False):
            hit = self._hit_test_peak(x)
            peaks = self.peaks_by_sweep.get(self.current_sweep, [])

            if hit is not None:
                self._push_undo()
                peaks.pop(hit)
                self.selected_peak_index = None
                self.dragging_peak = False
                self.update_plot()
                return

            # Add a new peak at x
            self._push_undo()
            peaks = self.peaks_by_sweep.setdefault(self.current_sweep, [])
            peaks.append({"peak_sec": float(x), "amp_pA": np.nan, "snr": np.nan, "parent_max_proba": np.nan})
            peaks.sort(key=lambda d: d["peak_sec"])
            self.selected_peak_index = self._hit_test_peak(x, tol_sec=1e-6)
            self.dragging_peak = False
            self.update_plot()
            return

        # Single click: select and drag
        hit = self._hit_test_peak(x)
        self._drag_start_snapshot = None
        if hit is not None:
            self._drag_start_snapshot = self._snapshot_peaks()
        self.selected_peak_index = hit
        self.dragging_peak = (hit is not None)
        self.update_plot()


    def on_motion(self, event):
        if not self.dragging_peak or not self.enable_peak_edit:
            return
        if (not self._in_trace_axes(event)) or (event.xdata is None):
            return
        if self.selected_peak_index is None:
            return

        x = float(event.xdata)
        x = max(self.t0, min(self.t0 + self.window_sec, x))

        peaks = self.peaks_by_sweep.get(self.current_sweep, [])
        if 0 <= self.selected_peak_index < len(peaks):
            peaks[self.selected_peak_index]["peak_sec"] = float(x)
            peaks.sort(key=lambda d: d["peak_sec"])
            self.selected_peak_index = self._hit_test_peak(x, tol_sec=1e-6)
            self.update_plot()

    def on_release(self, event):
        if self.dragging_peak:
            # If a drag occurred, push the pre-drag state once
            if getattr(self, "_drag_start_snapshot", None) is not None:
                # Compare current vs start snapshot; if different, push undo
                if self._snapshot_peaks() != self._drag_start_snapshot:
                    # Push the state BEFORE the drag
                    if not self._is_restoring:
                        self._undo_stack.append(self._drag_start_snapshot)
                        if len(self._undo_stack) > self._max_history:
                            self._undo_stack.pop(0)
                        self._redo_stack.clear()
            self._drag_start_snapshot = None

        self.dragging_peak = False

    def save_edited_peaks(self):
        all_refined = []
        for sw, plist in self.peaks_by_sweep.items():
            for d in plist:
                all_refined.append({"sweep": int(sw), **d})
        all_refined.sort(key=lambda dd: (dd["sweep"], dd["peak_sec"]))
        self.json_data["refined_events"] = all_refined

        out_path = self.json_path if self.json_path.name.endswith("_edited.json") else \
            self.json_path.with_name(self.json_path.stem + "_edited.json")

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(self.json_data, f, indent=2)

        print(f"Saved edited peaks to: {out_path}")

        if self.recompute_cmd_builder is None:
            return

        try:
            import subprocess

            cmd = self.recompute_cmd_builder(self.original_json_path, out_path)
            subprocess.run(cmd, check=True)

            with open(out_path, "r", encoding="utf-8") as f:
                self.json_data = json.load(f)

            new_peaks = {}
            for ev in self.json_data.get("refined_events", []):
                sw = int(ev["sweep"])
                new_peaks.setdefault(sw, []).append(
                    {
                        "peak_sec": float(ev["peak_sec"]),
                        "amp_pA": float(ev.get("amp_pA", np.nan)),
                        "snr": float(ev.get("snr", np.nan)),
                        "parent_max_proba": float(ev.get("parent_max_proba", np.nan)),
                    }
                )
            for sw in new_peaks:
                new_peaks[sw].sort(key=lambda d: d["peak_sec"])

            self.peaks_by_sweep = new_peaks
            self.selected_peak_index = None
            self.update_plot()

        except Exception as e:
            print(f"[WARN] recompute failed: {e}")

    def on_close(self, event):
        # Auto-save on window close
        if self.enable_save:
            try:
                self.save_edited_peaks()
            except Exception as e:
                print(f"[WARN] auto-save on close failed: {e}")

        # Go back to file/folder picker
        #if callable(self.on_close_callback):
        #    try:
        #        self.on_close_callback()
        #    except Exception as e:
        #        print(f"[WARN] on_close callback failed: {e}")

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button


@dataclass
class WindowInfo:
    t0: float
    t1: float
    idx0: int
    idx1: int
    t_seg: np.ndarray


class BaseSweepViewer:
    """
    Base class for sweep-based GUI viewers/editors.

    Design goals:
      - Scroll updates state immediately
      - Redraw is debounced (TkAgg safe)
      - No recursive update_plot
      - Avoid timer callbacks after close
    """

    def __init__(
        self,
        fig,
        ax,
        signals,
        fs: float,
        sweep_lengths,
        window_sec: float,
        y_range: float,
        ax_ref=None,
        signals_ref=None,
        t0_per_sweep: Optional[List[float]] = None,
        sharex_ref: bool = True,
    ):
        self.fig = fig
        self.ax = ax
        self.ax_ref = ax_ref

        self.signals = signals
        self.signals_ref = signals_ref
        self.axes = [self.ax] + ([self.ax_ref] if self.ax_ref is not None else [])

        self.fs = float(fs)
        self.sweep_lengths = sweep_lengths
        self.window_sec = float(window_sec)
        self.y_range = float(y_range)
        self.n_sweeps = len(signals)

        if t0_per_sweep is None:
            self.t0_per_sweep = [0.0 for _ in range(self.n_sweeps)]
        else:
            self.t0_per_sweep = list(map(float, t0_per_sweep))

        self.current_sweep = 0

        self.slider_t0: Optional[Slider] = None
        self.slider_window_patch = None

        self._in_update = False
        self._is_closed = False

        # Debounce timers
        self._scroll_timer = None
        self._scroll_delay_ms = 40  # ~25 FPS for scroll

        self._update_timer = None
        self._update_delay_ms = 30  # ~33 FPS for key-repeat / slider drag

        self._init_slider()
        self._setup_zoom_buttons(fig)

        self.fig.canvas.mpl_connect("scroll_event", self.on_scroll)
        self.fig.canvas.mpl_connect("close_event", self._on_close)

        if self.ax_ref is not None and sharex_ref:
            self.ax_ref.set_xlabel("")

    # ------------------------
    # lifecycle / guards
    # ------------------------
    def _on_close(self, event):
        self._is_closed = True
        for t in (self._scroll_timer, self._update_timer):
            if t is not None:
                try:
                    t.stop()
                except Exception:
                    pass
        self._scroll_timer = None
        self._update_timer = None

    @property
    def t0(self) -> float:
        return self.t0_per_sweep[self.current_sweep]

    @t0.setter
    def t0(self, value: float):
        L = float(self.sweep_lengths[self.current_sweep])
        max_t0 = max(L - self.window_sec, 0.0)
        self.t0_per_sweep[self.current_sweep] = max(0.0, min(float(value), max_t0))

    # ------------------------
    # helpers
    # ------------------------
    def _in_trace_axes(self, event) -> bool:
        return event.inaxes in self.axes

    def _get_window(self) -> WindowInfo:
        L = float(self.sweep_lengths[self.current_sweep])

        t0 = min(self.t0, max(L - self.window_sec, 0.0))
        t0 = max(t0, 0.0)
        t1 = min(t0 + self.window_sec, L)

        idx0 = int(t0 * self.fs)
        idx1 = max(idx0, int(t1 * self.fs))

        t_seg = (idx0 + np.arange(idx1 - idx0)) / self.fs
        return WindowInfo(t0, t1, idx0, idx1, t_seg)

    # ------------------------
    # debounced redraw
    # ------------------------
    def _schedule_update(self, delay_ms: Optional[int] = None):
        if self._is_closed:
            return

        if delay_ms is None:
            delay_ms = self._update_delay_ms

        if self._update_timer is not None:
            try:
                self._update_timer.stop()
            except Exception:
                pass

        self._update_timer = self.fig.canvas.new_timer(
            interval=int(delay_ms),
            callbacks=[(self._flush_update, [], {})],
        )
        self._update_timer.start()

    def _flush_update(self):
        if self._is_closed:
            return
        self.update_plot()

    # ------------------------
    # slider
    # ------------------------
    def _init_slider(self):
        ax_t0 = self.fig.add_axes([0.15, 0.02, 0.7, 0.03], facecolor="lightgoldenrodyellow")

        L0 = float(self.sweep_lengths[self.current_sweep])
        center0 = self.t0 + self.window_sec / 2.0

        self.slider_t0 = Slider(
            ax=ax_t0,
            label="t0 [s]",
            valmin=0.0,
            valmax=L0,
            valinit=center0,
            valstep=0.01,
        )

        self.slider_t0.poly.set_visible(False)
        if hasattr(self.slider_t0, "vline") and self.slider_t0.vline is not None:
            self.slider_t0.vline.set_visible(False)

        self.slider_t0.on_changed(self.on_slider_t0_changed)
        self._update_slider_window_patch()

    def _update_slider_window_patch(self):
        if self.slider_t0 is None:
            return

        ax = self.slider_t0.ax
        L = float(self.sweep_lengths[self.current_sweep])
        t0 = float(self.t0)
        t1 = min(t0 + self.window_sec, L)

        ax.set_xlim(0.0, L)

        if self.slider_window_patch is not None:
            try:
                self.slider_window_patch.remove()
            except Exception:
                pass

        self.slider_window_patch = ax.axvspan(t0, t1, facecolor="lightblue", alpha=0.4)

    def on_slider_t0_changed(self, val):
        if self._is_closed:
            return

        L = float(self.sweep_lengths[self.current_sweep])
        center = float(val)
        half = self.window_sec / 2.0

        if self.window_sec >= L:
            self.t0 = 0.0
        else:
            self.t0 = max(0.0, min(center - half, L - self.window_sec))

        self._update_slider_window_patch()
        self._schedule_update()

    # ------------------------
    # zoom buttons
    # ------------------------
    def _setup_zoom_buttons(self, fig):
        pos = self.ax.get_position()

        ax_tminus = fig.add_axes([pos.x0 + 0.04, pos.y0 - 0.06, 0.04, 0.035])
        ax_tplus  = fig.add_axes([pos.x0 + 0.09, pos.y0 - 0.06, 0.04, 0.035])

        Button(ax_tminus, "-").on_clicked(lambda _e: self._zoom_time(0.8))
        Button(ax_tplus,  "+").on_clicked(lambda _e: self._zoom_time(1.25))

        ax_yminus = fig.add_axes([pos.x0 - 0.08, pos.y0 + 0.01, 0.06, 0.035])
        ax_yplus  = fig.add_axes([pos.x0 - 0.08, pos.y0 + 0.055, 0.06, 0.035])

        Button(ax_yminus, "-").on_clicked(lambda _e: self._zoom_y(0.8))
        Button(ax_yplus,  "+").on_clicked(lambda _e: self._zoom_y(1.25))

    def _zoom_time(self, factor: float):
        if self._is_closed:
            return

        L = float(self.sweep_lengths[self.current_sweep])
        self.window_sec = float(np.clip(self.window_sec * factor, 0.005, max(L, 0.005)))
        self.t0 = min(self.t0, max(L - self.window_sec, 0.0))

        if self.slider_t0 is not None:
            center = self.t0 + self.window_sec / 2.0
            self.slider_t0.eventson = False
            self.slider_t0.set_val(center)
            self.slider_t0.eventson = True

        self._update_slider_window_patch()
        self._schedule_update()

    def _zoom_y(self, factor: float):
        if self._is_closed:
            return
        self.y_range = float(np.clip(self.y_range * factor, 1.0, 500.0))
        self._schedule_update()

    # ------------------------
    # scroll (xscale/yscale/trace)
    # ------------------------
    def on_scroll(self, event):
        if self._is_closed:
            return

        region = None

        if event.inaxes == self.ax:
            region = "trace"
        elif event.x is not None and event.y is not None:
            bbox = self.ax.bbox
            if bbox.x0 <= event.x <= bbox.x1 and event.y < bbox.y0:
                region = "xscale"
            elif bbox.y0 <= event.y <= bbox.y1 and event.x < bbox.x0:
                region = "yscale"

        if region is None:
            return

        if region == "xscale":
            factor = 0.8 if event.step > 0 else 1.25
            self._zoom_time(factor)
            return

        if region == "yscale":
            factor = 0.8 if event.step > 0 else 1.25
            self._zoom_y(factor)
            return

        # region == "trace"
        delta = self.window_sec * 0.25
        self.t0 += (-delta if event.step > 0 else +delta)

        if self.slider_t0 is not None:
            center = self.t0 + self.window_sec / 2.0
            self.slider_t0.eventson = False
            self.slider_t0.set_val(center)
            self.slider_t0.eventson = True

        self._update_slider_window_patch()

        if self._scroll_timer is not None:
            try:
                self._scroll_timer.stop()
            except Exception:
                pass

        self._scroll_timer = self.fig.canvas.new_timer(
            interval=self._scroll_delay_ms,
            callbacks=[(self._flush_scroll_update, [], {})],
        )
        self._scroll_timer.start()

    def _flush_scroll_update(self):
        if self._is_closed:
            return
        self.update_plot()

    # ------------------------
    # plot skeleton
    # ------------------------
    def _plot_axis_trace(self, ax, sig: np.ndarray, w: WindowInfo, title: str = ""):
        ax.clear()
        seg = sig[w.idx0:w.idx1]
        if len(seg):
            ax.plot(w.t_seg, seg, lw=0.8, color="black")
            y0 = float(np.median(seg))
            ax.set_ylim(y0 - self.y_range, y0 + self.y_range)

        ax.set_xlim(w.t0, w.t1)
        if title:
            ax.set_title(title, fontsize=10)

    def _main_title(self) -> str:
        return ""

    def _ref_title(self) -> str:
        return ""

    def _draw_overlays_main(self, ax_main, w: WindowInfo):
        raise NotImplementedError

    def _draw_overlays_ref(self, ax_ref, w: WindowInfo):
        return

    # ------------------------
    # update plot (reentry safe)
    # ------------------------
    def update_plot(self):
        if self._is_closed or self._in_update:
            return

        self._in_update = True
        try:
            w = self._get_window()

            self._plot_axis_trace(
                self.ax,
                self.signals[self.current_sweep],
                w,
                title=self._main_title(),
            )

            if self.ax_ref is not None and self.signals_ref is not None:
                self._plot_axis_trace(
                    self.ax_ref,
                    self.signals_ref[self.current_sweep],
                    w,
                    title=self._ref_title(),
                )
                self.ax_ref.set_xlabel("")

            self._draw_overlays_main(self.ax, w)
            if self.ax_ref is not None and self.signals_ref is not None:
                self._draw_overlays_ref(self.ax_ref, w)

            self.ax.set_xlabel("Time (s)")
            self._update_slider_window_patch()
            self.fig.canvas.draw_idle()

        finally:
            self._in_update = False

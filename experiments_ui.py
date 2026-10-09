"""掃頻 / 批次參數掃描 / 分岔圖 的介面（MainWindow 的 mixin）。"""
import math
import os
import re

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets

import config
from analysis import fmt_eng, period_label
from experiments import (STOP_ACTIONS, BatchWorker, BifSettings, BifurcationWorker, DIR_NAMES, SweepSettings,
                         SweepWorker, write_bifurcation_csv, write_sweep_csv)
from instruments import AFG_FREQ_RANGE, check_drive

DIR_COLORS = {"up": "#4FC3F7", "down": "#FF8A65"}
PERIOD_COLORS = {1: "#4FC3F7", 2: "#81C784", 4: "#FFB74D", 8: "#BA68C8", 0: "#E57373"}
SCOPE_AVG_CHOICES = [1, 2, 4, 8, 16, 32, 64, 128, 256]


def _row(*widgets):
    w = QtWidgets.QWidget()
    lay = QtWidgets.QHBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    for x in widgets:
        if x == "stretch":
            lay.addStretch(1)
        else:
            lay.addWidget(x)
    return w


def scrollable(w):
    sa = QtWidgets.QScrollArea()
    sa.setWidgetResizable(True)
    sa.setFrameShape(QtWidgets.QFrame.NoFrame)
    sa.setWidget(w)
    return sa


def _freq_spin(v):
    s = QtWidgets.QDoubleSpinBox()
    s.setDecimals(1)
    s.setRange(*AFG_FREQ_RANGE)
    s.setGroupSeparatorShown(True)
    s.setSuffix(" Hz")
    s.setValue(v)
    return s


def _dspin(v, lo, hi, dec=2, suffix="", step=None):
    s = QtWidgets.QDoubleSpinBox()
    s.setDecimals(dec)
    s.setRange(lo, hi)
    s.setValue(v)
    if suffix:
        s.setSuffix(suffix)
    if step:
        s.setSingleStep(step)
    return s


def _ispin(v, lo, hi, suffix=""):
    s = QtWidgets.QSpinBox()
    s.setRange(lo, hi)
    s.setValue(v)
    if suffix:
        s.setSuffix(suffix)
    return s


def _gray(text):
    lb = QtWidgets.QLabel(text)
    lb.setWordWrap(True)
    lb.setStyleSheet("color: gray;")
    return lb


def _fmt_f(x):
    return fmt_eng(x, "Hz", 6)


def parse_values(text):
    vals = [float(x) for x in re.split(r"[,\s，、;；]+", text.strip()) if x]
    if not vals:
        raise ValueError("請輸入至少一個數值")
    return vals


class ExperimentsMixin:
    """需要 MainWindow 提供：scope, afg, start_live(), stop_live(), sync_scope_settings(), _meta_lines(),
    _base_name(), _ensure_dir(), plot_tabs, lbl_saved, statusBar(), afg_high_z()。"""

    # ==================================================================
    # 繪圖分頁
    # ==================================================================
    def _build_bode_tab(self):
        tab = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(tab)
        self.bode_view = pg.GraphicsLayoutWidget()
        self.gain_plot = self.bode_view.addPlot(row=0, col=0)
        self.phase_plot = self.bode_view.addPlot(row=1, col=0)
        for p in (self.gain_plot, self.phase_plot):
            p.showGrid(x=True, y=True, alpha=0.3)
            p.setLabel("bottom", "頻率", units="Hz")
        self.gain_plot.addLegend(offset=(-10, 10))
        self.gain_plot.setLabel("left", "增益 CH2/CH1", units="dB")
        self.phase_plot.setLabel("left", "相位差 φ2−φ1", units="°")
        self.phase_plot.setXLink(self.gain_plot)
        self.phase_plot.addLine(y=0, pen=pg.mkPen("#666", style=QtCore.Qt.DashLine))
        self.bode_curves = {}
        for d, color in DIR_COLORS.items():
            kw = dict(pen=pg.mkPen(color, width=2), symbol="o", symbolSize=5, symbolBrush=color, symbolPen=None)
            self.bode_curves[d] = (self.gain_plot.plot([], [], name=DIR_NAMES[d], **kw),
                                   self.phase_plot.plot([], [], **kw))
        bad = dict(pen=None, symbol="x", symbolSize=10, symbolPen=pg.mkPen("#FF1744", width=2))
        self.bode_bad = (self.gain_plot.plot([], [], name="訊號太小", **bad), self.phase_plot.plot([], [], **bad))
        self.bode_lines = []
        lay.addWidget(self.bode_view, 1)
        self.lbl_bode = QtWidgets.QLabel("")
        self.lbl_bode.setTextFormat(QtCore.Qt.RichText)
        self.lbl_bode.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.lbl_bode.setStyleSheet("font-size: 10.5pt; padding: 4px;")
        lay.addWidget(self.lbl_bode)
        self.bode_tab = tab
        self.sweep_points, self.sweep_analyses, self.sweep_settings, self.sweep_warning = [], {}, None, None
        self._bode_log = True
        self._apply_bode_axis()
        return tab

    def _build_param_plot_tab(self):
        tab = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(tab)
        view = pg.GraphicsLayoutWidget()
        self.fres_plot = view.addPlot(row=0, col=0)
        self.ceff_plot = view.addPlot(row=1, col=0)
        self.fres_plot.addLegend(offset=(-10, 10))
        for p in (self.fres_plot, self.ceff_plot):
            p.showGrid(x=True, y=True, alpha=0.3)
        self.fres_plot.setLabel("left", "共振頻率", units="Hz")
        self.ceff_plot.setLabel("left", "C_eff", units="F")
        self.ceff_plot.setXLink(self.fres_plot)
        self.param_curves = {}
        for d, color in DIR_COLORS.items():
            self.param_curves[(d, "peak")] = self.fres_plot.plot(
                [], [], pen=pg.mkPen(color, width=2), symbol="o", symbolBrush=color, symbolPen=None,
                name=f"{DIR_NAMES[d]}：增益峰")
            self.param_curves[(d, "phase")] = self.fres_plot.plot(
                [], [], pen=pg.mkPen(color, width=1, style=QtCore.Qt.DashLine), symbol="s", symbolBrush=None,
                symbolPen=pg.mkPen(color), name=f"{DIR_NAMES[d]}：相位過零")
            self.param_curves[(d, "C")] = self.ceff_plot.plot(
                [], [], pen=pg.mkPen(color, width=2), symbol="o", symbolBrush=color, symbolPen=None)
        lay.addWidget(view, 1)
        self.lbl_param = QtWidgets.QLabel("")
        self.lbl_param.setTextFormat(QtCore.Qt.RichText)
        self.lbl_param.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.lbl_param.setStyleSheet("font-size: 10pt; padding: 4px;")
        lay.addWidget(scrollable(self.lbl_param), 0)
        self.param_view, self.param_tab = view, tab
        self.batch_rows = []
        return tab

    def _build_bif_plot_tab(self):
        tab = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(tab)
        view = pg.GraphicsLayoutWidget()
        self.bif_plot = view.addPlot(row=0, col=0)
        self.bif_plot.showGrid(x=True, y=True, alpha=0.3)
        self.bif_plot.setLabel("bottom", "驅動振幅", units="Vpp")
        self.bif_plot.setLabel("left", "CH2 每週期取樣值", units="V")
        self.bif_scatter = pg.ScatterPlotItem(size=3, pen=None)
        self.bif_plot.addItem(self.bif_scatter)
        self.bif_seq_plot = view.addPlot(row=1, col=0)
        self.bif_seq_plot.showGrid(x=True, y=True, alpha=0.3)
        self.bif_seq_plot.setLabel("bottom", "週期編號")
        self.bif_seq_plot.setLabel("left", "最新振幅的取樣值", units="V")
        self.bif_seq = self.bif_seq_plot.plot([], [], pen=pg.mkPen("#888"), symbol="o", symbolSize=4,
                                              symbolBrush="#4FC3F7", symbolPen=None)
        view.ci.layout.setRowStretchFactor(0, 3)
        lay.addWidget(view, 1)
        legend = "　".join(f"<span style='color:{c}'>●</span> {period_label(p)}" for p, c in PERIOD_COLORS.items())
        self.lbl_bif = QtWidgets.QLabel(legend)
        self.lbl_bif.setTextFormat(QtCore.Qt.RichText)
        self.lbl_bif.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.lbl_bif.setStyleSheet("font-size: 10pt; padding: 4px;")
        lay.addWidget(self.lbl_bif)
        self._bif_legend = legend
        self.bif_view, self.bif_tab = view, tab
        self.bif_results = []
        return tab

    # ==================================================================
    # 控制分頁
    # ==================================================================
    def _build_sweep_tab(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        v.addWidget(_gray("接線：產生器輸出 → 電路，同時接 CH1；CH2 量輸出（或感測電阻）。示波器觸發來源請設為 CH1。\n"
                          "產生器的振幅與偏移沿用「函數產生器」分頁目前的設定。"))

        g = QtWidgets.QGroupBox("頻率範圍")
        f = QtWidgets.QFormLayout(g)
        self.spn_sw_start = _freq_spin(1.0e6)
        self.spn_sw_stop = _freq_spin(2.5e6)
        self.cmb_sw_spacing = QtWidgets.QComboBox()
        self.cmb_sw_spacing.addItem("對數間隔", "log")
        self.cmb_sw_spacing.addItem("線性間隔", "lin")
        self.spn_sw_points = _ispin(31, 2, 2000)
        f.addRow("起點", self.spn_sw_start)
        f.addRow("終點", self.spn_sw_stop)
        f.addRow("間隔", self.cmb_sw_spacing)
        f.addRow("點數", self.spn_sw_points)
        v.addWidget(g)

        g = QtWidgets.QGroupBox("自動兩階段（粗掃 → 在增益峰附近細掃）")
        g.setCheckable(True)
        g.setChecked(True)
        self.grp_two_stage = g
        f = QtWidgets.QFormLayout(g)
        self.spn_fine_pct = _dspin(5.0, 0.1, 50, 2, " %", 0.5)
        self.spn_fine_n = _ispin(41, 3, 2000)
        f.addRow("細掃範圍 峰 × (1 ±", self.spn_fine_pct)
        f.addRow("細掃點數（線性）", self.spn_fine_n)
        v.addWidget(g)

        g = QtWidgets.QGroupBox("方向")
        f = QtWidgets.QFormLayout(g)
        self.cmb_sw_dir = QtWidgets.QComboBox()
        for key, label in (("up", "往上"), ("down", "往下"), ("updown", "往上再往下（檢查遲滯）")):
            self.cmb_sw_dir.addItem(label, key)
        f.addRow("掃描方向", self.cmb_sw_dir)
        f.addRow(_gray("往上再往下時，兩條曲線用不同顏色畫在同一張圖；中途不關輸出、不重設。"))
        v.addWidget(g)

        g = QtWidgets.QGroupBox("量測")
        f = QtWidgets.QFormLayout(g)
        self.spn_sw_periods = _dspin(4, 1, 50, 1)
        self.spn_sw_settle = _dspin(0.5, 0, 30, 2, " s", 0.1)
        self.chk_sw_autov = QtWidgets.QCheckBox("自動調整垂直刻度")
        self.chk_sw_autov.setChecked(True)
        self.spn_sw_avg = _ispin(1, 1, 64, " 次")
        self.spn_sw_avg.setToolTip("每個頻率點讀幾次波形，逐點平均後再做鎖相分析。")
        self.cmb_scope_avg = QtWidgets.QComboBox()
        for n in SCOPE_AVG_CHOICES:
            self.cmb_scope_avg.addItem("關閉" if n == 1 else f"{n} 次", n)
        self.chk_sw_ac = QtWidgets.QCheckBox("開始前設 CH1/CH2 為 AC 耦合、關閉頻寬限制")
        f.addRow("畫面顯示週期數", self.spn_sw_periods)
        f.addRow("每點等待", self.spn_sw_settle)
        f.addRow("程式平均", self.spn_sw_avg)
        f.addRow("示波器平均", self.cmb_scope_avg)
        f.addRow(self.chk_sw_autov)
        f.addRow(self.chk_sw_ac)
        f.addRow(_gray(f"CH1 或 CH2 峰對峰值小於 {config.SMALL_SIGNAL_DIVS:g} 格的點會標為「訊號太小，不可靠」（紅色 ×）。"))
        v.addWidget(g)

        g = QtWidgets.QGroupBox("分析與安全")
        f = QtWidgets.QFormLayout(g)
        self.spn_L = _dspin(config.DEFAULT_L_UH, 0, 1e6, 3, " µH")
        f.addRow("電感 L（算 C_eff）", self.spn_L)
        self.cmb_stop_action = QtWidgets.QComboBox()
        for key, label in STOP_ACTIONS.items():
            self.cmb_stop_action.addItem(label, key)
        f.addRow("中途停止時產生器", self.cmb_stop_action)
        self.cmb_bode_x = QtWidgets.QComboBox()
        self.cmb_bode_x.addItems(["對數頻率軸", "線性頻率軸"])
        self.cmb_bode_x.currentIndexChanged.connect(self._apply_bode_axis)
        f.addRow("圖的頻率軸", self.cmb_bode_x)
        v.addWidget(g)

        self.btn_sw_start = QtWidgets.QPushButton("▶ 開始掃頻")
        self.btn_sw_stop = QtWidgets.QPushButton("■ 停止")
        self.btn_sw_start.clicked.connect(self.start_sweep)
        self.btn_sw_stop.clicked.connect(self.stop_experiment)
        v.addWidget(_row(self.btn_sw_start, self.btn_sw_stop))
        self.prg_sweep = QtWidgets.QProgressBar()
        v.addWidget(self.prg_sweep)
        self.lbl_sweep = QtWidgets.QLabel("")
        self.lbl_sweep.setWordWrap(True)
        v.addWidget(self.lbl_sweep)
        self.btn_sw_save = QtWidgets.QPushButton("儲存掃頻結果（CSV + PNG）")
        self.btn_sw_clear = QtWidgets.QPushButton("清除")
        self.btn_sw_save.clicked.connect(self.save_sweep)
        self.btn_sw_clear.clicked.connect(self.clear_sweep)
        v.addWidget(_row(self.btn_sw_save, self.btn_sw_clear))
        v.addStretch(1)
        return scrollable(w)

    def _build_batch_tab(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        v.addWidget(_gray("對每個值各做一次掃頻（掃頻設定沿用「掃頻」分頁，建議開啟兩階段），"
                          "每個值存一個 CSV，最後存總表 summary.csv 與 f_res 圖。"))
        g = QtWidgets.QGroupBox("外層參數")
        f = QtWidgets.QFormLayout(g)
        self.cmb_batch_kind = QtWidgets.QComboBox()
        self.cmb_batch_kind.addItem("振幅 (Vpp)", "amp")
        self.cmb_batch_kind.addItem("DC offset (V)（C–V 量測）", "offset")
        self.cmb_batch_kind.currentIndexChanged.connect(self._batch_kind_changed)
        self.txt_batch_values = QtWidgets.QLineEdit("0.05, 0.1, 0.2, 0.4, 0.6, 1, 1.5, 2, 3, 4")
        self.lbl_batch_fixed = _gray("")
        f.addRow("參數", self.cmb_batch_kind)
        f.addRow("數值（逗號分隔）", self.txt_batch_values)
        f.addRow(self.lbl_batch_fixed)
        v.addWidget(g)

        g = QtWidgets.QGroupBox("沿用上一個 f_res 決定粗掃範圍")
        g.setCheckable(True)
        g.setChecked(True)
        self.grp_follow = g
        f = QtWidgets.QFormLayout(g)
        self.spn_follow_lo = _dspin(0.6, 0.05, 1.0, 2, "", 0.05)
        self.spn_follow_hi = _dspin(1.1, 1.0, 5.0, 2, "", 0.05)
        f.addRow("粗掃起點 = 上一個 f_res ×", self.spn_follow_lo)
        f.addRow("粗掃終點 = 上一個 f_res ×", self.spn_follow_hi)
        f.addRow(_gray("第一個值用「掃頻」分頁的起點、終點。"))
        v.addWidget(g)

        self.btn_batch_start = QtWidgets.QPushButton("▶ 開始批次掃描")
        self.btn_batch_stop = QtWidgets.QPushButton("■ 停止")
        self.btn_batch_start.clicked.connect(self.start_batch)
        self.btn_batch_stop.clicked.connect(self.stop_experiment)
        v.addWidget(_row(self.btn_batch_start, self.btn_batch_stop))
        v.addWidget(_gray("中途停止時產生器的處理，沿用「掃頻」分頁的設定。"))
        self.prg_batch = QtWidgets.QProgressBar()
        v.addWidget(self.prg_batch)
        self.lbl_batch = QtWidgets.QLabel("")
        self.lbl_batch.setWordWrap(True)
        v.addWidget(self.lbl_batch)
        self.btn_batch_open = QtWidgets.QPushButton("開啟結果資料夾")
        self.btn_batch_open.clicked.connect(self._open_batch_dir)
        self.btn_batch_open.setEnabled(False)
        v.addWidget(self.btn_batch_open)
        v.addStretch(1)
        self._batch_outdir = None
        self._batch_kind_changed()
        return scrollable(w)

    def _build_bif_tab(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        v.addWidget(_gray("固定頻率、逐步加大振幅；每個振幅抓一段長紀錄，每個驅動週期取一個 CH2 的值，"
                          "畫成分岔圖並自動判斷週期 1 / 2 / 4 / 8 / 混沌。"))
        g = QtWidgets.QGroupBox("驅動")
        f = QtWidgets.QFormLayout(g)
        self.spn_bif_f = _freq_spin(1.0e6)
        self.spn_bif_a0 = _dspin(0.1, 0.001, 20, 3, " Vpp", 0.1)
        self.spn_bif_a1 = _dspin(min(4.0, config.AMPLITUDE_MAX_VPP), 0.001, 20, 3, " Vpp", 0.1)
        self.spn_bif_steps = _ispin(40, 1, 2000)
        self.spn_bif_settle = _dspin(1.0, 0, 60, 2, " s", 0.1)
        f.addRow("固定頻率", self.spn_bif_f)
        f.addRow("振幅起點", self.spn_bif_a0)
        f.addRow("振幅終點", self.spn_bif_a1)
        f.addRow("步數", self.spn_bif_steps)
        f.addRow("每個振幅等待", self.spn_bif_settle)
        v.addWidget(g)

        g = QtWidgets.QGroupBox("擷取與判斷")
        f = QtWidgets.QFormLayout(g)
        self.cmb_bif_len = QtWidgets.QComboBox()
        for n in (1e4, 1e5, 1e6):
            self.cmb_bif_len.addItem(f"{int(n):,} 點", int(n))
        self.spn_bif_periods = _ispin(100, 10, 10000, " 個週期")
        self.cmb_bif_method = QtWidgets.QComboBox()
        self.cmb_bif_method.addItem("固定相位取樣（推薦）", "phase")
        self.cmb_bif_method.addItem("每週期最大值", "max")
        self.cmb_bif_ref = QtWidgets.QComboBox()
        self.cmb_bif_ref.addItem("CH2 基頻波峰（自動）", "ch2")
        self.cmb_bif_ref.addItem("CH1 驅動波峰", "ch1")
        self.cmb_bif_ref.setToolTip("相位 0 的定義。CH2 與 CH1 之間可能差到 ±90°（例如共振以下電流超前），\n"
                                    "以 CH1 為基準時若取樣點剛好落在 CH2 過零處，就只會量到雜訊。")
        self.spn_bif_phase = _dspin(0, -360, 360, 1, " °", 5)
        self.spn_bif_phase.setToolTip("相對於相位基準的偏移。\n每週期最大值模式下，視窗以這個相位為中心、前後各半週期。")
        self.spn_bif_tol = _dspin(config.BIF_TOL * 100, 0.1, 50, 1, " %", 0.5)
        f.addRow("記錄長度", self.cmb_bif_len)
        f.addRow("至少涵蓋", self.spn_bif_periods)
        f.addRow("每週期取值", self.cmb_bif_method)
        f.addRow("相位基準", self.cmb_bif_ref)
        f.addRow("相位偏移", self.spn_bif_phase)
        f.addRow("週期判斷容許誤差", self.spn_bif_tol)
        f.addRow(_gray("只取每週期最大值時，若一個週期內有兩個相近的峰，最大值可能在兩者間跳動而造成假分岔；"
                       "固定相位取樣可避免。容許誤差是相對於 CH2 峰對峰值。"))
        self.cmb_bif_stop = QtWidgets.QComboBox()
        for key, label in STOP_ACTIONS.items():
            self.cmb_bif_stop.addItem(label, key)
        f.addRow("中途停止時產生器", self.cmb_bif_stop)
        v.addWidget(g)

        self.btn_bif_start = QtWidgets.QPushButton("▶ 開始分岔圖")
        self.btn_bif_stop = QtWidgets.QPushButton("■ 停止")
        self.btn_bif_start.clicked.connect(self.start_bif)
        self.btn_bif_stop.clicked.connect(self.stop_experiment)
        v.addWidget(_row(self.btn_bif_start, self.btn_bif_stop))
        self.prg_bif = QtWidgets.QProgressBar()
        v.addWidget(self.prg_bif)
        self.lbl_bif_status = QtWidgets.QLabel("")
        self.lbl_bif_status.setWordWrap(True)
        v.addWidget(self.lbl_bif_status)
        self.btn_bif_save = QtWidgets.QPushButton("儲存分岔圖（CSV + PNG）")
        self.btn_bif_save.clicked.connect(self.save_bif)
        v.addWidget(self.btn_bif_save)
        v.addStretch(1)
        return scrollable(w)

    # ==================================================================
    # 共用
    # ==================================================================
    def _exp_buttons(self):
        return (self.btn_sw_start, self.btn_batch_start, self.btn_bif_start)

    def _exp_set_enabled(self, has_afg):
        running = getattr(self, "exp_worker", None) is not None
        for b in self._exp_buttons():
            b.setEnabled(has_afg and not running)
        for b in (self.btn_sw_stop, self.btn_batch_stop, self.btn_bif_stop):
            b.setEnabled(running)

    def _can_start(self):
        if not (self.scope and self.afg):
            QtWidgets.QMessageBox.information(self, "實驗", "需要同時連接示波器和函數產生器。")
            return False
        if getattr(self, "exp_worker", None) is not None:
            QtWidgets.QMessageBox.information(self, "實驗", "已有實驗在執行中，請先停止。")
            return False
        return True

    def _start_worker(self, worker, progress_bar, status_label):
        self.stop_live()
        self.exp_worker = worker
        worker.status.connect(status_label.setText)
        worker.status.connect(self.statusBar().showMessage)
        worker.progress.connect(lambda i, n: (progress_bar.setMaximum(max(n, 1)), progress_bar.setValue(i)))
        worker.error.connect(lambda m: QtWidgets.QMessageBox.warning(self, "實驗錯誤", m))
        worker.finished.connect(self._exp_finished)
        progress_bar.setValue(0)
        self._exp_set_enabled(True)
        worker.start()

    def _exp_finished(self):
        self.exp_worker = None
        self._exp_set_enabled(self.afg is not None)
        if self.scope:
            self.sync_scope_settings()
            self.start_live()

    def stop_experiment(self):
        if getattr(self, "exp_worker", None) is not None:
            self.exp_worker.stop()
            self.statusBar().showMessage("正在停止…（等目前這一點量完）")

    def stop_all_experiments(self, wait=False):
        w = getattr(self, "exp_worker", None)
        if w is not None:
            w.stop()
            if wait:
                w.wait(60000)

    def _afg_now(self):
        """讀回產生器目前的振幅、偏移（掃頻記錄用）。"""
        amp, off = self.afg.get_amplitude(), self.afg.get_offset()
        if not np.isfinite(amp):
            amp = self.spn_amp.value()
        if not np.isfinite(off):
            off = self.spn_off.value()
        return float(amp), float(off)

    def get_sweep_settings(self):
        return SweepSettings(
            f_start=self.spn_sw_start.value(), f_stop=self.spn_sw_stop.value(),
            n_coarse=self.spn_sw_points.value(), spacing=self.cmb_sw_spacing.currentData(),
            two_stage=self.grp_two_stage.isChecked(), fine_pct=self.spn_fine_pct.value(),
            n_fine=self.spn_fine_n.value(), direction=self.cmb_sw_dir.currentData(),
            periods=self.spn_sw_periods.value(), settle=self.spn_sw_settle.value(),
            auto_v=self.chk_sw_autov.isChecked(), sw_avg=self.spn_sw_avg.value(),
            scope_avg=self.cmb_scope_avg.currentData(), auto_ac=self.chk_sw_ac.isChecked(),
            L=self.spn_L.value() * 1e-6, stop_action=self.cmb_stop_action.currentData())

    # ==================================================================
    # 單次掃頻
    # ==================================================================
    def start_sweep(self):
        if not self._can_start():
            return
        s = self.get_sweep_settings()
        if s.f_stop == s.f_start:
            QtWidgets.QMessageBox.warning(self, "掃頻", "起點和終點不能相同。")
            return
        try:
            amp, off = self._afg_now()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "掃頻", f"讀取產生器設定失敗：{e}")
            return
        self.clear_sweep()
        self.plot_tabs.setCurrentWidget(self.bode_tab)
        worker = SweepWorker(self.scope, self.afg, s, amp, off, self.afg_high_z())
        worker.point.connect(self.on_sweep_point)
        worker.result.connect(self.on_sweep_result)
        self._start_worker(worker, self.prg_sweep, self.lbl_sweep)

    def on_sweep_point(self, p):
        self.sweep_points.append(p)
        self._redraw_bode()

    def _redraw_bode(self):
        pts = self.sweep_points
        for d, (cg, cp) in self.bode_curves.items():
            sel = sorted((p for p in pts if p["direction"] == d), key=lambda p: p["freq_Hz"])
            f = np.array([p["freq_Hz"] for p in sel])
            cg.setData(f, np.array([p["gain_dB"] for p in sel]))
            cp.setData(f, np.array([p["phase_deg"] for p in sel]))
        bad = [p for p in pts if p["unreliable"]]
        fb = np.array([p["freq_Hz"] for p in bad])
        self.bode_bad[0].setData(fb, np.array([p["gain_dB"] for p in bad]))
        self.bode_bad[1].setData(fb, np.array([p["phase_deg"] for p in bad]))
        self._draw_fres_lines()

    def _apply_bode_axis(self, *_):
        self._bode_log = self.cmb_bode_x.currentIndex() == 0 if hasattr(self, "cmb_bode_x") else True
        for p in (self.gain_plot, self.phase_plot):
            p.setLogMode(x=self._bode_log, y=False)
        self._draw_fres_lines()
        self.gain_plot.getViewBox().enableAutoRange()

    def _draw_fres_lines(self):
        for item, plot in self.bode_lines:
            plot.removeItem(item)
        self.bode_lines = []
        for d, a in self.sweep_analyses.items():
            f = a["f_peak"]
            if not (np.isfinite(f) and f > 0):
                continue
            x = math.log10(f) if self._bode_log else f
            for plot in (self.gain_plot, self.phase_plot):
                ln = pg.InfiniteLine(pos=x, angle=90, pen=pg.mkPen(DIR_COLORS[d], width=1, style=QtCore.Qt.DotLine),
                                     label=f"{_fmt_f(f)}" if plot is self.gain_plot else None,
                                     labelOpts={"position": 0.1, "color": DIR_COLORS[d], "fill": (0, 0, 0, 160)})
                plot.addItem(ln, ignoreBounds=True)
                self.bode_lines.append((ln, plot))

    def _analysis_html(self, analyses):
        rows = []
        for d, a in analyses.items():
            c = DIR_COLORS[d]
            rows.append(
                f"<b style='color:{c}'>{DIR_NAMES[d]}</b>　f_res（增益峰）<b>{_fmt_f(a['f_peak'])}</b>　"
                f"f_res（相位過零）<b>{_fmt_f(a['f_phase0'])}</b>　峰值增益 {a['gain_peak_db']:.2f} dB　"
                f"−3 dB 頻寬 {_fmt_f(a['bw'])}　Q = {a['Q']:.3g}　"
                f"C_eff = {fmt_eng(a['C_eff'], 'F')}（相位：{fmt_eng(a['C_eff_phase'], 'F')}）"
                + (f"　<span style='color:#FF1744'>不可靠 {a['n_unreliable']}/{a['n']} 點</span>"
                   if a["n_unreliable"] else ""))
        return "<br>".join(rows)

    def on_sweep_result(self, r):
        self.sweep_analyses = r["analyses"]
        self.sweep_settings = r["settings"]
        self.sweep_warning = r["warning"]
        html = self._analysis_html(self.sweep_analyses)
        if r["warning"]:
            html += f"<br><span style='color:#FF9800'>⚠ {r['warning']}</span>"
        self.lbl_bode.setText(html)
        self._draw_fres_lines()
        if r["warning"]:
            QtWidgets.QMessageBox.warning(self, "振幅檢查", r["warning"])

    def clear_sweep(self):
        self.sweep_points, self.sweep_analyses, self.sweep_warning = [], {}, None
        for p in (self.gain_plot, self.phase_plot):
            p.enableAutoRange()
        self._redraw_bode()
        self.lbl_bode.setText("")
        self.prg_sweep.setValue(0)

    def save_sweep(self):
        if not self.sweep_points:
            QtWidgets.QMessageBox.information(self, "儲存", "目前沒有掃頻資料。")
            return
        base = self._base_name("sweep")
        s = self.sweep_settings or self.get_sweep_settings()
        write_sweep_csv(base + ".csv", self.sweep_points, self.sweep_analyses, self._meta_lines(), s,
                        self.sweep_warning)
        self.bode_tab.grab().save(base + ".png")
        self.lbl_saved.setText(f"已儲存：\n{base}.csv\n{base}.png")
        self.statusBar().showMessage(f"已儲存 {os.path.basename(base)}.csv / .png")

    # ==================================================================
    # 批次參數掃描
    # ==================================================================
    def _batch_kind_changed(self, *_):
        amp = self.cmb_batch_kind.currentData() == "amp"
        if amp:
            self.txt_batch_values.setText("0.05, 0.1, 0.2, 0.4, 0.6, 1, 1.5, 2, 3, 4")
            self.spn_follow_lo.setValue(0.6)
            self.spn_follow_hi.setValue(1.1)
            self.lbl_batch_fixed.setText("DC offset 固定為「函數產生器」分頁目前的值。振幅越大 f_res 越低。")
        else:
            self.txt_batch_values.setText("0, 1, 2, 3, 4, 5")
            self.spn_follow_lo.setValue(0.9)
            self.spn_follow_hi.setValue(1.6)
            self.lbl_batch_fixed.setText("振幅固定為「函數產生器」分頁目前的值（建議用小振幅）。"
                                         "反向偏壓越大，電容越小，f_res 越高。建議同時勾選「AC 耦合」。")

    def start_batch(self):
        if not self._can_start():
            return
        kind = self.cmb_batch_kind.currentData()
        try:
            values = parse_values(self.txt_batch_values.text())
            amp, off = self._afg_now()
            for v in values:
                check_drive(v if kind == "amp" else amp, off if kind == "amp" else v)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "批次掃描", str(e))
            return
        s = self.get_sweep_settings()
        outdir = self._base_name("batch")
        os.makedirs(outdir, exist_ok=True)
        self._batch_outdir = outdir
        follow = (self.spn_follow_lo.value(), self.spn_follow_hi.value()) if self.grp_follow.isChecked() else None
        self.batch_rows = []
        for p in (self.fres_plot, self.ceff_plot):
            p.enableAutoRange()
        self._redraw_param()
        self.clear_sweep()
        self.fres_plot.setLabel("bottom", "振幅" if kind == "amp" else "DC offset", units="Vpp" if kind == "amp" else "V")
        self.ceff_plot.setLabel("bottom", "振幅" if kind == "amp" else "DC offset", units="Vpp" if kind == "amp" else "V")
        self.plot_tabs.setCurrentWidget(self.bode_tab)
        worker = BatchWorker(self.scope, self.afg, s, kind, values, amp, off, self.afg_high_z(), follow, outdir,
                             self._meta_lines())
        worker.point.connect(self.on_sweep_point)
        worker.value_started.connect(lambda _v: self.clear_sweep())
        worker.value_done.connect(self.on_batch_value)
        worker.result.connect(self.on_batch_result)
        self._start_worker(worker, self.prg_batch, self.lbl_batch)

    def on_batch_value(self, r):
        self.sweep_analyses = r["analyses"]
        self.lbl_bode.setText(f"<b>{r['value']:g}</b>：" + self._analysis_html(r["analyses"]))
        self._draw_fres_lines()
        self.batch_rows += r["rows"]
        self._redraw_param()

    def _redraw_param(self):
        rows = self.batch_rows
        for d in DIR_COLORS:
            sel = sorted((r for r in rows if r["direction"] == DIR_NAMES[d]), key=lambda r: r["param"])
            x = np.array([r["param"] for r in sel], float)
            self.param_curves[(d, "peak")].setData(x, np.array([r["f_res_peak_Hz"] for r in sel], float))
            self.param_curves[(d, "phase")].setData(x, np.array([r["f_res_phase0_Hz"] for r in sel], float))
            self.param_curves[(d, "C")].setData(x, np.array([r["C_eff_F"] for r in sel], float))
        lines = ["<tr><th>參數</th><th>方向</th><th>f_res 增益峰</th><th>f_res 相位過零</th><th>峰值增益</th>"
                 "<th>Q</th><th>C_eff</th><th>不可靠</th></tr>"]
        for r in rows:
            lines.append(f"<tr><td>{r['param']:g}</td><td>{r['direction']}</td><td>{_fmt_f(r['f_res_peak_Hz'])}</td>"
                         f"<td>{_fmt_f(r['f_res_phase0_Hz'])}</td><td>{r['peak_gain_dB']:.2f} dB</td>"
                         f"<td>{r['Q']:.3g}</td><td>{fmt_eng(r['C_eff_F'], 'F')}</td>"
                         f"<td>{r['n_unreliable']}/{r['n_points']}</td></tr>")
        self.lbl_param.setText("<table cellspacing=6>" + "".join(lines) + "</table>" if rows else "")

    def on_batch_result(self, r):
        self.plot_tabs.setCurrentWidget(self.param_tab)
        msg = f"完成 {len({x['param'] for x in r['rows']})} 個值。結果在：{r['outdir']}"
        if r["summary"]:
            QtCore.QTimer.singleShot(300, lambda: self.param_tab.grab().save(
                os.path.join(r["outdir"], "summary.png")))
        self.lbl_batch.setText(msg)
        self.btn_batch_open.setEnabled(True)
        self.lbl_saved.setText(f"批次結果：\n{r['outdir']}")
        if r["warnings"]:
            QtWidgets.QMessageBox.warning(self, "振幅檢查", "\n\n".join(r["warnings"]))

    def _open_batch_dir(self):
        if self._batch_outdir and os.path.isdir(self._batch_outdir) and hasattr(os, "startfile"):
            os.startfile(self._batch_outdir)

    # ==================================================================
    # 分岔圖
    # ==================================================================
    def start_bif(self):
        if not self._can_start():
            return
        b = BifSettings(freq=self.spn_bif_f.value(), amp_start=self.spn_bif_a0.value(),
                        amp_stop=self.spn_bif_a1.value(), steps=self.spn_bif_steps.value(),
                        settle=self.spn_bif_settle.value(), record_length=self.cmb_bif_len.currentData(),
                        min_periods=self.spn_bif_periods.value(), method=self.cmb_bif_method.currentData(),
                        phase_deg=self.spn_bif_phase.value(), phase_ref=self.cmb_bif_ref.currentData(), tol=self.spn_bif_tol.value() / 100,
                        stop_action=self.cmb_bif_stop.currentData())
        try:
            amp, off = self._afg_now()
            for a in (b.amp_start, b.amp_stop):
                check_drive(a, off)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "分岔圖", str(e))
            return
        self.bif_results = []
        self.bif_scatter.clear()
        self.bif_seq.setData([], [])
        for p in (self.bif_plot, self.bif_seq_plot):
            p.enableAutoRange()
        self.lbl_bif.setText(self._bif_legend)
        self.plot_tabs.setCurrentWidget(self.bif_tab)
        worker = BifurcationWorker(self.scope, self.afg, b)
        worker.amp_done.connect(self.on_bif_amp)
        worker.result.connect(lambda r: setattr(self, "bif_settings", r["settings"]))
        self.bif_settings = b
        self._start_worker(worker, self.prg_bif, self.lbl_bif_status)

    def on_bif_amp(self, r):
        self.bif_results.append(r)
        vals = r["values"]
        color = PERIOD_COLORS.get(r["period"], PERIOD_COLORS[0])
        self.bif_scatter.addPoints(x=np.full(len(vals), r["amp"]), y=vals, brush=pg.mkBrush(color))
        self.bif_seq.setData(np.arange(len(vals)), vals)
        self.bif_seq.setSymbolBrush(color)
        self.bif_seq_plot.setTitle(f"振幅 {r['amp']:.4g} Vpp：{r['label']}（{len(vals)} 個週期）")
        recent = self.bif_results[-6:]
        lines = [f"{x['amp']:.4g} Vpp → <b style='color:{PERIOD_COLORS.get(x['period'], PERIOD_COLORS[0])}'>"
                 f"{x['label']}</b>（f/2：{x['sub']['f/2']:.1f} dBc，f/4：{x['sub']['f/4']:.1f} dBc）"
                 for x in recent]
        self.lbl_bif.setText(self._bif_legend + "<br>" + "<br>".join(lines))

    def save_bif(self):
        if not self.bif_results:
            QtWidgets.QMessageBox.information(self, "儲存", "目前沒有分岔圖資料。")
            return
        base = self._base_name("bifurcation")
        write_bifurcation_csv(base, self.bif_results, self.bif_settings, self._meta_lines())
        self.bif_tab.grab().save(base + ".png")
        self.lbl_saved.setText(f"已儲存：\n{base}.csv\n{base}_summary.csv\n{base}.png")
        self.statusBar().showMessage(f"已儲存 {os.path.basename(base)}")

"""示波器 + 函數產生器 控制介面

GW Instek GDS-1072B 示波器 + AFG-2112 函數產生器（USB）
功能：即時波形、游標、FFT、儲存 CSV/PNG、設定產生器、共振掃頻（兩階段/遲滯）、批次參數掃描、分岔圖。

執行：python app.py      （沒有接儀器時勾選「模擬模式」即可試用）
"""
import csv
import dataclasses
import datetime
import math
import os
import sys
import time

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets

import config
from analysis import (FFT_WINDOWS, NOISE_MULT, estimate_frequency, find_peaks, fmt_eng, harmonics, lockin,
                      nearest_index, noise_floor, pick_scale, refine_peak, spectrum, suggest_vscale, thd, vpp,
                      wrap_deg)
from experiments_ui import ExperimentsMixin, scrollable
from instruments import (AFG_FREQ_RANGE, AFG_FUNCS, AFG2100, GDS1000B, SimBench, TIME_SCALES, VOLT_SCALES,
                         auto_detect, list_com_ports)

HERE = os.path.dirname(os.path.abspath(__file__))
CH_COLORS = {1: "#F5C400", 2: "#00B7EB"}     # 與 GDS 螢幕相同：CH1 黃、CH2 藍
NO_DEVICE = "(不使用)"


# ======================================================================
# 背景執行緒
# ======================================================================

class LiveWorker(QtCore.QThread):
    """持續向示波器讀波形。"""
    data = QtCore.pyqtSignal(object)
    error = QtCore.pyqtSignal(str)

    def __init__(self, scope, channels):
        super().__init__()
        self.scope = scope
        self.channels = frozenset(channels)
        self._running = False

    def run(self):
        self._running = True
        while self._running:
            chs = sorted(self.channels)
            try:
                waves = {ch: self.scope.read_waveform(ch) for ch in chs}
            except Exception as e:
                if self._running:
                    self.error.emit(f"讀取波形失敗：{e}")
                break
            if waves and self._running:
                self.data.emit(waves)
            self.msleep(int(config.LIVE_INTERVAL_S * 1000) if chs else 200)

    def stop(self):
        self._running = False
        self.wait(10000)


# ======================================================================
# 主視窗
# ======================================================================

def hbox(*widgets):
    w = QtWidgets.QWidget()
    lay = QtWidgets.QHBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    for x in widgets:
        if x == "stretch":
            lay.addStretch(1)
        else:
            lay.addWidget(x)
    return w


class MainWindow(ExperimentsMixin, QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("示波器 / 函數產生器 控制台 — GDS-1072B + AFG-2112")
        self.resize(1400, 820)
        self.scope = self.afg = None
        self.live = None
        self.exp_worker = None
        self.last_waves = {}
        self._frame_times = []

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.addWidget(self._build_connection_bar())

        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        split.addWidget(self._build_plot_area())
        split.addWidget(self._build_control_tabs())
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 1)
        split.setSizes([950, 450])
        root.addWidget(split, 1)

        self.statusBar().showMessage("請選擇 COM 埠後按「連線」，或勾選「模擬模式」試用。")
        self.refresh_ports()
        self._set_connected(False)

    # ---------------- 介面建立 ----------------
    def _build_connection_bar(self):
        box = QtWidgets.QGroupBox("連線")
        lay = QtWidgets.QHBoxLayout(box)
        self.scope_port = QtWidgets.QComboBox()
        self.afg_port = QtWidgets.QComboBox()
        for c in (self.scope_port, self.afg_port):
            c.setMinimumWidth(220)
        self.btn_refresh = QtWidgets.QPushButton("重新整理")
        self.btn_detect = QtWidgets.QPushButton("自動偵測")
        self.chk_sim = QtWidgets.QCheckBox("模擬模式（不接儀器）")
        self.btn_connect = QtWidgets.QPushButton("連線")
        self.btn_disconnect = QtWidgets.QPushButton("中斷連線")
        self.lbl_conn = QtWidgets.QLabel("未連線")
        self.btn_refresh.clicked.connect(self.refresh_ports)
        self.btn_detect.clicked.connect(self.detect_ports)
        self.btn_connect.clicked.connect(self.connect_instruments)
        self.btn_disconnect.clicked.connect(self.disconnect_instruments)
        for w in (QtWidgets.QLabel("示波器："), self.scope_port, QtWidgets.QLabel("函數產生器："), self.afg_port,
                  self.btn_refresh, self.btn_detect, self.chk_sim, self.btn_connect, self.btn_disconnect):
            lay.addWidget(w)
        lay.addWidget(self.lbl_conn, 1)
        return box

    def _build_plot_area(self):
        self.plot_tabs = QtWidgets.QTabWidget()

        # 即時波形
        live = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(live)
        self.plot = pg.PlotWidget()
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.setLabel("bottom", "時間", units="s")
        self.plot.setLabel("left", "電壓", units="V")
        self.plot.addLegend(offset=(10, 10))
        self.curves = {ch: self.plot.plot([], [], pen=pg.mkPen(CH_COLORS[ch], width=1.5), name=f"CH{ch}")
                       for ch in (1, 2)}
        vb = self.plot.getViewBox()
        vb.setMouseMode(pg.ViewBox.RectMode)              # 左鍵拖曳框選放大
        vb.sigRangeChangedManually.connect(self._on_manual_range)
        self.auto_view = True

        # 游標：兩條時間游標（直線）、兩條電壓游標（橫線）
        self.tcur = []
        for name, color in (("T1", "#FF5252"), ("T2", "#FF9800")):
            ln = pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen(color, width=1.5, style=QtCore.Qt.DashLine),
                                 hoverPen=pg.mkPen(color, width=3), label=name,
                                 labelOpts={"position": 0.95, "color": color, "fill": (0, 0, 0, 160)})
            ln.sigPositionChanged.connect(self.update_cursor_readout)
            ln.setVisible(False)
            self.plot.addItem(ln, ignoreBounds=True)
            self.tcur.append(ln)
        self.vcur = []
        for name, color in (("V1", "#69F0AE"), ("V2", "#E040FB")):
            ln = pg.InfiniteLine(angle=0, movable=True, pen=pg.mkPen(color, width=1.5, style=QtCore.Qt.DashLine),
                                 hoverPen=pg.mkPen(color, width=3), label=name,
                                 labelOpts={"position": 0.03, "color": color, "fill": (0, 0, 0, 160)})
            ln.sigPositionChanged.connect(self.update_cursor_readout)
            ln.setVisible(False)
            self.plot.addItem(ln, ignoreBounds=True)
            self.vcur.append(ln)
        lay.addWidget(self.plot, 1)

        self.chk_auto_view = QtWidgets.QCheckBox("自動視圖")
        self.chk_auto_view.setChecked(True)
        self.chk_auto_view.setToolTip("跟著示波器的時基與刻度自動調整顯示範圍。\n用滑鼠縮放後會自動關閉。")
        self.chk_auto_view.toggled.connect(self._on_auto_view_toggled)
        btn_reset = QtWidgets.QPushButton("重設視圖")
        btn_reset.clicked.connect(lambda: self.chk_auto_view.setChecked(True) or self._apply_auto_view())
        self.chk_tcur = QtWidgets.QCheckBox("時間游標 T1/T2")
        self.chk_vcur = QtWidgets.QCheckBox("電壓游標 V1/V2")
        self.chk_tcur.toggled.connect(lambda on: self._show_cursors(self.tcur, on, vertical=True))
        self.chk_vcur.toggled.connect(lambda on: self._show_cursors(self.vcur, on, vertical=False))
        btn_zoom = QtWidgets.QPushButton("放大到 T1~T2")
        btn_zoom.clicked.connect(self.zoom_to_cursors)
        self.chk_sel = QtWidgets.QCheckBox("量測與存檔只用 T1~T2 之間")
        self.chk_sel.toggled.connect(lambda _on: self.on_waves(self.last_waves, force=True))
        lay.addWidget(hbox(self.chk_auto_view, btn_reset, self.chk_tcur, self.chk_vcur, btn_zoom,
                           self.chk_sel, "stretch"))

        self.lbl_cursor = QtWidgets.QLabel("")
        self.lbl_cursor.setTextFormat(QtCore.Qt.RichText)
        self.lbl_cursor.setStyleSheet("font-size: 10.5pt; padding: 2px 4px;")
        self.lbl_cursor.setVisible(False)
        lay.addWidget(self.lbl_cursor)

        self.lbl_meas = QtWidgets.QLabel("—")
        self.lbl_meas.setTextFormat(QtCore.Qt.RichText)
        self.lbl_meas.setStyleSheet("font-size: 11pt; padding: 4px;")
        lay.addWidget(self.lbl_meas)

        self.btn_save_csv = QtWidgets.QPushButton("儲存波形 CSV")
        self.btn_save_png = QtWidgets.QPushButton("儲存截圖 PNG")
        self.btn_save_both = QtWidgets.QPushButton("兩者都存")
        self.btn_freeze = QtWidgets.QPushButton("凍結畫面")
        self.btn_freeze.setCheckable(True)
        self.btn_save_csv.clicked.connect(lambda: self.save_waveform(csv_=True, png=False))
        self.btn_save_png.clicked.connect(lambda: self.save_waveform(csv_=False, png=True))
        self.btn_save_both.clicked.connect(lambda: self.save_waveform(csv_=True, png=True))
        self.lbl_fps = QtWidgets.QLabel("")
        lay.addWidget(hbox(self.btn_save_csv, self.btn_save_png, self.btn_save_both, self.btn_freeze,
                           "stretch", self.lbl_fps))
        self.plot_tabs.addTab(live, "即時波形")
        self.plot_tabs.addTab(self._build_fft_tab(), "頻譜 FFT")

        self.plot_tabs.addTab(self._build_bode_tab(), "波德圖")
        self.plot_tabs.addTab(self._build_param_plot_tab(), "參數掃描")
        self.plot_tabs.addTab(self._build_bif_plot_tab(), "分岔圖")
        self.plot_tabs.addTab(self._build_feig_tab(), "分岔分析")
        return self.plot_tabs

    def _build_fft_tab(self):
        tab = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(tab)
        self.fft_tab = tab
        self.fft_plot = pg.PlotWidget()
        self.fft_plot.showGrid(x=True, y=True, alpha=0.3)
        self.fft_plot.setLabel("bottom", "頻率", units="Hz")
        self.fft_plot.addLegend(offset=(-10, 10))
        self.fft_plot.getViewBox().setMouseMode(pg.ViewBox.RectMode)
        self.fft_curves = {ch: self.fft_plot.plot([], [], pen=pg.mkPen(CH_COLORS[ch], width=1.3), name=f"CH{ch}")
                           for ch in (1, 2)}
        self.fft_marks = {ch: self.fft_plot.plot([], [], pen=None, symbol="t", symbolSize=9,
                                                 symbolBrush=CH_COLORS[ch], symbolPen=None) for ch in (1, 2)}
        self.fft_cursor = pg.InfiniteLine(angle=90, movable=True,
                                          pen=pg.mkPen("#FF5252", width=1.5, style=QtCore.Qt.DashLine),
                                          hoverPen=pg.mkPen("#FF5252", width=3), label="F",
                                          labelOpts={"position": 0.95, "color": "#FF5252", "fill": (0, 0, 0, 160)})
        self.fft_cursor.setVisible(False)
        self.fft_cursor.sigPositionChanged.connect(self.update_fft_readout)
        self.fft_plot.addItem(self.fft_cursor, ignoreBounds=True)
        lay.addWidget(self.fft_plot, 1)

        self.cmb_fft_win = QtWidgets.QComboBox()
        for name, (_fn, _bw, desc) in FFT_WINDOWS.items():
            self.cmb_fft_win.addItem(name)
            self.cmb_fft_win.setItemData(self.cmb_fft_win.count() - 1, desc, QtCore.Qt.ToolTipRole)
        self.cmb_fft_unit = QtWidgets.QComboBox()
        self.cmb_fft_unit.addItems(["dBV (rms)", "Vrms（線性）"])
        self.cmb_fft_x = QtWidgets.QComboBox()
        self.cmb_fft_x.addItems(["線性頻率", "對數頻率"])
        self.cmb_fft_pad = QtWidgets.QComboBox()
        for z in (1, 4, 8):
            self.cmb_fft_pad.addItem(f"補零 ×{z}", z)
        self.cmb_fft_pad.setToolTip("補零只讓曲線較平滑、峰值位置較準，不會提高真正的頻率解析度。\n"
                                    "要提高解析度，請加大時基或 config.py 的 RECORD_LENGTH。")
        self.spn_fft_avg = QtWidgets.QSpinBox()
        self.spn_fft_avg.setRange(1, 100)
        self.spn_fft_avg.setPrefix("平均 ")
        self.spn_fft_avg.setSuffix(" 張")
        self.spn_fft_avg.setToolTip("對最近 N 張頻譜做功率平均，可降低雜訊底線。")
        for w in (self.cmb_fft_win, self.cmb_fft_unit, self.cmb_fft_pad, self.spn_fft_avg):
            (w.currentIndexChanged if isinstance(w, QtWidgets.QComboBox) else w.valueChanged).connect(
                self._fft_settings_changed)
        self.cmb_fft_x.currentIndexChanged.connect(self._fft_xmode_changed)
        self.chk_fft_cursor = QtWidgets.QCheckBox("頻率游標")
        self.chk_fft_cursor.toggled.connect(self._toggle_fft_cursor)
        btn_reset = QtWidgets.QPushButton("重設視圖")
        btn_reset.clicked.connect(lambda: self.fft_plot.getViewBox().enableAutoRange())
        lay.addWidget(hbox(QtWidgets.QLabel("視窗"), self.cmb_fft_win, self.cmb_fft_unit, self.cmb_fft_x,
                           self.cmb_fft_pad, self.spn_fft_avg, self.chk_fft_cursor, btn_reset, "stretch"))

        self.lbl_fft = QtWidgets.QLabel("—")
        self.lbl_fft.setTextFormat(QtCore.Qt.RichText)
        self.lbl_fft.setStyleSheet("font-size: 10.5pt; padding: 4px;")
        self.lbl_fft.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        lay.addWidget(self.lbl_fft)

        btn_csv = QtWidgets.QPushButton("儲存頻譜 CSV")
        btn_png = QtWidgets.QPushButton("儲存頻譜截圖 PNG")
        btn_csv.clicked.connect(self.save_fft_csv)
        btn_png.clicked.connect(self.save_fft_png)
        self.lbl_fft_info = QtWidgets.QLabel("")
        self.lbl_fft_info.setStyleSheet("color: gray;")
        lay.addWidget(hbox(btn_csv, btn_png, "stretch", self.lbl_fft_info))

        self.fft_avg = {}        # ch -> list of amp² arrays
        self.fft_data = {}       # ch -> (f, amp_peak)（平均後）
        return tab

    def _build_control_tabs(self):
        tabs = QtWidgets.QTabWidget()
        tabs.addTab(scrollable(self._build_scope_tab()), "示波器")
        tabs.addTab(scrollable(self._build_afg_tab()), "產生器")
        tabs.addTab(self._build_sweep_tab(), "掃頻")
        tabs.addTab(self._build_batch_tab(), "批次")
        tabs.addTab(self._build_bif_tab(), "分岔圖")
        tabs.addTab(scrollable(self._build_save_tab()), "儲存")
        tabs.setMinimumWidth(420)
        return tabs

    def _build_scope_tab(self):
        w = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(w)

        self.btn_live_start = QtWidgets.QPushButton("▶ 開始即時顯示")
        self.btn_live_stop = QtWidgets.QPushButton("■ 停止即時顯示")
        self.btn_live_start.clicked.connect(self.start_live)
        self.btn_live_stop.clicked.connect(self.stop_live)
        form.addRow(hbox(self.btn_live_start, self.btn_live_stop))

        self.chk_ch = {}
        for ch in (1, 2):
            c = QtWidgets.QCheckBox(f"CH{ch}")
            c.setChecked(True)
            c.setStyleSheet(f"color: {CH_COLORS[ch]}; font-weight: bold;")
            c.toggled.connect(lambda on, ch=ch: self.on_channel_toggled(ch, on))
            self.chk_ch[ch] = c
        form.addRow("顯示通道", hbox(self.chk_ch[1], self.chk_ch[2], "stretch"))

        self.cmb_time = QtWidgets.QComboBox()
        for s in TIME_SCALES:
            self.cmb_time.addItem(fmt_eng(s, "s") + "/div", s)
        self.cmb_time.currentIndexChanged.connect(self.on_timebase_changed)
        form.addRow("時基", self.cmb_time)

        self.cmb_volt = {}
        for ch in (1, 2):
            c = QtWidgets.QComboBox()
            for s in VOLT_SCALES:
                c.addItem(fmt_eng(s, "V") + "/div", s)
            c.currentIndexChanged.connect(lambda _i, ch=ch: self.on_vscale_changed(ch))
            self.cmb_volt[ch] = c
            form.addRow(f"CH{ch} 垂直刻度", c)

        self.scope_setting_widgets = []
        self.cmb_coup, self.cmb_probe, self.chk_bw = {}, {}, {}
        for ch in (1, 2):
            c = QtWidgets.QComboBox()
            c.addItems(["DC", "AC", "GND"])
            c.setToolTip("耦合方式：有大 DC offset、只想看交流小訊號時用 AC")
            c.currentTextChanged.connect(lambda m, ch=ch: self._scope_set("set_coupling", ch, m))
            p = QtWidgets.QComboBox()
            p.addItem("探棒 ×1", 1.0)
            p.addItem("探棒 ×10", 10.0)
            p.currentIndexChanged.connect(
                lambda _i, ch=ch: self._scope_set("set_probe", ch, self.cmb_probe[ch].currentData()))
            b = QtWidgets.QCheckBox("頻寬限制")
            b.toggled.connect(lambda on, ch=ch: self._scope_set("set_bwlimit", ch, on))
            self.cmb_coup[ch], self.cmb_probe[ch], self.chk_bw[ch] = c, p, b
            self.scope_setting_widgets += [c, p, b]
            form.addRow(f"CH{ch} 輸入", hbox(c, p, b))
        self.cmb_acq = QtWidgets.QComboBox()
        for n in (1, 2, 4, 8, 16, 32, 64, 128, 256):
            self.cmb_acq.addItem("一般取樣" if n == 1 else f"平均 {n} 次", n)
        self.cmb_acq.currentIndexChanged.connect(
            lambda _i: self._scope_set("set_acquire_average", self.cmb_acq.currentData()))
        self.scope_setting_widgets.append(self.cmb_acq)
        form.addRow("擷取模式", self.cmb_acq)

        self.btn_run = QtWidgets.QPushButton("RUN")
        self.btn_stop = QtWidgets.QPushButton("STOP")
        self.btn_single = QtWidgets.QPushButton("SINGLE")
        self.btn_autoset = QtWidgets.QPushButton("AUTOSET")
        self.btn_run.clicked.connect(lambda: self._scope_cmd("run"))
        self.btn_stop.clicked.connect(lambda: self._scope_cmd("stop"))
        self.btn_single.clicked.connect(lambda: self._scope_cmd("single"))
        self.btn_autoset.clicked.connect(self.on_autoset)
        form.addRow("示波器", hbox(self.btn_run, self.btn_stop, self.btn_single))
        self.btn_pos0 = QtWidgets.QPushButton("垂直位置歸零")
        self.btn_pos0.setToolTip("把 CH1、CH2 的垂直位置設為 0 V。\n位置不是 0 時，小訊號可能被推到螢幕邊緣，自動刻度也無法縮小。")
        self.btn_pos0.clicked.connect(lambda: [self._scope_set("set_position", ch, 0.0) for ch in (1, 2)])
        self.scope_setting_widgets.append(self.btn_pos0)
        form.addRow("", hbox(self.btn_autoset, self.btn_pos0, "stretch"))

        tip = QtWidgets.QLabel("提示：即時顯示的更新率受 USB 傳輸速度限制，約每秒數張。"
                               "記錄長度可在 config.py 的 RECORD_LENGTH 調整。")
        tip.setWordWrap(True)
        tip.setStyleSheet("color: gray;")
        form.addRow(tip)
        return w

    def _build_afg_tab(self):
        w = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(w)

        self.cmb_func = QtWidgets.QComboBox()
        self.cmb_func.addItems(list(AFG_FUNCS))
        form.addRow("波形", self.cmb_func)

        self.spn_freq = QtWidgets.QDoubleSpinBox()
        self.spn_freq.setDecimals(4)
        self.spn_freq.setRange(0.0001, 1e9)
        self.spn_freq.setValue(1.0)
        self.spn_freq.setKeyboardTracking(False)
        self.cmb_funit = QtWidgets.QComboBox()
        for name, mult in (("Hz", 1.0), ("kHz", 1e3), ("MHz", 1e6)):
            self.cmb_funit.addItem(name, mult)
        self.cmb_funit.setCurrentIndex(1)
        form.addRow("頻率", hbox(self.spn_freq, self.cmb_funit))

        quick = []
        for label, factor in (("÷10", 0.1), ("÷2", 0.5), ("×2", 2.0), ("×10", 10.0)):
            b = QtWidgets.QPushButton(label)
            b.setMaximumWidth(60)
            b.clicked.connect(lambda _c, f=factor: self.scale_frequency(f))
            quick.append(b)
        form.addRow("快速調整", hbox(*quick, "stretch"))

        self.spn_amp = QtWidgets.QDoubleSpinBox()
        self.spn_amp.setDecimals(3)
        self.spn_amp.setRange(0.001, 20)
        self.spn_amp.setValue(2.0)
        self.spn_amp.setSuffix(" Vpp")
        form.addRow("振幅", self.spn_amp)

        self.spn_off = QtWidgets.QDoubleSpinBox()
        self.spn_off.setDecimals(3)
        self.spn_off.setRange(-10, 10)
        self.spn_off.setSuffix(" V")
        form.addRow("直流偏移", self.spn_off)

        self.cmb_load = QtWidgets.QComboBox()
        self.cmb_load.addItem("High-Z（高阻抗）", True)
        self.cmb_load.addItem("50 Ω", False)
        self.cmb_load.setToolTip("產生器假設的負載。設 50 Ω 但實際接高阻抗時，實際輸出會是設定值的 2 倍。")
        self.cmb_load.currentIndexChanged.connect(self.on_load_changed)
        form.addRow("負載設定", self.cmb_load)

        self.btn_afg_apply = QtWidgets.QPushButton("套用設定")
        self.btn_afg_apply.clicked.connect(self.apply_afg)
        self.btn_output = QtWidgets.QPushButton("輸出：OFF")
        self.btn_output.setCheckable(True)
        self.btn_output.toggled.connect(self.on_output_toggled)
        form.addRow(hbox(self.btn_afg_apply, self.btn_output))

        self.chk_auto_apply = QtWidgets.QCheckBox("修改頻率後立即套用（按 Enter 或離開輸入框時）")
        self.chk_auto_apply.setChecked(True)
        self.spn_freq.editingFinished.connect(self._maybe_apply_freq)
        form.addRow(self.chk_auto_apply)

        tip = QtWidgets.QLabel("AFG-2112 頻率範圍：0.1 Hz ~ 12 MHz（正弦/方波）。\n"
                               "振幅依產生器的負載設定（50Ω / High-Z）而定，接示波器等高阻抗負載時請設為 High-Z。\n"
                               f"安全上限：振幅 {config.AMPLITUDE_MAX_VPP:g} Vpp、|偏移|+振幅/2 ≤ {config.PEAK_MAX_V:g} V"
                               "（config.py）。")
        tip.setWordWrap(True)
        tip.setStyleSheet("color: gray;")
        form.addRow(tip)
        return w

    def _build_save_tab(self):
        w = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(w)
        self.txt_dir = QtWidgets.QLineEdit(os.path.join(HERE, config.DATA_DIR))
        btn = QtWidgets.QPushButton("瀏覽…")
        btn.clicked.connect(self.browse_dir)
        form.addRow("儲存資料夾", hbox(self.txt_dir, btn))
        self.txt_prefix = QtWidgets.QLineEdit("wave")
        form.addRow("檔名開頭", self.txt_prefix)
        self.txt_note = QtWidgets.QLineEdit()
        self.txt_note.setPlaceholderText("例如：RC 電路，R=1kΩ，C=0.1µF")
        form.addRow("備註（寫進 CSV）", self.txt_note)
        btn_open = QtWidgets.QPushButton("開啟資料夾")
        btn_open.clicked.connect(lambda: os.startfile(self._ensure_dir()) if sys.platform == "win32" else None)
        form.addRow(btn_open)
        self.lbl_saved = QtWidgets.QLabel("")
        self.lbl_saved.setWordWrap(True)
        form.addRow(self.lbl_saved)
        return w

    # ---------------- 連線 ----------------
    def refresh_ports(self):
        ports = list_com_ports()
        for combo, pref, allow_none in ((self.scope_port, config.SCOPE_PORT, False),
                                        (self.afg_port, config.AFG_PORT, True)):
            cur = combo.currentData() or pref
            combo.clear()
            if allow_none:
                combo.addItem(NO_DEVICE, None)
            for dev, desc in ports:
                combo.addItem(f"{dev} — {desc}", dev)
            i = combo.findData(cur)
            if i >= 0:
                combo.setCurrentIndex(i)
        if not ports:
            self.statusBar().showMessage("找不到任何 COM 埠：請確認 USB 已接上且已安裝 GW Instek 驅動。")

    def detect_ports(self):
        self.statusBar().showMessage("正在偵測儀器…")
        QtWidgets.QApplication.processEvents()
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            found = auto_detect()
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self.refresh_ports()
        msg = []
        for key, combo, name in (("scope", self.scope_port, "示波器"), ("afg", self.afg_port, "函數產生器")):
            if key in found:
                combo.setCurrentIndex(combo.findData(found[key][0]))
                msg.append(f"{name}：{found[key][0]}（{found[key][1]}）")
            else:
                msg.append(f"{name}：找不到")
        self.statusBar().showMessage("；".join(msg))

    def connect_instruments(self):
        self.disconnect_instruments()
        try:
            if self.chk_sim.isChecked():
                bench = SimBench()
                self.scope, self.afg = bench.scope, bench.afg
            else:
                port = self.scope_port.currentData()
                if not port:
                    raise RuntimeError("請先選擇示波器的 COM 埠")
                self.scope = GDS1000B(port)
                afg_port = self.afg_port.currentData()
                if afg_port:
                    if afg_port == port:
                        raise RuntimeError("示波器和函數產生器不能用同一個 COM 埠")
                    self.afg = AFG2100(afg_port)
            idns = [f"示波器：{self.scope.idn()}"]
            self.scope.setup()
            if self.afg:
                idns.append(f"產生器：{self.afg.idn()}")
        except Exception as e:
            self.disconnect_instruments()
            QtWidgets.QMessageBox.critical(self, "連線失敗", str(e))
            return
        self.lbl_conn.setText("  |  ".join(idns))
        self.statusBar().showMessage("連線成功")
        self._set_connected(True)
        self.sync_scope_settings()
        if self.afg:
            self.sync_afg_settings()
        if self.chk_sim.isChecked():
            self.btn_output.setChecked(True)   # 模擬模式直接打開輸出，方便試用
        self.start_live()

    def disconnect_instruments(self):
        self.stop_all_experiments(wait=True)
        self.stop_live()
        for inst in (self.scope, self.afg):
            if inst:
                inst.close()
        self.scope = self.afg = None
        self.lbl_conn.setText("未連線")
        self._set_connected(False)

    def _set_connected(self, on):
        has_afg = on and self.afg is not None
        for w in (self.btn_live_start, self.btn_live_stop, self.cmb_time, *self.cmb_volt.values(),
                  self.btn_run, self.btn_stop, self.btn_single, self.btn_autoset):
            w.setEnabled(on)
        for w in (self.btn_afg_apply, self.btn_output, self.cmb_load):
            w.setEnabled(has_afg)
        for w in self.scope_setting_widgets:
            w.setEnabled(on)
        self._exp_set_enabled(has_afg)
        self.btn_disconnect.setEnabled(on)

    def sync_scope_settings(self):
        """把示波器目前的時基與垂直刻度讀回介面。"""
        try:
            self._select_scale(self.cmb_time, TIME_SCALES, self.scope.get_timebase())
            for ch in (1, 2):
                self._select_scale(self.cmb_volt[ch], VOLT_SCALES, self.scope.get_ch_scale(ch))
        except Exception as e:
            self.statusBar().showMessage(f"讀取示波器設定失敗：{e}")
        for ch in (1, 2):                      # 個別讀，某一項不支援也不影響其他
            for getter, apply in (
                    (self.scope.get_coupling, lambda v, ch=ch: self.cmb_coup[ch].setCurrentText(v[:3].strip())),
                    (self.scope.get_probe, lambda v, ch=ch: self.cmb_probe[ch].setCurrentIndex(1 if v >= 5 else 0)),
                    (self.scope.get_bwlimit, lambda v, ch=ch: self.chk_bw[ch].setChecked(bool(v)))):
                try:
                    value = getter(ch)
                    for w in (self.cmb_coup[ch], self.cmb_probe[ch], self.chk_bw[ch]):
                        w.blockSignals(True)
                    apply(value)
                except Exception:
                    pass
                finally:
                    for w in (self.cmb_coup[ch], self.cmb_probe[ch], self.chk_bw[ch]):
                        w.blockSignals(False)

    def _scope_set(self, name, *args):
        if self.scope:
            try:
                getattr(self.scope, name)(*args)
                self.statusBar().showMessage(f"示波器：{name} {args}")
            except Exception as e:
                self.statusBar().showMessage(f"示波器設定失敗：{e}")

    def sync_afg_settings(self):
        try:
            f = self.afg.get_frequency()
            a = self.afg.get_amplitude()
            if np.isfinite(f):
                self.set_freq_hz(f)
            if np.isfinite(a):
                self.spn_amp.setValue(a)
            o = self.afg.get_offset()
            if np.isfinite(o):
                self.spn_off.setValue(o)
            self.cmb_load.blockSignals(True)
            self.cmb_load.setCurrentIndex(0 if self.afg.get_load() else 1)
            self.cmb_load.blockSignals(False)
        except Exception as e:
            self.statusBar().showMessage(f"讀取產生器設定失敗：{e}")

    @staticmethod
    def _select_scale(combo, scales, value):
        if value and np.isfinite(value) and value > 0:
            combo.blockSignals(True)
            combo.setCurrentIndex(nearest_index(scales, value))
            combo.blockSignals(False)

    # ---------------- 即時顯示 ----------------
    def _channels(self):
        return [ch for ch in (1, 2) if self.chk_ch[ch].isChecked()]

    def start_live(self):
        if not self.scope or (self.live and self.live.isRunning()):
            return
        self.live = LiveWorker(self.scope, self._channels())
        self.live.data.connect(self.on_waves)
        self.live.error.connect(self.on_live_error)
        self._frame_times.clear()
        self.live.start()
        self.statusBar().showMessage("即時顯示中")

    def stop_live(self):
        if self.live:
            self.live.stop()
            self.live = None
            self.lbl_fps.setText("")

    def on_live_error(self, msg):
        self.stop_live()
        self.statusBar().showMessage(msg)
        QtWidgets.QMessageBox.warning(self, "讀取錯誤", msg + "\n\n可按「開始即時顯示」重試，或執行 test_connection.py 檢查連線。")

    def on_waves(self, waves, force=False):
        if self.btn_freeze.isChecked() and not force:
            return
        self.last_waves = waves
        for ch in (1, 2):
            w = waves.get(ch)
            if w is None:
                self.curves[ch].setData([], [])
            else:
                self.curves[ch].setData(w.t, w.v)
        if self.auto_view:
            self._apply_auto_view()
        self.update_measurements(self._selected(waves), waves)
        self.update_cursor_readout()
        self.update_fft(self._selected(waves), accumulate=not force)
        if force:
            return
        now = time.monotonic()
        self._frame_times = [t for t in self._frame_times if now - t < 2.0] + [now]
        if len(self._frame_times) > 1:
            fps = (len(self._frame_times) - 1) / (self._frame_times[-1] - self._frame_times[0])
            self.lbl_fps.setText(f"更新率 {fps:.1f} 張/秒")

    # ---------------- 視圖與游標 ----------------
    def _on_manual_range(self, *_):
        """使用者用滑鼠縮放/平移 → 關掉自動視圖，保留使用者選的範圍。"""
        if self.chk_auto_view.isChecked():
            self.chk_auto_view.setChecked(False)

    def _on_auto_view_toggled(self, on):
        self.auto_view = on
        if on:
            self._apply_auto_view()

    def _apply_auto_view(self):
        """顯示範圍 = 示波器螢幕範圍（水平 10 格、垂直 ±4 格）。"""
        waves = self.last_waves
        if not waves:
            return
        t0 = min(float(w.t[0]) for w in waves.values() if len(w.t))
        t1 = max(float(w.t[-1]) for w in waves.values() if len(w.t))
        h = config.SCREEN_HALF_DIVS
        lo = min(-h * w.vscale - w.vpos for w in waves.values())
        hi = max(h * w.vscale - w.vpos for w in waves.values())
        # setRange 不會觸發 sigRangeChangedManually（只有滑鼠操作才會），所以不會關掉自動視圖
        self.plot.getViewBox().setRange(xRange=(t0, t1), yRange=(lo, hi), padding=0.02)

    def _show_cursors(self, lines, on, vertical):
        if on:
            (x0, x1), (y0, y1) = self.plot.getViewBox().viewRange()
            a, b = (x0, x1) if vertical else (y0, y1)
            for ln, frac in zip(lines, (1 / 3, 2 / 3)):
                ln.setValue(a + (b - a) * frac)
        for ln in lines:
            ln.setVisible(on)
        self.lbl_cursor.setVisible(self.chk_tcur.isChecked() or self.chk_vcur.isChecked())
        if self.chk_sel.isChecked():
            self.on_waves(self.last_waves, force=True)
        else:
            self.update_cursor_readout()

    def _t_range(self):
        a, b = (ln.value() for ln in self.tcur)
        return min(a, b), max(a, b)

    def _selected(self, waves):
        """勾選「只用 T1~T2」時，回傳只含游標之間資料的波形。"""
        if not (self.chk_sel.isChecked() and self.chk_tcur.isChecked()):
            return waves
        a, b = self._t_range()
        out = {}
        for ch, w in waves.items():
            m = (w.t >= a) & (w.t <= b)
            out[ch] = dataclasses.replace(w, t=w.t[m], v=w.v[m], screen_div=w.screen_div[m])
        return out

    def zoom_to_cursors(self):
        if not self.chk_tcur.isChecked():
            self.chk_tcur.setChecked(True)
            return
        a, b = self._t_range()
        if b > a:
            self.chk_auto_view.setChecked(False)
            self.plot.getViewBox().setXRange(a, b, padding=0.02)

    def update_cursor_readout(self, *_):
        if not self.lbl_cursor.isVisible():
            return
        parts = []
        if self.chk_tcur.isChecked():
            t1, t2 = (ln.value() for ln in self.tcur)
            dt = t2 - t1
            line = (f"<b style='color:#FF5252'>T1</b> {fmt_eng(t1, 's')}　<b style='color:#FF9800'>T2</b> "
                    f"{fmt_eng(t2, 's')}　<b>Δt</b> {fmt_eng(dt, 's')}　<b>1/Δt</b> "
                    f"{fmt_eng(1 / abs(dt), 'Hz') if dt else '—'}")
            for ch, w in sorted(self.last_waves.items()):
                if len(w.t) < 2:
                    continue
                v1 = float(np.interp(t1, w.t, w.v)) if w.t[0] <= t1 <= w.t[-1] else float("nan")
                v2 = float(np.interp(t2, w.t, w.v)) if w.t[0] <= t2 <= w.t[-1] else float("nan")
                line += (f"　｜　<span style='color:{CH_COLORS[ch]}'><b>CH{ch}</b></span> "
                         f"@T1 {fmt_eng(v1, 'V')}，@T2 {fmt_eng(v2, 'V')}，Δ {fmt_eng(v2 - v1, 'V')}")
            parts.append(line)
        if self.chk_vcur.isChecked():
            v1, v2 = (ln.value() for ln in self.vcur)
            parts.append(f"<b style='color:#69F0AE'>V1</b> {fmt_eng(v1, 'V')}　<b style='color:#E040FB'>V2</b> "
                         f"{fmt_eng(v2, 'V')}　<b>ΔV</b> {fmt_eng(v2 - v1, 'V')}")
        self.lbl_cursor.setText("<br>".join(parts))

    # ---------------- FFT 頻譜 ----------------
    def _fft_db(self):
        return self.cmb_fft_unit.currentIndex() == 0

    def _fft_logx(self):
        return self.cmb_fft_x.currentIndex() == 1

    @staticmethod
    def _to_unit(amp_peak, db):
        vrms = np.asarray(amp_peak) / math.sqrt(2)
        return 20 * np.log10(np.maximum(vrms, 1e-9)) if db else vrms

    def update_fft(self, waves, accumulate=True):
        win = self.cmb_fft_win.currentText()
        zp = self.cmb_fft_pad.currentData()
        navg = self.spn_fft_avg.value()
        self.fft_data = {}
        for ch in (1, 2):
            w = waves.get(ch)
            if w is None or len(w.v) < 8:
                self.fft_avg.pop(ch, None)
                continue
            f, amp = spectrum(w.t, w.v, win, zp)
            buf = self.fft_avg.get(ch, [])
            if buf and (len(buf[0][0]) != len(f) or not np.isclose(buf[0][0][-1], f[-1])):
                buf = []                       # 頻率軸變了（改時基/選取範圍）→ 重新平均
            buf = (buf + [(f, amp ** 2)])[-navg:] if accumulate else [(f, amp ** 2)]
            self.fft_avg[ch] = buf
            self.fft_data[ch] = (f, np.sqrt(np.mean([p for _, p in buf], axis=0)))
        self._redraw_fft()

    def _redraw_fft(self):
        db, win = self._fft_db(), self.cmb_fft_win.currentText()
        zp = self.cmb_fft_pad.currentData()
        bw = FFT_WINDOWS[win][1]
        axis = self.fft_plot.getAxis("left")
        axis.enableAutoSIPrefix(not db)
        self.fft_plot.setLabel("left", "振幅", units="dBV" if db else "Vrms")
        html, info = [], ""
        for ch in (1, 2):
            if ch not in self.fft_data:
                self.fft_curves[ch].setData([], [])
                self.fft_marks[ch].setData([], [])
                continue
            f, a = self.fft_data[ch]
            y = self._to_unit(a, db)
            self.fft_curves[ch].setData(f[1:], y[1:])          # 不畫 DC
            idx = find_peaks(f, a, count=5, min_sep_bins=2 * bw * zp)
            self.fft_marks[ch].setData(f[idx], y[idx])
            if not len(idx):
                continue
            ref = a[idx[0]]
            items = []
            for i in idx:
                rel = 20 * math.log10(a[i] / ref) if a[i] > 0 and ref > 0 else float("nan")
                val = f"{y[i]:.1f} dBV" if db else fmt_eng(y[i], "Vrms")
                items.append(f"{fmt_eng(f[i], 'Hz')}：{val}" + (f"（{rel:.1f} dBc）" if i != idx[0] else ""))
            line = (f"<span style='color:{CH_COLORS[ch]}'><b>CH{ch}</b></span> 峰值　" + "　｜　".join(items))
            harm = harmonics(f, a, refine_peak(f, a, idx[0]), 10, search_bins=zp + 1, lobe_bins=bw * zp)
            d, upper = thd(harm, NOISE_MULT * noise_floor(a))
            if not harm:
                line += "<br>　　THD：頻率解析度不足（請加大時基，讓畫面多幾個週期）"
            elif np.isfinite(d):
                line += (f"<br>　　THD（{len(harm)} 次諧波內）{'&lt;' if upper else '='} {d * 100:.3g} %"
                         f"（{20 * math.log10(max(d, 1e-12)):.1f} dB）"
                         + ("　<i>諧波都在雜訊以下</i>" if upper else ""))
            html.append(line)
            n = len(f)
            df_real = f[-1] * 2 / ((n - 1) * 2 / zp) if n > 1 else float("nan")   # = fs / N_原始
            info = (f"取樣率 {fmt_eng(2 * f[-1], 'Sa/s')}　最高頻率 {fmt_eng(f[-1], 'Hz')}　"
                    f"解析度 Δf ≈ {fmt_eng(df_real, 'Hz')}　平均 {len(self.fft_avg.get(ch, []))}/{self.spn_fft_avg.value()}")
        self._fft_peak_html = "<br>".join(html) if html else "—"
        self.lbl_fft_info.setText(info)
        self.update_fft_readout()

    def _fft_settings_changed(self, *_):
        self.fft_avg.clear()
        if self.last_waves:
            self.update_fft(self._selected(self.last_waves), accumulate=True)

    def _fft_xmode_changed(self, *_):
        logx = self._fft_logx()
        v = self.fft_cursor.value()
        self.fft_plot.setLogMode(x=logx, y=False)
        self.fft_cursor.setValue(math.log10(max(v, 1e-3)) if logx else 10 ** v)
        self._redraw_fft()
        self.fft_plot.getViewBox().enableAutoRange()

    def _toggle_fft_cursor(self, on):
        if on:
            pos = None
            for ch in (1, 2):
                if ch in self.fft_data:
                    f, a = self.fft_data[ch]
                    idx = find_peaks(f, a, count=1)
                    if len(idx):
                        pos = f[idx[0]]
                        break
            if pos is None:
                x0, x1 = self.fft_plot.getViewBox().viewRange()[0]
                self.fft_cursor.setValue((x0 + x1) / 2)
            else:
                self.fft_cursor.setValue(math.log10(pos) if self._fft_logx() else pos)
        self.fft_cursor.setVisible(on)
        self.update_fft_readout()

    def update_fft_readout(self, *_):
        text = getattr(self, "_fft_peak_html", "—")
        if self.chk_fft_cursor.isChecked():
            v = self.fft_cursor.value()
            fc = 10 ** v if self._fft_logx() else v
            line = f"<b style='color:#FF5252'>游標 F</b> {fmt_eng(fc, 'Hz')}"
            for ch, (f, a) in sorted(self.fft_data.items()):
                if len(f) > 1:
                    i = int(np.clip(round(fc / (f[1] - f[0])), 0, len(f) - 1))
                    y = self._to_unit(a[i], self._fft_db())
                    val = f"{float(y):.1f} dBV" if self._fft_db() else fmt_eng(float(y), "Vrms")
                    line += f"　｜　<span style='color:{CH_COLORS[ch]}'><b>CH{ch}</b></span> {val}"
            text = line + "<br>" + text
        self.lbl_fft.setText(text)

    def save_fft_csv(self):
        if not self.fft_data:
            QtWidgets.QMessageBox.information(self, "儲存", "目前沒有頻譜資料。")
            return
        base = self._base_name("fft")
        chs = sorted(self.fft_data)
        same_f = len({len(self.fft_data[c][0]) for c in chs}) == 1
        with open(base + ".csv", "w", newline="", encoding="utf-8-sig") as fp:
            for line in self._meta_lines():
                fp.write(line + "\n")
            fp.write(f"# FFT 視窗,{self.cmb_fft_win.currentText()}\n# 補零,{self.cmb_fft_pad.currentData()}\n"
                     f"# 平均張數,{self.spn_fft_avg.value()}\n# 振幅定義,Vrms = Vpeak/√2；dBV = 20·log10(Vrms/1V)\n")
            wr = csv.writer(fp)
            if same_f:
                wr.writerow(["freq_Hz"] + sum([[f"CH{c}_Vrms", f"CH{c}_dBV"] for c in chs], []))
                cols = [self.fft_data[chs[0]][0]]
            else:
                wr.writerow(sum([[f"CH{c}_freq_Hz", f"CH{c}_Vrms", f"CH{c}_dBV"] for c in chs], []))
                cols = []
            for c in chs:
                f, a = self.fft_data[c]
                cols += ([] if same_f else [f]) + [self._to_unit(a, False), self._to_unit(a, True)]
            n = max(len(c) for c in cols)
            for i in range(n):
                wr.writerow([f"{c[i]:.9g}" if i < len(c) else "" for c in cols])
        self.lbl_saved.setText(f"已儲存：\n{base}.csv")
        self.statusBar().showMessage(f"已儲存 {os.path.basename(base)}.csv")

    def save_fft_png(self):
        base = self._base_name("fft")
        self.fft_tab.grab().save(base + ".png")
        self.lbl_saved.setText(f"已儲存：\n{base}.png")
        self.statusBar().showMessage(f"已儲存 {os.path.basename(base)}.png")

    def update_measurements(self, waves, full=None):
        """量測 waves；頻率一律用完整波形 full 估計（選取範圍太短時才不會失準）。"""
        parts = []
        if self.chk_sel.isChecked() and self.chk_tcur.isChecked():
            parts.append("<i>（以下量測只計算 T1~T2 之間）</i>")
        freqs = {}
        for ch, w in sorted(waves.items()):
            ref = (full or waves).get(ch, w)
            f = estimate_frequency(ref.t, ref.v)
            freqs[ch] = f
            parts.append(f"<span style='color:{CH_COLORS[ch]}'><b>CH{ch}</b></span> "
                         f"Vpp {fmt_eng(vpp(w.v), 'V')}　頻率 {fmt_eng(f, 'Hz')}　平均 {fmt_eng(float(np.mean(w.v)), 'V')}")
        if 1 in waves and 2 in waves:
            f = freqs.get(1)
            if not np.isfinite(f):
                f = freqs.get(2)
            a1, p1 = lockin(waves[1].t, waves[1].v, f)
            a2, p2 = lockin(waves[2].t, waves[2].v, f)
            if np.isfinite(a1) and a1 > 0 and a2 > 0:
                parts.append(f"<b>CH2/CH1</b> 增益 {20 * math.log10(a2 / a1):.2f} dB（{a2 / a1:.4g} 倍）　"
                             f"相位差 {wrap_deg(math.degrees(p2 - p1)):.1f}°")
        self.lbl_meas.setText("<br>".join(parts) if parts else "—")

    # ---------------- 示波器設定 ----------------
    def _scope_cmd(self, name):
        if not self.scope:
            return
        try:
            getattr(self.scope, name)()
            self.statusBar().showMessage(f"示波器：{name.upper()}")
        except Exception as e:
            self.statusBar().showMessage(f"指令失敗：{e}")

    def on_autoset(self):
        self._scope_cmd("autoset")
        QtCore.QTimer.singleShot(3000 if not self.chk_sim.isChecked() else 100, self.sync_scope_settings)

    def on_channel_toggled(self, ch, on):
        if self.live:
            self.live.channels = frozenset(self._channels())
        if not on:
            self.curves[ch].setData([], [])
        if self.scope:
            try:
                self.scope.set_ch_display(ch, on)
            except Exception as e:
                self.statusBar().showMessage(f"設定 CH{ch} 失敗：{e}")

    def on_timebase_changed(self, _i):
        if self.scope:
            try:
                self.scope.set_timebase(self.cmb_time.currentData())
            except Exception as e:
                self.statusBar().showMessage(f"設定時基失敗：{e}")

    def on_vscale_changed(self, ch):
        if self.scope:
            try:
                self.scope.set_ch_scale(ch, self.cmb_volt[ch].currentData())
            except Exception as e:
                self.statusBar().showMessage(f"設定 CH{ch} 刻度失敗：{e}")

    # ---------------- 函數產生器 ----------------
    def freq_hz(self):
        return self.spn_freq.value() * self.cmb_funit.currentData()

    def set_freq_hz(self, f):
        idx = 2 if f >= 1e6 else 1 if f >= 1e3 else 0
        self.cmb_funit.setCurrentIndex(idx)
        self.spn_freq.blockSignals(True)
        self.spn_freq.setValue(f / self.cmb_funit.itemData(idx))
        self.spn_freq.blockSignals(False)

    def _check_freq(self, f):
        lo, hi = AFG_FREQ_RANGE
        if not lo <= f <= hi:
            QtWidgets.QMessageBox.warning(self, "頻率超出範圍", f"AFG-2112 頻率範圍為 {fmt_eng(lo, 'Hz')} ~ {fmt_eng(hi, 'Hz')}")
            return False
        return True

    def apply_afg(self):
        if not self.afg:
            return
        f = self.freq_hz()
        if not self._check_freq(f):
            return
        try:
            self.afg.set_function(self.cmb_func.currentText())
            self.afg.set_frequency(f)
            self.afg.set_amplitude(self.spn_amp.value())
            self.afg.set_offset(self.spn_off.value())
            err = self.afg.check_error() if hasattr(self.afg, "check_error") else None
            if err:
                QtWidgets.QMessageBox.warning(self, "產生器回報錯誤", f"產生器拒絕了部分設定：{err}\n請確認數值在產生器允許的範圍內。")
            self.statusBar().showMessage(
                f"產生器：{self.cmb_func.currentText()} {fmt_eng(f, 'Hz')}，{self.spn_amp.value():g} Vpp，偏移 {self.spn_off.value():g} V")
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self, "超出安全限制", str(e))
        except Exception as e:
            self.statusBar().showMessage(f"設定產生器失敗：{e}")

    def on_load_changed(self, *_):
        if self.afg:
            try:
                self.afg.set_load(self.afg_high_z())
                self.statusBar().showMessage(f"產生器負載 → {self.cmb_load.currentText()}")
            except Exception as e:
                self.statusBar().showMessage(f"設定負載失敗：{e}")

    def afg_high_z(self):
        return bool(self.cmb_load.currentData())

    def _maybe_apply_freq(self):
        if self.afg and self.chk_auto_apply.isChecked():
            f = self.freq_hz()
            if self._check_freq(f):
                try:
                    self.afg.set_frequency(f)
                    self.statusBar().showMessage(f"產生器頻率 → {fmt_eng(f, 'Hz')}")
                except Exception as e:
                    self.statusBar().showMessage(f"設定頻率失敗：{e}")

    def scale_frequency(self, factor):
        f = self.freq_hz() * factor
        lo, hi = AFG_FREQ_RANGE
        f = min(max(f, lo), hi)
        self.set_freq_hz(f)
        if self.afg:
            try:
                self.afg.set_frequency(f)
                self.statusBar().showMessage(f"產生器頻率 → {fmt_eng(f, 'Hz')}")
            except Exception as e:
                self.statusBar().showMessage(f"設定頻率失敗：{e}")

    def on_output_toggled(self, on):
        self.btn_output.setText(f"輸出：{'ON' if on else 'OFF'}")
        self.btn_output.setStyleSheet("background:#2e7d32; color:white;" if on else "")
        if self.afg:
            try:
                self.afg.set_output(on)
            except Exception as e:
                self.statusBar().showMessage(f"切換輸出失敗：{e}")

    # ---------------- 儲存 ----------------
    def browse_dir(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "選擇儲存資料夾", self.txt_dir.text())
        if d:
            self.txt_dir.setText(d)

    def _ensure_dir(self):
        d = self.txt_dir.text().strip() or os.path.join(HERE, config.DATA_DIR)
        os.makedirs(d, exist_ok=True)
        return d

    def _base_name(self, prefix):
        """單張波形、FFT 的檔名：<儲存資料夾>/<config.WAVE_SUBDIR>/<prefix>_<時間>。"""
        d = os.path.join(self._ensure_dir(), config.WAVE_SUBDIR)
        os.makedirs(d, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(d, f"{prefix}_{stamp}")

    def _meta_lines(self):
        lines = [f"# 儲存時間,{datetime.datetime.now().isoformat(timespec='seconds')}"]
        if self.txt_note.text().strip():
            lines.append(f"# 備註,{self.txt_note.text().strip()}")
        if self.chk_sel.isChecked() and self.chk_tcur.isChecked():
            a, b = self._t_range()
            lines.append(f"# 資料範圍,只含 T1~T2：{a:.9g} s ~ {b:.9g} s")
        lines.append(f"# 時基,{self.cmb_time.currentText()}")
        for ch in (1, 2):
            lines.append(f"# CH{ch} 垂直刻度,{self.cmb_volt[ch].currentText()}")
        if self.afg:
            lines.append(f"# 產生器設定,{self.cmb_func.currentText()} {fmt_eng(self.freq_hz(), 'Hz')} "
                         f"{self.spn_amp.value():g} Vpp 偏移 {self.spn_off.value():g} V")
        return lines

    def save_waveform(self, csv_=True, png=True):
        waves = self._selected(self.last_waves)
        if not waves:
            QtWidgets.QMessageBox.information(self, "儲存", "目前沒有波形資料。")
            return
        base = self._base_name(self.txt_prefix.text().strip() or "wave")
        saved = []
        try:
            if csv_:
                path = base + ".csv"
                chs = sorted(waves)
                same_t = len({len(waves[c].t) for c in chs}) == 1 and all(
                    np.allclose(waves[c].t, waves[chs[0]].t) for c in chs)
                with open(path, "w", newline="", encoding="utf-8-sig") as fp:
                    for line in self._meta_lines():
                        fp.write(line + "\n")
                    wr = csv.writer(fp)
                    if same_t:
                        wr.writerow(["time_s"] + [f"CH{c}_V" for c in chs])
                        cols = [waves[chs[0]].t] + [waves[c].v for c in chs]
                    else:
                        wr.writerow(sum([[f"CH{c}_time_s", f"CH{c}_V"] for c in chs], []))
                        cols = sum([[waves[c].t, waves[c].v] for c in chs], [])
                    n = max(len(c) for c in cols)
                    for i in range(n):
                        wr.writerow([f"{c[i]:.9g}" if i < len(c) else "" for c in cols])
                saved.append(path)
            if png:
                path = base + ".png"
                self.plot_tabs.widget(0).grab().save(path)
                saved.append(path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "儲存失敗", str(e))
            return
        self.lbl_saved.setText("已儲存：\n" + "\n".join(saved))
        self.statusBar().showMessage("已儲存 " + "、".join(os.path.basename(p) for p in saved))

    def closeEvent(self, ev):
        self.disconnect_instruments()
        super().closeEvent(ev)


def _fix_qt_plugin_path():
    """路徑含中文時 Qt 會把它變成 '????' 而找不到 platform plugin，這裡由 Python 直接指定。"""
    import PyQt5
    plugins = os.path.join(os.path.dirname(PyQt5.__file__), "Qt5", "plugins")
    if os.path.isdir(plugins):
        QtCore.QCoreApplication.setLibraryPaths([plugins])


def main():
    _fix_qt_plugin_path()
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
    app = QtWidgets.QApplication(sys.argv)
    app.setFont(QtGui.QFont("Microsoft JhengHei UI", 10))
    pg.setConfigOptions(antialias=True)
    win = MainWindow()
    if "--sim" in sys.argv:
        win.chk_sim.setChecked(True)
        win.connect_instruments()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

"""實驗流程（在背景執行緒執行）：
- 共振掃頻：往上 / 往下 / 往上再往下、對數 / 線性、自動兩階段（粗掃 → 峰附近細掃）
- 批次參數掃描：對每個振幅（或 DC offset）各做一次掃頻，輸出總表
- 分岔圖：固定頻率逐步加大振幅，每週期取一個值，判斷週期 1 / 2 / 4 / 8 / 混沌
"""
import cmath
import copy
import csv
import math
import os
import time
from dataclasses import asdict, dataclass

import numpy as np
from PyQt5 import QtCore

import config
from analysis import (POW2_ORDERS, amplitude_rel_error, analyze_resonance, branch_stats, classify_period,
                      detect_period, feigenbaum_analysis, fmt_eng, lockin, lyapunov_kantz, SPEC_GRID,
                      spectrum_map_row,
                      per_cycle_values,
                      period_label, pick_scale,
                      subharmonic_dbc, suggest_vscale, vpp, wrap_deg)
from instruments import AFG_FREQ_RANGE, TIME_SCALES, VOLT_SCALES

DIR_NAMES = {"up": "往上", "down": "往下"}
STOP_ACTIONS = {"keep": "保持輸出", "min": "降到最小振幅", "off": "關閉輸出"}


class StopRequested(Exception):
    pass


@dataclass
class SweepSettings:
    f_start: float = 1.0e6
    f_stop: float = 2.5e6
    n_coarse: int = 31
    spacing: str = "log"          # "log" / "lin"
    two_stage: bool = True
    fine_pct: float = 5.0         # 細掃範圍：峰 × (1 ± fine_pct %)
    n_fine: int = 41
    direction: str = "up"         # "up" / "down" / "updown"
    periods: float = 4.0          # 畫面上顯示幾個週期
    settle: float = 0.5           # 每點等待 (s)
    auto_v: bool = True
    sw_avg: int = config.DEFAULT_SW_AVG   # 程式平均：每點讀幾張，各自鎖相後平均（不受觸發抖動影響）
    scope_avg: int = config.DEFAULT_SCOPE_AVG   # 示波器硬體平均次數（1 = 關）
    auto_ac: bool = False         # 開始前設 AC 耦合、關頻寬限制
    L: float = config.DEFAULT_L_UH * 1e-6
    stop_action: str = "keep"

    def directions(self):
        return ["up", "down"] if self.direction == "updown" else [self.direction]

    def total_points(self):
        nd = len(self.directions())
        return self.n_coarse + self.n_fine * nd if self.two_stage else self.n_coarse * nd


def make_freqs(f0, f1, n, spacing):
    lo, hi = sorted((float(f0), float(f1)))
    lo, hi = max(lo, AFG_FREQ_RANGE[0]), min(hi, AFG_FREQ_RANGE[1])
    n = max(int(n), 2)
    return np.geomspace(lo, hi, n) if spacing == "log" else np.linspace(lo, hi, n)


def apply_stop_action(afg, action):
    if not afg:
        return
    if action == "min":
        afg.set_amplitude(config.AFG_MIN_VPP)
    elif action == "off":
        afg.set_output(False)


# ======================================================================
# 量測一個頻率點
# ======================================================================

class Measurer:
    def __init__(self, scope, afg, s, stop_flag):
        self.scope, self.afg, self.s, self.stop_flag = scope, afg, s, stop_flag

    def wait(self, sec):
        """可中斷的等待。"""
        end = time.time() + sec
        while True:
            if self.stop_flag():
                raise StopRequested
            left = end - time.time()
            if left <= 0:
                return
            time.sleep(min(0.05, left))

    def acquire(self):
        self.scope.stop()                 # 停下來再讀，確保兩個通道是同一次擷取
        try:
            return self.scope.read_waveform(1), self.scope.read_waveform(2)
        finally:
            self.scope.run()

    def measure(self, f, direction, stage):
        s = self.s
        self.afg.set_frequency(f)
        tscale = pick_scale(TIME_SCALES, s.periods / (10 * f))
        self.scope.set_timebase(tscale)
        frame = 10 * tscale + 0.01
        wait = s.settle + 2 * frame + (s.scope_avg * frame if s.scope_avg > 1 else 0)
        self.wait(wait)
        for _ in range(6):
            w1, w2 = self.acquire()
            if not s.auto_v:
                break
            changed = False
            for ch, w in ((1, w1), (2, w2)):
                new = suggest_vscale(w, VOLT_SCALES, config.SCREEN_HALF_DIVS)
                if new is not None:
                    self.scope.set_ch_scale(ch, new)
                    changed = True
            if not changed:
                break
            self.wait(wait)
        frames = [(w1, w2)]
        for _ in range(max(int(s.sw_avg), 1) - 1):
            self.wait(frame)
            frames.append(self.acquire())

        # 資料不完整的張數（傳輸掉了一塊）：有完整的就只用完整的；全部不完整才標記，留待掃描結束後補測
        good = [fr for fr in frames if not (fr[0].incomplete or fr[1].incomplete)]
        incomplete = not good
        if good:
            frames = good
            w1, w2 = frames[-1]

        # 程式平均：每一張各自做鎖相，CH2 以同一張 CH1 的相位為基準再平均。
        # 小訊號時觸發會抖動（實測每張相位差到 ±35°），直接把波形逐點平均會讓振幅變小；
        # 這樣做則振幅、增益、相位都不受觸發抖動影響。
        amp1, rel2 = [], []
        for a, b in frames:
            a1_, p1_ = lockin(a.t, a.v, f)
            a2_, p2_ = lockin(b.t, b.v, f)
            amp1.append(a1_)
            rel2.append(a2_ * cmath.exp(1j * (p2_ - p1_)))
        amp1, rel2 = np.array(amp1), np.array(rel2)
        n = len(frames)
        a1 = float(np.mean(amp1))
        h = complex(np.mean(rel2))
        a2 = abs(h)
        ok = a1 > 0 and a2 > 0
        if n >= 3:       # 用張與張之間的離散程度估計誤差
            err1 = float(np.std(amp1, ddof=1) / math.sqrt(n) / a1) if a1 > 0 else float("inf")
            err2 = float(math.sqrt(np.sum(np.abs(rel2 - h) ** 2) / (n - 1)) / math.sqrt(n) / a2) if a2 > 0 \
                else float("inf")
        else:            # 張數太少：用殘差雜訊估計
            err1 = amplitude_rel_error(w1.t, w1.v, f) / math.sqrt(n)
            err2 = amplitude_rel_error(w2.t, w2.v, f) / math.sqrt(n)
        div1 = 2 * a1 / w1.vscale if w1.vscale else 0.0
        div2 = 2 * a2 / w2.vscale if w2.vscale else 0.0
        clip = max(float(np.mean(np.abs(w.screen_div) >= config.ADC_LIMIT_DIVS)) for w in (w1, w2))
        reasons = []
        if not ok:
            reasons.append("沒有訊號")
        for name, err, div in (("CH1", err1, div1), ("CH2", err2, div2)):
            if not err <= config.MAX_REL_ERROR:
                reasons.append(f"{name} 誤差 {err * 100:.0f}%")
            elif div < config.MIN_SIGNAL_DIVS:
                reasons.append(f"{name} 只有 {div:.2f} 格")
        if clip > 0.01:
            reasons.append(f"波形超出範圍 {clip * 100:.0f}%")
        if incomplete:
            reasons.append("資料不完整（傳輸遺失）")
        return dict(
            freq_Hz=float(f), direction=direction, stage=stage,
            gain_dB=20 * math.log10(a2 / a1) if ok else float("nan"),
            phase_deg=wrap_deg(math.degrees(cmath.phase(h))) if ok else float("nan"),
            CH1_amp_V=a1, CH2_amp_V=a2, CH1_Vpp=2 * a1, CH2_Vpp=2 * a2,
            CH1_div_pp=div1, CH2_div_pp=div2, CH1_rel_err=err1, CH2_rel_err=err2, n_frames=n,
            unreliable=bool(reasons), unreliable_reason="；".join(reasons), incomplete=incomplete, redone=False)


def run_resonance_sweep(meas, s, extra, on_point, on_status):
    """依設定執行掃頻（含兩階段與方向）。回傳 (points, analyses)。"""
    pts = []

    def sweep(freqs, direction, stage, approach):
        seq = list(freqs) if direction == "up" else list(freqs)[::-1]
        if approach:     # 先停在掃描起點外側，讓電路從正確的一側接近（遲滯）
            pad = 1 - 2 * s.fine_pct / 100 if direction == "up" else 1 + 2 * s.fine_pct / 100
            fa = float(np.clip(seq[0] * pad, *AFG_FREQ_RANGE))
            on_status(f"{stage}：移到起點外側 {fmt_eng(fa, 'Hz')}")
            meas.afg.set_frequency(fa)
            meas.wait(max(s.settle, 0.2))
        for i, f in enumerate(seq):
            on_status(f"{stage}（{DIR_NAMES[direction]}）{fmt_eng(f, 'Hz')}　{i + 1}/{len(seq)}")
            p = meas.measure(f, direction, stage)
            p.update(extra)
            pts.append(p)
            on_point(p)

    dirs = s.directions()
    if s.two_stage:
        first = "down" if s.direction == "down" else "up"
        sweep(make_freqs(s.f_start, s.f_stop, s.n_coarse, s.spacing), first, "粗掃", False)
        cand = [p for p in pts if np.isfinite(p["gain_dB"]) and not p["unreliable"]] or \
               [p for p in pts if np.isfinite(p["gain_dB"])]
        if not cand:
            raise RuntimeError("粗掃沒有任何有效的點，請檢查接線與示波器設定")
        fp = max(cand, key=lambda p: p["gain_dB"])["freq_Hz"]
        x = s.fine_pct / 100
        fine = make_freqs(fp * (1 - x), fp * (1 + x), s.n_fine, "lin")
        on_status(f"粗掃峰值約 {fmt_eng(fp, 'Hz')}，開始細掃 ±{s.fine_pct:g}%")
        for d in dirs:
            sweep(fine, d, "細掃", True)
    else:
        freqs = make_freqs(s.f_start, s.f_stop, s.n_coarse, s.spacing)
        for k, d in enumerate(dirs):
            sweep(freqs, d, "掃描", k > 0)

    # 補測資料不完整的點：先停在掃描方向的前一側，再回到該頻率重量，結果取代原本的點
    for rnd in range(config.REDO_ROUNDS if config.AUTO_REDO_INCOMPLETE else 0):
      bad = [i for i, p in enumerate(pts) if p.get("incomplete")]
      if not bad:
          break
      for k, i in enumerate(bad):
        old = pts[i]
        f, d = old["freq_Hz"], old["direction"]
        on_status(f"補測資料不完整的點 {fmt_eng(f, 'Hz')}（{k + 1}/{len(bad)}）")
        meas.afg.set_frequency(float(np.clip(f * (0.99 if d == "up" else 1.01), *AFG_FREQ_RANGE)))
        meas.wait(max(s.settle, 0.2))
        p = meas.measure(f, d, old["stage"])
        p.update(extra)
        p["redone"], p["redo_of"] = True, i
        pts[i] = p
        on_point(p)
    return pts, analyze_points(pts, s.L)


def analyze_points(pts, L):
    out = {}
    for d in ("up", "down"):
        sel = [p for p in pts if p["direction"] == d]
        if sel:
            out[d] = analyze_resonance([p["freq_Hz"] for p in sel], [p["gain_dB"] for p in sel],
                                       [p["phase_deg"] for p in sel], [p["unreliable"] for p in sel], L)
    return out


def amplitude_check(pts):
    """CH1 實測 vs 產生器設定：取整個掃描中 CH1 最大的振幅（遠離共振時最接近開路電壓）。
    回傳警告字串或 None。"""
    good = [p for p in pts if np.isfinite(p.get("CH1_Vpp", np.nan)) and p.get("AFG_set_Vpp")]
    if not good:
        return None
    p = max(good, key=lambda q: q["CH1_Vpp"])
    expect = p["AFG_set_Vpp"] * (1.0 if p.get("AFG_load") == "High-Z" else 2.0)
    ratio = p["CH1_Vpp"] / expect
    if abs(ratio - 1) > config.AMP_MISMATCH_WARN:
        return (f"CH1 實測最大 {p['CH1_Vpp']:.3g} Vpp，但產生器設定 {p['AFG_set_Vpp']:g} Vpp、負載 {p['AFG_load']}"
                f"（預期開路約 {expect:.3g} Vpp），相差 {ratio:.2f} 倍。"
                "請確認產生器的負載設定（50 Ω / High-Z）與探棒倍率。")
    return None


# ======================================================================
# CSV
# ======================================================================

POINT_FIELDS = ["freq_Hz", "direction", "stage", "gain_dB", "phase_deg", "CH1_amp_V", "CH2_amp_V", "CH1_Vpp",
                "CH2_Vpp", "CH1_div_pp", "CH2_div_pp", "CH1_rel_err", "CH2_rel_err", "n_frames", "AFG_set_Vpp", "AFG_offset_V",
                "AFG_load", "unreliable", "unreliable_reason", "incomplete", "redone"]


def _fmt(x):
    if isinstance(x, bool):
        return "1" if x else "0"
    if isinstance(x, (float, np.floating)):
        return f"{x:.9g}"
    return str(x)


def analysis_lines(analyses):
    lines = []
    for d, a in analyses.items():
        tag = f"# [{DIR_NAMES[d]}]"
        lines += [f"{tag} f_res_peak_Hz,{_fmt(a['f_peak'])}",
                  f"{tag} f_res_phase0_Hz,{_fmt(a['f_phase0'])}",
                  f"{tag} peak_gain_dB,{_fmt(a['gain_peak_db'])}",
                  f"{tag} bw_3dB_Hz,{_fmt(a['bw'])}",
                  f"{tag} Q,{_fmt(a['Q'])}",
                  f"{tag} C_eff_F（由增益峰）,{_fmt(a['C_eff'])}",
                  f"{tag} C_eff_F（由相位過零）,{_fmt(a['C_eff_phase'])}",
                  f"{tag} 不可靠點數,{a['n_unreliable']}/{a['n']}"]
    return lines


def write_sweep_csv(path, pts, analyses, meta_lines, s, warning=None):
    with open(path, "w", newline="", encoding="utf-8-sig") as fp:
        for line in meta_lines:
            fp.write(line + "\n")
        fp.write("# 掃描設定," + "；".join(f"{k}={v}" for k, v in asdict(s).items()) + "\n")
        for line in analysis_lines(analyses):
            fp.write(line + "\n")
        if s.scope_avg > 1:
            fp.write("# 注意,使用示波器平均時若觸發不穩，CH1/CH2 的絕對振幅可能偏小；增益與相位不受影響\n")
        if warning:
            fp.write(f"# 警告,{warning}\n")
        fields = POINT_FIELDS + [k for k in (pts[0].keys() if pts else []) if k not in POINT_FIELDS]
        wr = csv.writer(fp)
        wr.writerow(fields)
        for p in pts:
            wr.writerow([_fmt(p.get(k, "")) for k in fields])


# ======================================================================
# 背景執行緒
# ======================================================================

class _Worker(QtCore.QThread):
    status = QtCore.pyqtSignal(str)
    progress = QtCore.pyqtSignal(int, int)
    error = QtCore.pyqtSignal(str)
    result = QtCore.pyqtSignal(object)

    def __init__(self, scope, afg):
        super().__init__()
        self.scope, self.afg = scope, afg
        self._stop = False
        self.stop_action = "keep"

    def stop(self):
        self._stop = True

    def stopped(self):
        return self._stop

    def _prepare_scope(self, s):
        for ch in (1, 2):
            self.scope.set_ch_display(ch, True)
            if s.auto_v:
                # 垂直位置的單位是伏特：不歸零的話，波形會被位置推到螢幕邊緣，自動刻度就不敢縮小，
                # 縮小後波形也會跑出螢幕外。
                self.scope.set_position(ch, 0.0)
            if s.auto_ac:
                self.scope.set_coupling(ch, "AC")
                self.scope.set_bwlimit(ch, False)
        if s.scope_avg > 1:
            self.scope.set_acquire_average(s.scope_avg)
        time.sleep(0.5)                 # 讓示波器套用新設定，避免第一點讀到舊波形

    def _finish(self, s, stopped):
        if s.scope_avg > 1:
            try:
                self.scope.set_acquire_average(1)
            except Exception:
                pass
        if stopped:
            try:
                apply_stop_action(self.afg, self.stop_action)
            except Exception as e:
                self.status.emit(f"停止時調整產生器失敗：{e}")

    def _afg_info(self, amp, offset, high_z):
        return dict(AFG_set_Vpp=amp, AFG_offset_V=offset, AFG_load="High-Z" if high_z else "50Ω")


class SweepWorker(_Worker):
    """單次共振掃頻。"""
    point = QtCore.pyqtSignal(object)

    def __init__(self, scope, afg, s, amp, offset, high_z):
        super().__init__(scope, afg)
        self.s, self.amp, self.offset, self.high_z = s, amp, offset, high_z
        self.stop_action = s.stop_action
        self._n = 0

    def _on_point(self, p):
        self._n += 1
        self.point.emit(p)
        self.progress.emit(self._n, self.s.total_points())

    def run(self):
        s, pts, analyses, stopped = self.s, [], {}, False
        meas = Measurer(self.scope, self.afg, s, self.stopped)
        try:
            self._prepare_scope(s)
            self.afg.set_output(True)
            pts, analyses = run_resonance_sweep(meas, s, self._afg_info(self.amp, self.offset, self.high_z),
                                                self._on_point, self.status.emit)
            self.status.emit("掃頻完成")
        except StopRequested:
            stopped = True
            self.status.emit("已停止")
        except Exception as e:
            stopped = True
            self.error.emit(f"掃頻失敗：{e}")
        finally:
            self._finish(s, stopped)
        if pts and not analyses:
            analyses = analyze_points(pts, s.L)
        self.result.emit(dict(points=pts, analyses=analyses, warning=amplitude_check(pts) if s.scope_avg <= 1 else None,
                                settings=s))


class BatchWorker(_Worker):
    """外層參數掃描：對每個振幅（kind="amp"）或 DC offset（kind="offset"）各做一次掃頻。"""
    point = QtCore.pyqtSignal(object)
    value_started = QtCore.pyqtSignal(float)
    value_done = QtCore.pyqtSignal(object)

    def __init__(self, scope, afg, s, kind, values, fixed_amp, fixed_offset, high_z, follow, outdir, meta_lines):
        super().__init__(scope, afg)
        self.s, self.kind, self.values = s, kind, list(values)
        self.fixed_amp, self.fixed_offset, self.high_z = fixed_amp, fixed_offset, high_z
        self.follow = follow            # None 或 (下限倍數, 上限倍數)
        self.outdir, self.meta_lines = outdir, meta_lines
        self.stop_action = s.stop_action
        self._n = 0

    def _total(self):
        return self.s.total_points() * len(self.values)

    def _on_point(self, p):
        self._n += 1
        self.point.emit(p)
        self.progress.emit(self._n, self._total())

    def run(self):
        s0 = self.s
        rows, warnings, stopped, prev_f = [], [], False, None
        unit = "Vpp" if self.kind == "amp" else "V"
        name = "振幅" if self.kind == "amp" else "DC offset"
        try:
            # 先檢查所有值都在安全範圍內，避免跑到一半才出錯
            from instruments import check_drive
            for v in self.values:
                check_drive(v if self.kind == "amp" else self.fixed_amp, self.fixed_offset if self.kind == "amp" else v)
            self._prepare_scope(s0)
            self.afg.set_output(True)
            for idx, val in enumerate(self.values):
                if self._stop:
                    raise StopRequested
                amp = val if self.kind == "amp" else self.fixed_amp
                off = self.fixed_offset if self.kind == "amp" else val
                self.value_started.emit(float(val))
                self.status.emit(f"[{idx + 1}/{len(self.values)}] 設定{name} = {val:g} {unit}")
                if self.kind == "amp":
                    self.afg.set_amplitude(amp)
                else:
                    self.afg.set_offset(off)
                s = copy.copy(s0)
                if self.follow and prev_f and np.isfinite(prev_f):
                    s.f_start, s.f_stop = prev_f * self.follow[0], prev_f * self.follow[1]
                meas = Measurer(self.scope, self.afg, s, self.stopped)
                meas.wait(max(s.settle, 0.3))
                extra = dict(param=val, **self._afg_info(amp, off, self.high_z))
                pts, analyses = run_resonance_sweep(
                    meas, s, extra, self._on_point,
                    lambda m, i=idx, v=val: self.status.emit(f"[{i + 1}/{len(self.values)}] {name} {v:g} {unit}：{m}"))
                warn = amplitude_check(pts) if s.scope_avg <= 1 else None
                if warn:
                    warnings.append(f"{name} {val:g} {unit}：{warn}")
                fname = os.path.join(self.outdir, f"{self.kind}_{val:g}{unit}.csv")
                write_sweep_csv(fname, pts, analyses, self.meta_lines + [f"# {name},{val:g} {unit}"], s, warn)
                new_rows = [dict(param=val, direction=DIR_NAMES[d], f_res_peak_Hz=a["f_peak"],
                                 f_res_phase0_Hz=a["f_phase0"], peak_gain_dB=a["gain_peak_db"], Q=a["Q"],
                                 C_eff_F=a["C_eff"], C_eff_phase_F=a["C_eff_phase"], bw_3dB_Hz=a["bw"],
                                 n_unreliable=a["n_unreliable"], n_points=a["n"],
                                 sweep_from_Hz=s.f_start, sweep_to_Hz=s.f_stop, file=os.path.basename(fname))
                            for d, a in analyses.items()]
                rows += new_rows
                first = analyses.get("up") or next(iter(analyses.values()), None)
                if first and np.isfinite(first["f_peak"]):
                    prev_f = first["f_peak"]
                self.value_done.emit(dict(value=val, rows=new_rows, points=pts, analyses=analyses))
            self.status.emit("批次掃描完成")
        except StopRequested:
            stopped = True
            self.status.emit("已停止")
        except Exception as e:
            stopped = True
            self.error.emit(f"批次掃描失敗：{e}")
        finally:
            self._finish(s0, stopped)
        summary = None
        if rows:
            summary = os.path.join(self.outdir, "summary.csv")
            fields = list(rows[0].keys())
            with open(summary, "w", newline="", encoding="utf-8-sig") as fp:
                for line in self.meta_lines:
                    fp.write(line + "\n")
                fp.write(f"# 外層參數,{name} ({unit})\n")
                fp.write("# 掃描設定," + "；".join(f"{k}={v}" for k, v in asdict(s0).items()) + "\n")
                for w in warnings:
                    fp.write(f"# 警告,{w}\n")
                wr = csv.writer(fp)
                wr.writerow(["param_" + unit] + fields[1:])
                for r in rows:
                    wr.writerow([_fmt(r[k]) for k in fields])
        self.result.emit(dict(rows=rows, warnings=warnings, outdir=self.outdir, summary=summary, kind=self.kind))


@dataclass
class BifSettings:
    freq: float = 1.0e6
    amp_start: float = 0.1
    amp_stop: float = 4.0
    steps: int = 40
    settle: float = 1.0
    record_length: int = config.BIF_RECORD_LENGTH
    n_frames: int = 1             # 每個振幅擷取幾張（A：取樣值合併判斷、B：功率頻譜平均）
    min_periods: int = 100
    method: str = "max"          # "max" / "phase"
    phase_deg: float = 0.0
    phase_ref: str = "ch2"       # "ch2" = 以 CH2 基頻波峰為相位 0（自動）；"ch1" = 以 CH1 驅動波峰為相位 0
    tol: float = config.BIF_TOL
    max_period: int = config.BIF_MAX_PERIOD
    chaos_method: str = config.BIF_CHAOS_METHOD
    stop_action: str = "keep"

    def amps(self):
        return np.linspace(self.amp_start, self.amp_stop, max(int(self.steps), 1))

    def periods(self):
        return tuple(p for p in config.BIF_PERIODS if p <= self.max_period)

    def pow2_divs(self):
        """要鎖相記錄的次諧波 f/2、f/4、…（到最高判斷週期，至少到 f/8）。"""
        return tuple(q for q in POW2_ORDERS if q <= max(self.max_period, 8))

    def cycles_needed(self):
        return max(int(self.min_periods), config.BIF_CYCLES_PER_PERIOD * int(self.max_period))


class BifurcationWorker(_Worker):
    amp_done = QtCore.pyqtSignal(object)

    def __init__(self, scope, afg, b):
        super().__init__(scope, afg)
        self.b = b
        self.stop_action = b.stop_action

    def _acquire(self, meas):
        for _ in range(5):
            w1, w2 = meas.acquire()
            changed = False
            for ch, w in ((1, w1), (2, w2)):
                new = suggest_vscale(w, VOLT_SCALES, config.SCREEN_HALF_DIVS, fill=config.BIF_SCALE_FILL)
                if new is not None:
                    self.scope.set_ch_scale(ch, new)
                    changed = True
            if not changed:
                break
            meas.wait(0.3 + 20 * self.tscale)
        return w1, w2

    def run(self):
        b = self.b
        stopped = False
        bench = getattr(self.scope, "bench", None)        # 模擬模式：打開倍週期模擬
        dummy = SweepSettings(settle=b.settle)
        meas = Measurer(self.scope, self.afg, dummy, self.stopped)
        results = []
        try:
            from instruments import check_drive
            check_drive(max(b.amp_start, b.amp_stop), 0.0)
            if bench is not None:
                bench.chaos = True
            for ch in (1, 2):
                self.scope.set_ch_display(ch, True)
                self.scope.set_position(ch, 0.0)       # 見 _prepare_scope 的說明
                self.scope.set_bwlimit(ch, config.BIF_BWLIMIT)
            self.scope.set_record_length(b.record_length)
            need = b.cycles_needed()      # 最高判斷週期越高，需要越多週期（f/p 譜線才分得開）
            self.tscale = pick_scale(TIME_SCALES, need / (10 * b.freq))
            self.scope.set_timebase(self.tscale)
            spp = b.record_length / (10 * self.tscale * b.freq)
            self.status.emit(f"時基 {fmt_eng(self.tscale, 's')}/div，畫面約 {10 * self.tscale * b.freq:.0f} 個週期"
                             + (f"（判斷到週期 {b.max_period} 需要 {need} 個）" if need > b.min_periods else "")
                             + f"，每週期約 {spp:.0f} 點" + ("（點數偏少，建議加大記錄長度）" if spp < 20 else ""))
            self.afg.set_frequency(b.freq)
            self.afg.set_output(True)
            amps = b.amps()

            def one(a):
                self.afg.set_amplitude(float(a))
                meas.wait(b.settle + 20 * self.tscale)
                w1, w2 = self._acquire(meas)
                # 多張擷取（刻度已調好，之後不再改）：每張各自處理，不把波形疊起來平均（會抵消倍週期）
                frames = [(w1, w2)]
                for _ in range(max(int(b.n_frames), 1) - 1):
                    meas.wait(0.05 + 20 * self.tscale)
                    frames.append(meas.acquire())
                good = [fr for fr in frames if not (fr[0].incomplete or fr[1].incomplete)
                        and len(fr[1].v) == len(w2.v)]
                incomplete = not good
                if good:
                    frames = good
                w1, w2 = frames[0]

                def refof(fr):
                    return (fr[1].t, fr[1].v) if b.phase_ref == "ch2" else (fr[0].t, fr[0].v)

                ref = refof(frames[0])
                # A：每週期取樣值 = 所有張合在一起（畫圖用；週期判斷時各張分開比較）
                seqs = [per_cycle_values(fr[1].t, fr[1].v, b.freq, b.method, b.phase_deg, ref=refof(fr))
                        for fr in frames]
                vals = np.concatenate(seqs)
                scale = float(np.median([vpp(fr[1].v) for fr in frames]))
                # 最大 Lyapunov 指數：用畫圖的那一串每週期取樣值（各張分開演化）
                ly = lyapunov_kantz(seqs, scale, tol=b.tol, shell=config.BIF_LYAP_SHELL, dim=config.BIF_LYAP_DIM,
                                    max_sat=config.BIF_LYAP_MAX_SAT)
                # 週期判斷：多相位時域 + 頻譜確認；多張時，時域合併各張、頻譜做功率平均（B）；混沌用 λ 判斷
                cls = classify_period(w2.t, w2.v, b.freq, ref=ref, tol=b.tol, sub_dbc=config.BIF_SUB_DBC,
                                      confirm_db=config.BIF_CONFIRM_DB, chaos_db=config.BIF_CHAOS_DB,
                                      n_phases=config.BIF_PHASES, chaos_floor_dbc=config.BIF_CHAOS_FLOOR_DBC,
                                      periods=b.periods(), extra_frames=[(fr[1].v, refof(fr)) for fr in frames[1:]],
                                      lyap=ly["lyap"] if ly["valid"] else float("nan"), lyap_thr=config.BIF_LYAP_THR,
                                      chaos_method=b.chaos_method, weak_upgrade=config.BIF_WEAK_UPGRADE)
                p = cls["period"]
                subs = [subharmonic_dbc(fr[1].t, fr[1].v, b.freq, divs=b.pow2_divs()) for fr in frames]
                sub = {k: 10 * math.log10(np.mean([10 ** (s[k] / 10) for s in subs if np.isfinite(s[k])]))
                       if any(np.isfinite(s[k]) for s in subs) else float("nan") for k in subs[0]}
                return dict(amp=float(a), values=vals, branches=branch_stats(seqs, p), period=p, label=period_label(p),
                            sub=sub, ch2_vpp=scale, cls=cls, lyap=ly["lyap"], lyap_n=ly["n_ref"],
                            lyap_valid=ly["valid"], lyap_sat=ly["sat"],
                            spec=spectrum_map_row(w2.t, [fr[1].v for fr in frames], b.freq),
                            ch1_vpp=vpp(w1.v), n_cycles=len(vals), n_frames=len(frames), t=w2.t, v=w2.v,
                            incomplete=incomplete, redone=False)

            for i, a in enumerate(amps):
                if self._stop:
                    raise StopRequested
                self.status.emit(f"振幅 {a:.4g} Vpp（{i + 1}/{len(amps)}）")
                r = one(a)
                results.append(r)
                self.amp_done.emit(r)
                self.progress.emit(i + 1, len(amps))

            # 補測資料不完整的振幅：先回到前一個振幅再升上來，保持「由小往大」的掃描歷程
            for rnd in range(config.REDO_ROUNDS if config.AUTO_REDO_INCOMPLETE else 0):
              bad = [i for i, r in enumerate(results) if r["incomplete"]]
              if not bad:
                  break
              for k, i in enumerate(bad):
                if self._stop:
                    raise StopRequested
                self.status.emit(f"補測資料不完整的振幅 {amps[i]:.4g} Vpp（{k + 1}/{len(bad)}）")
                self.afg.set_amplitude(float(amps[max(i - 1, 0)]))
                meas.wait(b.settle + 20 * self.tscale)
                r = one(amps[i])
                r["redone"], r["redo_of"] = True, i
                results[i] = r
                self.amp_done.emit(r)
            still = sum(r["incomplete"] for r in results)
            self.status.emit("分岔圖完成" + (f"（補測後仍有 {still} 個振幅資料不完整）" if still else ""))
        except StopRequested:
            stopped = True
            self.status.emit("已停止")
        except Exception as e:
            stopped = True
            self.error.emit(f"分岔圖失敗：{e}")
        finally:
            if bench is not None:
                bench.chaos = False
            try:
                self.scope.set_record_length(config.RECORD_LENGTH)
            except Exception:
                pass
            if stopped:
                try:
                    apply_stop_action(self.afg, self.stop_action)
                except Exception as e:
                    self.status.emit(f"停止時調整產生器失敗：{e}")
        self.result.emit(dict(results=results, settings=b))


EXTRA_SPEC = (12, 16, 32, 64, 128)     # summary CSV 後面追加的頻譜譜線欄位


def _cls_cols(c):
    """summary CSV 的週期判斷細節欄位（classify_period 的結果）。"""
    if not c:
        return [""] * (11 + len(EXTRA_SPEC))
    L = c["lines"]
    g = lambda k: f"{L.get(k, float('nan')):.2f}"
    return [c["p_time"], c["p_spec"], g("f/2"), g("f/4"), g("f/8"),
            f"{c['floor']:.2f}", f"{c['floor_hf']:.2f}", c["reason"],
            g("f/3"), g("f/5"), g("f/6")] + [g(f"f/{q}") for q in EXTRA_SPEC]


def bif_feigenbaum(results, b):
    """分岔圖結果 → 費根鮑姆分析（analysis.feigenbaum_analysis）。依振幅排序；次諧波用鎖相值。"""
    rs = sorted(results, key=lambda r: r["amp"])
    if len(rs) < 3:
        return None
    amps = [r["amp"] for r in rs]
    per = [r["period"] for r in rs]
    lev = {q: np.array([r["sub"].get(f"f/{q}", float("nan")) for r in rs]) for q in b.pow2_divs()}
    return feigenbaum_analysis(amps, per, lev, gates=config.BIF_FEIG_GATES, nfit=config.BIF_FEIG_NFIT)


def write_bifurcation_csv(base, results, b, meta_lines, feig=None):
    """base.csv：每個週期的取樣值（長格式）；base_summary.csv：每個振幅的週期判斷、次諧波、Lyapunov 指數；
    base_branches.csv：每個振幅、每個分支的平均 ± 標準差 / 標準誤（週期 p 有 p 列；混沌與無法判斷不列）；
    base_spectra.npz：每個振幅的平均功率頻譜（amp_Vpp、f_over_fdrive、dBc[振幅, 頻率]）；
    base_feigenbaum.txt：倍週期閾值與費根鮑姆常數（feig = bif_feigenbaum 的結果）。"""
    specs = [r for r in results if r.get("spec") is not None]
    if specs:
        np.savez_compressed(base + "_spectra.npz", amp_Vpp=np.array([r["amp"] for r in specs]),
                            f_over_fdrive=SPEC_GRID, dBc=np.array([r["spec"] for r in specs]))
    if feig is not None:
        with open(base + "_feigenbaum.txt", "w", encoding="utf-8-sig") as fp:
            for line in meta_lines:
                fp.write(line + "\n")
            fp.write(f"# 驅動頻率,{b.freq:.9g} Hz；混沌判斷 = {b.chaos_method}（λ 門檻 {config.BIF_LYAP_THR:g}）\n\n")
            fp.write(feig["text"] + "\n")
    head = meta_lines + ["# 分岔圖設定," + "；".join(f"{k}={v}" for k, v in asdict(b).items()),
                         "# 取樣方式," + ("每週期最大值" if b.method == "max" else f"固定相位 {b.phase_deg:g}°")
                         + ("（相位 0 = CH2 基頻波峰）" if b.phase_ref == "ch2" else "（相位 0 = CH1 驅動波峰）")]
    with open(base + ".csv", "w", newline="", encoding="utf-8-sig") as fp:
        for line in head:
            fp.write(line + "\n")
        wr = csv.writer(fp)
        wr.writerow(["amp_Vpp", "period", "cycle_index", "CH2_value_V"])
        for r in results:
            for i, x in enumerate(r["values"]):
                wr.writerow([f"{r['amp']:.6g}", r["period"], i, f"{x:.6g}"])
    with open(base + "_summary.csv", "w", newline="", encoding="utf-8-sig") as fp:
        for line in head:
            fp.write(line + "\n")
        fp.write("# period：多相位時域判斷 + 頻譜確認（analysis.classify_period）；0 = 混沌"
                 f"（判斷方式 {b.chaos_method}），-1 = 雜訊太大無法判斷\n")
        fp.write("# p_time = 時域判斷，p_spec = 頻譜判斷，spec_* = Hann FFT 的譜線/底噪 (dBc)；f/2_dBc、f/4_dBc 為鎖相值\n")
        fp.write("# lyapunov = 最大 Lyapunov 指數（每驅動週期，自然對數；analysis.lyapunov_kantz），"
                 "lyap_n_ref = 有鄰居的參考點數，lyap_sat = 鄰居一步分開的程度（> "
                 f"{config.BIF_LYAP_MAX_SAT:g} 視為雜訊，lyap_valid = 0，不用來判斷混沌）；f/8_dBc 之後為鎖相值\n")
        divs = [q for q in b.pow2_divs() if q > 4]
        wr = csv.writer(fp)
        wr.writerow(["amp_Vpp", "period", "label", "f/2_dBc", "f/4_dBc", "CH1_Vpp", "CH2_Vpp", "n_cycles", "incomplete", "redone",
                     "p_time", "p_spec", "spec_f2_dBc", "spec_f4_dBc", "spec_f8_dBc", "floor_dBc", "floor_hf_dBc", "judge",
                     "spec_f3_dBc", "spec_f5_dBc", "spec_f6_dBc"] + [f"spec_f{q}_dBc" for q in EXTRA_SPEC]
                    + ["lyapunov", "lyap_n_ref", "lyap_sat", "lyap_valid"] + [f"f/{q}_dBc" for q in divs])
        for r in results:
            wr.writerow([f"{r['amp']:.6g}", r["period"], r["label"], f"{r['sub']['f/2']:.2f}",
                         f"{r['sub']['f/4']:.2f}", f"{r['ch1_vpp']:.6g}", f"{r['ch2_vpp']:.6g}", r["n_cycles"],
                         int(r.get("incomplete", False)), int(r.get("redone", False))]
                        + _cls_cols(r.get("cls"))
                        + [f"{r.get('lyap', float('nan')):.4f}", r.get("lyap_n", 0),
                           f"{r.get('lyap_sat', float('nan')):.3f}", int(r.get("lyap_valid", False))]
                        + [f"{r['sub'].get(f'f/{q}', float('nan')):.2f}" for q in divs])
    with open(base + "_branches.csv", "w", newline="", encoding="utf-8-sig") as fp:
        for line in head:
            fp.write(line + "\n")
        fp.write("# 週期 p 的每週期取樣值分成 p 個分支（第 i 個週期屬於分支 i mod p，多張以循環平移對齊），"
                 "branch 依平均值由小到大編號；std = 樣本標準差，sem = std/√n\n")
        wr = csv.writer(fp)
        wr.writerow(["amp_Vpp", "period", "label", "branch", "mean_V", "std_V", "sem_V", "n"])
        for r in results:
            for k, br in enumerate(r.get("branches") or []):
                wr.writerow([f"{r['amp']:.6g}", r["period"], r["label"], k, f"{br['mean']:.6g}", f"{br['std']:.6g}",
                             f"{br['sem']:.6g}", br["n"]])

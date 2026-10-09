"""實驗流程（在背景執行緒執行）：
- 共振掃頻：往上 / 往下 / 往上再往下、對數 / 線性、自動兩階段（粗掃 → 峰附近細掃）
- 批次參數掃描：對每個振幅（或 DC offset）各做一次掃頻，輸出總表
- 分岔圖：固定頻率逐步加大振幅，每週期取一個值，判斷週期 1 / 2 / 4 / 8 / 混沌
"""
import copy
import csv
import math
import os
import time
from dataclasses import asdict, dataclass

import numpy as np
from PyQt5 import QtCore

import config
from analysis import (analyze_resonance, detect_period, fmt_eng, lockin, per_cycle_values, period_label, pick_scale,
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
    sw_avg: int = 1               # 程式平均：每點讀幾次再平均
    scope_avg: int = 1            # 示波器硬體平均次數（1 = 關）
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

    def acquire_avg(self, n, gap):
        """讀 n 次，波形逐點平均（觸發在 CH1，所以每次的相位一致）。"""
        w1, w2 = self.acquire()
        if n <= 1:
            return w1, w2
        s1, s2 = [w1], [w2]
        for _ in range(n - 1):
            self.wait(gap)
            a, b = self.acquire()
            if len(a.v) == len(w1.v) and len(b.v) == len(w2.v):
                s1.append(a)
                s2.append(b)
        out = []
        for lst in (s1, s2):
            w = copy.copy(lst[-1])
            w.v = np.mean([x.v for x in lst], axis=0)
            w.screen_div = np.mean([x.screen_div for x in lst], axis=0)
            out.append(w)
        return out[0], out[1]

    def measure(self, f, direction, stage):
        s = self.s
        self.afg.set_frequency(f)
        tscale = pick_scale(TIME_SCALES, s.periods / (10 * f))
        self.scope.set_timebase(tscale)
        frame = 10 * tscale + 0.01
        wait = s.settle + 2 * frame + (s.scope_avg * frame if s.scope_avg > 1 else 0)
        self.wait(wait)
        for _ in range(6):
            w1, w2 = self.acquire_avg(s.sw_avg, frame)
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
        a1, p1 = lockin(w1.t, w1.v, f)
        a2, p2 = lockin(w2.t, w2.v, f)
        ok = a1 > 0 and a2 > 0
        div1 = 2 * a1 / w1.vscale if w1.vscale else 0.0
        div2 = 2 * a2 / w2.vscale if w2.vscale else 0.0
        return dict(
            freq_Hz=float(f), direction=direction, stage=stage,
            gain_dB=20 * math.log10(a2 / a1) if ok else float("nan"),
            phase_deg=wrap_deg(math.degrees(p2 - p1)) if ok else float("nan"),
            CH1_amp_V=a1, CH2_amp_V=a2, CH1_Vpp=2 * a1, CH2_Vpp=2 * a2,
            CH1_div_pp=div1, CH2_div_pp=div2,
            unreliable=bool(not ok or div1 < config.SMALL_SIGNAL_DIVS or div2 < config.SMALL_SIGNAL_DIVS))


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
                "CH2_Vpp", "CH1_div_pp", "CH2_div_pp", "AFG_set_Vpp", "AFG_offset_V", "AFG_load", "unreliable"]


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
            if s.auto_ac:
                self.scope.set_coupling(ch, "AC")
                self.scope.set_bwlimit(ch, False)
        if s.scope_avg > 1:
            self.scope.set_acquire_average(s.scope_avg)

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
        self.result.emit(dict(points=pts, analyses=analyses, warning=amplitude_check(pts), settings=s))


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
                warn = amplitude_check(pts)
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
    record_length: int = 10000
    min_periods: int = 100
    method: str = "max"          # "max" / "phase"
    phase_deg: float = 0.0
    phase_ref: str = "ch2"       # "ch2" = 以 CH2 基頻波峰為相位 0（自動）；"ch1" = 以 CH1 驅動波峰為相位 0
    tol: float = config.BIF_TOL
    stop_action: str = "keep"

    def amps(self):
        return np.linspace(self.amp_start, self.amp_stop, max(int(self.steps), 1))


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
                new = suggest_vscale(w, VOLT_SCALES, config.SCREEN_HALF_DIVS)
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
            self.scope.set_record_length(b.record_length)
            self.tscale = pick_scale(TIME_SCALES, b.min_periods / (10 * b.freq))
            self.scope.set_timebase(self.tscale)
            spp = b.record_length / (10 * self.tscale * b.freq)
            self.status.emit(f"時基 {fmt_eng(self.tscale, 's')}/div，畫面約 {10 * self.tscale * b.freq:.0f} 個週期，"
                             f"每週期約 {spp:.0f} 點" + ("（點數偏少，建議加大記錄長度）" if spp < 20 else ""))
            self.afg.set_frequency(b.freq)
            self.afg.set_output(True)
            amps = b.amps()
            for i, a in enumerate(amps):
                if self._stop:
                    raise StopRequested
                self.afg.set_amplitude(float(a))
                self.status.emit(f"振幅 {a:.4g} Vpp（{i + 1}/{len(amps)}）")
                meas.wait(b.settle + 20 * self.tscale)
                w1, w2 = self._acquire(meas)
                ref = (w2.t, w2.v) if b.phase_ref == "ch2" else (w1.t, w1.v)
                vals = per_cycle_values(w2.t, w2.v, b.freq, b.method, b.phase_deg, ref=ref)
                scale = vpp(w2.v)
                p = detect_period(vals, scale, b.tol)
                sub = subharmonic_dbc(w2.t, w2.v, b.freq)
                r = dict(amp=float(a), values=vals, period=p, label=period_label(p), sub=sub, ch2_vpp=scale,
                         ch1_vpp=vpp(w1.v), n_cycles=len(vals), t=w2.t, v=w2.v)
                results.append(r)
                self.amp_done.emit(r)
                self.progress.emit(i + 1, len(amps))
            self.status.emit("分岔圖完成")
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


def write_bifurcation_csv(base, results, b, meta_lines):
    """base.csv：每個週期的取樣值（長格式）；base_summary.csv：每個振幅的週期判斷與 f/2、f/4 分量。"""
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
        fp.write("# period = 0 表示混沌或無法判斷\n")
        wr = csv.writer(fp)
        wr.writerow(["amp_Vpp", "period", "label", "f/2_dBc", "f/4_dBc", "CH1_Vpp", "CH2_Vpp", "n_cycles"])
        for r in results:
            wr.writerow([f"{r['amp']:.6g}", r["period"], r["label"], f"{r['sub']['f/2']:.2f}",
                         f"{r['sub']['f/4']:.2f}", f"{r['ch1_vpp']:.6g}", f"{r['ch2_vpp']:.6g}", r["n_cycles"]])

"""波形分析與格式化小工具。"""
import math

import numpy as np


def series_125(lo, hi):
    """產生 1-2-5 數列，例如 1m, 2m, 5m, 10m ..."""
    vals = []
    e = math.floor(math.log10(lo)) - 1
    while True:
        for m in (1, 2, 5):
            v = float(f"{m * 10.0 ** e:.3g}")
            if v > hi * (1 + 1e-9):
                return vals
            if v >= lo * (1 - 1e-9):
                vals.append(v)
        e += 1


def pick_scale(scales, x):
    """回傳 >= x 的最小刻度；超過範圍就回傳最大的。"""
    for s in scales:
        if s >= x * (1 - 1e-6):
            return s
    return scales[-1]


def nearest_index(scales, x):
    return min(range(len(scales)), key=lambda i: abs(math.log(scales[i] / x)))


def fmt_eng(x, unit="", digits=4):
    """工程記號格式，例如 0.00025 -> '250 µs'。"""
    if x is None or not np.isfinite(x):
        return "—"
    if x == 0:
        return f"0 {unit}"
    for f, p in ((1e9, "G"), (1e6, "M"), (1e3, "k"), (1, ""), (1e-3, "m"), (1e-6, "µ"), (1e-9, "n")):
        if abs(x) >= f * 0.9995:
            return f"{x / f:.{digits}g} {p}{unit}"
    return f"{x / 1e-12:.{digits}g} p{unit}"


def vpp(v):
    return float(np.max(v) - np.min(v)) if len(v) else float("nan")


def estimate_frequency(t, v):
    """用 FFT 找主頻率（含拋物線內插）。訊號太小或不到一個週期時回傳 nan。"""
    n = len(v)
    if n < 16:
        return float("nan")
    x = np.asarray(v, float) - np.mean(v)
    if np.std(x) < 1e-12:
        return float("nan")
    dt = (t[-1] - t[0]) / (n - 1)
    nfft = 1 << int(math.ceil(math.log2(n * 4)))
    spec = np.abs(np.fft.rfft(x * np.hanning(n), nfft))
    spec[:2] = 0
    k = int(np.argmax(spec))
    if 0 < k < len(spec) - 1:
        a, b, c = np.log(spec[k - 1:k + 2] + 1e-300)
        denom = a - 2 * b + c
        k = k + (0.5 * (a - c) / denom if denom != 0 else 0.0)
    f = k / (nfft * dt)
    if f * (t[-1] - t[0]) < 0.8:   # 視窗內不到一個週期，不可靠
        return float("nan")
    return float(f)


def lockin(t, v, f):
    """在頻率 f 做鎖相分析，回傳 (振幅, 相位[rad])，模型 v = A cos(2πft + φ)。
    只取整數個週期以減少誤差。"""
    t = np.asarray(t, float)
    v = np.asarray(v, float)
    if not np.isfinite(f) or f <= 0 or len(t) < 4:
        return float("nan"), float("nan")
    span = t[-1] - t[0]
    periods = math.floor(span * f)
    if periods >= 1:
        mask = t <= t[0] + periods / f
        t, v = t[mask], v[mask]
    v = v - np.mean(v)
    w = 2 * math.pi * f * t
    i = 2 * np.mean(v * np.cos(w))
    q = 2 * np.mean(v * np.sin(w))
    return float(math.hypot(i, q)), float(math.atan2(-q, i))


def wrap_deg(d):
    """把角度包到 (-180, 180]。"""
    d = (d + 180.0) % 360.0 - 180.0
    return 180.0 if d == -180.0 else d


# ---------------- FFT 頻譜 ----------------

def _flattop(n):
    k = np.arange(n)
    x = 2 * np.pi * k / max(n - 1, 1)
    a = (0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368)
    return a[0] - a[1] * np.cos(x) + a[2] * np.cos(2 * x) - a[3] * np.cos(3 * x) + a[4] * np.cos(4 * x)


# 名稱 -> (視窗函數, 主瓣半寬（bins），說明)
FFT_WINDOWS = {
    "Hanning": (np.hanning, 2, "通用，頻率解析與振幅精度平衡"),
    "Flat-top": (_flattop, 5, "振幅最準，適合量諧波大小"),
    "Blackman": (np.blackman, 3, "旁瓣低，適合看小訊號"),
    "矩形": (np.ones, 1, "不加窗，頻率剛好整數週期時最清楚"),
}


def spectrum(t, v, window="Hanning", zero_pad=1, remove_dc=True):
    """回傳 (頻率 Hz, 振幅 V_peak)。已做視窗振幅修正，正弦波的峰值即為其振幅。"""
    v = np.asarray(v, float)
    n = len(v)
    if n < 8:
        return np.array([]), np.array([])
    dt = (t[-1] - t[0]) / (n - 1)
    w = FFT_WINDOWS[window][0](n)
    x = v - np.mean(v) if remove_dc else v
    nfft = int(n * zero_pad)
    amp = 2 * np.abs(np.fft.rfft(x * w, nfft)) / np.sum(w)
    amp[0] /= 2
    return np.fft.rfftfreq(nfft, dt), amp


def noise_floor(amp):
    """雜訊底線估計：頻譜振幅的中位數（峰只佔少數 bin，不影響中位數）。"""
    return float(np.median(amp[1:])) if len(amp) > 1 else 0.0


NOISE_MULT = 6.0   # 高於雜訊底線幾倍（約 15.6 dB）才算是真的峰


def find_peaks(f, amp, count=6, min_sep_bins=4, floor_ratio=1e-3):
    """找出最大的幾個峰（局部最大值），彼此至少相隔 min_sep_bins，且明顯高於雜訊底線。
    回傳 index 陣列（由大到小）。"""
    if len(amp) < 3:
        return np.array([], int)
    a = amp.copy()
    a[0] = 0
    cand = np.where((a[1:-1] > a[:-2]) & (a[1:-1] >= a[2:]))[0] + 1
    cand = cand[a[cand] > max(a.max() * floor_ratio, NOISE_MULT * noise_floor(amp))]
    cand = cand[np.argsort(a[cand])[::-1]]
    picked = []
    for i in cand:
        if all(abs(i - j) > min_sep_bins for j in picked):
            picked.append(i)
            if len(picked) >= count:
                break
    return np.array(picked, int)


def refine_peak(f, amp, i):
    """用拋物線內插（對數振幅）求峰的精確頻率。"""
    if 0 < i < len(amp) - 1:
        a, b, c = np.log(amp[i - 1:i + 2] + 1e-300)
        d = a - 2 * b + c
        if d != 0:
            return float(f[i] + 0.5 * (a - c) / d * (f[1] - f[0]))
    return float(f[i])


def harmonics(f, amp, f0, n_harm=10, search_bins=2, lobe_bins=0):
    """以 f0 為基頻，找第 1~n 次諧波的 (k, 頻率, 振幅)。超出頻寬的略過。
    若基頻離 0 太近（諧波會落在基頻主瓣裡，解析度不足）回傳空串列。"""
    out = []
    if not np.isfinite(f0) or f0 <= 0 or len(f) < 2:
        return out
    df = f[1] - f[0]
    if f0 / df < lobe_bins + search_bins:      # 相鄰諧波相距 f0，必須在主瓣之外
        return out
    for k in range(1, n_harm + 1):
        fk = k * f0
        if fk > f[-1]:
            break
        i = int(round(fk / df))
        lo, hi = max(i - search_bins, 1), min(i + search_bins + 1, len(amp))
        j = lo + int(np.argmax(amp[lo:hi]))
        out.append((k, float(f[j]), float(amp[j])))
    return out


def thd(harm, floor=0.0):
    """總諧波失真 = sqrt(ΣA_k², k≥2) / A_1。低於 floor（雜訊）的諧波不計。
    回傳 (THD, 是否只是上限)：所有諧波都在雜訊以下時，回傳以雜訊底線估的上限。"""
    if len(harm) < 2 or harm[0][2] <= 0:
        return float("nan"), False
    real = [a for _, _, a in harm[1:] if a > floor]
    if not real:
        return floor / harm[0][2], True
    return math.sqrt(sum(a * a for a in real)) / harm[0][2], False


# ---------------- 共振分析 ----------------

def parabolic_vertex(x, y, i):
    """以 i 和左右兩點做拋物線，回傳頂點 (x, y)；無法內插時回傳原點。"""
    if 0 < i < len(y) - 1 and np.all(np.isfinite(y[i - 1:i + 2])):
        x0 = float(x[i])
        h = max(abs(x[i + 1] - x0), abs(x0 - x[i - 1]))
        u = (np.asarray(x[i - 1:i + 2], float) - x0) / h     # 平移、縮放到 ~±1，避免數值問題
        y0, y1, y2 = (float(v) for v in y[i - 1:i + 2])
        d0, d2 = u[0], u[2]                                  # u[1] = 0
        if not (d0 < 0 < d2):                                # 頻率重複或未排序
            return float(x[i]), float(y[i])
        # 過三點的拋物線 y = a u² + b u + y1
        a = ((y0 - y1) / d0 - (y2 - y1) / d2) / (d0 - d2)
        b = (y0 - y1) / d0 - a * d0
        if a < 0:
            uv = -b / (2 * a)
            if d0 <= uv <= d2:
                return x0 + uv * h, y1 - b * b / (4 * a)
    return float(x[i]), float(y[i])


def _interp_cross(x0, y0, x1, y1, level):
    if y1 == y0:
        return float((x0 + x1) / 2)
    return float(x0 + (level - y0) * (x1 - x0) / (y1 - y0))


def analyze_resonance(freq, gain_db, phase_deg, unreliable=None, L=0.0):
    """共振分析。回傳 dict：
    f_peak, gain_peak_db（增益峰，3 點拋物線內插）、f_phase0（峰附近相位由正變負的過零點）、
    f_lo, f_hi, bw（−3 dB 頻寬）、Q、C_eff、C_eff_phase、n、n_unreliable。找不到的項目為 nan。"""
    nan = float("nan")
    out = dict(f_peak=nan, gain_peak_db=nan, f_phase0=nan, f_lo=nan, f_hi=nan, bw=nan, Q=nan,
               C_eff=nan, C_eff_phase=nan, n=0, n_unreliable=0)
    f = np.asarray(freq, float)
    g = np.asarray(gain_db, float)
    p = np.asarray(phase_deg, float)
    bad = np.zeros(len(f), bool) if unreliable is None else np.asarray(unreliable, bool)
    out["n"], out["n_unreliable"] = int(len(f)), int(bad.sum())
    # 同一頻率量了不只一次（例如細掃中心點 = 粗掃峰值）→ 合併：增益、相位取平均，有一次不可靠就算不可靠
    f, inv = np.unique(f, return_inverse=True)

    def merged_mean(a):
        ok_ = np.isfinite(a)
        s = np.bincount(inv, weights=np.where(ok_, a, 0.0), minlength=len(f))
        n = np.bincount(inv, weights=ok_.astype(float), minlength=len(f))
        return np.where(n > 0, s / np.maximum(n, 1), np.nan)

    g, p = merged_mean(g), merged_mean(p)
    bad = np.bincount(inv, weights=bad.astype(float), minlength=len(f)) > 0
    ok = np.isfinite(g)
    if ok.sum() < 3:
        return out
    use = ok & ~bad if (ok & ~bad).sum() >= 3 else ok
    i = int(np.argmax(np.where(use, g, -np.inf)))
    out["f_peak"], out["gain_peak_db"] = parabolic_vertex(f, g, i)

    # 相位過零（+ → −），取最接近增益峰的那一個
    cands = []
    for j in range(len(f) - 1):
        if np.isfinite(p[j]) and np.isfinite(p[j + 1]) and p[j] > 0 >= p[j + 1] and p[j] - p[j + 1] < 180:
            cands.append(_interp_cross(f[j], p[j], f[j + 1], p[j + 1], 0.0))
    if cands:
        out["f_phase0"] = min(cands, key=lambda x: abs(x - out["f_peak"]))

    # −3 dB 頻寬
    level = out["gain_peak_db"] - 3.0
    for k in range(i, 0, -1):
        if ok[k - 1] and g[k - 1] < level <= g[k]:
            out["f_lo"] = _interp_cross(f[k - 1], g[k - 1], f[k], g[k], level)
            break
    for k in range(i, len(f) - 1):
        if ok[k + 1] and g[k + 1] < level <= g[k]:
            out["f_hi"] = _interp_cross(f[k], g[k], f[k + 1], g[k + 1], level)
            break
    out["bw"] = out["f_hi"] - out["f_lo"]
    if np.isfinite(out["bw"]) and out["bw"] > 0:
        out["Q"] = out["f_peak"] / out["bw"]
    if L and L > 0:
        for key, fk in (("C_eff", out["f_peak"]), ("C_eff_phase", out["f_phase0"])):
            if np.isfinite(fk) and fk > 0:
                out[key] = 1.0 / ((2 * math.pi * fk) ** 2 * L)
    return out


# ---------------- 分岔圖 ----------------

def per_cycle_values(t, v, f, method="max", phase_deg=0.0, ref=None):
    """每個驅動週期取一個值。
    method="phase"：在相位 phase_deg 取樣；method="max"：以相位 phase_deg 為中心、前後各半週期的視窗內取最大值。
    相位 0 = 參考訊號 ref=(t, v) 基頻的波峰；沒有 ref 時以 t=0 為相位 0。"""
    t = np.asarray(t, float)
    v = np.asarray(v, float)
    if len(t) < 4 or not f > 0:
        return np.array([])
    ph0 = 0.0
    if ref is not None:
        _a, ph0 = lockin(ref[0], ref[1], f)        # 驅動 = A cos(2πft + ph0)
        if not np.isfinite(ph0):
            ph0 = 0.0
    offset = (math.radians(phase_deg) - ph0) / (2 * math.pi)   # 單位：週期
    if method != "phase":
        offset -= 0.5      # 最大值模式：視窗以該相位為中心（±半週期），避免視窗邊界切在波峰上
    k0 = math.ceil(t[0] * f - offset)
    k1 = math.floor(t[-1] * f - offset)
    tk = (np.arange(k0, k1 + 1) + offset) / f
    if method == "phase":
        return np.interp(tk, t, v)
    idx = np.searchsorted(t, tk)
    return np.array([v[a:b].max() for a, b in zip(idx[:-1], idx[1:]) if b > a])


def detect_period(values, scale, tol=0.03, periods=(1, 2, 4, 8)):
    """比較 x[i] 與 x[i+p]，回傳最小的符合週期 p；都不符合回傳 0（混沌 / 無法判斷）。"""
    x = np.asarray(values, float)
    if not scale or scale <= 0:
        return 0
    for p in periods:
        if len(x) < 3 * p + 2:
            break
        d = np.abs(x[p:] - x[:-p])
        if np.median(d) <= tol * scale and np.percentile(d, 90) <= 2 * tol * scale:
            return p
    return 0


def period_label(p):
    return f"週期 {p}" if p else "混沌"


def subharmonic_dbc(t, v, f):
    """f/2、f/4 分量相對於 f 分量的大小 (dBc)。"""
    a1, _ = lockin(t, v, f)
    out = {}
    for name, div in (("f/2", 2), ("f/4", 4)):
        a, _ = lockin(t, v, f / div)
        out[name] = 20 * math.log10(a / a1) if a1 > 0 and a > 0 else float("nan")
    return out


def suggest_vscale(wave, scales, half_divs):
    """依目前波形建議新的垂直刻度；不需要改就回傳 None。"""
    peak = float(np.max(np.abs(wave.screen_div))) if len(wave.screen_div) else 0.0
    idx = nearest_index(scales, wave.vscale)
    if peak > half_divs * 0.95:           # 被截掉了 → 放大刻度
        if idx < len(scales) - 1:
            return scales[min(idx + 2, len(scales) - 1)]
        return None
    if peak < half_divs * 0.3:            # 太小 → 縮小刻度，讓波形約佔 3/4 螢幕
        new = pick_scale(scales, peak * wave.vscale / (half_divs * 0.75))
        if new < wave.vscale * 0.99:
            return new
    return None

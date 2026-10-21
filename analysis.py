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


def amplitude_rel_error(t, v, f):
    """以最小平方擬合 v = a·cos + b·sin + c，用殘差估計振幅的相對誤差 σ_A / A。
    （白雜訊下 σ_A ≈ rms(殘差)·√(2/N)；實測與重複量測的離散程度相符。）"""
    t = np.asarray(t, float)
    v = np.asarray(v, float)
    if len(t) < 8 or not f > 0:
        return float("nan")
    w = 2 * math.pi * f * t
    m = np.column_stack([np.cos(w), np.sin(w), np.ones_like(w)])
    coef, *_ = np.linalg.lstsq(m, v, rcond=None)
    amp = math.hypot(coef[0], coef[1])
    if amp <= 0:
        return float("inf")
    return float(np.std(v - m @ coef) * math.sqrt(2.0 / len(v)) / amp)


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


def detect_period_multi(seqs, scale, tol=0.03, periods=(1, 2, 4, 8)):
    """多張擷取版的 detect_period：每張各自比較 x[i] 與 x[i+p]（不把不同張接起來，順序不會亂），
    把所有張的差值合在一起判斷。張數越多，中位數與百分位越穩。"""
    seqs = [np.asarray(x, float) for x in seqs]
    if not scale or scale <= 0:
        return 0
    for p in periods:
        ds = [np.abs(x[p:] - x[:-p]) for x in seqs if len(x) >= 3 * p + 2]
        if not ds:
            break
        d = np.concatenate(ds)
        if np.median(d) <= tol * scale and np.percentile(d, 90) <= 2 * tol * scale:
            return p
    return 0


def branch_stats(seqs, p):
    """把每週期取樣值分成 p 個分支，算各分支的平均與離散程度（分岔圖的「平均成數據點」）。
    seqs：各張的每週期取樣值（依時間順序）。每張第 i 個值屬於分支 i mod p；各張的分支起點是隨機的，
    所以以第一張為基準，對其他張做循環平移，使各分支平均值最接近後再合併（不會把不同分支平均在一起）。
    回傳依平均值排序的 [dict(mean, std, sem, n), ...]；p < 1（混沌 / 無法判斷）回傳 []。"""
    if p is None or p < 1:
        return []
    groups, ref = None, None
    for x in seqs:
        x = np.asarray(x, float)
        if len(x) < p:
            continue
        br = [x[k::p] for k in range(p)]
        m = np.array([b.mean() for b in br])
        if groups is None:
            groups, ref = [list(b) for b in br], m
            continue
        s = min(range(p), key=lambda s: float(np.sum((np.roll(m, -s) - ref) ** 2)))
        for k in range(p):
            groups[k].extend(br[(k + s) % p])
    out = []
    for g in groups or []:
        g = np.asarray(g, float)
        n = len(g)
        sd = float(np.std(g, ddof=1)) if n > 1 else 0.0
        out.append(dict(mean=float(g.mean()), std=sd, sem=sd / math.sqrt(n), n=n))
    return sorted(out, key=lambda d: d["mean"])


def period_label(p):
    if p is None or p < 0:
        return "無法判斷"
    return f"週期 {p}" if p else "混沌"


SUB_ORDERS = (2, 3, 4, 5, 6, 8, 12, 16, 32, 64, 128)    # 會檢查的次諧波族 f/q（q 階週期）
POW2_ORDERS = (2, 4, 8, 16, 32, 64, 128)


def orders_for(periods):
    """週期清單 → 需要檢查的次諧波族（2～8 一律包含，與舊版相容）。"""
    pmax = max(max(periods), 8)
    return tuple(q for q in SUB_ORDERS if q <= pmax)


def spectrum_levels(t, v, f, more=(), orders=(2, 3, 4, 5, 6, 8)):
    """Hann 視窗 FFT：回傳各次諧波族 f/q（q 屬於 orders）相對 f 的大小 (dBc)，以及兩個底噪 (dBc)。
    f/q 族 = m·f/q（0 < m < q、m 與 q 互質）中最大的一條，例如 f/4 取 f/4、3f/4；f/3 取 f/3、2f/3。
    紀錄太短、f/q 族的譜線間隔小於 4 個 bin（分不開）時，該族回傳 nan。
    floor     = 0.05f–1.45f 之間、避開 f/Q、f/6（或 f/12）、f/5 整數倍 ±3 bin 後的頻譜中位數（混沌時這裡會被寬頻填滿）；
                Q = 16，或 orders 中譜線間隔 ≥ 10 bin 的最大 2 的冪次（例如 640 個週期、檢查到 f/64 時 Q = 64）
    floor_hf  = 3f–6f 之間（同樣避開譜線）的中位數（白雜訊在各頻段差不多，混沌的寬頻主要在低頻）
    more：同樣取樣設定的其他張波形（長度要和 v 相同）。給了就把各張的「功率」頻譜平均（與相位無關，
    不會抵消次諧波），底噪的起伏約降為 1/√張數，微弱譜線比較容易從底噪中分辨（底噪的平均高度不變）。"""
    t = np.asarray(t, float)
    v = np.asarray(v, float)
    n = len(v)
    nan = dict(lines={f"f/{q}": float("nan") for q in orders}, floor=float("nan"), floor_hf=float("nan"))
    if n < 64 or not f > 0:
        return nan
    dt = (t[-1] - t[0]) / (n - 1)
    win = np.hanning(n)
    vs = [v] + [np.asarray(x, float) for x in more if len(x) == n]
    spec = np.sqrt(np.mean([np.abs(np.fft.rfft((x - x.mean()) * win)) ** 2 for x in vs], axis=0))
    df = 1.0 / (n * dt)

    def peak(fx, w=2):
        k = int(round(fx / df))
        if k - w < 1 or k + w >= len(spec):
            return float("nan")
        return float(spec[k - w:k + w + 1].max())

    a1 = peak(f)
    if not a1 > 0:
        return nan
    db = lambda a: 20 * math.log10(a / a1) if a > 0 else float("nan")
    bins_per_f = f / df                     # = 紀錄涵蓋的週期數
    lines = {f"f/{q}": db(max(peak(m * f / q) for m in range(1, q) if math.gcd(m, q) == 1))
             if bins_per_f / q >= 4 else float("nan") for q in orders}
    fk = np.arange(len(spec)) * df
    off = np.ones(len(spec), bool)
    big = max([16] + [q for q in orders if q in POW2_ORDERS and bins_per_f / q >= 10])
    six = 12 if 12 in orders and bins_per_f / 12 >= 10 else 6
    for step in (f / big, f / six, f / 5):
        off &= np.abs(fk - np.round(fk / step) * step) > 3 * df

    def med(lo, hi):
        sel = spec[(fk > lo * f) & (fk < hi * f) & off]
        return db(float(np.median(sel))) if len(sel) else float("nan")

    return dict(lines=lines, floor=med(0.05, 1.45), floor_hf=med(3.0, 6.0))


def classify_period(t, v, f, ref=None, tol=0.03, sub_dbc=-30.0, confirm_db=12.0, chaos_db=15.0,
                    n_phases=16, min_votes=2, chaos_floor_dbc=-45.0, periods=(1, 2, 3, 4, 5, 6, 8),
                    extra_frames=(), lyap=None, lyap_thr=0.1, chaos_method="spectrum", weak_upgrade=False):
    """判斷週期，避免單一取樣相位剛好落在分支交叉處而誤判。

    1. 時域：在 n_phases 個等間隔相位各取一串「每週期取樣值」，再加上每週期最大值，共 n_phases+1 串，
       各自用 detect_period（periods 中的週期，例如 1、2、3、4、5、6、8、12、16、32、64）判斷；
       取「至少 min_votes 串同意」的最大週期 p_time（沒有則 0）。
    2. 頻譜確認（p_conf）：週期 p 需要對應的次諧波族 f/p 高於底噪 confirm_db 以上；
       2 的冪次（2、4、8、16、32、64…）依序逐級確認；3→6→12 逐級（需要明顯譜線）；5 直接看 f/5；
       確認不了就退回已確認的因數（最少為 1），避免雜訊造成假週期。
    3. 頻譜升級（p_spec）：次諧波高於 sub_dbc（很明顯）時判為對應週期：2→4→8→16… 逐級；或 f/3（→ f/6 → f/12）、f/5。
       weak_upgrade=True：時域判出 2 的冪次週期 p 後，若 f/2p、f/4p… 逐級都高於底噪 confirm_db（不必高於 sub_dbc），
       也往上升級。高階倍週期的分支分裂很小（每級約縮小 2.5 倍），時域的容許誤差分不出來，但譜線還看得到。
    4. 綜合：
       - p_time > 0 → period = p_spec（若 p_conf 整除 p_spec），否則 p_conf
       - p_time = 0：判斷是否混沌（chaos_method）：
         "spectrum"：低頻底噪高於 chaos_floor_dbc，且比 3f–6f 底噪高 chaos_db 以上 → 0（混沌，寬頻）；
           （2026-10-21 實測：混沌時低頻底噪約 −33～−40 dBc，比高頻底噪高 25–30 dB；週期態約 −55～−62 dBc）
         "lyapunov"：最大 Lyapunov 指數 lyap > lyap_thr（每驅動週期，自然對數）→ 0（混沌）；
           lyap 算不出來（nan，鄰居太少）時改用 "spectrum" 的判斷；
         "either"：兩者任一成立就算混沌。
         都不成立 → 雜訊太大（通常是訊號太小）→ -1（無法判斷）。週期 3、5、6、12 需要明顯譜線（高於 sub_dbc）才成立。
    extra_frames：同一個振幅多擷取的其他張 [(v_k, ref_k), ...]（時間軸與 t 相同）。
       時域：每個相位用所有張的每週期取樣值一起判斷（detect_period_multi，各張不接在一起）；
       頻譜：各張功率頻譜平均（spectrum_levels(more=...)）。不給就和單張完全相同。
    回傳 dict：period, p_time, p_conf, p_spec, votes, lines, floor, floor_hf, reason, n_frames, lyap。"""
    t = np.asarray(t, float)
    v = np.asarray(v, float)
    frames = [(v, ref)] + [(np.asarray(x, float), r) for x, r in extra_frames if len(x) == len(v)]
    scale = float(np.median([vpp(x) for x, _ in frames]))
    phases = list(np.arange(n_phases) * 360.0 / n_phases)
    series = [[per_cycle_values(t, x, f, "phase", ph, ref=r) for x, r in frames] for ph in phases]
    series.append([per_cycle_values(t, x, f, "max", 0.0, ref=r) for x, r in frames])
    votes = [detect_period_multi(xs, scale, tol, periods=periods) for xs in series]
    p_time = 0
    for p in sorted(periods, reverse=True):
        if sum(1 for q in votes if q == p) >= min_votes:
            p_time = p
            break

    orders = orders_for(periods)
    sp = spectrum_levels(t, v, f, more=[x for x, _ in frames[1:]], orders=orders)
    lines, floor, floor_hf = sp["lines"], sp["floor"], sp["floor_hf"]
    pow2 = [q for q in POW2_ORDERS if q in orders and q <= max(periods)]

    def above(x, ref_db):
        return np.isfinite(x) and (not np.isfinite(ref_db) or x > ref_db)

    def seen(p):          # 譜線高於底噪 confirm_db
        return above(lines.get(f"f/{p}", float("nan")), floor + confirm_db)

    def strong(p):        # 譜線明顯（高於 sub_dbc，且高於底噪 confirm_db）
        return seen(p) and above(lines.get(f"f/{p}", float("nan")), sub_dbc)

    # 2. 頻譜確認
    clean = np.isfinite(floor) and floor < chaos_floor_dbc
    if p_time == 0:
        p_conf = 0
    elif p_time == 1 or p_time in POW2_ORDERS:
        p_conf = 1
        for p in pow2:
            if p > p_time or not seen(p):
                break
            p_conf = p
    elif p_time in (6, 12):   # 非 2 的冪次週期，雜訊也容易湊出來 → 要求明顯的譜線，且頻譜乾淨（底噪低於 chaos_floor_dbc）
        p_conf = 2 if seen(2) else 1
        if clean and strong(3):
            p_conf = 3
            for p in (6, 12):
                if p > p_time or not strong(p):
                    break
                p_conf = p
    else:                     # 3、5
        p_conf = p_time if clean and strong(p_time) else 1
    # 3. 頻譜升級
    p_spec = 1
    for p in pow2:
        if not strong(p):
            break
        p_spec = p
    if p_spec == 1 and strong(3):
        p_spec = 3
        for p in (6, 12):
            if p > max(periods) or not strong(p):
                break
            p_spec = p
    elif p_spec == 1 and strong(5):
        p_spec = 5
    p_weak = p_conf
    if weak_upgrade and p_time > 0 and (p_conf == 1 or p_conf in POW2_ORDERS) and p_conf == p_time:
        for p in pow2:
            if p <= p_weak:
                continue
            if p != 2 * p_weak or not seen(p):
                break
            p_weak = p

    if p_time > 0:
        period = p_spec if (p_spec > p_conf and p_spec % p_conf == 0) else p_conf
        if p_weak > period:
            period, reason = p_weak, f"頻譜（f/{p_weak} 高於底噪，時域分不出）"
        elif period == p_time:
            reason = "時域"
        elif period > p_time:
            reason = "頻譜（次諧波明顯）"
        else:
            reason = "時域（頻譜未確認較高週期）"
    else:
        broad = bool(np.isfinite(floor) and floor > chaos_floor_dbc
                     and (not np.isfinite(floor_hf) or floor - floor_hf > chaos_db))
        has_l = lyap is not None and np.isfinite(lyap)
        pos_l = has_l and lyap > lyap_thr
        if chaos_method == "lyapunov" and has_l:
            chaos, why = pos_l, f"Lyapunov λ = {lyap:+.2f} > {lyap_thr:g}"
        elif chaos_method == "either":
            chaos = broad or pos_l
            why = f"Lyapunov λ = {lyap:+.2f}" if pos_l else "寬頻"
        else:
            chaos, why = broad, "寬頻" + ("（λ 算不出來）" if chaos_method == "lyapunov" else "")
        if chaos:
            period, reason = 0, f"混沌（{why}）"
        else:
            period, reason = -1, "雜訊太大，無法判斷" + (f"（λ = {lyap:+.2f}）" if has_l else "")
    return dict(period=period, p_time=p_time, p_conf=p_conf, p_spec=p_spec, votes=votes, lines=lines,
                floor=floor, floor_hf=floor_hf, reason=reason, n_frames=len(frames), lyap=lyap)


def subharmonic_dbc(t, v, f, divs=(2, 4)):
    """f/2、f/4（divs 可加 8、16…）分量相對於 f 分量的大小 (dBc)，鎖相值（只取 m = 1 那一條）。"""
    a1, _ = lockin(t, v, f)
    out = {}
    for div in divs:
        a, _ = lockin(t, v, f / div)
        out[f"f/{div}"] = 20 * math.log10(a / a1) if a1 > 0 and a > 0 else float("nan")
    return out


# ---------------- Lyapunov 指數 ----------------

def lyapunov_kantz(seqs, scale, tol=0.03, shell=2.5, dim=2, nmax=3, min_ref=10, max_ref=400, max_sat=0.8):
    """由每週期取樣值估計最大 Lyapunov 指數 λ（每個驅動週期，自然對數；Kantz 法）。
    嵌入向量 y_n = (x_{n−dim+1}, …, x_n)；對參考點 i 找距離（最大範數）在 [tol·scale, shell·tol·scale) 的鄰居 j，
    S(Δn) = < ln( mean_j |x_{i+Δn} − x_{j+Δn}| ) >_i，λ = S(Δn) 在 Δn = 0…nmax 的斜率。
    鄰居距離的下限 = 週期判斷的容許誤差：比這更近的點和雜訊分不開。若用更近的點，雜訊一步就讓它們
    分開到雜訊大小，週期態也會得到假的正 λ（2026-10-21 實測資料上試過：週期 1 會得到 +0.9）。
    在 10/21 的 2.2 MHz、1.9 MHz 實測：混沌 λ 中位數約 +0.34，週期 1、2、3、4、6、8 約 −0.2～0。
    seqs：各張的每週期取樣值（依時間順序，各張分開演化，鄰居可以來自別張）；scale：CH2 峰對峰值。
    雜訊檢查：純雜訊（雜訊比容許誤差大）時，鄰居一步就分開到「隨機兩點的平均距離」，之後不再變大；
    混沌則是一步一步變大。sat = (S(1) − S(0)) / (S_隨機 − S(0))，sat > max_sat 視為雜訊 → valid = False。
    回傳 dict(lyap, n_ref, S, sat, valid)：有鄰居的參考點少於 min_ref（例如週期態的點都擠在雜訊範圍內）時 lyap = nan。"""
    nan = dict(lyap=float("nan"), n_ref=0, S=None, sat=float("nan"), valid=False)
    seqs = [np.asarray(x, float) for x in seqs]
    seqs = [x for x in seqs if len(x) >= dim + nmax + 2]
    if not seqs or not scale or scale <= 0:
        return nan
    lo, hi = tol * scale, shell * tol * scale
    Y, fut = [], []
    for x in seqs:
        for n in range(dim - 1, len(x) - nmax):
            Y.append(x[n - dim + 1:n + 1])
            fut.append(x[n:n + nmax + 1])
    Y, fut = np.array(Y), np.array(fut)
    refs = np.unique(np.linspace(0, len(Y) - 1, min(max_ref, len(Y))).astype(int))
    S, cnt = np.zeros(nmax + 1), 0
    for i in refs:
        d = np.max(np.abs(Y - Y[i]), axis=1)
        nb = (d >= lo) & (d < hi)
        if not nb.any():
            continue
        sep = np.abs(fut[nb] - fut[i]).mean(axis=0)
        S += np.log(np.maximum(sep, 1e-6 * lo))
        cnt += 1
    if cnt < min_ref:
        return dict(nan, n_ref=cnt)
    S /= cnt
    allx = np.concatenate(seqs)
    rng = np.random.default_rng(0)
    a, b = rng.integers(0, len(allx), (2, 4000))
    s_rand = math.log(max(float(np.mean(np.abs(allx[a] - allx[b]))), 1e-6 * lo))
    sat = (S[1] - S[0]) / (s_rand - S[0]) if s_rand > S[0] else float("inf")
    return dict(lyap=float(np.polyfit(np.arange(nmax + 1), S, 1)[0]), n_ref=cnt, S=S - S[0],
                sat=float(sat), valid=bool(sat <= max_sat))


# ---------------- 頻譜圖（分岔分析的 FFT 圖） ----------------

SPEC_GRID = np.linspace(0.0, 2.0, 801)     # f / f_drive


def spectrum_map_row(t, vs, f, grid=SPEC_GRID):
    """多張波形的平均功率頻譜（Hann），換成 dBc（相對 f 分量），重新取樣到 f/f_drive 的固定格點上
    （每格取最大值，保留窄譜線）。給分岔分析畫「頻譜 vs 振幅」圖用。"""
    t = np.asarray(t, float)
    vs = [np.asarray(x, float) for x in vs]
    n = len(vs[0])
    if n < 64:
        return np.full(len(grid), np.nan)
    dt = (t[-1] - t[0]) / (n - 1)
    win = np.hanning(n)
    p = np.mean([np.abs(np.fft.rfft((x - x.mean()) * win)) ** 2 for x in vs if len(x) == n], axis=0)
    fk = np.arange(len(p)) / (n * dt) / f
    k1 = int(round(1.0 / fk[1])) if len(fk) > 1 else 0
    a1 = p[max(k1 - 2, 1):k1 + 3].max() if 0 < k1 < len(p) - 3 else p.max()
    db = 10 * np.log10(np.maximum(p / a1, 1e-15))
    half = (grid[1] - grid[0]) / 2
    idx = np.searchsorted(fk, np.r_[grid - half, grid[-1] + half])
    out = np.full(len(grid), np.nan)
    for g in range(len(grid)):
        a, b = idx[g], idx[g + 1]
        out[g] = db[a:b].max() if b > a else np.interp(grid[g], fk, db)
    return out.astype(np.float32)


# ---------------- 費根鮑姆常數 ----------------

FEIGENBAUM_DELTA = 4.669201609


def period_runs(amps, periods, keep=2):
    """週期判斷 → 連續區段 [(週期, 起點振幅, 終點振幅, 起點索引, 終點索引)]。
    忽略「無法判斷」(-1)；少於 keep 點、而且和前後區段不同的零星判斷視為雜訊並略過。"""
    amps = np.asarray(amps, float)
    per = np.asarray(periods, int)
    use = np.where(per >= 0)[0]
    out, s = [], 0
    for j in range(1, len(use) + 1):
        if j == len(use) or per[use[j]] != per[use[s]]:
            p = int(per[use[s]])
            if j - s >= keep or (out and out[-1][0] == p):
                if out and out[-1][0] == p:
                    out[-1] = (p, out[-1][1], amps[use[j - 1]], out[-1][3], int(use[j - 1]))
                else:
                    out.append((p, amps[use[s]], amps[use[j - 1]], int(use[s]), int(use[j - 1])))
            s = j
    return out


def _threshold(amps, level_db, k_on, win, hi_idx, gate, nfit):
    """倍週期閾值：在週期判斷的起點 k_on ± win 點內找次諧波「連續 2 點」超過 gate (dBc) 的第一點當起點
    （找不到就用 k_on），取起點後 nfit 點（不超過 hi_idx）的鎖相振幅平方 a² 對 A 做線性外插到 0
    （倍週期分岔附近 a ∝ √(A − A_n)）。外插值不在 [前一點 − 2 步距, 起點] 內、或點數不足時，改用前一點與起點的中點。
    回傳 dict(A, err, how, k, fit=(x, y, slope, icpt) 或 None)。"""
    amps = np.asarray(amps, float)
    lev = np.asarray(level_db, float)
    step = float(np.median(np.diff(amps))) if len(amps) > 1 else 0.0
    cand = [j for j in range(max(k_on - win, 1), min(k_on + win, hi_idx, len(amps) - 2) + 1)
            if np.isfinite(lev[j]) and lev[j] > gate and np.isfinite(lev[j + 1]) and lev[j + 1] > gate]
    k = cand[0] if cand else k_on
    k = max(k, 1)
    sel = np.arange(k, min(k + nfit, hi_idx + 1, len(amps)))
    sel = sel[np.isfinite(lev[sel])]
    mid = dict(A=0.5 * (amps[k - 1] + amps[k]), err=0.5 * abs(amps[k] - amps[k - 1]), how="中點", k=k, fit=None)
    if len(sel) < 3:
        return mid
    x, y = amps[sel], (10 ** (lev[sel] / 20)) ** 2
    (sl, ic), cov = np.polyfit(x, y, 1, cov=True)
    if not sl > 0:
        return mid
    a0 = -ic / sl
    if not (amps[k - 1] - 2 * step <= a0 <= amps[k]):
        return mid
    # 外插值的誤差：由擬合的共變異數傳遞（A0 = −b/a）
    g = np.array([ic / sl ** 2, -1 / sl])
    err = float(math.sqrt(max(g @ cov @ g, 0.0)))
    return dict(A=float(a0), err=max(err, 0.05 * step), how=f"f/{{}} a² 外插（{len(sel)} 點）", k=k, fit=(x, y, sl, ic))


def feigenbaum_analysis(amps, periods, sublevels, gates=None, nfit=4, keep=2):
    """由分岔圖的週期判斷與次諧波鎖相值，求倍週期閾值 A_n 與費根鮑姆常數 δ（方法與 linsay_freq_analysis.py 相同）。
    amps、periods：各振幅（由小到大）；sublevels：{2: f/2 的 dBc 陣列, 4: …, 8: …, 16: …}。
    1. 週期判斷 → 連續區段；找第一個混沌區段，往前找通往它的倍週期序列 …、8、4、2、1（之前倒退的「泡泡」另外列出）。
       沒有混沌時，用最後一個 2 的冪次區段往前找。
    2. 每個 1→2、2→4、… 的閾值 A_n：次諧波 f/2^n 的 a² 線性外插（_threshold）。
    3. A_c（混沌起點）= 最後一個週期點與第一個混沌點的中點。
    4. δ_n = (A_n − A_{n−1}) / (A_{n+1} − A_n)；δ（用 A_c）≈ 1 + (A_N − A_{N−1}) / (A_c − A_N)（假設 A_∞ − A_n ∝ δ^−n）。
    回傳 dict(runs, cascade, bubbles, thresholds=[dict(n, p, A, err, how, …)], Ac, Ac_err, deltas, delta_c, text)。"""
    gates = gates or {}
    amps = np.asarray(amps, float)
    per = np.asarray(periods, int)
    runs = period_runs(amps, per, keep)
    out = dict(runs=runs, cascade=[], bubbles=[], thresholds=[], Ac=float("nan"), Ac_err=float("nan"),
               deltas=[], delta_c=None, text="")
    lines = ["週期區段：" + "  ".join(f"[{period_label(p)}: {a:.3f}–{b:.3f}]" for p, a, b, _, _ in runs)]
    pw = lambda p: p >= 1 and (p & (p - 1)) == 0
    ic = next((k for k, r in enumerate(runs) if r[0] == 0), None)
    before = runs[:ic] if ic is not None else runs
    # 序列的終點：混沌之前「最高的」2 的冪次區段（第一次出現的那個）；之後倒退的區段當作泡泡
    top = max((r[0] for r in before if pw(r[0])), default=None)
    last = next((r for r in before if r[0] == top), None)
    if last is None:
        out["text"] = "\n".join(lines + ["找不到倍週期序列，無法計算。"])
        return out
    # 往前找 …、4、2、1；步距太粗時某一級可能沒有區段（例如 4 直接跳到 16），就跳過那一級
    seq, want = [], last[0]
    for r in reversed(before[:before.index(last) + 1]):
        if pw(r[0]) and r[0] <= want:
            seq.append(r)
            want = r[0] // 2
            if want == 0:
                break
    seq = seq[::-1]                       # 週期由小到大
    out["cascade"] = seq
    out["bubbles"] = [r for r, nx in zip(before, before[1:] + ([runs[ic]] if ic is not None else []))
                      if nx[0] and r[0] and nx[0] < r[0] and r not in seq]
    if out["bubbles"]:
        lines.append("注意：通往混沌之前另有倒退回較低週期的區段（泡泡），不列入序列：" +
                     "、".join(f"{period_label(p)} @ {a:.3f}–{b:.3f}" for p, a, b, _, _ in out["bubbles"]))
    for prev, cur in zip(seq, seq[1:]):
        p = cur[0]
        if p != 2 * prev[0]:
            lines.append(f"週期 {prev[0]} → {p} 之間缺了中間的倍週期（步距太粗或判斷不出），這幾個閾值不計算。")
            continue
        lev = sublevels.get(p)
        gate = gates.get(p, -35.0)
        # 起點 = 前一級區段結束後的第一點（中間若有倒退的泡泡，cur 是泡泡之後的區段，但分岔發生在前一級結束時）
        k_on = min(prev[4] + 1, len(amps) - 1)
        if lev is None:
            th = dict(A=0.5 * (amps[k_on - 1] + amps[k_on]), err=0.5 * abs(amps[k_on] - amps[k_on - 1]),
                      how="中點", k=k_on, fit=None)
        else:
            th = _threshold(amps, lev, k_on, 3, cur[4], gate, nfit)
        th["how"] = th["how"].replace("{}", str(p))
        n = int(round(math.log2(p)))
        th.update(n=n, p=p, gate=gate)
        out["thresholds"].append(th)
        lines.append(f"A{n}（{p // 2}→{p}）= {th['A']:.4f} ± {th['err']:.4f} Vpp　［{th['how']}；"
                     f"週期判斷起點 {amps[k_on]:.4f}］")
    if ic is not None:
        k0 = runs[ic][3]
        prev_ok = [j for j in range(k0) if per[j] > 0]
        if prev_ok:
            j = prev_ok[-1]
            out["Ac"], out["Ac_err"] = 0.5 * (amps[j] + amps[k0]), 0.5 * (amps[k0] - amps[j])
        else:
            out["Ac"], out["Ac_err"] = amps[k0], 0.0
        lines.append(f"Ac（混沌起點）= {out['Ac']:.4f} ± {out['Ac_err']:.4f} Vpp（最後一個週期點與第一個混沌點的中點）")
    TH = {th["n"]: th for th in out["thresholds"]}

    def ratio(x1, e1, x2, e2, x3, e3):
        """δ = (x2 − x1) / (x3 − x2) 與誤差（x2 同時出現在分子分母）。"""
        d1, d2 = x2 - x1, x3 - x2
        if not (d1 > 0 and d2 > 0):
            return None
        g = (-1 / d2, 1 / d2 + d1 / d2 ** 2, -d1 / d2 ** 2)
        return d1 / d2, math.sqrt(sum((gi * ei) ** 2 for gi, ei in zip(g, (e1, e2, e3))))

    for n1 in sorted(TH):
        if n1 + 1 in TH and n1 + 2 in TH:
            r = ratio(*(v for k in (n1, n1 + 1, n1 + 2) for v in (TH[k]["A"], TH[k]["err"])))
            if r:
                out["deltas"].append(dict(n=n1, delta=r[0], err=r[1]))
                lines.append(f"δ{n1} = (A{n1 + 1} − A{n1}) / (A{n1 + 2} − A{n1 + 1}) = {r[0]:.3f} ± {r[1]:.3f}")
    nl = max(TH) if TH else None
    if nl is not None and nl - 1 in TH and np.isfinite(out["Ac"]) and out["Ac"] > TH[nl]["A"]:
        r = ratio(TH[nl - 1]["A"], TH[nl - 1]["err"], TH[nl]["A"], TH[nl]["err"], out["Ac"], out["Ac_err"])
        if r:
            out["delta_c"] = dict(delta=1 + r[0], err=r[1])
            lines.append(f"δ（用 Ac 近似 A∞）≈ 1 + (A{nl} − A{nl - 1}) / (Ac − A{nl}) = {1 + r[0]:.3f} ± {r[1]:.3f}")
    if not out["deltas"] and out["delta_c"] is None:
        lines.append(f"連續的閾值只有 {len(TH)} 個（至少要 3 個連續的，或最後 2 個加上混沌起點），無法計算 δ。")
    lines.append(f"理論 δ = {FEIGENBAUM_DELTA:.7f}")
    out["text"] = "\n".join(lines)
    return out


def suggest_vscale(wave, scales, half_divs, fill=None):
    """依目前波形建議新的垂直刻度；不需要改就回傳 None。
    fill=None：只在波形太小（< 0.3 個半螢幕）時才縮小（掃頻用，對雜訊尖峰比較保守）。
    fill=0.9 之類：直接選「波形峰值 ≤ fill × 半螢幕」的最小刻度，盡量填滿畫面以降低量化雜訊（分岔圖用）。"""
    peak = float(np.max(np.abs(wave.screen_div))) if len(wave.screen_div) else 0.0
    idx = nearest_index(scales, wave.vscale)
    if peak > half_divs * 0.95:           # 被截掉了 → 放大刻度
        if idx < len(scales) - 1:
            return scales[min(idx + 2, len(scales) - 1)]
        return None
    if fill is not None:
        if peak <= 0:
            return None
        best = pick_scale(scales, peak * wave.vscale / (half_divs * fill))
        return best if abs(best - wave.vscale) > wave.vscale * 0.01 else None
    if peak < half_divs * 0.3:            # 太小 → 縮小刻度，讓波形約佔 3/4 螢幕
        new = pick_scale(scales, peak * wave.vscale / (half_divs * 0.75))
        if new < wave.vscale * 0.99:
            return new
    return None

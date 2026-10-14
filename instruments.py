"""儀器控制：GW Instek GDS-1000B 示波器、AFG-2100 系列函數產生器，以及不接儀器時用的模擬器。

兩台儀器透過 USB 連線時，在電腦上都是虛擬 COM 埠（USB CDC），用 SCPI 文字指令溝通。
"""
import math
import re
import threading
import time
from dataclasses import dataclass, field

import numpy as np
import serial
from serial.tools import list_ports

import config
from analysis import estimate_frequency, series_125, vpp

VOLT_SCALES = series_125(1e-3, 10)    # GDS-1000B：1 mV/div ~ 10 V/div
TIME_SCALES = series_125(5e-9, 50)    # GDS-1000B：5 ns/div ~ 50 s/div

AFG_FUNCS = {           # 介面顯示名稱 -> (SCPI 函數, 對稱度)
    "正弦波": ("SIN", None),
    "方波": ("SQU", None),
    "三角波": ("RAMP", 50),
    "鋸齒波": ("RAMP", 100),
}
AFG_FREQ_RANGE = (0.1, 12e6)    # AFG-2112：0.1 Hz ~ 12 MHz（正弦/方波）


def list_com_ports():
    """回傳 [(埠, 說明)]，USB（GW Instek 優先）排前面、藍牙排最後。"""
    def key(p):
        return (p.vid is None, p.vid != GW_INSTEK_VID, p.device)
    out = []
    for p in sorted(list_ports.comports(), key=key):
        desc = p.description
        if p.vid == GW_INSTEK_VID:
            desc = f"GW Instek USB — {desc}"
        elif "BTHENUM" in (p.hwid or "").upper():
            desc = f"藍牙（不是儀器）— {desc}"
        out.append((p.device, desc))
    return out


@dataclass
class Waveform:
    ch: int
    t: np.ndarray            # 時間 (s)
    v: np.ndarray            # 電壓 (V)
    screen_div: np.ndarray   # 在螢幕上的位置（格），用來判斷是否被截掉
    vscale: float            # V/div
    dt: float                # 取樣間隔 (s)
    header: dict = field(default_factory=dict)
    vpos: float = 0.0        # 垂直位置偏移 (V)；螢幕上的位置 = (v + vpos) / vscale


def _first_float(text):
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text or "")
    return float(m.group()) if m else None


def parse_header(raw):
    """把 'Key,Value;Key,Value;...' 形式的標頭解析成 dict（key 轉小寫）。"""
    text = raw.decode("ascii", errors="replace").replace("\r", "").replace("\n", ";")
    out = {}
    for item in text.split(";"):
        if "," in item:
            k, v = item.split(",", 1)
            out[k.strip().lower()] = v.strip()
    return out


def header_float(header, *names):
    for name in names:                       # 先找完全相同的 key
        if name in header:
            v = _first_float(header[name])
            if v is not None:
                return v
    for name in names:                       # 再找包含該字串的 key
        for k, val in header.items():
            if name in k:
                v = _first_float(val)
                if v is not None:
                    return v
    return None


class SerialInstrument:
    def __init__(self, port, timeout=config.TIMEOUT_S):
        self.port = port
        self.ser = serial.Serial(port, config.BAUDRATE, timeout=timeout)
        self.lock = threading.RLock()
        time.sleep(0.1)
        self.ser.reset_input_buffer()

    def write(self, cmd):
        with self.lock:
            self.ser.write((cmd + "\n").encode("ascii"))
            self.ser.flush()
            time.sleep(config.WRITE_DELAY_S)

    def readline(self):
        line = self.ser.readline()
        if not line:
            raise TimeoutError(f"{self.port} 沒有回應")
        return line.decode("ascii", errors="replace").strip()

    def query(self, cmd):
        with self.lock:
            self.write(cmd)
            return self.readline()

    def query_float(self, cmd):
        v = _first_float(self.query(cmd))
        return float("nan") if v is None else v

    def _read_exact(self, n):
        buf = bytearray()
        while len(buf) < n:
            chunk = self.ser.read(n - len(buf))
            if not chunk:
                raise TimeoutError(f"{self.port} 資料傳輸逾時（收到 {len(buf)}/{n} bytes）")
            buf += chunk
        return bytes(buf)

    def _drain(self, wait=0.05):
        old = self.ser.timeout
        self.ser.timeout = wait
        try:
            while self.ser.read(256):
                pass
        finally:
            self.ser.timeout = old

    def idn(self):
        return self.query("*IDN?")

    def close(self):
        try:
            self.ser.close()
        except Exception:
            pass


class GDS1000B(SerialInstrument):
    """GW Instek GDS-1000B 系列示波器（例如 GDS-1072B）。"""

    def setup(self):
        self.set_record_length(config.RECORD_LENGTH)

    def read_raw(self, ch):
        """送出讀波形指令，回傳 (標頭 bytes, 資料 bytes)。"""
        with self.lock:
            self.write(config.WAVEFORM_QUERY.format(ch=ch))
            head = self.ser.read_until(b"#")
            if not head.endswith(b"#"):
                raise TimeoutError(f"讀 CH{ch} 波形逾時，收到：{head[:80]!r}")
            n = int(self._read_exact(1))
            length = int(self._read_exact(n))
            data = self._read_exact(length)
            self._drain()
        return head[:-1], data

    def read_waveform(self, ch):
        head, data = self.read_raw(ch)
        header = parse_header(head)
        dtype = np.dtype(config.DATA_DTYPE)
        data = data[: len(data) // dtype.itemsize * dtype.itemsize]
        raw = np.frombuffer(data, dtype=dtype).astype(float)
        n = len(raw)
        vscale = header_float(header, "vertical scale") or self.get_ch_scale(ch)
        vpos = header_float(header, "vertical position") or 0.0
        dt = header_float(header, "sampling period")
        if not dt:
            dt = self.get_timebase() * 10 / max(n, 1)
        if config.SUBTRACT_POSITION:      # 原始值是螢幕位置（含垂直位置偏移）
            screen_div = raw / config.COUNTS_PER_DIV
            v = screen_div * vscale - vpos
        else:                             # 原始值以接地為基準（GDS-1072B V1.21 實測如此）
            v = raw / config.COUNTS_PER_DIV * vscale
            screen_div = (v + vpos) / vscale
        t = (np.arange(n) - n / 2) * dt
        return Waveform(ch, t, v, screen_div, vscale, dt, header, vpos)

    def measure(self, ch, item):
        """item 例如 'PK2PK'、'FREQuency'、'AMPlitude'、'MEAN'。"""
        with self.lock:
            self.write(f":MEASure:SOURce1 CH{ch}")
            return self.query_float(f":MEASure:{item}?")

    def run(self):
        self.write(":RUN")

    def stop(self):
        self.write(":STOP")

    def single(self):
        self.write(":SINGle")

    def autoset(self):
        self.write(":AUTOSet")

    def set_timebase(self, sec_per_div):
        self.write(f":TIMebase:SCALe {sec_per_div:.3e}")

    def get_timebase(self):
        return self.query_float(":TIMebase:SCALe?")

    def set_ch_scale(self, ch, volt_per_div):
        self.write(f":CHANnel{ch}:SCALe {volt_per_div:.3e}")

    def get_ch_scale(self, ch):
        return self.query_float(f":CHANnel{ch}:SCALe?")

    def set_ch_display(self, ch, on):
        self.write(f":CHANnel{ch}:DISPlay {'ON' if on else 'OFF'}")

    def set_coupling(self, ch, mode):
        """mode：'AC'、'DC' 或 'GND'。"""
        self.write(f":CHANnel{ch}:COUPling {mode}")

    def get_coupling(self, ch):
        return self.query(f":CHANnel{ch}:COUPling?").strip().upper()

    def set_probe(self, ch, ratio):
        self.write(f":CHANnel{ch}:PROBe:RATio {ratio:.3e}")

    def get_probe(self, ch):
        return self.query_float(f":CHANnel{ch}:PROBe:RATio?")

    def set_bwlimit(self, ch, on):
        self.write(f":CHANnel{ch}:BWLimit {config.BWLIMIT_ON if on else config.BWLIMIT_OFF}")

    def get_bwlimit(self, ch):
        r = self.query(f":CHANnel{ch}:BWLimit?").strip().upper()
        return r not in ("OFF", "0", "FULL", config.BWLIMIT_OFF.upper())

    def set_acquire_average(self, n):
        """n <= 1：一般取樣；n = 2, 4, ..., 256：示波器硬體平均。"""
        if n and n > 1:
            self.write(":ACQuire:MODe AVERage")
            self.write(f":ACQuire:AVERage {int(n)}")
        else:
            self.write(":ACQuire:MODe SAMPle")

    def set_record_length(self, n):
        self.write(f":ACQuire:RECOrdlength {float(n):.0e}")


def check_drive(vpp_, offset):
    """產生器輸出的安全檢查，超過 config 的上限就丟出 ValueError。"""
    if vpp_ > config.AMPLITUDE_MAX_VPP + 1e-12:
        raise ValueError(f"振幅 {vpp_:g} Vpp 超過上限 {config.AMPLITUDE_MAX_VPP:g} Vpp"
                         "（可在 config.py 的 AMPLITUDE_MAX_VPP 修改）")
    if abs(offset) + vpp_ / 2 > config.PEAK_MAX_V + 1e-12:
        raise ValueError(f"|偏移| + 振幅/2 = {abs(offset) + vpp_ / 2:g} V 超過上限 {config.PEAK_MAX_V:g} V"
                         "（可在 config.py 的 PEAK_MAX_V 修改）")


class AFG2100(SerialInstrument):
    """GW Instek AFG-2100/2000 系列函數產生器（例如 AFG-2112）。"""

    def __init__(self, port, timeout=config.TIMEOUT_S):
        super().__init__(port, timeout)
        self._amp, self._off = 0.0, 0.0    # 記住目前設定，用來做安全檢查

    def set_function(self, name):
        func, symm = AFG_FUNCS[name]
        self.write(f"SOUR1:FUNC {func}")
        if symm is not None:
            self.write(f"SOUR1:RAMP:SYMM {symm}")

    def set_frequency(self, hz):
        self.write(f"SOUR1:FREQ {hz:.9g}")

    def set_amplitude(self, vpp_):
        check_drive(vpp_, self._off)
        self.write(f"SOUR1:AMPL {vpp_:.6g}")
        self._amp = vpp_

    def set_offset(self, volt):
        check_drive(self._amp, volt)
        self.write(f"SOUR1:DCO {volt:.6g}")
        self._off = volt

    def set_output(self, on):
        self.write(f"OUTP {'ON' if on else 'OFF'}")

    def set_load(self, high_z):
        """True = High-Z（INFinity），False = 50 Ω（DEFault）。"""
        self.write(f"OUTP:LOAD {'INF' if high_z else 'DEF'}")

    def get_load(self):
        """回傳 True = High-Z，False = 50 Ω。"""
        return "INF" in self.query("OUTP:LOAD?").upper()

    def get_frequency(self):
        return self.query_float("SOUR1:FREQ?")

    def get_amplitude(self):
        v = self.query_float("SOUR1:AMPL?")
        if np.isfinite(v):
            self._amp = v
        return v

    def get_offset(self):
        v = self.query_float("SOUR1:DCO?")
        if np.isfinite(v):
            self._off = v
        return v

    def error(self):
        return self.query("SYST:ERR?")


GW_INSTEK_VID = 0x2184


def candidate_ports():
    """可能是儀器的 COM 埠：只取 USB 裝置（GW Instek 的排最前面），略過藍牙序列埠。
    藍牙序列埠一打開就會嘗試連線配對的裝置，可能卡住很久。"""
    usb = [p for p in list_ports.comports() if p.vid is not None]
    usb.sort(key=lambda p: (p.vid != GW_INSTEK_VID, p.device))
    return [p.device for p in usb]


def _probe_idn(port, result):
    try:
        s = serial.Serial(port, config.BAUDRATE, timeout=0.8)
    except Exception:
        return
    try:
        s.reset_input_buffer()
        s.write(b"*IDN?\n")
        result[port] = s.readline().decode("ascii", errors="replace").strip()
    except Exception:
        pass
    finally:
        s.close()


def auto_detect(ports=None, per_port_timeout=2.0):
    """對每個候選 COM 埠送 *IDN?，找出示波器與函數產生器。回傳 {'scope': (port, idn), 'afg': (...)}
    每個埠最多等 per_port_timeout 秒，卡住的埠直接略過。"""
    found, result = {}, {}
    for port in ports or candidate_ports():
        th = threading.Thread(target=_probe_idn, args=(port, result), daemon=True)
        th.start()
        th.join(per_port_timeout)
        up = result.get(port, "").upper()
        if "GDS" in up:
            found["scope"] = (port, result[port])
        elif "AFG" in up or "SFG" in up:
            found["afg"] = (port, result[port])
        if "scope" in found and "afg" in found:
            break
    return found


# ======================================================================
# 模擬器：不接儀器時用來測試介面。
#   SIM_MODEL = "rlc"：產生器(50 Ω) → L → C → R_sense → GND；CH1 = 產生器輸出端，CH2 = R_sense 上的電壓
#   SIM_MODEL = "rc" ：CH1 = 產生器輸出，CH2 = RC 低通濾波後的訊號
# ======================================================================

class SimBench:
    def __init__(self):
        self.func, self.symm = "SIN", None
        self.freq, self.amp, self.offset, self.output = 1000.0, 2.0, 0.0, False
        self.high_z = True        # 產生器的負載設定
        self.chaos = False        # 分岔圖模式時由程式打開：CH2 振幅依 logistic map 逐週期變化
        self.rng = np.random.default_rng()
        self.afg = SimAFG(self)
        self.scope = SimScope(self)

    # ---- 電路模型 ----
    def f0(self):
        f0 = config.SIM_RLC_F0
        if config.SIM_NONLINEAR:
            a, fr = config.SIM_FRES_TABLE
            f0 = float(np.exp(np.interp(np.log(max(self.amp, 1e-3)), np.log(a), np.log(fr))))
            f0 *= (1 + max(self.offset, 0.0) / 0.7) ** 0.25      # 反向偏壓越大，電容越小
        return f0

    def transfer(self, freqs, ch):
        """CH 上的電壓 / 產生器開路電壓（複數）。"""
        freqs = np.asarray(freqs, float)
        if config.SIM_MODEL == "rc":
            return np.ones(len(freqs), complex) if ch == 1 else 1 / (1 + 1j * freqs / config.SIM_RC_CUTOFF_HZ)
        w = 2 * np.pi * freqs
        f0, L = self.f0(), config.SIM_L
        C = 1 / ((2 * np.pi * f0) ** 2 * L)
        r_total = max(2 * np.pi * f0 * L / config.SIM_RLC_Q, config.SIM_R_SENSE)
        z = r_total + 1j * w * L + 1 / (1j * w * C)
        node = z / (config.SIM_SOURCE_R + z)
        return node if ch == 1 else node * config.SIM_R_SENSE / z

    def _harmonics(self):
        f = self.freq
        kmax = int(max(1, min(199, 2e7 / f)))
        if self.func == "SIN":
            return np.array([1]), np.array([1.0])
        if self.func == "SQU":
            k = np.arange(1, kmax + 1, 2)
            return k, 4 / (np.pi * k)
        if self.func == "RAMP" and self.symm == 50:
            k = np.arange(1, kmax + 1, 2)
            return k, 8 / np.pi ** 2 * (-1.0) ** ((k - 1) // 2) / k ** 2
        k = np.arange(1, kmax + 1)
        return k, 2 / np.pi * (-1.0) ** (k + 1) / k

    def _chaos_env(self, t, ph):
        """每個週期一個振幅係數（logistic map），振幅越大越接近混沌。週期交界在正弦過零處，波形連續。"""
        a0, a1 = config.SIM_CHAOS_RANGE_VPP
        r = 2.8 + 1.2 * float(np.clip((self.amp - a0) / (a1 - a0), 0, 1))
        cyc = np.floor((2 * np.pi * self.freq * t + ph) / (2 * np.pi)).astype(int)
        c0, n = cyc.min(), cyc.max() - cyc.min() + 1
        x = 0.3 + 0.4 * self.rng.random()
        for _ in range(500):
            x = r * x * (1 - x)
        xs = np.empty(n)
        for i in range(n):
            x = r * x * (1 - x)
            xs[i] = x
        return 0.5 + xs[cyc - c0]

    def signal(self, t, ch, include_dc=True):
        if not self.output:
            return np.zeros_like(t)
        voc = 1.0 if self.high_z else 2.0          # 50 Ω 設定時，開路電壓是設定值的 2 倍
        k, a = self._harmonics()
        h = self.transfer(k * self.freq, ch)
        phase = 2 * np.pi * self.freq * np.outer(t, k) + np.angle(h)
        x = (np.sin(phase) * (a * np.abs(h))).sum(axis=1) * self.amp / 2 * voc
        if ch == 2 and self.chaos and self.func == "SIN":
            x = x * self._chaos_env(t, float(np.angle(h[0])))
        dc_gain = 1.0 if ch == 1 or config.SIM_MODEL == "rc" else 0.0   # 電容擋直流
        return x + (self.offset * voc * dc_gain if include_dc else 0.0)


class _SimBase:
    port = "模擬"

    def __init__(self, bench):
        self.bench = bench
        self.lock = threading.RLock()

    def close(self):
        pass


class SimAFG(_SimBase):
    def idn(self):
        return "GW INSTEK,AFG-2112(模擬),0,0"

    def set_function(self, name):
        self.bench.func, self.bench.symm = AFG_FUNCS[name]

    def set_frequency(self, hz):
        self.bench.freq = float(hz)

    def set_amplitude(self, vpp_):
        check_drive(vpp_, self.bench.offset)
        self.bench.amp = float(vpp_)

    def set_offset(self, volt):
        check_drive(self.bench.amp, volt)
        self.bench.offset = float(volt)

    def set_output(self, on):
        self.bench.output = bool(on)

    def set_load(self, high_z):
        self.bench.high_z = bool(high_z)

    def get_load(self):
        return self.bench.high_z

    def get_frequency(self):
        return self.bench.freq

    def get_amplitude(self):
        return self.bench.amp

    def get_offset(self):
        return self.bench.offset

    def error(self):
        return '+0, "No error"'


class SimScope(_SimBase):
    def __init__(self, bench):
        super().__init__(bench)
        self.tscale = 2.5e-4
        self.vscale = {1: 0.5, 2: 0.5}
        self.coupling = {1: "DC", 2: "DC"}
        self.probe = {1: 1.0, 2: 1.0}
        self.bw = {1: False, 2: False}
        self.avg = 1
        self.record_length = int(config.RECORD_LENGTH)
        self.rng = np.random.default_rng()

    def idn(self):
        return "GW,GDS-1072B(模擬),0,0"

    def setup(self):
        self.record_length = int(config.RECORD_LENGTH)

    def read_waveform(self, ch):
        n = self.record_length
        time.sleep(0.02 + n * 1e-6)
        dt = self.tscale * 10 / n
        t = (np.arange(n) - n / 2) * dt
        s = self.vscale[ch]
        v = self.bench.signal(t, ch, include_dc=self.coupling[ch] != "AC")
        v = v + self.rng.normal(0, (0.01 * s + 1e-3) / math.sqrt(self.avg), n)
        lim = config.SCREEN_HALF_DIVS * 1.05
        div = np.clip(v / s, -lim, lim)
        return Waveform(ch, t, div * s, div, s, dt, {"source": f"CH{ch}", "模擬": "是"})

    def measure(self, ch, item):
        w = self.read_waveform(ch)
        item = item.upper()
        if item.startswith("PK2PK"):
            return vpp(w.v)
        if item.startswith("FREQ"):
            return estimate_frequency(w.t, w.v)
        if item.startswith("MEAN"):
            return float(np.mean(w.v))
        return float("nan")

    def run(self):
        pass

    def stop(self):
        pass

    def single(self):
        pass

    def autoset(self):
        f = max(self.bench.freq, 1e-3)
        from analysis import pick_scale
        self.tscale = pick_scale(TIME_SCALES, 3 / (10 * f))
        for ch in (1, 2):
            peak = np.max(np.abs(self.bench.signal(np.linspace(0, 1 / f, 400), ch))) + 1e-3
            self.vscale[ch] = pick_scale(VOLT_SCALES, peak / 3)

    def set_timebase(self, sec_per_div):
        self.tscale = float(sec_per_div)

    def get_timebase(self):
        return self.tscale

    def set_ch_scale(self, ch, volt_per_div):
        self.vscale[ch] = float(volt_per_div)

    def get_ch_scale(self, ch):
        return self.vscale[ch]

    def set_ch_display(self, ch, on):
        pass

    def set_coupling(self, ch, mode):
        self.coupling[ch] = mode

    def get_coupling(self, ch):
        return self.coupling[ch]

    def set_probe(self, ch, ratio):
        self.probe[ch] = float(ratio)

    def get_probe(self, ch):
        return self.probe[ch]

    def set_bwlimit(self, ch, on):
        self.bw[ch] = bool(on)

    def get_bwlimit(self, ch):
        return self.bw[ch]

    def set_acquire_average(self, n):
        self.avg = int(n) if n and n > 1 else 1

    def set_record_length(self, n):
        self.record_length = int(n)

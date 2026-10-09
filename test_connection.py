"""連線測試：第一次接上儀器時先跑這個。

用法：
    python test_connection.py                       # 自動偵測
    python test_connection.py --scope COM3 --afg COM4

結果會同時存到 test_log.txt（以及示波器原始資料 scope_raw_ch1.bin），
如果有問題，把 test_log.txt 的內容貼給 Claude 即可。
不加參數時不會改變儀器的任何設定（只做查詢）；加 --calibrate 會設定產生器並按示波器 AUTOSET。
"""
import argparse
import os
import time
import traceback

import numpy as np

import config
from analysis import vpp
from instruments import AFG2100, GDS1000B, auto_detect, list_com_ports

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = []


def log(msg=""):
    print(msg)
    LOG.append(str(msg))


def check_voltage(scope, ch, w=None):
    """比對程式換算的電壓與示波器自己的量測值。"""
    w = w or scope.read_waveform(ch)
    span_counts = (w.screen_div.max() - w.screen_div.min()) * config.COUNTS_PER_DIV
    data_vpp, data_mean = vpp(w.v), float(np.mean(w.v))
    meas_vpp, meas_mean = scope.measure(ch, "PK2PK"), scope.measure(ch, "MEAN")
    log(f"CH{ch}：程式 Vpp = {data_vpp:.4g} V / 示波器 Vpp = {meas_vpp:.4g} V；"
        f"程式平均 = {data_mean:.4g} V / 示波器平均 = {meas_mean:.4g} V（原始值跨度 {span_counts:.0f}）")
    if span_counts < 50:
        log(f"  （CH{ch} 訊號太小，無法校正電壓；請接上訊號後用 --calibrate 再測）")
        return
    if np.isfinite(meas_vpp) and meas_vpp > 0:
        ratio = data_vpp / meas_vpp
        if abs(ratio - 1) < 0.05:
            log(f"  ✔ CH{ch} Vpp 換算正確")
        else:
            log(f"  ✘ CH{ch} Vpp 差了 {ratio:.3g} 倍 → 建議把 config.py 的 COUNTS_PER_DIV 改成 "
                f"{config.COUNTS_PER_DIV * ratio:.4g}")
    if np.isfinite(meas_mean):
        tol = 0.1 * w.vscale + 0.02 * abs(meas_mean)
        if abs(data_mean - meas_mean) <= tol:
            log(f"  ✔ CH{ch} 直流位準正確")
        else:
            log(f"  ✘ CH{ch} 直流位準不符（差 {data_mean - meas_mean:.4g} V，垂直位置 = "
                f"{w.header.get('vertical position')}）→ 可試著把 config.py 的 SUBTRACT_POSITION 改成 "
                f"{not config.SUBTRACT_POSITION}")


def calibrate(scope_port, afg_port):
    """讓產生器輸出已知訊號，檢查示波器兩個通道的電壓換算。"""
    log("\n===== 校正（會改變儀器設定）=====")
    scope, afg = GDS1000B(scope_port), AFG2100(afg_port)
    try:
        afg.set_function("正弦波")
        afg.set_frequency(1000)
        afg.set_amplitude(2.0)
        afg.set_offset(0.5)
        afg.set_output(True)
        log("產生器：1 kHz 正弦波，2 Vpp，偏移 +0.5 V，輸出 ON")
        for ch in (1, 2):
            scope.set_ch_display(ch, True)
        scope.autoset()
        log("示波器 AUTOSET 中，等待 5 秒…")
        time.sleep(5)
        scope.setup()
        time.sleep(1)
        log(f"時基 = {scope.get_timebase():g} s/div；CH1 = {scope.get_ch_scale(1):g} V/div；"
            f"CH2 = {scope.get_ch_scale(2):g} V/div")
        for ch in (1, 2):
            check_voltage(scope, ch)
        afg.set_offset(0)
        log("（產生器偏移已改回 0 V）")
    finally:
        scope.close()
        afg.close()


def test_scope(port):
    log(f"\n===== 示波器 {port} =====")
    scope = GDS1000B(port)
    try:
        log(f"*IDN?  -> {scope.idn()}")
        log(f"時基   -> {scope.query(':TIMebase:SCALe?')}")
        log(f"CH1 V/div -> {scope.query(':CHANnel1:SCALe?')}")
        # 新增功能用到的指令（只查詢）：若回應逾時或怪怪的，表示指令格式要調整
        for cmd in (":CHANnel1:COUPling?", ":CHANnel1:PROBe:RATio?", ":CHANnel1:BWLimit?",
                    ":ACQuire:MODe?", ":ACQuire:AVERage?", ":ACQuire:RECOrdlength?"):
            try:
                log(f"{cmd:24s} -> {scope.query(cmd)}")
            except Exception as e:
                log(f"{cmd:24s} -> ✘ {e}")
                scope._drain()

        # 1) 原始回應
        with scope.lock:
            scope.write(config.WAVEFORM_QUERY.format(ch=1))
            t0 = time.time()
            raw = bytearray()
            old = scope.ser.timeout
            scope.ser.timeout = 0.5
            while True:
                chunk = scope.ser.read(65536)
                if not chunk:
                    break
                raw += chunk
            scope.ser.timeout = old
        log(f"\n波形指令 {config.WAVEFORM_QUERY.format(ch=1)} 收到 {len(raw)} bytes，耗時 {time.time() - t0:.2f} s")
        with open(os.path.join(HERE, "scope_raw_ch1.bin"), "wb") as fp:
            fp.write(raw)
        cut = raw.find(b"#")
        log("---- 標頭（文字部分）----")
        log(raw[:cut if cut >= 0 else 600].decode("ascii", errors="replace"))
        log("---- '#' 之後前 48 bytes（hex）----")
        if cut >= 0:
            log(raw[cut:cut + 48].hex(" "))

        # 2) 用正式程式解析
        w = scope.read_waveform(1)
        log(f"\n解析結果：{len(w.v)} 點，dt = {w.dt:.4g} s，V/div = {w.vscale:g}")
        log(f"原始值範圍：{w.screen_div.min() * config.COUNTS_PER_DIV:.0f} ~ {w.screen_div.max() * config.COUNTS_PER_DIV:.0f}")
        check_voltage(scope, 1, w)

        # 3) 讀取速度
        t0 = time.time()
        for _ in range(5):
            scope.read_waveform(1)
        log(f"連續讀 5 次 CH1：平均 {(time.time() - t0) / 5:.3f} s/次")
    finally:
        scope.close()


def test_afg(port):
    log(f"\n===== 函數產生器 {port} =====")
    afg = AFG2100(port)
    try:
        log(f"*IDN?       -> {afg.idn()}")
        log(f"SOUR1:FREQ? -> {afg.query('SOUR1:FREQ?')}")
        log(f"SOUR1:AMPL? -> {afg.query('SOUR1:AMPL?')}")
        log(f"SOUR1:FUNC? -> {afg.query('SOUR1:FUNC?')}")
        for cmd in ("SOUR1:DCO?", "OUTP:LOAD?", "OUTP?"):
            try:
                log(f"{cmd:11s} -> {afg.query(cmd)}")
            except Exception as e:
                log(f"{cmd:11s} -> ✘ {e}")
                afg._drain()
        log(f"SYST:ERR?   -> {afg.error()}")
    finally:
        afg.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope")
    ap.add_argument("--afg")
    ap.add_argument("--calibrate", action="store_true",
                    help="讓產生器輸出 1 kHz 正弦波並校正示波器電壓換算（AFG 輸出需接到 CH1 與 CH2）")
    args = ap.parse_args()

    log("可用的 COM 埠：")
    ports = list_com_ports()
    for dev, desc in ports:
        log(f"  {dev}: {desc}")
    if not ports:
        log("  （沒有）→ 請確認 USB 線已接上，並已安裝 GW Instek 的 USB 驅動。")

    if not (args.scope or args.afg):
        log("\n自動偵測中（對每個埠送 *IDN?）…")
        found = auto_detect()
        for k, (p, idn) in found.items():
            log(f"  {k}: {p}  {idn}")
        args.scope = found.get("scope", (None,))[0]
        args.afg = found.get("afg", (None,))[0]

    for name, port, fn in (("示波器", args.scope, test_scope), ("函數產生器", args.afg, test_afg)):
        if not port:
            log(f"\n（沒有找到{name}，略過）")
            continue
        try:
            fn(port)
        except Exception:
            log(f"\n✘ {name}測試發生錯誤：")
            log(traceback.format_exc())

    if args.calibrate:
        if args.scope and args.afg:
            try:
                calibrate(args.scope, args.afg)
            except Exception:
                log("\n✘ 校正發生錯誤：")
                log(traceback.format_exc())
        else:
            log("\n校正需要同時找到示波器和函數產生器。")

    path = os.path.join(HERE, "test_log.txt")
    with open(path, "w", encoding="utf-8") as fp:
        fp.write("\n".join(LOG))
    print(f"\n結果已存到 {path}")


if __name__ == "__main__":
    main()

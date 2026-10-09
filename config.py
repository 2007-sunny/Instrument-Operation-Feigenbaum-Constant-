"""使用者設定。接上真實儀器後，若有需要調整的地方大多在這裡改。"""

# ---- 連線 ----
# 填入 COM 埠（例如 "COM3"）可讓程式啟動時預先選好；None 表示在介面上選或按「自動偵測」
SCOPE_PORT = None
AFG_PORT = None
BAUDRATE = 115200        # USB CDC 虛擬 COM 埠其實不看鮑率，保持預設即可
TIMEOUT_S = 2.0          # 等待儀器回應的逾時（秒）
WRITE_DELAY_S = 0.02     # 每送一個指令後稍等一下，避免儀器來不及處理

# ---- 示波器 GDS-1000B ----
# 讀波形的指令，{ch} 會換成通道編號
WAVEFORM_QUERY = ":ACQuire{ch}:MEMory?"
# 記錄長度：越短傳輸越快（GDS-1000B 可選 1e3, 1e4, 1e5, 1e6, 1e7）
RECORD_LENGTH = 1000
# 原始資料格式：">i2" = 16-bit 有號整數、big-endian
DATA_DTYPE = ">i2"
# 原始數值每一格（div）對應多少 count。
# 若存下來的電壓和示波器螢幕上的不一致，執行 test_connection.py，它會建議正確的數值。
# GDS-1072B 實測：25 正確（43 counts @ 5 mV/div = 8.6 mV，與示波器量測一致）
COUNTS_PER_DIV = 25.0
# 換算電壓時是否扣掉垂直位置（position）偏移。
# GDS-1072B（韌體 V1.21）實測：原始值已以接地為基準，不需扣除 → False
SUBTRACT_POSITION = False
# 螢幕垂直方向從中心到邊緣的格數（GDS-1000B 共 8 格，所以是 4）
SCREEN_HALF_DIVS = 4.0

# ---- 即時顯示 ----
LIVE_INTERVAL_S = 0.05   # 每次讀完波形後的最短間隔（秒）

# ---- 儲存 ----
DATA_DIR = "data"        # 預設儲存資料夾（相對於本程式所在位置）

# 頻寬限制的指令參數（:CHANnel<n>:BWLimit <值>）。
# 若示波器不接受，執行 test_connection.py 看 :CHANnel1:BWLimit? 回傳什麼，照格式修改。
BWLIMIT_ON = "ON"
BWLIMIT_OFF = "OFF"

# ---- 安全限制（產生器）----
AMPLITUDE_MAX_VPP = 5.0    # 振幅上限（MV2109 反向耐壓 30 V；4 Vpp 時二極體已擺到約 −12.5 V）
PEAK_MAX_V = 6.0           # |直流偏移| + 振幅/2 的上限 (V)
AFG_MIN_VPP = 0.01         # 停止時「降到最小振幅」用的值

# ---- 掃頻 ----
SMALL_SIGNAL_DIVS = 0.5    # CH1 或 CH2 峰對峰值小於這麼多格 → 標註「訊號太小，不可靠」
AMP_MISMATCH_WARN = 0.30   # CH1 實測與產生器設定值相差超過 30% → 警告（50 Ω / High-Z 設定錯誤）
DEFAULT_L_UH = 91.67       # 電感預設值 (µH)，用來算 C_eff

# ---- 分岔圖 ----
BIF_TOL = 0.03             # 判斷週期：|x[i+p] − x[i]| 的中位數 < BIF_TOL × CH2 峰對峰值 → 視為週期 p

# ---- 模擬模式 ----
SIM_MODEL = "rlc"          # "rlc" = 串聯 RLC 共振（Linsay 電路）；"rc" = RC 低通濾波器
SIM_RC_CUTOFF_HZ = 1000.0  # RC 模型的截止頻率 (Hz)
SIM_RLC_F0 = 1.9e6         # RLC 模型的共振頻率 (Hz)
SIM_RLC_Q = 50.0           # RLC 模型的 Q 值
SIM_L = 91.67e-6           # 電感 (H)
SIM_R_SENSE = 10.0         # 感測電阻 (Ω)
SIM_SOURCE_R = 50.0        # 產生器輸出阻抗 (Ω)
# True：共振頻率隨振幅改變（用 LTspice 預期值內插），也隨 DC offset 改變（模擬變容二極體）
SIM_NONLINEAR = False
SIM_FRES_TABLE = ([0.05, 0.1, 0.2, 0.4, 0.6, 1.0, 1.5, 2.0, 3.0, 4.0],
                  [1930e3, 1845e3, 1706e3, 1501e3, 1340e3, 1075e3, 877e3, 783e3, 665e3, 602e3])
SIM_CHAOS_RANGE_VPP = (0.5, 4.0)  # 分岔圖模擬：振幅在此範圍內由週期 1 走到混沌

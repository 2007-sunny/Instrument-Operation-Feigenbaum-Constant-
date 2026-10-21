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
# 讀波形失敗時的重試次數。GDS-1072B 在 STOP 狀態下偶爾會掉一整塊資料（少 511/1023 bytes），重讀即可。
WAVEFORM_RETRIES = 5
# 重試都失敗時：True = 先用收到的部分資料（標記「資料不完整」）讓量測繼續；False = 停止並報錯
WAVEFORM_ALLOW_PARTIAL = True
# 掃頻 / 分岔圖結束後，自動回頭補測「資料不完整」的點，並用新數據取代
AUTO_REDO_INCOMPLETE = True
REDO_ROUNDS = 3            # 補測最多重複幾輪（每輪只補仍不完整的點）
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
# 實驗結果存在儲存資料夾底下的子資料夾；檔名自動帶頻率、振幅等設定，例如
#   掃頻/掃頻_1MHz-2.5MHz_0.5Vpp_偏移0V_往上_兩階段_20261021_190000.csv
#   分岔圖/分岔_1.4MHz_0.05-4Vpp_40步_8張_20261021_190000/（一次量測一個資料夾，內含 CSV、頻譜、δ、報告圖）
SWEEP_SUBDIR = "掃頻"      # 單次掃頻與批次掃描
BIF_SUBDIR = "分岔圖"
WAVE_SUBDIR = "波形"       # 單張波形（wave_*）與 FFT（fft_*）
# data/ 根目錄只放依實驗整理的資料夾與 資料索引.md（2026-10-21 使用者整理），程式不會直接存到根目錄

# 頻寬限制的指令參數（:CHANnel<n>:BWLimit <值>）。
# GDS-1072B（V1.21）實測：20E+6 = 開（20 MHz），FULL = 關；"OFF" 不被接受。
BWLIMIT_ON = "20E+6"
BWLIMIT_OFF = "FULL"

# ---- 安全限制（產生器）----
AMPLITUDE_MAX_VPP = 20.0    # 振幅上限（MV2109 反向耐壓 30 V；4 Vpp 時二極體已擺到約 −12.5 V）
PEAK_MAX_V = 20.0           # |直流偏移| + 振幅/2 的上限 (V)
AFG_MIN_VPP = 0.01         # 停止時「降到最小振幅」用的值
# AFG-2112（V2.04）實測：負載 High-Z 時，SOUR1:AMPL? / SOUR1:DCO? 回傳實際值的一半（設定則正確）。
# True = 程式讀回時自動乘 2。若換了產生器且讀回的值和面板顯示不同，改成 False 試試。
AFG_HIGHZ_QUERY_HALF = True

# ---- 掃頻 ----
# 「不可靠」的判斷：由殘差雜訊估計的振幅相對誤差 > MAX_REL_ERROR，或訊號小於 MIN_SIGNAL_DIVS 格，
# 或超過 1% 的取樣點碰到 ADC 上下限（±ADC_LIMIT_DIVS 格，GDS-1072B 約 ±5.1 格）。
MAX_REL_ERROR = 0.05
MIN_SIGNAL_DIVS = 0.05
ADC_LIMIT_DIVS = 5.0
# 掃頻時的平均（2026-10-14 在 C–V 電路上實測）：
# - 程式平均：每點讀 N 張，各自鎖相後以 CH1 相位為基準平均 → 振幅、增益、相位都正確，誤差約 ∝ 1/√N。
# - 示波器平均：增益、相位很穩，但小訊號時 CH1 觸發會抖動（每張相位差 ±35°），
#   平均後 CH1/CH2 的「絕對振幅」會偏小（0.05 Vpp 只剩約一半）。用產生器 SYNC 接 EXT TRIG 觸發則無此問題。
DEFAULT_SW_AVG = 4
DEFAULT_SCOPE_AVG = 1
AMP_MISMATCH_WARN = 0.30   # CH1 實測與產生器設定值相差超過 30% → 警告（50 Ω / High-Z 設定錯誤）
DEFAULT_L_UH = 91.67       # 電感預設值 (µH)，用來算 C_eff

# ---- 分岔圖 ----
BIF_TOL = 0.03             # 判斷週期：|x[i+p] − x[i]| 的中位數 < BIF_TOL × CH2 峰對峰值 → 視為週期 p
# 降低雜訊（2026-10-21 雜訊測試的建議）：
BIF_RECORD_LENGTH = 100000 # 分岔圖預設記錄長度（1e5：每個頻率格的雜訊比 1e4 再低約 10 dB）
BIF_SCALE_FILL = 0.9       # 自動刻度讓波形峰值填到半螢幕的 90%（約 ±3.6 格），量化雜訊最小；超過 3.8 格會自動放大
BIF_BWLIMIT = True         # 分岔圖固定開啟 20 MHz 頻寬限制
# 週期判斷（analysis.classify_period，2026-10-21）：16 個取樣相位 + 最大值法各自判斷，取至少 2 串同意的最大週期，
# 再用頻譜確認：f/2、f/4、f/8 需高於底噪 BIF_CONFIRM_DB 才算；高於 BIF_SUB_DBC 時直接升級為該週期。
# 時域判斷不出時，低頻底噪高於 BIF_CHAOS_FLOOR_DBC 且比 3f–6f 底噪高 BIF_CHAOS_DB 以上 → 混沌，否則 → 無法判斷（雜訊）。
BIF_PHASES = 16
BIF_SUB_DBC = -30.0
BIF_CONFIRM_DB = 12.0
BIF_CHAOS_DB = 15.0
BIF_CHAOS_FLOOR_DBC = -45.0
# 會判斷的週期（介面上的「最高判斷週期」再把上限截掉）。12 = 週期 3 窗口裡的 3→6→12。
BIF_PERIODS = (1, 2, 3, 4, 5, 6, 8, 12, 16, 32, 64, 128)
BIF_MAX_PERIOD = 16
# 判斷週期 p 需要的最少週期數 = BIF_CYCLES_PER_PERIOD × p（f/p 譜線才分得開；時域也要至少 3p+2 個週期）
BIF_CYCLES_PER_PERIOD = 10
# 混沌判斷：時域找不到週期時，"lyapunov" = λ > BIF_LYAP_THR 才算混沌（λ 算不出來時改用寬頻判斷）；
# "spectrum" = 舊的寬頻判斷；"either" = 任一成立
BIF_CHAOS_METHOD = "lyapunov"
BIF_LYAP_THR = 0.1         # 每驅動週期（自然對數）。10/21 實測：混沌中位數 +0.34，週期態 −0.2～0
BIF_LYAP_DIM = 2           # 嵌入維度
BIF_LYAP_SHELL = 2.5       # 鄰居距離在 [tol, SHELL × tol] × CH2 峰對峰值 之間
BIF_LYAP_MAX_SAT = 0.8     # 鄰居一步就分開到隨機距離的 80% 以上 → 視為雜訊，不用 λ 判斷
# 高階倍週期（8、16、32…）：時域判出 p 後，f/2p 逐級高於底噪 BIF_CONFIRM_DB 就升級（分支分裂太小，時域分不出）
BIF_WEAK_UPGRADE = True
# 費根鮑姆常數：倍週期閾值用次諧波鎖相振幅 a² 外插，次諧波要連續 2 點超過這些門檻 (dBc) 才算開始
# （與 linsay/HW7_analysis/linsay_freq_analysis.py 相同；16 以上沒有實測過，先用 −35）
BIF_FEIG_GATES = {2: -30.0, 4: -25.0, 8: -33.0}
BIF_FEIG_NFIT = 4
BIF_AUTO_SAVE = True       # 分岔圖跑完（或中途停止）自動存檔並做分岔分析

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

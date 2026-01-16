"""
System Ver16 - MACD Divergence Strategy 設定ファイル
Ver16: 5分足RCI±70反転 + 5M PO + 1H PO + 1H RCI±60 + 暫定ピボットDiv

検討事項:
- ダイバージェンス有効期間の判断方法
  - 現状: 時間制（12時間）
  - 代替案: ダウ理論による判断（高値/安値の更新で無効化など）
"""

# ==========================================
# データファイルパス
# ==========================================
from pathlib import Path
_PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = str(_PROJECT_ROOT / "ローソク足データ")
DATA_5M = f"{DATA_DIR}/GBPJPY_5M_2015-2025.csv"
DATA_1H = f"{DATA_DIR}/GBPJPY_1H_2013-2025.csv"
DATA_4H = f"{DATA_DIR}/GBPJPY_4H_2013-2025.csv"

# ==========================================
# 取引設定
# ==========================================
SYMBOL = "GBPJPY"
INITIAL_CAPITAL_USD = 10000
INITIAL_CAPITAL_JPY = 1500000
POSITION_SIZE = 10000
LOT_SIZE = 0.1

# ==========================================
# 取引時間帯
# ==========================================
TRADING_START_HOUR = 8   # 日本時間 08:00
TRADING_END_HOUR = 24    # 日本時間 24:00

# ==========================================
# インジケーター設定
# ==========================================

# RCI (5分足エントリートリガー用)
RCI_SHORT = 9

# Ver16: 5分足RCI閾値を±70に変更
RCI_5M_THRESHOLD = 70  # ±70到達後、反転でエントリー

# 1時間足RCI閾値
RCI_1H_THRESHOLD = 60  # ±60

# EMA (1時間足パーフェクトオーダー用)
EMA_1H_SHORT = 20
EMA_1H_MID = 30
EMA_1H_LONG = 40

# EMA (5分足パーフェクトオーダー用)
EMA_5M_SHORT = 20
EMA_5M_MID = 30
EMA_5M_LONG = 40

# MACD (1時間足ダイバージェンス検出用)
MACD_FAST = 6
MACD_SLOW = 13
MACD_SIGNAL = 4

# ZigZag - ダイバージェンス検出用（1時間足）
ZIGZAG_1H_DEPTH = 12
ZIGZAG_1H_DEVIATION = 5
ZIGZAG_1H_BACKSTEP = 3

# ZigZag - 損切り基準（5分足）
ZIGZAG_5M_DEPTH = 5
ZIGZAG_5M_DEVIATION = 3
ZIGZAG_5M_BACKSTEP = 2

# ==========================================
# ダイバージェンス設定
# ==========================================
# 検討事項: 時間制 vs ダウ理論判断
DIVERGENCE_VALID_HOURS = 12  # 現状: 時間制（12時間）

# 暫定ピボットモード
USE_PROSPECTIVE_PIVOT = True

# ダイバージェンスタイプ
USE_HIDDEN_DIVERGENCE = True    # Hidden（トレンド継続）
USE_REGULAR_DIVERGENCE = True   # Regular（反転）

# ==========================================
# リスク管理
# ==========================================
RISK_REWARD_RATIO = 1.5

# 1 divergence = 1 entry 制限
ENABLE_1DIV1ENTRY = True

# Stop Loss設定（ZigZag Depth5ベース）
SL_BUFFER_PIPS = 2  # ZigZag高値/安値からのバッファ

# ATRベースSL設定（フォールバック用）
SL_ATR_LENGTH = 14
SL_ATR_MULT = 1.0
SL_FALLBACK_ATR_MULT = 2.0
SL_MIN_ATR_MULT = 0.5

# スプレッド
SPREAD_PIPS = 0.5

# ==========================================
# バックテスト期間
# ==========================================
BACKTEST_START = "2015-11-15"
BACKTEST_END = "2025-11-24"

# ==========================================
# 出力設定
# ==========================================
RESULTS_DIR = "./scripts/results_v16"
REPORT_FILE = f"{RESULTS_DIR}/backtest_report_v16.txt"
TRADES_CSV = f"{RESULTS_DIR}/trades_v16.csv"
DIVERGENCES_CSV = f"{RESULTS_DIR}/divergences_v16.csv"

# ==========================================
# その他
# ==========================================
POINT_VALUE = 0.01  # GBPJPYの1ポイント（0.01円）

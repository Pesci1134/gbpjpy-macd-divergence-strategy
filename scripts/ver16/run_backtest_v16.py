"""
System Ver16 - バックテスト実行スクリプト

Ver16: 5分足RCI±70反転 + 5M PO + 1H PO + 1H RCI±60 + 暫定ピボットDiv
"""

import sys
from pathlib import Path

script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))
sys.path.insert(0, str(script_dir.parent))

import pandas as pd
from datetime import datetime

import config_v16 as config
from backtest_engine_v16 import BacktestEngineV16


def load_data():
    """ローソク足データを読み込む"""
    print("データを読み込み中...")

    df_5m = pd.read_csv(config.DATA_5M, parse_dates=['Time'], index_col='Time')
    df_1h = pd.read_csv(config.DATA_1H, parse_dates=['Time'], index_col='Time')
    df_4h = pd.read_csv(config.DATA_4H, parse_dates=['Time'], index_col='Time')

    # バックテスト期間でフィルタ
    start_date = pd.to_datetime(config.BACKTEST_START)
    end_date = pd.to_datetime(config.BACKTEST_END)

    df_5m = df_5m[(df_5m.index >= start_date) & (df_5m.index <= end_date)]
    df_1h = df_1h[df_1h.index <= end_date]
    df_4h = df_4h[df_4h.index <= end_date]

    print(f"  5分足: {len(df_5m)} 件 ({df_5m.index[0]} ~ {df_5m.index[-1]})")
    print(f"  1時間足: {len(df_1h)} 件")
    print(f"  4時間足: {len(df_4h)} 件")

    return df_5m, df_1h, df_4h


def save_results(engine: BacktestEngineV16, trades_df: pd.DataFrame):
    """結果を保存"""
    results_dir = Path(config.RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)

    # トレード一覧CSV
    if not trades_df.empty:
        trades_df.to_csv(results_dir / "trades_v16.csv", index=False)
        print(f"トレード一覧を保存: {results_dir / 'trades_v16.csv'}")

    # ダイバージェンス一覧CSV
    div_df = engine.get_divergences_dataframe()
    if not div_df.empty:
        div_df.to_csv(results_dir / "divergences_v16.csv", index=False)
        print(f"ダイバージェンス一覧を保存: {results_dir / 'divergences_v16.csv'}")

    # レポート
    stats = engine.get_statistics()
    monthly = engine.get_monthly_summary()
    yearly = engine.get_yearly_summary()

    report_path = results_dir / "backtest_report_v16.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("System Ver16 - MACD Divergence Strategy\n")
        f.write("バックテストレポート\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("【Ver16の特徴】\n")
        f.write("  5分足トリガー:\n")
        f.write(f"    - RCI短期(9): ±{config.RCI_5M_THRESHOLD}到達後、反転\n")
        f.write("    - パーフェクトオーダー: 20>30>40 EMA\n")
        f.write("    - エントリー: ローソク足確定時\n")
        f.write("  1時間足環境認識:\n")
        f.write("    - パーフェクトオーダー: 20>30>40 EMA\n")
        f.write(f"    - RCI条件: ±{config.RCI_1H_THRESHOLD}\n")
        f.write("    - ダイバージェンス: Hidden+Regular (暫定ピボット)\n")
        f.write("  4時間足: なし\n\n")

        f.write("【検討事項】\n")
        f.write("  - ダイバージェンス有効期間: 時間制(12H) vs ダウ理論判断\n\n")

        f.write("【設定】\n")
        f.write(f"  通貨ペア:         {config.SYMBOL}\n")
        f.write(f"  バックテスト期間: {config.BACKTEST_START} ~ {config.BACKTEST_END}\n")
        f.write(f"  取引時間帯:       {config.TRADING_START_HOUR}:00 ~ {config.TRADING_END_HOUR}:00\n")
        f.write(f"  初期資金:         ¥{config.INITIAL_CAPITAL_JPY:,}\n")
        f.write(f"  ロットサイズ:     {config.LOT_SIZE}ロット ({config.POSITION_SIZE:,}通貨)\n")
        f.write(f"  リスクリワード:   1:{config.RISK_REWARD_RATIO}\n")
        f.write(f"  ダイバージェンス有効期間: {config.DIVERGENCE_VALID_HOURS}時間\n\n")

        f.write("【インジケーター設定】\n")
        f.write(f"  5M RCI短期:       {config.RCI_SHORT} (閾値: ±{config.RCI_5M_THRESHOLD})\n")
        f.write(f"  1H RCI閾値:       ±{config.RCI_1H_THRESHOLD}\n")
        f.write(f"  MACD:             ({config.MACD_FAST}, {config.MACD_SLOW}, {config.MACD_SIGNAL})\n")
        f.write(f"  1H EMA (PO):      ({config.EMA_1H_SHORT}, {config.EMA_1H_MID}, {config.EMA_1H_LONG})\n")
        f.write(f"  5M EMA (PO):      ({config.EMA_5M_SHORT}, {config.EMA_5M_MID}, {config.EMA_5M_LONG})\n")
        f.write(f"  ZigZag 1H:        Depth {config.ZIGZAG_1H_DEPTH}\n")
        f.write(f"  ZigZag 5M:        Depth {config.ZIGZAG_5M_DEPTH}\n\n")

        if stats:
            f.write("【サマリー統計】\n")
            f.write(f"  総トレード数:         {stats['total_trades']}件\n")
            f.write(f"  勝率:                 {stats['win_rate']:.2f}%\n")
            f.write(f"  期待値:               {stats['expectancy_pct']:.4f}%/トレード\n")
            f.write(f"  プロフィットファクター: {stats['profit_factor']:.2f}\n")
            f.write(f"  総損益 (pips):        {stats['total_profit_pips']:.2f} pips\n")
            f.write(f"  総損益 (円):          ¥{stats['total_profit_jpy']:,.0f}\n")
            f.write(f"  最大ドローダウン:     ¥{stats['max_drawdown_jpy']:,.0f} ({stats['max_drawdown_pct']:.2f}%)\n")
            f.write(f"  平均保有時間:         {stats['avg_holding_hours']:.1f}時間\n\n")

            f.write("【勝敗内訳】\n")
            f.write(f"  勝ち:     {stats['wins']}件 (平均 +{stats['avg_win']:.2f} pips)\n")
            f.write(f"  負け:     {stats['losses']}件 (平均 -{stats['avg_loss']:.2f} pips)\n\n")

            f.write("【決済内訳】\n")
            f.write(f"  TP決済:   {stats['tp_exits']}件\n")
            f.write(f"  SL決済:   {stats['sl_exits']}件\n\n")

            f.write("【ダイバージェンス内訳】\n")
            f.write(f"  総検出数:     {stats['total_divergences']}件\n")
            f.write(f"  Hidden型:     {stats['hidden_divergence_trades']}件\n")
            f.write(f"  Regular型:    {stats['regular_divergence_trades']}件\n\n")

        if not yearly.empty:
            f.write("【年別サマリー】\n")
            f.write(yearly.to_string())
            f.write("\n\n")

        if not monthly.empty:
            f.write("【月別サマリー】\n")
            f.write(monthly.to_string())
            f.write("\n")

    print(f"レポートを保存: {report_path}")


def main():
    """メイン処理"""
    print("=" * 60)
    print("System Ver16 - MACD Divergence Strategy")
    print("5M RCI±70反転 + 5M/1H PO + 1H RCI±60 + Div(暫定ピボット)")
    print("=" * 60)

    df_5m, df_1h, df_4h = load_data()

    engine = BacktestEngineV16(df_5m, df_1h, df_4h)
    trades_df = engine.run_backtest()

    save_results(engine, trades_df)

    stats = engine.get_statistics()
    if stats:
        print("\n" + "=" * 60)
        print("【結果サマリー】")
        print(f"  総トレード数: {stats['total_trades']}件")
        print(f"  勝率: {stats['win_rate']:.2f}%")
        print(f"  PF: {stats['profit_factor']:.2f}")
        print(f"  総損益: {stats['total_profit_pips']:.2f} pips (¥{stats['total_profit_jpy']:,.0f})")
        print("=" * 60)


if __name__ == "__main__":
    main()

"""
System Ver18 - バックテスト実行スクリプト
Ver18: 1H POなし / 1H RCIあり
"""

import sys
from pathlib import Path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))
sys.path.insert(0, str(script_dir.parent))

import pandas as pd
from datetime import datetime
import config_v18 as config
from backtest_engine_v18 import BacktestEngineV18


def load_data():
    print("データを読み込み中...")
    df_5m = pd.read_csv(config.DATA_5M, parse_dates=['Time'], index_col='Time')
    df_1h = pd.read_csv(config.DATA_1H, parse_dates=['Time'], index_col='Time')
    df_4h = pd.read_csv(config.DATA_4H, parse_dates=['Time'], index_col='Time')
    start_date = pd.to_datetime(config.BACKTEST_START)
    end_date = pd.to_datetime(config.BACKTEST_END)
    df_5m = df_5m[(df_5m.index >= start_date) & (df_5m.index <= end_date)]
    df_1h = df_1h[df_1h.index <= end_date]
    df_4h = df_4h[df_4h.index <= end_date]
    print(f"  5分足: {len(df_5m)} 件, 1時間足: {len(df_1h)} 件")
    return df_5m, df_1h, df_4h


def save_results(engine, trades_df):
    results_dir = Path(config.RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)
    if not trades_df.empty:
        trades_df.to_csv(results_dir / "trades_v18.csv", index=False)
    div_df = engine.get_divergences_dataframe()
    if not div_df.empty:
        div_df.to_csv(results_dir / "divergences_v18.csv", index=False)

    stats = engine.get_statistics()
    yearly = engine.get_yearly_summary()
    with open(results_dir / "backtest_report_v18.txt", 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("System Ver18 - MACD Divergence Strategy\n")
        f.write("1H POなし / 1H RCIあり\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"【設定】 1H PO={config.USE_1H_PO}, 1H RCI={config.USE_1H_RCI}\n\n")
        if stats:
            f.write("【サマリー】\n")
            f.write(f"  総トレード: {stats['total_trades']}件, 勝率: {stats['win_rate']:.2f}%\n")
            f.write(f"  PF: {stats['profit_factor']:.2f}, 総損益: {stats['total_profit_pips']:.2f} pips\n")
            f.write(f"  最大DD: ¥{stats['max_drawdown_jpy']:,.0f}\n\n")
        if not yearly.empty:
            f.write("【年別】\n" + yearly.to_string() + "\n")
    print(f"結果を保存: {results_dir}")


def main():
    print("=" * 60)
    print("Ver18: 1H POなし / 1H RCIあり")
    print("=" * 60)
    df_5m, df_1h, df_4h = load_data()
    engine = BacktestEngineV18(df_5m, df_1h, df_4h)
    trades_df = engine.run_backtest()
    save_results(engine, trades_df)
    stats = engine.get_statistics()
    if stats:
        print(f"\n【結果】トレード: {stats['total_trades']}件, 勝率: {stats['win_rate']:.2f}%, PF: {stats['profit_factor']:.2f}")


if __name__ == "__main__":
    main()

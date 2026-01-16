"""
System Ver19 - バックテストエンジン
Ver19: 5分足RCI±70反転 + 5M PO + 暫定ピボットDiv（1H PO/RCI両方なし）
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

import config_v19 as config

from indicators import (
    calculate_rci, calculate_ema, calculate_macd,
    calculate_zigzag, calculate_zigzag_with_prospective, calculate_atr,
    get_latest_zigzag_level, check_perfect_order, check_1h_rci_condition
)


@dataclass
class Trade:
    entry_time: datetime
    exit_time: datetime
    direction: str
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    profit_pips: float
    result: str
    exit_reason: str
    divergence_type: str


class BacktestEngineV19:
    def __init__(self, df_5m: pd.DataFrame, df_1h: pd.DataFrame, df_4h: pd.DataFrame):
        self.df_5m = df_5m.copy()
        self.df_1h = df_1h.copy()
        self.df_4h = df_4h.copy()
        self.trades: List[Trade] = []
        self.current_position: Optional[Dict] = None
        self.divergences: List[Dict] = []
        self.current_divergence: Optional[Dict] = None
        self.current_divergence_used = False
        self.prev_prospective_low = np.nan
        self.prev_prospective_low_idx = np.nan
        self.prev_macd_at_prospective_low = np.nan
        self.prev_prospective_high = np.nan
        self.prev_prospective_high_idx = np.nan
        self.prev_macd_at_prospective_high = np.nan
        self._calculate_indicators()

    def _calculate_indicators(self):
        print("インジケーターを計算中...")
        self.df_5m['RCI_9'] = calculate_rci(self.df_5m, config.RCI_SHORT)
        zz_highs_5m, zz_lows_5m = calculate_zigzag(
            self.df_5m, depth=config.ZIGZAG_5M_DEPTH,
            deviation=config.ZIGZAG_5M_DEVIATION, backstep=config.ZIGZAG_5M_BACKSTEP)
        self.df_5m['ZZ_High'] = zz_highs_5m
        self.df_5m['ZZ_Low'] = zz_lows_5m
        self.df_5m['ATR'] = calculate_atr(self.df_5m, config.SL_ATR_LENGTH)
        self.df_5m['EMA_20'] = calculate_ema(self.df_5m, config.EMA_5M_SHORT)
        self.df_5m['EMA_30'] = calculate_ema(self.df_5m, config.EMA_5M_MID)
        self.df_5m['EMA_40'] = calculate_ema(self.df_5m, config.EMA_5M_LONG)

        macd_line, signal_line, histogram = calculate_macd(
            self.df_1h, fast=config.MACD_FAST, slow=config.MACD_SLOW, signal=config.MACD_SIGNAL)
        self.df_1h['MACD'] = macd_line
        self.df_1h['MACD_Signal'] = signal_line
        self.df_1h['MACD_Hist'] = histogram

        (zz_highs_1h, zz_lows_1h, prospective_high, prospective_low,
         prospective_high_idx, prospective_low_idx) = calculate_zigzag_with_prospective(
            self.df_1h, depth=config.ZIGZAG_1H_DEPTH,
            deviation=config.ZIGZAG_1H_DEVIATION, backstep=config.ZIGZAG_1H_BACKSTEP)
        self.df_1h['ZZ_High'] = zz_highs_1h
        self.df_1h['ZZ_Low'] = zz_lows_1h
        self.df_1h['Prospective_High'] = prospective_high
        self.df_1h['Prospective_Low'] = prospective_low
        self.df_1h['Prospective_High_Idx'] = prospective_high_idx
        self.df_1h['Prospective_Low_Idx'] = prospective_low_idx

        self.df_1h['EMA_20'] = calculate_ema(self.df_1h, config.EMA_1H_SHORT)
        self.df_1h['EMA_30'] = calculate_ema(self.df_1h, config.EMA_1H_MID)
        self.df_1h['EMA_40'] = calculate_ema(self.df_1h, config.EMA_1H_LONG)
        self.df_1h['RCI_9'] = calculate_rci(self.df_1h, config.RCI_SHORT)

        print(f"  5分足: {len(self.df_5m)} 件, 1時間足: {len(self.df_1h)} 件")

    def _get_1h_data_at_time(self, current_time: pd.Timestamp) -> Optional[Dict]:
        mask = self.df_1h.index <= current_time
        if mask.sum() == 0:
            return None
        return self.df_1h.loc[mask].iloc[-1].to_dict()

    def _check_1h_perfect_order(self, data_1h: Dict) -> str:
        if data_1h is None or not config.USE_1H_PO:
            return 'none' if config.USE_1H_PO else 'skip'
        ema_20 = data_1h.get('EMA_20', np.nan)
        ema_30 = data_1h.get('EMA_30', np.nan)
        ema_40 = data_1h.get('EMA_40', np.nan)
        return check_perfect_order(ema_20, ema_30, ema_40)

    def _check_1h_rci(self, data_1h: Dict, direction: str) -> bool:
        if not config.USE_1H_RCI:
            return True
        if data_1h is None:
            return False
        rci_value = data_1h.get('RCI_9', np.nan)
        return check_1h_rci_condition(rci_value, direction, config.RCI_1H_THRESHOLD)

    def _update_divergences(self, current_time: pd.Timestamp):
        mask_1h = self.df_1h.index <= current_time
        if mask_1h.sum() == 0:
            return
        idx_1h = mask_1h.sum() - 1

        current_prospective_low = self.df_1h['Prospective_Low'].iloc[idx_1h]
        current_prospective_low_idx = self.df_1h['Prospective_Low_Idx'].iloc[idx_1h]
        current_prospective_high = self.df_1h['Prospective_High'].iloc[idx_1h]
        current_prospective_high_idx = self.df_1h['Prospective_High_Idx'].iloc[idx_1h]

        current_macd_at_low = np.nan
        if not pd.isna(current_prospective_low_idx):
            current_macd_at_low = self.df_1h['MACD'].iloc[int(current_prospective_low_idx)]

        current_macd_at_high = np.nan
        if not pd.isna(current_prospective_high_idx):
            current_macd_at_high = self.df_1h['MACD'].iloc[int(current_prospective_high_idx)]

        if (not pd.isna(current_prospective_low) and not pd.isna(current_prospective_low_idx) and
            current_prospective_low_idx != self.prev_prospective_low_idx):
            if not pd.isna(self.prev_prospective_low) and not pd.isna(self.prev_macd_at_prospective_low):
                if (config.USE_HIDDEN_DIVERGENCE and
                    current_prospective_low > self.prev_prospective_low and
                    current_macd_at_low < self.prev_macd_at_prospective_low):
                    self._set_divergence(current_time, 'hidden', 'long', current_prospective_low, current_macd_at_low)
                elif (config.USE_REGULAR_DIVERGENCE and
                      current_prospective_low < self.prev_prospective_low and
                      current_macd_at_low > self.prev_macd_at_prospective_low):
                    self._set_divergence(current_time, 'regular', 'long', current_prospective_low, current_macd_at_low)
            self.prev_prospective_low = current_prospective_low
            self.prev_prospective_low_idx = current_prospective_low_idx
            self.prev_macd_at_prospective_low = current_macd_at_low

        if (not pd.isna(current_prospective_high) and not pd.isna(current_prospective_high_idx) and
            current_prospective_high_idx != self.prev_prospective_high_idx):
            if not pd.isna(self.prev_prospective_high) and not pd.isna(self.prev_macd_at_prospective_high):
                if (config.USE_HIDDEN_DIVERGENCE and
                    current_prospective_high < self.prev_prospective_high and
                    current_macd_at_high > self.prev_macd_at_prospective_high):
                    self._set_divergence(current_time, 'hidden', 'short', current_prospective_high, current_macd_at_high)
                elif (config.USE_REGULAR_DIVERGENCE and
                      current_prospective_high > self.prev_prospective_high and
                      current_macd_at_high < self.prev_macd_at_prospective_high):
                    self._set_divergence(current_time, 'regular', 'short', current_prospective_high, current_macd_at_high)
            self.prev_prospective_high = current_prospective_high
            self.prev_prospective_high_idx = current_prospective_high_idx
            self.prev_macd_at_prospective_high = current_macd_at_high

        if self.current_divergence is not None:
            valid_time_threshold = current_time - timedelta(hours=config.DIVERGENCE_VALID_HOURS)
            if self.current_divergence['detected_time'] < valid_time_threshold:
                self.current_divergence = None
                self.current_divergence_used = False

    def _set_divergence(self, current_time, div_type, direction, price, macd_value):
        divergence = {
            'type': div_type, 'direction': direction, 'detected_time': current_time,
            'price': price, 'macd': macd_value, 'pivot_type': 'prospective'
        }
        self.current_divergence = divergence
        self.current_divergence_used = False
        self.divergences.append(divergence)

    def _get_active_setup(self, current_time: pd.Timestamp, direction: str) -> bool:
        if self.current_divergence is None:
            return False
        valid_time_threshold = current_time - timedelta(hours=config.DIVERGENCE_VALID_HOURS)
        if self.current_divergence['detected_time'] < valid_time_threshold:
            return False
        if config.ENABLE_1DIV1ENTRY and self.current_divergence_used:
            return False
        return self.current_divergence['direction'] == direction

    def _check_5m_entry(self, idx: int, direction: str) -> bool:
        if idx < 2:
            return False
        current = self.df_5m.iloc[idx]
        prev1 = self.df_5m.iloc[idx - 1]
        prev2 = self.df_5m.iloc[idx - 2]
        rci_current, rci_prev1, rci_prev2 = current['RCI_9'], prev1['RCI_9'], prev2['RCI_9']
        if any(pd.isna([rci_current, rci_prev1, rci_prev2])):
            return False
        ema_20, ema_30, ema_40 = current.get('EMA_20'), current.get('EMA_30'), current.get('EMA_40')
        po_5m = check_perfect_order(ema_20, ema_30, ema_40)

        if direction == 'long':
            reached = rci_prev1 <= -config.RCI_5M_THRESHOLD or rci_prev2 <= -config.RCI_5M_THRESHOLD
            hook_up = rci_current > rci_prev1
            return reached and hook_up and po_5m == 'uptrend'
        elif direction == 'short':
            reached = rci_prev1 >= config.RCI_5M_THRESHOLD or rci_prev2 >= config.RCI_5M_THRESHOLD
            hook_down = rci_current < rci_prev1
            return reached and hook_down and po_5m == 'downtrend'
        return False

    def _calculate_sl_tp(self, idx: int, direction: str, entry_price: float) -> Tuple[float, float]:
        atr = self.df_5m['ATR'].iloc[idx]
        if pd.isna(atr):
            atr = 0.0
        sl_buffer = config.SL_BUFFER_PIPS * config.POINT_VALUE
        sl_fallback = atr * config.SL_FALLBACK_ATR_MULT
        sl_min = atr * config.SL_MIN_ATR_MULT
        pivot_idx = idx - config.ZIGZAG_5M_DEPTH

        if direction == 'long':
            zz_level = get_latest_zigzag_level(self.df_5m['ZZ_Low'], pivot_idx) if pivot_idx >= 0 else None
            sl = (zz_level - sl_buffer) if zz_level else (entry_price - sl_fallback)
            sl = min(sl, entry_price - sl_min)
        else:
            zz_level = get_latest_zigzag_level(self.df_5m['ZZ_High'], pivot_idx) if pivot_idx >= 0 else None
            sl = (zz_level + sl_buffer) if zz_level else (entry_price + sl_fallback)
            sl = max(sl, entry_price + sl_min)

        risk = abs(entry_price - sl)
        tp = entry_price + risk * config.RISK_REWARD_RATIO if direction == 'long' else entry_price - risk * config.RISK_REWARD_RATIO
        return sl, tp

    def _is_trading_hours(self, timestamp: pd.Timestamp) -> bool:
        return config.TRADING_START_HOUR <= timestamp.hour < config.TRADING_END_HOUR

    def _check_exit(self, idx: int) -> Optional[Tuple[str, float]]:
        if self.current_position is None:
            return None
        current = self.df_5m.iloc[idx]
        d, sl, tp = self.current_position['direction'], self.current_position['stop_loss'], self.current_position['take_profit']
        if d == 'long':
            if current['High'] >= tp: return ('tp', tp)
            if current['Low'] <= sl: return ('sl', sl)
        else:
            if current['Low'] <= tp: return ('tp', tp)
            if current['High'] >= sl: return ('sl', sl)
        return None

    def run_backtest(self) -> pd.DataFrame:
        print("バックテスト開始...")
        print(f"Ver19: 1H PO={config.USE_1H_PO}, 1H RCI={config.USE_1H_RCI} (Divのみ)")

        for idx in range(len(self.df_5m)):
            current_time = self.df_5m.index[idx]
            self._update_divergences(current_time)

            if self.current_position is not None:
                exit_result = self._check_exit(idx)
                if exit_result:
                    self._close_position(current_time, exit_result[1], exit_result[0])

            if self.current_position is None and self._is_trading_hours(current_time):
                # Ver19: 1H PO/RCI両方なし、Divのみ
                if self._get_active_setup(current_time, 'long'):
                    if self._check_5m_entry(idx, 'long'):
                        self._open_position(idx, 'long')

                if self._get_active_setup(current_time, 'short'):
                    if self._check_5m_entry(idx, 'short'):
                        self._open_position(idx, 'short')

            if idx % 10000 == 0:
                print(f"進捗: {idx}/{len(self.df_5m)} ({idx/len(self.df_5m)*100:.1f}%) - トレード: {len(self.trades)}")

        if self.current_position:
            self._close_position(self.df_5m.index[-1], self.df_5m.iloc[-1]['Close'], 'end_of_data')

        print(f"完了: {len(self.trades)}件のトレード, {len(self.divergences)}件のDiv")
        return self._create_trades_dataframe()

    def _open_position(self, idx: int, direction: str):
        current = self.df_5m.iloc[idx]
        entry_price = current['Close']
        sl, tp = self._calculate_sl_tp(idx, direction, entry_price)
        div_type = self.current_divergence['type'] if self.current_divergence else 'unknown'
        if config.ENABLE_1DIV1ENTRY:
            self.current_divergence_used = True
        self.current_position = {
            'direction': direction, 'entry_time': current.name, 'entry_price': entry_price,
            'stop_loss': sl, 'take_profit': tp, 'entry_idx': idx, 'divergence_type': div_type
        }

    def _close_position(self, exit_time, exit_price, exit_reason):
        pos = self.current_position
        profit_pips = ((exit_price - pos['entry_price']) if pos['direction'] == 'long'
                      else (pos['entry_price'] - exit_price)) / config.POINT_VALUE
        result = 'win' if profit_pips > 0 else ('loss' if profit_pips < 0 else 'breakeven')
        self.trades.append(Trade(
            entry_time=pos['entry_time'], exit_time=exit_time, direction=pos['direction'],
            entry_price=pos['entry_price'], exit_price=exit_price, stop_loss=pos['stop_loss'],
            take_profit=pos['take_profit'], profit_pips=profit_pips, result=result,
            exit_reason=exit_reason, divergence_type=pos['divergence_type']
        ))
        self.current_position = None

    def _create_trades_dataframe(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame()
        trades_data = []
        cumulative = 0
        for t in self.trades:
            profit_jpy = t.profit_pips * 100 * config.LOT_SIZE
            cumulative += profit_jpy
            trades_data.append({
                'entry_time': t.entry_time, 'exit_time': t.exit_time, 'direction': t.direction,
                'entry_price': t.entry_price, 'exit_price': t.exit_price,
                'stop_loss': t.stop_loss, 'take_profit': t.take_profit,
                'holding_hours': round((t.exit_time - t.entry_time).total_seconds() / 3600, 2),
                'profit_pips': t.profit_pips, 'profit_jpy': round(profit_jpy, 0),
                'cumulative_profit_jpy': round(cumulative, 0),
                'result': t.result, 'exit_reason': t.exit_reason, 'divergence_type': t.divergence_type
            })
        return pd.DataFrame(trades_data)

    def get_divergences_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.divergences) if self.divergences else pd.DataFrame()

    def get_statistics(self) -> Dict:
        if not self.trades:
            return {}
        total = len(self.trades)
        wins = len([t for t in self.trades if t.result == 'win'])
        losses = len([t for t in self.trades if t.result == 'loss'])
        total_profit = sum(t.profit_pips for t in self.trades)
        gross_profit = sum(t.profit_pips for t in self.trades if t.profit_pips > 0)
        gross_loss = abs(sum(t.profit_pips for t in self.trades if t.profit_pips < 0))
        pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        total_jpy = total_profit * 100 * config.LOT_SIZE

        cumulative, peak, max_dd = 0, 0, 0
        for t in self.trades:
            cumulative += t.profit_pips * 100 * config.LOT_SIZE
            peak = max(peak, cumulative)
            max_dd = max(max_dd, peak - cumulative)

        return {
            'total_trades': total, 'wins': wins, 'losses': losses,
            'win_rate': wins / total * 100 if total else 0,
            'total_profit_pips': total_profit, 'profit_factor': pf,
            'total_profit_jpy': total_jpy,
            'avg_win': gross_profit / wins if wins else 0,
            'avg_loss': gross_loss / losses if losses else 0,
            'max_drawdown_jpy': max_dd,
            'max_drawdown_pct': (max_dd / (config.INITIAL_CAPITAL_JPY + peak)) * 100 if peak else 0,
            'total_divergences': len(self.divergences),
            'hidden_divergence_trades': len([t for t in self.trades if t.divergence_type == 'hidden']),
            'regular_divergence_trades': len([t for t in self.trades if t.divergence_type == 'regular']),
            'tp_exits': len([t for t in self.trades if t.exit_reason == 'tp']),
            'sl_exits': len([t for t in self.trades if t.exit_reason == 'sl']),
            'expectancy_pct': (total_jpy / config.INITIAL_CAPITAL_JPY) * 100 / total if total else 0,
            'avg_holding_hours': np.mean([(t.exit_time - t.entry_time).total_seconds() / 3600 for t in self.trades])
        }

    def get_yearly_summary(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame()
        data = {}
        for t in self.trades:
            y = t.entry_time.strftime('%Y')
            if y not in data:
                data[y] = {'trades': 0, 'wins': 0, 'profit_pips': 0}
            data[y]['trades'] += 1
            if t.result == 'win':
                data[y]['wins'] += 1
            data[y]['profit_pips'] += t.profit_pips
        df = pd.DataFrame.from_dict(data, orient='index')
        df['win_rate'] = (df['wins'] / df['trades'] * 100).round(2)
        df['profit_jpy'] = (df['profit_pips'] * 100 * config.LOT_SIZE).round(0).astype(int)
        return df[['trades', 'wins', 'win_rate', 'profit_pips', 'profit_jpy']].sort_index()

    def get_monthly_summary(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame()
        data = {}
        for t in self.trades:
            m = t.entry_time.strftime('%Y-%m')
            if m not in data:
                data[m] = {'trades': 0, 'wins': 0, 'profit_pips': 0}
            data[m]['trades'] += 1
            if t.result == 'win':
                data[m]['wins'] += 1
            data[m]['profit_pips'] += t.profit_pips
        df = pd.DataFrame.from_dict(data, orient='index')
        df['win_rate'] = (df['wins'] / df['trades'] * 100).round(2)
        df['profit_jpy'] = (df['profit_pips'] * 100 * config.LOT_SIZE).round(0).astype(int)
        return df[['trades', 'wins', 'win_rate', 'profit_pips', 'profit_jpy']].sort_index()

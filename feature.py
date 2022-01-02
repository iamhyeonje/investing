import pandas as pd
import numpy as np


def moving_average(df_price, window):
    moving_avg = df_price.rolling(window=window).mean()
    moving_std = df_price.rolling(window=window).std()
    upper_band = moving_avg + 2 * moving_std
    lower_band = moving_avg - 2 * moving_std
    return moving_avg, moving_std, upper_band, lower_band


def moving_average_convergence_divergence(df_price, short=12, long=26, signal=9):
    macd_short = df_price.ewm(span=short, adjust=False).mean()
    macd_long = df_price.ewm(span=long, adjust=False).mean()
    macd = macd_short - macd_long
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_oscillator = macd - macd_signal
    return macd_short, macd_long, macd_signal, macd, macd_oscillator


def volume_average(df_volume, window):
    volume_avg = df_volume.rolling(window=window).mean()
    v_va = df_volume / volume_avg
    return volume_avg, v_va


def get_stochastic_fast_k(close_price, low, high, n=5):
    fast_k = ((close_price - low.rolling(n).min()) / (high.rolling(n).max() - low.rolling(n).min())) * 100
    return fast_k


# Slow %K = Fast %K의 m기간 이동평균(SMA)
def get_stochastic_slow_k(fast_k, n=3):
    slow_k = fast_k.rolling(n).mean()
    return slow_k

# Slow %D = Slow %K의 t기간 이동평균(S170MA)
def get_stochastic_slow_d(slow_k, n=3):
    slow_d = slow_k.rolling(n).mean()
    return slow_d


def get_rsi(price, period=14):
    delta = price.diff()
    gains, declines = delta.copy(), delta.copy()
    gains[gains < 0] = 0
    declines[declines > 0] = 0

    _gain = gains.ewm(com=(period - 1), min_periods=period).mean()
    _loss = declines.abs().ewm(com=(period - 1), min_periods=period).mean()
    RS = _gain / _loss

    return pd.Series(100 - (100 / (1 + RS)), name="RSI")

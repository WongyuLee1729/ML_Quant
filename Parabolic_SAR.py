import binance as bn
import pyupbit
from numpy import nan as npNaN
from pandas import DataFrame, Series

def PSAR(high, low, close=None, af0=0.02, af=0.02, max_af=0.2):
    """
    Parabolic Stop and Reverse (PSAR)
    
    - input:
        af0 : initial acceleration factor
        af : final acceleration factor
        max_af : maximum acceleration factor
    
    - output:
        Dataframe
    """
    
    # Validate Arguments
    def trend(high, low, drift:int=1):
        up = high - high.shift(drift)
        dn = low.shift(drift) - low
        _dmn = (((dn > up) & (dn > 0)) * dn).iloc[-1]
        return _dmn > 0

    # Falling if the first NaN -DM is positive
    trend = trend(high.iloc[:2], low.iloc[:2])
    if trend:
        sar = high.max()
        ep = low.min()
    else:
        sar = low.min()
        ep = high.max()

    if close is not None:
        sar = close.iloc[0]

    long = Series(npNaN, index=high.index)
    short = long.copy()
    reversal = Series(0, index=high.index)
    _af = long.copy()
    _af.iloc[0:2] = af0

    # Calculate Result
    m = high.shape[0]
    for row in range(1, m):
        high_ = high.iloc[row]
        low_ = low.iloc[row]

        if trend:
            _sar = sar + af * (ep - sar)
            reverse = high_ > _sar

            if low_ < ep:
                ep = low_
                af = min(af + af0, max_af)

            _sar = max(high.iloc[row - 1], high.iloc[row - 2], _sar)
        else:
            _sar = sar + af * (ep - sar)
            reverse = low_ < _sar

            if high_ > ep:
                ep = high_
                af = min(af + af0, max_af)

            _sar = min(low.iloc[row - 1], low.iloc[row - 2], _sar)

        if reverse:
            _sar = ep
            af = af0
            trend = not trend # Must come before next line
            ep = low_ if trend else high_

        sar = _sar # Update SAR

        # Seperate long/short sar based on falling
        if trend:
            short.iloc[row] = sar
        else:
            long.iloc[row] = sar

        _af.iloc[row] = af
        reversal.iloc[row] = int(reverse)



    # Prepare DataFrame to return
    _params = f"_{af0}_{max_af}"
    data = {
        f"long": long,
        f"short": short,
        f"af": _af,
        f"reversal": reversal,
    }
    psardf = DataFrame(data)
    psardf.name = f"PSAR{_params}"
    psardf.category = long.category = short.category = "trend"

    return psardf

if __name__ == '__main__':
    df = pyupbit.get_ohlcv(ticker='KRW-BTC',interval='minute60',count=200)
    high = df['high']
    low = df['low']
    close = df['close']
    
    # df = bn.get("BTC","1d")
    # high = df['high']
    # low = df['low']
    # close = df['close']
    
    print(PSAR(high, low, af0=0.02, af= 0.02, max_af=0.2))




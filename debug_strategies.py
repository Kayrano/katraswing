from agents.intraday_strategies import _STRATEGIES_5M
from data.fetcher_intraday import fetch_intraday_data

df = fetch_intraday_data('EURUSD=X', interval='5m')
for fn in _STRATEGIES_5M:
    sig = fn(df)
    print(f'{fn.__name__:35s} {sig.signal:6s} conf={sig.confidence:.2f}  {sig.reason[:70]}')

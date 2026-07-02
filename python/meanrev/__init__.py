"""meanrev — hybrid C++/Python microstructure mean-reversion system.

Layering (latency-critical → coordination):

    microcore (C++ / pybind11)      : OrderBookManager, SignalGenerator,
                                      mmap Checkpointer — microsecond path
    meanrev.router / exchanges      : asyncio ExchangeRouter implementations
                                      (CoinbaseCEX CLOB, PumpFunDEX bonding
                                      curve via Solana RPC + Jito bundles)
    meanrev.regime                  : NightMarketRegime (Crypto Dead Zone)
    meanrev.risk                    : drawdown breaker, cross-correlation
                                      halt, inverse-volatility sizing
    meanrev.toxicity                : Pump.fun bundled-launch / dev-wallet
                                      toxicity screen
    meanrev.engine                  : per-instrument strategy loop
"""

__version__ = "0.1.0"

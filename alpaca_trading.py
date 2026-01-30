from alpaca.trading.client import TradingClient
from alpaca.trading.requests import LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
from alpaca.trading.requests import GetOrdersRequest
from alpaca.data.live import StockDataStream, CryptoDataStream

trading_client = TradingClient("PKHBCOPOFEFD6HIEZ6Z4BI47Y5","P1WPGimEqMo78QcwBjDTvZLasb2hCSspVT6dgfNTjeX")

stream = StockDataStream("PKHBCOPOFEFD6HIEZ6Z4BI47Y5","P1WPGimEqMo78QcwBjDTvZLasb2hCSspVT6dgfNTjeX")

async def handle_trade(data):
    print(data)

stream.subscribe_trades(handle_trade,"SPY")

stream.run()
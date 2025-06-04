from fastapi import FastAPI, HTTPException , Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
import yfinance as yf
import pandas as pd
import math
import numpy as np
import uuid
import csv
import os
import logging
from fastapi.templating import Jinja2Templates
from fastapi_cache import FastAPICache
from fastapi_cache.backends.inmemory import InMemoryBackend
from fastapi_cache.decorator import cache
import asyncio
from datetime import datetime, timedelta

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mount static files for frontend
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/")
async def read_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.on_event("startup")
async def startup():
    FastAPICache.init(InMemoryBackend(), prefix="fastapi-cache")


# Data Models
class StockData(BaseModel):
    ticker: str
    date: str
    close: float
    volume: float

class Trade(BaseModel):
    ticker: str
    quantity: int
    type: str  # "BUY" or "SELL"

class Holding(BaseModel):
    ticker: str
    quantity: int
    avg_price: float
    buy_date: str
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

class Transaction(BaseModel):
    id: str
    date: str
    ticker: str
    action: str
    quantity: int
    price: float
    commission: float
    signal: str
    pnl: float
    notes: str

class HistoricalDataRequest(BaseModel):
    ticker: str
    period: str = "1mo"  # Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
    interval: str = "1d"  # Valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo

# Database (in-memory)
db = {
    "account": {
        "capital": 100000,
        "cash_balance": 100000,
        "holdings": [],
        "transactions": []
    },
    "rules": {
        "commission": 20,
        "max_position_size": 0.2,
        "min_trade_value": 1000,
        "stop_loss_pct": 0.05,  # 5% stop loss
        "take_profit_pct": 0.10  # 10% take profit
    },
    "nifty50": ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS",
        "ITC.NS", "KOTAKBANK.NS", "HINDUNILVR.NS", "BHARTIARTL.NS",
        "LT.NS", "SBIN.NS", "BAJFINANCE.NS", "ASIANPAINT.NS", "HCLTECH.NS",
        "MARUTI.NS", "TITAN.NS", "SUNPHARMA.NS", "AXISBANK.NS", "NTPC.NS",
        "ONGC.NS", "POWERGRID.NS", "NESTLEIND.NS", "ULTRACEMCO.NS", "WIPRO.NS",
        "ADANIPORTS.NS", "TECHM.NS", "JSWSTEEL.NS", "HINDALCO.NS", "GRASIM.NS",
        "IOC.NS", "COALINDIA.NS", "BPCL.NS", "UPL.NS", "TATASTEEL.NS",
        "BAJAJ-AUTO.NS", "SHREECEM.NS", "INDUSINDBK.NS", "DRREDDY.NS", "EICHERMOT.NS",
        "HEROMOTOCO.NS", "DIVISLAB.NS", "CIPLA.NS", "BAJAJFINSV.NS", "BRITANNIA.NS",
        "HDFCLIFE.NS", "SBILIFE.NS", "TATACONSUM.NS", "APOLLOHOSP.NS", "ADANIENT.NS"]  # Top 5 NIFTY 50
}


async def get_current_price(ticker: str) -> float:
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="1d")
        if data.empty or 'Close' not in data or data['Close'].empty:
            return 0.0
        return float(data['Close'].iloc[-1])
    except Exception as e:
        logger.error(f"Error fetching price for {ticker}: {str(e)}")
        return 0.0  # Fallback value

def convert_types(obj):
    if isinstance(obj, dict):
        return {k: convert_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_types(i) for i in obj]
    elif isinstance(obj, (np.floating, float)):
        v = float(obj)
        if math.isnan(v) or math.isinf(v):
            return 0.0
        return v
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif obj is None:
        return 0.0
    else:
        return obj
    

# Helper function to get historical data
def get_historical_data(ticker: str, period: str = "1mo", interval: str = "1d") -> pd.DataFrame:
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period, interval=interval)
        return data
    except Exception as e:
        logger.error(f"Error fetching historical data for {ticker}: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Could not fetch historical data for {ticker}")

async def calculate_holding_values(holding: dict) -> tuple:
    current_price = await get_current_price(holding["ticker"])
    value = holding["quantity"] * current_price
    pnl = (current_price - holding["avg_price"]) * holding["quantity"]
    return value, pnl


# Technical Indicators Calculation
def calculate_technical_indicators(ticker: str, period: str = "3mo") -> dict:
    try:
        df = get_historical_data(ticker, period=period)
        
        if df.empty:
            raise HTTPException(status_code=404, detail="No data available for this ticker")
        
        # Calculate moving averages
        df['MA_9'] = df['Close'].rolling(window=9).mean()
        df['MA_21'] = df['Close'].rolling(window=21).mean()
        df['MA_50'] = df['Close'].rolling(window=50).mean()
        
        # Calculate RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI_14'] = 100 - (100 / (1 + rs))
        
        # Calculate MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Get the latest data point
        latest = df.iloc[-1]
        
        # Generate trading signals
        signal = "HOLD"
        reasons = []
        
        # RSI based signals
        if latest['RSI_14'] < 30:
            reasons.append("RSI indicates oversold")
            signal = "BUY"
        elif latest['RSI_14'] > 70:
            reasons.append("RSI indicates overbought")
            signal = "SELL"
        
        # Moving average crossover
        if latest['MA_9'] > latest['MA_21'] and df['MA_9'].iloc[-2] <= df['MA_21'].iloc[-2]:
            reasons.append("Golden cross (9MA crossed above 21MA)")
            signal = "BUY"
        elif latest['MA_9'] < latest['MA_21'] and df['MA_9'].iloc[-2] >= df['MA_21'].iloc[-2]:
            reasons.append("Death cross (9MA crossed below 21MA)")
            signal = "SELL"
        
        # MACD crossover
        if latest['MACD'] > latest['Signal_Line'] and df['MACD'].iloc[-2] <= df['Signal_Line'].iloc[-2]:
            reasons.append("MACD crossed above Signal Line")
            signal = "BUY"
        elif latest['MACD'] < latest['Signal_Line'] and df['MACD'].iloc[-2] >= df['Signal_Line'].iloc[-2]:
            reasons.append("MACD crossed below Signal Line")
            signal = "SELL"
        
        return {
            "ticker": ticker,
            "price": latest['Close'],
            "ma_9": latest['MA_9'],
            "ma_21": latest['MA_21'],
            "ma_50": latest['MA_50'],
            "rsi_14": latest['RSI_14'],
            "macd": latest['MACD'],
            "signal_line": latest['Signal_Line'],
            "signal": signal,
            "reasons": reasons,
            "last_updated": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error calculating indicators for {ticker}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Technical indicator calculation failed for {ticker}")


price_cache = {}

# API Endpoints
@app.get("/debug/db")
async def debug_db():
    return {
        "account": db["account"],
        "nifty50": len(db["nifty50"])
    }
@app.get("/api/stocks")
@cache(expire=60)
async def get_all_stocks():
    """Fetch all NIFTY 50 stock prices in one batch"""
    try:
        tickers = " ".join(db["nifty50"])  # Join tickers with spaces
        data = yf.download(tickers, period="1d", group_by="ticker")
        
        stocks = []
        for ticker in db["nifty50"]:
            try:
                price = data[ticker]['Close'].iloc[-1]
                stocks.append({"ticker": ticker, "price": price})
            except:
                stocks.append({"ticker": ticker, "price": 0.0})  # Fallback
        
        return {"stocks": stocks}
    except Exception as e:
        logger.error(f"Error fetching stock data: {str(e)}")
        raise HTTPException(500, "Could not fetch stock data")


def get_stock_name(ticker: str) -> str:
    """Helper function to get stock name"""
    stock = yf.Ticker(ticker)
    return stock.info.get('shortName', ticker)

@app.get("/api/stock/{ticker}")
async def get_stock_data(ticker: str):
    if ticker not in db["nifty50"]:
        raise HTTPException(404, "Stock not found")
    return calculate_technical_indicators(ticker)


@app.get('/favicon.ico', include_in_schema=False)
async def favicon():
    return ""

@app.get("/api/stock/{ticker}/price")
@cache(expire=60)
async def get_stock_price(ticker: str):
    """Get current price for a specific stock"""
    if ticker not in db["nifty50"]:
        raise HTTPException(404, "Stock not found")
    return {"price": await get_current_price(ticker)}


@app.post("/historical")
async def get_historical_data_endpoint(request: HistoricalDataRequest):
    if request.ticker not in db["nifty50"]:
        raise HTTPException(404, "Stock not found in our universe")
    try:
        data = get_historical_data(request.ticker, request.period, request.interval)
        # Convert DataFrame to dictionary for JSON response
        data = data.replace([np.inf, -np.inf, np.nan], 0)
        data = data.astype(float)
        return {
            "ticker": request.ticker,
            "period": request.period,
            "interval": request.interval,
            "data": data.reset_index().to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(400, str(e))

@app.get("/indicators/{ticker}")
async def get_indicators(ticker: str):
    if ticker not in db["nifty50"]:
        raise HTTPException(404, "Stock not found in our universe")
    return calculate_technical_indicators(ticker)

@app.post("/trade")
async def execute_trade(trade: Trade):
    if trade.ticker not in db["nifty50"]:
        raise HTTPException(404, "Stock not found in our universe")
    
    try:
        indicators = calculate_technical_indicators(trade.ticker)
        current_price = indicators["price"]
        
        # Validate trade against rules
        trade_value = trade.quantity * current_price
        if trade_value < db["rules"]["min_trade_value"]:
            raise HTTPException(400, f"Trade value below minimum ₹{db['rules']['min_trade_value']}")
        
        account = db["account"]
        
        if trade.type == "BUY":
            if trade_value > account["capital"] * db["rules"]["max_position_size"]:
                raise HTTPException(400, "Exceeds maximum position size")
            
            total_cost = trade_value + db["rules"]["commission"]
            if total_cost > account["cash_balance"]:
                raise HTTPException(400, "Insufficient funds")
            
            # Execute buy
            account["cash_balance"] -= total_cost
            
            # Update holdings
            existing = next((h for h in account["holdings"] if h["ticker"] == trade.ticker), None)
            if existing:
                existing["quantity"] += trade.quantity
                existing["avg_price"] = ((existing["avg_price"] * (existing["quantity"] - trade.quantity)) + 
                                        (trade.quantity * current_price)) / existing["quantity"]
            else:
                account["holdings"].append({
                    "ticker": trade.ticker,
                    "quantity": trade.quantity,
                    "avg_price": current_price,
                    "buy_date": datetime.now().isoformat(),
                    "stop_loss": current_price * (1 - db["rules"]["stop_loss_pct"]),
                    "take_profit": current_price * (1 + db["rules"]["take_profit_pct"])
                })
            
            # Record transaction
            transaction = {
                "id": str(uuid.uuid4()),
                "date": datetime.now().isoformat(),
                "ticker": trade.ticker,
                "action": "BUY",
                "quantity": trade.quantity,
                "price": current_price,
                "commission": db["rules"]["commission"],
                "signal": indicators["signal"],
                "pnl": 0,
                "notes": f"Signals: {', '.join(indicators['reasons'])}" if indicators['reasons'] else ""
            }
            account["transactions"].append(transaction)
        
        elif trade.type == "SELL":
            holding = next((h for h in account["holdings"] if h["ticker"] == trade.ticker), None)
            if not holding or holding["quantity"] < trade.quantity:
                raise HTTPException(400, "Insufficient holdings")
            
            # Calculate P&L
            pnl = (current_price - holding["avg_price"]) * trade.quantity - db["rules"]["commission"]
            
            # Execute sell
            account["cash_balance"] += (trade.quantity * current_price) - db["rules"]["commission"]
            
            # Update holdings
            holding["quantity"] -= trade.quantity
            if holding["quantity"] == 0:
                account["holdings"].remove(holding)
            
            # Record transaction
            transaction = {
                "id": str(uuid.uuid4()),
                "date": datetime.now().isoformat(),
                "ticker": trade.ticker,
                "action": "SELL",
                "quantity": trade.quantity,
                "price": current_price,
                "commission": db["rules"]["commission"],
                "signal": indicators["signal"],
                "pnl": pnl,
                "notes": f"Signals: {', '.join(indicators['reasons'])}" if indicators['reasons'] else ""
            }
            account["transactions"].append(transaction)
        
        # Generate updated portfolio reports
        await generate_csv_exports()
        
        return {
            "message": "Trade executed successfully",
            "cash_balance": account["cash_balance"],
            "current_price": current_price,
            "signal": indicators["signal"]
        }
    
    except Exception as e:
        logger.error(f"Trade execution failed: {str(e)}")
        raise HTTPException(400, str(e))

@app.get("/portfolio")
async def get_portfolio():
    account = db["account"]
    
    try:
        print("Starting portfolio calculation...")  # Debug log
        print(f"Current holdings: {account['holdings']}")  # Debug log

        tasks = [calculate_holding_values(holding) for holding in account["holdings"]]
        print(f"Created {len(tasks)} tasks") 
        # Run all calculations concurrently
        results = await asyncio.gather(*tasks)
        print("Got results:", results)
        
        # Sum up the results
        holdings_value = sum(r[0] for r in results)
        unrealized_pnl = sum(r[1] for r in results)
        print(f"Calculated values - holdings: {holdings_value}, pnl: {unrealized_pnl}")  # Debug log
        
        total_value = holdings_value + account["cash_balance"]
        realized_pnl = sum(t["pnl"] for t in account["transactions"] if t["action"] == "SELL")        
        print(f"Final calculations - total: {total_value}, realized: {realized_pnl}")  # Debug log
        # Generate CSV exports
        await generate_csv_exports()
        
        response = {
            "holdings": account["holdings"],
            "transactions": account["transactions"][-10:],  # Return last 10 transactions
            "metrics": {
                "total_value": total_value,
                "cash_balance": account["cash_balance"],
                "holdings_value": holdings_value,
                "realized_pnl": realized_pnl,
                "unrealized_pnl": unrealized_pnl,
                "total_return_pct": ((total_value - account["capital"]) / account["capital"]) * 100
            }
        }
        return convert_types(response)
    except Exception as e:
        logger.error(f"Portfolio calculation failed: {str(e)}" , exc_info=True)
        raise HTTPException(500, "Could not calculate portfolio metrics")

@app.get("/check-stop-losses")
async def check_stop_losses():
    try:
        account = db["account"]
        trades_executed = []
        
        for holding in account["holdings"]:
            current_price = await get_current_price(holding["ticker"])
            
            # Check stop loss
            if current_price <= holding["stop_loss"]:
                # Execute sell order
                trade = Trade(ticker=holding["ticker"], quantity=holding["quantity"], type="SELL")
                await execute_trade(trade)
                trades_executed.append({
                    "ticker": holding["ticker"],
                    "reason": f"Stop loss triggered at {current_price}",
                    "quantity": holding["quantity"]
                })
            
            # Check take profit
            elif current_price >= holding["take_profit"]:
                # Execute sell order
                trade = Trade(ticker=holding["ticker"], quantity=holding["quantity"], type="SELL")
                await execute_trade(trade)
                trades_executed.append({
                    "ticker": holding["ticker"],
                    "reason": f"Take profit triggered at {current_price}",
                    "quantity": holding["quantity"]
                })
        
        return {
            "message": "Stop loss/take profit check completed",
            "trades_executed": trades_executed
        }
    except Exception as e:
        logger.error(f"Stop loss check failed: {str(e)}")
        raise HTTPException(500, "Could not check stop losses")

async def generate_csv_exports():
    try:
        os.makedirs("data/exports", exist_ok=True)
        
        # Export holdings
        holdings_df = pd.DataFrame(db["account"]["holdings"])
        if not holdings_df.empty:
            prices = await asyncio.gather(*[get_current_price(ticker) for ticker in holdings_df['ticker']])
            holdings_df['current_price'] = prices
            holdings_df['current_value'] = holdings_df['quantity'] * holdings_df['current_price']
            holdings_df['unrealized_pnl'] = (holdings_df['current_price'] - holdings_df['avg_price']) * holdings_df['quantity']
            holdings_df['unrealized_pnl_pct'] = np.where(
                holdings_df['avg_price'] != 0,
                ((holdings_df['current_price'] / holdings_df['avg_price']) - 1) * 100,
                0
            )
            holdings_df = holdings_df.replace([np.inf, -np.inf, np.nan], 0)
            holdings_df.to_csv("data/exports/holdings.csv", index=False)
        else:
            # Always create the file, even if empty
            pd.DataFrame(columns=["ticker", "quantity", "avg_price", "buy_date", "stop_loss", "take_profit", "current_price", "current_value", "unrealized_pnl", "unrealized_pnl_pct"]).to_csv("data/exports/holdings.csv", index=False)
        
        # Export transactions (no async operations needed here)
        transactions_df = pd.DataFrame(db["account"]["transactions"])
        if not transactions_df.empty:
            transactions_df.to_csv("data/exports/transactions.csv", index=False)

        
        logger.info("CSV exports generated successfully")
    except Exception as e:
        logger.error(f"CSV export generation failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
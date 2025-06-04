# 📈 StockAstra – Smart Stock Signal Generator

Live App: **[https://stockastra1.onrender.com](https://stockastra1.onrender.com)**

## 🚀 Overview

**StockAstra** is a web application designed to provide smart, research-backed buy signals for NSE stocks. Leveraging **FastAPI**, the app calculates key technical indicators like **RSI (14-day)** and **9-day Moving Average** to generate actionable insights.

---

## 🛠️ Tech Stack

* **Backend**: FastAPI (Python)
* **Frontend**: HTML, CSS, JavaScript

---

## ⚙️ Working of the App

1. **User selects** a stock from the list of **50 NSE stocks**.
2. The **FastAPI backend** fetches price data and computes:

   * **RSI (Relative Strength Index)** over the past 14 days.
   * **9-Day Moving Average** of the closing price.
3. Based on research-backed heuristics, the app provides a **Buy Signal** or **Sell Signal** or **Hold Signal** if the stock meets certain criteria derived from these indicators.

---

## 📋 App Rules

* **Initial Capital**: ₹1,00,000
* **Minimum 1,000 per trade**
* Buy signals are purely **technical-based**, not financial advice.
* Helps beginner and intermediate traders to identify potential entries based on momentum indicators.

---

## 🔮 Future Improvements

* Add **Sell Signals** and **Stop-Loss Recommendations**
* Include **more technical indicators** like MACD, Bollinger Bands, etc.
* Track **portfolio performance** over time
* Add **historical backtesting** functionality

---

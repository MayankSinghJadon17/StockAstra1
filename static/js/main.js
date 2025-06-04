document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const stockSelect = document.getElementById('stockSelect');
    const quantityInput = document.getElementById('quantity');
    const refreshBtn = document.getElementById('refreshBtn');
    const indicatorContainer = document.getElementById('indicatorContainer');
    const holdingsBody = document.getElementById('holdingsBody');
    const transactionBody = document.getElementById('transactionBody');
    const totalValueEl = document.getElementById('totalValue');
    const cashBalanceEl = document.getElementById('cashBalance');
    const totalPnLEl = document.getElementById('totalPnL');
    const exportBtn = document.getElementById('exportBtn');
    const exportModal = document.getElementById('exportModal');
    const closeModal = document.querySelector('.close');
    const exportHoldings = document.getElementById('exportHoldings');
    const exportTransactions = document.getElementById('exportTransactions');

    // State
    let selectedStock = null;
    let portfolioData = null;
    let nifty50Stocks = [];

    // Initialize
    loadNifty50Stocks();
    loadPortfolio();

    // Event Listeners
    stockSelect.addEventListener('change', function() {
        selectedStock = this.value;
        if (selectedStock) {
            loadStockData(selectedStock);
        }
    });

    refreshBtn.addEventListener('click', function() {
        if (selectedStock) {
            loadStockData(selectedStock);
        } else {
            alert('Please select a stock first');
        }
    });

    exportBtn.addEventListener('click', function() {
        exportModal.style.display = 'block';
    });

    closeModal.addEventListener('click', function() {
        exportModal.style.display = 'none';
    });

    window.addEventListener('click', function(event) {
        if (event.target === exportModal) {
            exportModal.style.display = 'none';
        }
    });

    // Functions
    async function loadNifty50Stocks() {
        try {
            const response = await fetch('/api/stocks');
            const data = await response.json();
            nifty50Stocks = data.stocks;
            
            // Populate dropdown with real data
            stockSelect.innerHTML = '<option disabled selected>Loading stocks... <span class="spinner"></span></option>';
            nifty50Stocks.forEach(stock => {
                const option = document.createElement('option');
                option.value = stock.ticker;
                option.textContent = `${stock.ticker} (₹${safeToFixed(stock.price)})`;
                stockSelect.appendChild(option);
            });
        } catch (error) {
            console.error('Error loading NIFTY 50 stocks:', error);
        }
    }


    async function loadStockData(ticker) {
        try {
            indicatorContainer.innerHTML = '<div class="spinner"></div>';
            const response = await fetch(`/api/stock/${ticker}`);
            const data = await response.json();
            renderIndicators(data);
        } catch (error) {
            console.error('Error loading stock data:', error);
            indicatorContainer.innerHTML = '<div class="error">Failed to load stock data</div>';
        }
    }

    function renderIndicators(data) {
        indicatorContainer.innerHTML = `
            <div class="indicator-card">
                <h3>Current Price</h3>
                <p>₹${safeToFixed(data.price)}</p>
            </div>
            <div class="indicator-card">
                <h3>9-Day MA</h3>
                <p>₹${safeToFixed(data.ma_9)}</p>
            </div>
            <div class="indicator-card">
                <h3>14-Day RSI</h3>
                <p>${safeToFixed(data.rsi_14)}</p>
            </div>
            <div class="indicator-card">
                <h3>Signal</h3>
                <p class="signal-${data.signal.toLowerCase()}">${data.signal}</p>
                <button id="tradeBtn" class="btn btn-${data.signal === 'BUY' ? 'success' : data.signal === 'SELL' ? 'danger' : 'warning'}">
                    ${data.signal === 'BUY' ? 'Buy' : data.signal === 'SELL' ? 'Sell' : 'Hold'}
                </button>
            </div>
        `;

        // Add trade button event
        document.getElementById('tradeBtn').addEventListener('click', function() {
            executeTrade(data.signal);
        });
    }
    async function loadPortfolio() {
        try {
            holdingsBody.innerHTML = `
                <tr>
                    <td colspan="8" style="text-align:center;">
                        <div class="spinner"></div>
                        <div class="loading">Loading portfolio...</div>
                    </td>
                </tr>
            `;
            const response = await fetch('/portfolio');
            const data = await response.json();
            portfolioData = data;
            renderPortfolio(data);
        } catch (error) {
            console.error('Error loading portfolio:', error);
            holdingsBody.innerHTML = '<tr><td colspan="8" class="error">Failed to load portfolio</td></tr>';
        }
    }
    
    async function executeTrade(action) {
        const quantity = parseInt(quantityInput.value);
        if (!quantity || quantity <= 0) {
            alert('Please enter a valid quantity');
            return;
        }

        if (!selectedStock) {
            alert('Please select a stock');
            return;
        }

        try {
            // Get current price first
            const priceResponse = await fetch(`/api/stock/${selectedStock}/price`);
            const priceData = await priceResponse.json();
            const currentPrice = priceData.price;

            // Then execute trade
            const tradeResponse = await fetch('/trade', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    ticker: selectedStock,
                    quantity: quantity,
                    type: action,
                    currentPrice: currentPrice  // Pass the actual price
                })
            });


            const tradeResult = await tradeResponse.json();
                
                if (tradeResponse.ok) {
                    alert(`Trade executed successfully: ${action} ${quantity} shares of ${selectedStock}`);
                    // Refresh portfolio and stock data
                    loadPortfolio();
                    loadStockData(selectedStock);
                } else {
                    alert(`Trade failed: ${tradeResult.message || 'Unknown error'}`);
                }
        }   
        catch (error) {
        console.error('Trade error:', error);
        alert('Failed to execute trade');
        }
    }

    function safeToFixed(value, digits = 2) {
    if (typeof value !== "number" || isNaN(value)) return "0.00";
    return value.toFixed(digits);
    }

    async function renderPortfolio(data) {
        // Show spinner while loading transactions
        transactionBody.innerHTML = `
        <tr>
            <td colspan="8" style="text-align:center; padding: 32px;">
            <div class="spinner"></div>
            </td>
        </tr>
        `;
        // Update metrics
        totalValueEl.textContent = `₹${safeToFixed(data.metrics.total_value)}`;
        cashBalanceEl.textContent = `₹${safeToFixed(data.metrics.cash_balance)}`;

        const totalPnL = data.metrics.realized_pnl + data.metrics.unrealized_pnl;
        const pnlPercent = data.metrics.total_value !== 0
            ? ((totalPnL / data.metrics.total_value) * 100).toFixed(2)
            : "0.00";
        totalPnLEl.textContent = `₹${safeToFixed(totalPnL)} (${pnlPercent}%)`;
        totalPnLEl.className = totalPnL >= 0 ? 'pl-positive' : 'pl-negative';

        // Fetch all current prices in parallel
        const pricePromises = data.holdings.map(h => getCurrentPrice(h.ticker));
        const prices = await Promise.all(pricePromises);

        // Render holdings
        holdingsBody.innerHTML = data.holdings.map((holding, idx) => {
            const currentPrice = typeof prices[idx] === "number" && !isNaN(prices[idx]) ? prices[idx] : 0;
            const marketValue = holding.quantity * currentPrice;
            const costBasis = holding.quantity * holding.avg_price;
            const pnl = marketValue - costBasis;
            const holdingPnLPercent = costBasis !== 0 ? ((pnl / costBasis) * 100).toFixed(2) : "0.00";
            const daysHeld = Math.floor((new Date() - new Date(holding.buy_date)) / (1000 * 60 * 60 * 24));
            console.log('currentPrice:', currentPrice, typeof currentPrice);
            return `
                <tr>
                    <td>${holding.ticker}</td>
                    <td>${holding.quantity}</td>
                    <td data-currency="₹">${safeToFixed(holding.avg_price)}</td>
                    <td data-currency="₹">${safeToFixed(currentPrice)}</td>
                    <td class="${pnl >= 0 ? 'pl-positive' : 'pl-negative'}">₹${safeToFixed(pnl)}</td>
                    <td class="${pnl >= 0 ? 'pl-positive' : 'pl-negative'}">${holdingPnLPercent}%</td>
                    <td>${daysHeld}</td>
                    <td><button class="btn-exit" data-ticker="${holding.ticker}">Exit</button></td>
                </tr>
            `;
        }).join('');

        // Add exit handlers
        document.querySelectorAll('.btn-exit').forEach(btn => {
            btn.addEventListener('click', function() {
                const ticker = this.getAttribute('data-ticker');
                const holding = data.holdings.find(h => h.ticker === ticker);

                if (confirm(`Sell all ${holding.quantity} shares of ${ticker}?`)) {
                    selectedStock = ticker;
                    quantityInput.value = holding.quantity;
                    executeTrade('SELL');
                }
            });
        });

        // Render transactions
        if (!data.transactions || data.transactions.length === 0) {
            transactionBody.innerHTML = `
            <tr>
                <td colspan="8" style="text-align:center; padding: 32px;">
                <div class="spinner"></div>
                </td>
            </tr>
            `;
        } else {
            transactionBody.innerHTML = data.transactions.map(tx => {
                return `
                    <tr>
                        <td>${new Date(tx.date).toLocaleString()}</td>
                        <td>${tx.ticker}</td>
                        <td class="${tx.action === 'BUY' ? 'type-buy' : 'type-sell'}">${tx.action}</td>
                        <td>${tx.quantity}</td>
                        <td data-currency="₹">${safeToFixed(tx.price)}</td>
                        <td data-currency="₹">${safeToFixed(tx.commission)}</td>
                        <td>${tx.signal}</td>
                        <td class="${tx.pnl >= 0 ? 'pl-positive' : 'pl-negative'}">₹${safeToFixed(tx.pnl)}</td>
                    </tr>
                `;
            }).join('');
        }

        // Update export links
        exportHoldings.href = `/data/exports/holdings.csv?t=${Date.now()}`;
        exportTransactions.href = `/data/exports/transactions.csv?t=${Date.now()}`;
    }

    async function getCurrentPrice(ticker) {
        try {
            const response = await fetch(`/api/stock/${ticker}/price`);
            const data = await response.json();
            return typeof data.price === "number" && !isNaN(data.price) ? data.price : 0;
        } catch (error) {
            console.error(`Error getting price for ${ticker}:`, error);
            return 0;
        }
    }

});
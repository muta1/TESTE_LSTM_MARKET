import yfinance as yf
import pandas as pd
import datetime
import numpy as np
from pyportfolioopt import EfficientFrontier
from pyportfolioopt import risk_models
from pyportfolioopt import expected_returns


def get_data(tickers, start_date, end_date):
    """
    Obtém dados de preço ajustado do Yahoo Finance para uma lista de ações.

    Parâmetros:
    tickers (list): lista de tickers das ações
    start_date (str): data de início no formato "YYYY-MM-DD"
    end_date (str): data de fim no formato "YYYY-MM-DD"

    Retorna:
    DataFrame com os preços de fechamento ajustados das ações.
    """
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    return data


def optimize_portfolio(data):
    """
    Otimiza a carteira utilizando a biblioteca PyPortfolioOpt.

    Parâmetros:
    data (DataFrame): DataFrame com os preços de fechamento ajustados das ações.

    Retorna:
    Ações selecionadas pela otimização.
    """
    returns = expected_returns.mean_historical_return(data)
    cov_matrix = risk_models.sample_cov(data)
    ef = EfficientFrontier(returns, cov_matrix)
    weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()
    selected_tickers = list(cleaned_weights.keys())
    return selected_tickers


def simulate_trades(selected_tickers, data):
    """
    Simula a compra e venda de ações.

    Parâmetros:
    selected_tickers (list): lista de tickers das ações selecionadas pela otimização.
    data (DataFrame): DataFrame com os preços de fechamento ajustados das ações.

    Retorna:
    Lucro gerado pela simulação.
    """
    capital = 1000000
    n_stocks = len(selected_tickers)
    weights = np.array([1/n_stocks] * n_stocks)
    prices = data[selected_tickers].iloc[-1]
    n_shares = np.floor((capital * weights) / prices)
    total_cost = np.sum(n_shares * prices)
    cash = capital - total_cost
    n_shares = np.array(n_shares, dtype=np.int64)
    portfolio = pd.DataFrame({
        'Ticker': selected_tickers,
        'Number of Shares': n_shares,
        'Price': prices,
        'Value': n_shares * prices
    })
    portfolio = portfolio.set_index('Ticker')
    print("\nCarteira otimizada:")
    print(portfolio)
    print(f"\nValor total da carteira: R${portfolio['Value'].sum():,.2f}")
    
    print(f"\nSimulando venda das ações no dia seguinte...")
    for i, ticker in enumerate(selected_tickers):
        preco = yf.download(ticker, start=data.index[-2], end=data.index[-2] + datetime.timedelta(hours=24))['Adj Close'][0]
        portfolio.iloc[i, 3] = preco * portfolio.iloc[i, 1]
    portfolio['Profit'] = portfolio['Value'] - total_cost * np.array([1] * n_stocks)
    total_profit = np.sum(portfolio['Profit'])
    print("\nLucro gerado pela simulação:")
    print(f"R${total_profit:,.2f}")
    return total_profit
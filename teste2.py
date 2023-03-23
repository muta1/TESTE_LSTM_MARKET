import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
from pypfopt import expected_returns, risk_models, EfficientFrontier

# Define as ações que você deseja analisar
acoes = ['PETR4.SA', 'VALE3.SA', 'BBAS3.SA', 'ITUB4.SA', 'B3SA3.SA',
         'BBDC4.SA', 'ABEV3.SA', 'RENT3.SA', 'ABCB4.SA', 'CSNA3.SA']

# Baixa os dados históricos das ações
dados = yf.download(acoes, start="2016-01-01", end="2022-01-01")['Adj Close']

# Cria um DataFrame para armazenar as previsões de preço futuro
previsoes = pd.DataFrame()

# Aplica uma regressão linear a cada ação para prever o preço futuro
for acao in acoes:
    x = pd.DataFrame({'dias': range(len(dados[acao]))})
    y = dados[acao]
    modelo = LinearRegression()
    modelo.fit(x, y)
    proximo_dia = pd.DataFrame({'dias': [len(dados[acao]) + 1]})
    proximo_preco = modelo.predict(proximo_dia)
    previsoes[acao] = proximo_preco

# Calcula a matriz de retornos esperados e a matriz de covariância
retornos = expected_returns.mean_historical_return(dados)
covariancia = risk_models.sample_cov(dados)

# Otimiza a carteira com base nas previsões de preço futuro
ef = EfficientFrontier(retornos, covariancia)
pesos = ef.max_sharpe()
pesos = ef.clean_weights()

# Imprime a alocação de ativos otimizada
print(pesos)
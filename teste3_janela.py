import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
from pypfopt import expected_returns, risk_models, EfficientFrontier

# Define as ações que você deseja analisar
acoes = ['PETR4.SA', 'VALE3.SA', 'BBAS3.SA', 'ITUB4.SA', 'B3SA3.SA',
         'BBDC4.SA', 'ABEV3.SA', 'RENT3.SA', 'ABCB4.SA', 'CSNA3.SA']

# Baixa os dados históricos das ações
dados = yf.download(acoes, start="2016-01-01", end="2022-01-01")['Adj Close']
# Define a data do dia anterior ao último registro retornado pelo yfinance
ultimo_dia = dados.index[-1]
dia_anterior = ultimo_dia - pd.DateOffset(days=1)

# Cria um DataFrame contendo apenas os preços de fechamento do dia anterior
precos_dia_anterior = pd.DataFrame(index=acoes)
for acao in acoes:
    preco = yf.download(acao, start=dia_anterior, end=dia_anterior)['Adj Close'][0]
    precos_dia_anterior.loc[acao, 'preco'] = preco

# Calcula o retorno esperado de cada ação com base nas previsões de preço futuro
retornos_esperados = previsoes.pct_change().iloc[-1]

# Seleciona a ação com o maior retorno esperado
melhor_acao = retornos_esperados.idxmax()

# Calcula o preço de compra e o preço de venda para a melhor ação
preco_compra = precos_dia_anterior.loc[melhor_acao, 'preco']
preco_venda = dados.loc[ultimo_dia, melhor_acao]

# Calcula o lucro obtido se tivéssemos comprado a ação no dia anterior e vendido no último dia retornado pelo yfinance
lucro = preco_venda - preco_compra
print(f"Se tivesse comprado {melhor_acao} no dia {dia_anterior} e vendido no dia {ultimo_dia}, teria obtido um lucro de R${lucro:.2f}")
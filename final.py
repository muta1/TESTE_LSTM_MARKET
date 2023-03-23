import yfinance as yf
import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
import ta
import pickle


# Etapa 1: Obter a lista de ações da Ibovespa
def obter_acoes_ibovespa():
    url = "http://bvmf.bmfbovespa.com.br/indices/ResumoCarteiraQuadrimestre.aspx?Indice=IBOV&idioma=pt-br"
    acoes = pd.read_html(url, encoding="utf-8")[0][:-1]
    tickers = [acao.replace(" ", "") + '.SA' for acao in acoes['Código']]
    return tickers

def obter_precos_historicos(tickers, start_date='2021-01-01', end_date='2023-03-21', file_name='precos_historicos.csv'):
    if os.path.exists(file_name):
        precos = pd.read_csv(file_name, index_col=0, parse_dates=True)
        #if pd.to_datetime(end_date) > precos.index[-1]:
        #    precos_novos = pd.DataFrame()
        #    for ticker in tickers:
        #        try:
        #            data = yf.download(ticker, start=precos.index[-1], end=end_date)
        #            if not data.empty:
        #                precos_novos[ticker] = data['Adj Close']
        #        except Exception as e:
        #            print(f"Erro ao baixar dados do ticker {ticker}: {e}")
        #    precos = precos.combine_first(precos_novos)
        #    precos.to_csv(file_name)
    else:
        precos = pd.DataFrame()
        for ticker in tickers:
            try:
                data = yf.download(ticker, start=start_date, end=end_date)
                if not data.empty:
                    precos[ticker] = data['Adj Close']
            except Exception as e:
                print(f"Erro ao baixar dados do ticker {ticker}: {e}")
        precos.to_csv(file_name)
    return precos

# Etapa 3: Calcular o retorno médio das ações
def calcular_retorno_medio(precos):
    retornos = precos.pct_change().mean()
    return retornos



def treinar_modelo(X_train, y_train, epochs=50, batch_size=64, force_retrain=False, model_path='model.pkl'):
    if os.path.exists(model_path) and not force_retrain:
        # Carregar o modelo salvo
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    else:
        # Ajustar o formato de entrada para o formato exigido pelo LSTM
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2])

        # Definir a arquitetura do modelo LSTM
        model = Sequential()
        model.add(LSTM(100, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(50, activation='relu', return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')

        # Treinar o modelo
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

        # Salvar o modelo treinado
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

    return model

def selecionar_melhores_acoes(precos, model, n_acoes=10):
    previsoes = {}

    # Prever o retorno de cada ação usando o modelo LSTM
    for ticker in precos.columns:
        X, _, _2 = criar_features(precos[[ticker]])
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2])
        previsoes[ticker] = model.predict(X)[-1][0]

    # Selecionar as 10 melhores ações com base nas previsões
    melhores_acoes = sorted(previsoes, key=previsoes.get, reverse=True)[:n_acoes]

    return melhores_acoes




def criar_features(precos, window=30):
    X, y, tickers = [], [], []
    scaler = MinMaxScaler()

    for ticker in precos.columns:
        preco_ticker = precos[ticker].dropna()

        sma = ta.trend.sma_indicator(preco_ticker, window=window)
        ema = ta.trend.ema_indicator(preco_ticker, window=window)
        rsi = ta.momentum.rsi(preco_ticker, window=window)
        macd = ta.trend.macd(preco_ticker)
        bb_high, bb_mid, bb_low = ta.volatility.bollinger_hband(preco_ticker), ta.volatility.bollinger_mavg(preco_ticker), ta.volatility.bollinger_lband(preco_ticker)

        df_indicadores = pd.concat([preco_ticker, sma, ema, rsi, macd, bb_high, bb_mid, bb_low], axis=1)
        df_indicadores.columns = ['preco', 'sma', 'ema', 'rsi', 'macd', 'bb_high', 'bb_mid', 'bb_low']
        df_indicadores.dropna(inplace=True)

        # Escalonando os indicadores
        df_indicadores_scaled = scaler.fit_transform(df_indicadores.values)

        for i in range(window, len(df_indicadores_scaled)):
            X.append(df_indicadores_scaled[i-window:i, :])
            y.append(df_indicadores_scaled[i, 0])
            tickers.append(ticker)

    return np.array(X), np.array(y), tickers


def otimizar_portfolio(precos, melhores_acoes):
    # Filtrar os preços das ações selecionadas
    precos_selecionados = precos[melhores_acoes]

    # Calcular retornos esperados e matriz de covariância
    retornos_esperados = expected_returns.mean_historical_return(precos_selecionados)
    matriz_cov = risk_models.sample_cov(precos_selecionados)

    # Criar um portfólio eficiente com base nos retornos esperados e matriz de covariância
    ef = EfficientFrontier(retornos_esperados, matriz_cov)

    # Otimizar o portfólio para maximizar o índice Sharpe
    pesos_otimizados = ef.max_sharpe()

    # Converter os pesos brutos em pesos ajustados (porcentagens)
    pesos_otimizados = ef.clean_weights()

    return pesos_otimizados



def simular_lucro_periodo(melhores_acoes, precos, periodo_dias):
    investimento_inicial = 10000  # 10.000 unidades monetárias
    investimento_por_acao = investimento_inicial / len(melhores_acoes)
    lucros = []

    for dia in range(1, periodo_dias + 1):
        lucro_dia = 0
        for ticker in melhores_acoes:
            preco_ontem = precos.loc[precos.index[-(dia + 1)], ticker]
            preco_hoje = precos.loc[precos.index[-dia], ticker]
            acoes_compradas = investimento_por_acao / preco_ontem
            lucro_dia += acoes_compradas * (preco_hoje - preco_ontem)
            print("",dia," - ",ticker, " : ", lucro_dia)
        if (not np.isnan(lucro_dia)):
            lucros.append(lucro_dia)

    return lucros


def plotar_lucro(lucros):
    plt.plot(np.cumsum(lucros))
    plt.xlabel('Dias')
    plt.ylabel('Lucro acumulado')
    plt.title('Lucro acumulado durante o período de simulação')
    plt.show()

def main():
    tickers = obter_acoes_ibovespa()
    precos = obter_precos_historicos(tickers)
    
    X, y, tickers_dataset = criar_features(precos)
    X_train, X_test, y_train, y_test, tickers_train, tickers_test = train_test_split(
        X, y, tickers_dataset, test_size=0.3, random_state=42
    )
    model = treinar_modelo(X_train, y_train)

    # Selecionar as 10 melhores ações
    melhores_acoes = selecionar_melhores_acoes(precos, model)

    # Otimizar a alocação do portfólio entre as melhores ações
    alocação_otimizada = otimizar_portfolio(precos, melhores_acoes)

    alocação_otimizada = filter(lambda x : x != 0,alocação_otimizada)
    print("Alocação de recursos otimizada:")
    for ação, peso in alocação_otimizada.items():
        print(f"{ação}: {peso:.2%}")

    print("As 10 melhores ações para investir são:")
    print(melhores_acoes)

    periodo_dias = 20
    lucros = simular_lucro_periodo(melhores_acoes, precos, periodo_dias)
    print(f"Lucro acumulado após {periodo_dias} dias: {sum(lucros):.2f}")

    plotar_lucro(lucros)


if __name__ == "__main__":
    main()

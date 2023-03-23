import yfinance as yf
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam


# Etapa 1: Obter a lista de ações da Ibovespa
def obter_acoes_ibovespa():
    url = "http://bvmf.bmfbovespa.com.br/indices/ResumoCarteiraQuadrimestre.aspx?Indice=IBOV&idioma=pt-br"
    acoes = pd.read_html(url, encoding="utf-8")[0][:-1]
    tickers = [acao.replace(" ", "") + '.SA' for acao in acoes['Código']]
    return tickers

def obter_precos_historicos(tickers, start_date='2021-01-01', end_date='2023-03-21', file_name='precos_historicos.csv'):
    if os.path.exists(file_name):
        precos = pd.read_csv(file_name, index_col=0, parse_dates=True)
        if pd.to_datetime(end_date) > precos.index[-1]:
            precos_novos = pd.DataFrame()
            for ticker in tickers:
                try:
                    data = yf.download(ticker, start=precos.index[-1], end=end_date)
                    if not data.empty:
                        precos_novos[ticker] = data['Adj Close']
                except Exception as e:
                    print(f"Erro ao baixar dados do ticker {ticker}: {e}")
            precos = precos.combine_first(precos_novos)
            precos.to_csv(file_name)
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


def criar_features_old(precos, window=30):
    X, y, tickers = [], [], []
    for ticker in precos.columns:
        preco_ticker = precos[ticker].dropna()
        for i in range(window, len(preco_ticker)):
            X.append(preco_ticker.iloc[i-window:i].values)
            y.append(preco_ticker.iloc[i])
            tickers.append(ticker)
    return np.array(X), np.array(y), tickers

def criar_features_v1(precos, window=30):
    X, y, tickers = [], [], []
    scaler = MinMaxScaler()
    for ticker in precos.columns:
        print("Criando feature")
        preco_ticker = precos[ticker].dropna()
        preco_ticker_scaled = scaler.fit_transform(preco_ticker.values.reshape(-1, 1))
        for i in range(window, len(preco_ticker)):
            X.append(preco_ticker_scaled[i-window:i, 0])
            y.append(preco_ticker_scaled[i, 0])
            tickers.append(ticker)
    return np.array(X), np.array(y), tickers


def treinar_modelo_v1(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def treinar_modelo(X_train, y_train, epochs=100, batch_size=64, dropout_rate=0.2, learning_rate=0.001):
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    return model

def selecionar_melhores_acoes(model, X_val, y_val, tickers_val):
    tickers_unicos = list(set(tickers_val))
    erros = {}

    for ticker in tickers_unicos:
        indices = [i for i, t in enumerate(tickers_val) if t == ticker]
        X_ticker = X_val[indices]
        y_ticker = y_val[indices]

        y_pred = model.predict(X_ticker)
        erro = mean_squared_error(y_ticker, y_pred)
        erros[ticker] = erro

    erros = pd.Series(erros)
    melhores_acoes = erros.nsmallest(10)
    return melhores_acoes



def criar_features_LSTM(precos, window=30):
    X, y, tickers = [], [], []
    scaler = MinMaxScaler(feature_range=(0, 1))
    for ticker in precos.columns:
        preco_ticker = precos[ticker].dropna()
        preco_ticker_scaled = scaler.fit_transform(preco_ticker.values.reshape(-1, 1))
        for i in range(window, len(preco_ticker_scaled)):
            X.append(preco_ticker_scaled[i-window:i])
            y.append(preco_ticker_scaled[i])
            tickers.append(ticker)
    return np.array(X), np.array(y), tickers, scaler

import ta

def criar_features(precos, window=30):
    X, y, tickers = [], [], []
    scaler = MinMaxScaler()

    for ticker in precos.columns:
        preco_ticker = precos[ticker].dropna()

        # Adicionando indicadores técnicos
        sma = ta.trend.sma_indicator(preco_ticker, window=window)
        ema = ta.trend.ema_indicator(preco_ticker, window=window)
        rsi = ta.momentum.rsi(preco_ticker, window=window)

        # Concatenando os indicadores e os preços
        df_indicadores = pd.concat([preco_ticker, sma, ema, rsi], axis=1)
        df_indicadores.columns = ['preco', 'sma', 'ema', 'rsi']
        df_indicadores.dropna(inplace=True)

        # Escalonando os indicadores
        df_indicadores_scaled = scaler.fit_transform(df_indicadores.values)

        for i in range(window, len(df_indicadores_scaled)):
            X.append(df_indicadores_scaled[i-window:i, :])
            y.append(df_indicadores_scaled[i, 0])
            tickers.append(ticker)

    return np.array(X), np.array(y), tickers


def treinar_modelo_LSTM(X_train, y_train, epochs=50, batch_size=64):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    return model

def simular_lucro_periodo(melhores_acoes, precos, periodo_dias):
    investimento_inicial = 10000  # 10.000 unidades monetárias
    investimento_por_acao = investimento_inicial / len(melhores_acoes)
    lucros = []

    for dia in range(1, periodo_dias + 1):
        lucro_dia = 0
        for ticker in melhores_acoes.index:
            preco_ontem = precos.loc[precos.index[-(dia + 1)], ticker]
            preco_hoje = precos.loc[precos.index[-dia], ticker]
            acoes_compradas = investimento_por_acao / preco_ontem
            lucro_dia += acoes_compradas * (preco_hoje - preco_ontem)
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
    print("Treinando1 ")
    X_train, X_test, y_train, y_test, tickers_train, tickers_test = train_test_split(
        X, y, tickers_dataset, test_size=0.3, random_state=42
    )
    print("Treinando2")
    model = treinar_modelo(X_train, y_train)
    melhores_acoes = selecionar_melhores_acoes(model, X_test, y_test, tickers_test)

    print("As 10 melhores ações para investir são:")
    print(melhores_acoes)

    periodo_dias = 2
    lucros = simular_lucro_periodo(melhores_acoes, precos, periodo_dias)
    print(f"Lucro acumulado após {periodo_dias} dias: {sum(lucros):.2f}")

    plotar_lucro(lucros)


if __name__ == "__main__":
    main()

import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
import numpy as np

def select_cols(dataframe):
    return dataframe[['Frete por kg', 'Mode', 'UF Origem', 'Origem Cap. Inte.',
       'UF Destino', 'Destivo Cap. Inte.2',
       'Status', 'Renda per Capta', 'Populacao',
       'Distancia', 'Cabotagem']]

def complete_cols(dataframe):
    cols = ['Frete por kg', 'Renda per Capta', 'Populacao', 'Distancia', 'MOD_Baixo Valor',
     'MOD_Médio Alto Valor', 'UFO_BA', 'UFO_CE', 'UFO_GO', 'UFO_MG', 'UFO_PA', 'UFO_PE', 'UFO_PR',
     'UFO_RS', 'UFO_SP', 'CIO_Capital', 'CIO_Interior', 'UFD_AC', 'UFD_AL', 'UFD_AM', 'UFD_AP', 'UFD_BA',
     'UFD_CE', 'UFD_DF', 'UFD_ES', 'UFD_GO', 'UFD_MA', 'UFD_MG', 'UFD_MS', 'UFD_MT', 'UFD_PA', 'UFD_PB',
     'UFD_PE', 'UFD_PI', 'UFD_PR', 'UFD_RJ', 'UFD_RN', 'UFD_RO', 'UFD_RR', 'UFD_RS', 'UFD_SC', 'UFD_SE',
     'UFD_SP', 'UFD_TO', 'CID_Capital', 'CID_Interior', 'STA_Atual', 'STA_Cotacao', 'CAB_0', 'CAB_1']
    needed = set(cols) - set(dataframe)
    for col in needed:
        dataframe[col] = 0
    return dataframe[cols]

def make_dummies(dataframe):
    dataframe['Populacao'] = np.clip(dataframe['Populacao'], a_min=0, a_max=2000000)
    return pd.get_dummies(dataframe, columns=['Mode', 'UF Origem', 'Origem Cap. Inte.',
                                              'UF Destino', 'Destivo Cap. Inte.2',
                                              'Status', 'Cabotagem'],
                          prefix=['MOD', 'UFO', 'CIO', 'UFD', 'CID', 'STA', 'CAB'])

def rnn_model(inp):
    model = Sequential()
    model.add(Dense(500, activation='relu', input_dim=inp.shape[1]))
    model.add(Dropout(0.3))
    model.add(Dense(300, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='tanh'))
    model.compile(loss=mean_cubic_error, optimizer='adam')
    return model

from keras import backend as K


def mean_cubic_error(y_true, y_pred):
    return K.mean(K.abs(K.pow(y_pred - y_true, 3)), axis=-1)



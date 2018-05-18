import numpy as np
from sklearn import preprocessing
import pickle
from sklearn.model_selection import train_test_split
from util import *
from keras.callbacks import ModelCheckpoint

df = pd.read_excel('Análise Aula AI v2.xlsx')

data = select_cols(df)
data = make_dummies(data)


norm_data = np.asarray(data)

y = norm_data[:, :1]
y = np.clip(y, 0, 6)
X = norm_data[:, 1:]

y_norm = preprocessing.MinMaxScaler(feature_range=(-1, 1))
y = y_norm.fit_transform(y)
X_norm = preprocessing.MinMaxScaler()
X_norm.fit(X)
X = X_norm.transform(X)

pickle.dump(y_norm, open('y_norm.pkl', 'wb'))
pickle.dump(X_norm, open('X_norm.pkl', 'wb'))

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=True)

model = rnn_model(X_train)

checkpointer = ModelCheckpoint(filepath='model_frete_weights.hdf5', verbose=1, save_best_only=True)
model.fit(X_train, y_train, batch_size=500, epochs=2000, verbose=2, validation_data=(X_test, y_test), callbacks=[checkpointer], shuffle=True)

print(model.evaluate(X_test, y_test))


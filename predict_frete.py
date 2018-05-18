from util import *
import pickle
import numpy as np

df = pd.read_excel('novas_rotas.xlsx')
data = select_cols(df)
data = make_dummies(data)
data = complete_cols(data)
norm_data = np.asarray(data)
X = norm_data[:, 1:]
y_norm = pickle.load(open('y_norm.pkl', 'rb'))
X_norm = pickle.load(open('X_norm.pkl', 'rb'))

X = X_norm.transform(X)

model = rnn_model(X)
model.load_weights('model_frete_weights.hdf5')

result = model.predict(X)
result = y_norm.inverse_transform(result)

df['Frete por kg'] = result

df.to_csv('predict.csv', encoding='latin1')

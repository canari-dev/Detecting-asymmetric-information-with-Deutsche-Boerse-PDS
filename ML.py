#!/root/miniconda3/envs/py38/bin/python3.8


from SetUp import *

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline



# load dataset
df = pd.read_pickle(
    folder3 + '/XY_all_stocks -st_' + str(st) + '-lt_' + str(lt) + '-type_' + str(filter_type) + '-cap_' + str(
        cap) + '.pkl')
Xcol = ['dt-EWMA_ATF', 'dt-TotalSensiATF', 'dt-NumberOfTrades']
df['RY'] = 1 - df['Y-EWMA_ATF']
Ycol = ['Y-EWMA_ATF', 'RY']
df = df.dropna(subset=Xcol + Ycol, how='any')
X = df[Xcol].values.astype(float)
y = df[Ycol].values.astype(float)
sep = int(X.shape[0] / 2)
X_train = X[:sep]
y_train = y[:sep]
X_test = X[sep:]
y_test = y[sep:]

model = Sequential()
model.add(Dense(4, activation='relu', input_dim=3))
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=1)

pred_train = model.predict(X_train)
scores = model.evaluate(X_train, y_train, verbose=0)
print('Accuracy on training data: {}% \n Error on training data: {}'.format(scores[1], 1 - scores[1]))

pred_test = model.predict(X_test)
scores2 = model.evaluate(X_test, y_test, verbose=0)
print('Accuracy on test data: {}% \n Error on test data: {}'.format(scores2[1], 1 - scores2[1]))

res = pd.DataFrame(scores)
res.to_csv(folder4 + '/res1.csv')

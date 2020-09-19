import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv1D
from keras.layers import  MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.optimizers import Adam
import tensorflow as tf
from sklearn import metrics
from sklearn.metrics import classification_report, accuracy_score

df = pd.read_csv("C:/Users/Admin/Desktop/Machine Learning/Project_ML_DL/Projects/CNN_Project_3/datasets_33180_43520_heart.csv")
df.info()

# searching the missing values
print(df.isnull().sum())

# labeling taget column
df['target'] = df.target.map({0: 0, 1: 1, 2: 1, 3: 1, 4: 1})

# converting column into int
df['age'] = df['age'].astype(int)
df['sex'] = df['sex'].astype(int)
df['cp'] = df['cp'].astype(int)
df['trestbps'] = df['trestbps'].astype(int)
df['chol'] = df['chol'].astype(int)
df['fbs'] = df['fbs'].astype(int)
df['restecg'] = df['restecg'].astype(int)
df['thalach'] = df['thalach'].astype(int)
df['exang'] = df['exang'].astype(int)
df['slope'] = df['slope'].astype(int)
df['ca'] = df['ca'].astype(float)
df['ca'] = df['ca'].astype(int)
df['thal'] = df['thal'].astype(float)
df['thal'] = df['thal'].astype(int)
df['target'] = df['target'].astype(int)

#X = data.copy()
data = df.drop(['target'], axis= 1)
X = data.copy()
X = X.to_numpy()
print(X)

y = df.target
y = y.to_numpy()
print(y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify= y)
print(X_train.shape, X_test.shape)
print(X_train.ndim)

# Dropinig target variable from X_train
X_train = np.delete(X_train,[13],1)
print(X_train.shape)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

X_train = X_train.reshape(212,13,1)
X_test = X_test.reshape(91,13,1)

epochs = 90
model = Sequential()
model.add(Conv1D(filters = 32, kernel_size=2, activation ='relu', input_shape= (13,1)))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Conv1D(filters = 64, kernel_size=2, activation ='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# Converting values into vector
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid'))

print(model.summary())

model.compile(optimizer=Adam(lr=0.00005), loss='binary_crossentropy',metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), verbose=1)

# generate classification report using predictions for binary model
binary_pred = np.round(model.predict(X_test)).astype(int)

print('Results for Binary Model')
print(accuracy_score(y_test, binary_pred))

def plot_learningCurve(history, epoch):
  # Plot training & validation accuracy values
  epoch_range = range(1, epoch+1)
  plt.plot(epoch_range, history.history['accuracy'])
  plt.plot(epoch_range, history.history['val_accuracy'])
  plt.title('Model accuracy')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Val'], loc='upper left')
  plt.show()

  # Plot training & validation loss values
  plt.plot(epoch_range, history.history['loss'])
  plt.plot(epoch_range, history.history['val_loss'])
  plt.title('Model loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Val'], loc='upper left')
  plt.show()

  history.history

  plt_learningCurve(history, epochs)
  plt.imshow()
  plt.show(block=True)
  plt.interactive(False)
  #myDataFrame.plot()
  #plt.interactive(False)
  #plt.show(block=True)







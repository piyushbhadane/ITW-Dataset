import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
db = pd.read_csv('/content/drive/MyDrive/dataset/diabetes.csv')
db.head()
db.shape
db.describe()
db['Outcome'].value_counts()
db.groupby('Outcome').mean()
X = db.drop(columns='Outcome',axis=1)
Y = db['Outcome']
print(X)
print(Y)
scaler = StandardScaler()
standardized_data = scaler.transform(X)
print(standardized_data)
X = standardized_data
Y = db['Outcome']
print(X)
print(Y)
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,stratify = Y,random_state = 2)
print(X.shape,X_train.shape,X_test.shape)
classifier = svm.SVC(kernel = 'linear')
classifier.fit(X_train,Y_train)
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy score of the training data : ',training_data_accuracy)
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score of the test data : ',test_data_accuracy)
input_data = (5,166,72,19,175,25.8,0.587,51)

input_data_as_numpy_array = np.asarray(input_data)

input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

std_data = scaler.transform(input_data_reshaped)
print(std_data)

prediction = classifier.predict(std_data)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')

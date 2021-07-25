import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import neighbors, metrics
from sklearn.metrics import max_error
from sklearn import svm
from sklearn import tree
from sklearn.linear_model import LinearRegression

train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')
example_submission = pd.read_csv('data/gender_submission.csv')
print(example_submission)
test_data_array = test_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']].values

Le = LabelEncoder()
for i in range(len(test_data_array[0])):
    test_data_array[:, i] = Le.fit_transform(test_data_array[:, i])

test_data_array = np.array(test_data_array)


X = train_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']].values
y = train_data.Survived

Le = LabelEncoder()
for i in range(len(X[0])):
    X[:, i] = Le.fit_transform(X[:, i])
X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# KNN MODEL
knn = neighbors.KNeighborsClassifier(n_neighbors=25, weights='uniform')

knn.fit(X_train, y_train)

prediction = knn.predict(X_test)

acc = metrics.accuracy_score(y_test, prediction)

print(f'The knn accuracy is: {acc}')




# SVC MODEL
svc = svm.SVC(kernel='linear')

svc.fit(X_train, y_train)

prediction = svc.predict(test_data_array)
# acc = metrics.accuracy_score(y_test, prediction)
# print(f'The SVC accuracy is: {acc}')

# print(prediction)

# concatenate predictions and persons
# passID = pd.DataFrame(test_data.PassengerId, columns=['PassengerId'])
# # print(passID)
# survived = pd.DataFrame(prediction, columns=['Survived'])
# # print(survived)
# resultDF = pd.merge(passID, survived, left_index=True, right_index=True)
# resultDF.to_csv('resultData.csv', index=False)
#print(passID)
#print(survived)
# print(resultDF)







# Decision tree model
DT = tree.DecisionTreeClassifier()
DT.fit(X_train, y_train)

prediction = DT.predict(test_data_array)

# acc = metrics.accuracy_score(y_test, prediction)
passID = pd.DataFrame(test_data.PassengerId, columns=['PassengerId'])
# print(passID)
survived = pd.DataFrame(prediction, columns=['Survived'])
# print(survived)
resultDF = pd.merge(passID, survived, left_index=True, right_index=True)
resultDF.to_csv('resultData.csv', index=False)
# print(f'The Decision Tree accuracy is: {acc}')



# Linear regression model

Lr = LinearRegression()

Lr.fit(X_train, y_train)

prediction = Lr.predict(X_test)

# acc = max_error(y_test, prediction)

print(f'Linear Regression max error: {acc}')
print(f'r^2 valueL {Lr.score(y_test, prediction)}')
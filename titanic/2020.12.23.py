import numpy as np 
import pandas as pd

##### 1. 데이터 처리
train = pd.read_csv('../input/titanic/train.csv', header=0, index_col=None)
test = pd.read_csv('../input/titanic/test.csv', header=0, index_col=None)

train_x=train[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
train_y=train[['Survived']]
test_x=test[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]

em_change = {'S':1,'C':2,'Q':0}
train_x.Embarked=train_x.Embarked.map(em_change)
test_x.Embarked=test_x.Embarked.map(em_change)


sex_change={'male':0, 'female':1}
train_x.Sex=train_x.Sex.map(sex_change)
test_x.Sex=test_x.Sex.map(sex_change)


train_x=train_x.interpolate(method='linear', limit_direction='forward')
test_x=test_x.interpolate(method='linear', limit_direction='forward')
print(train_x)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

train_x=train_x.to_numpy().astype('float32')
train_y=train_y.to_numpy().astype('float32')
test_x=test_x.to_numpy().astype('float32')

train_y=train_y.reshape(891,)

print(train_x.shape)
print(train_y.shape)
print(test_x.shape)

print(train_y)

##### 2. 모델 구성
model=Sequential()
model.add(Dense(64, activation='relu', input_shape=(7,)))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(4, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))


##### 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping

es=EarlyStopping(monitor='val_loss', patience=30, mode='min')

model.fit(train_x, train_y, epochs=10000, verbose=1, validation_split=0.2, callbacks=[es])

##### 4. 예측
predict=model.predict(test_x)
predict = (predict > 0.5).astype(int).ravel()
print('predict :', predict)


##### 제출
submit = pd.DataFrame({"PassengerId":test.PassengerId, 'Survived':predict})
submit.to_csv("final_submission.csv",index = False)
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

from xgboost import XGBClassifier

train_x=train_x.to_numpy().astype('float32')
train_y=train_y.to_numpy()
test_x=test_x.to_numpy().astype('float32')

train_y=train_y.reshape(891,)

print(train_x.shape)
print(train_y.shape)
print(test_x.shape)

print(train_y)


##### 2. 모델 구성
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split, KFold, cross_val_score

parameters= [
    {'n_estimators' : [100,200, 300],
    'learning_rate' : [0.1,0.3,0.001,0.01],
    'max_depth' : [4,5,6]}, 
    {'n_estimators' : [100,200, 300],
    'learning_rate' : [0.1, 0.001, 0.01],
    'max_depth' : [4,5,6],
    'colsample_bytree' :[0.6, 0.9, 1]},
    {'n_estimators' : [90, 110],
    'learning_rate' : [0.1, 0.001, 0.5],
    'max_depth' : [4,5,6],
    'colsample_bytree' :[0.6, 0.9, 1],
    'colsample_bylevel' :[0.6, 0.7, 0.9]}
] 

kfold=KFold(n_splits=5, shuffle=True)
search=RandomizedSearchCV(XGBClassifier(), parameters, cv=kfold)

search.fit(train_x, train_y) 

model=XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.9, gamma=0, gpu_id=-1,
              importance_type='gain', interaction_constraints='',
              learning_rate=0.01, max_delta_step=0, max_depth=6,
              min_child_weight=1, monotone_constraints='()',
              n_estimators=200, n_jobs=0, num_parallel_tree=1, random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
              tree_method='exact', validate_parameters=1, verbosity=None)


##### 3. 컴파일, 훈련
model.fit(train_x, train_y)


##### 4. 예측
predict=model.predict(test_x)

predict = (predict > 0.5).astype('int').ravel()
print('predict :', predict)


##### 3. 컴파일, 훈련
model.fit(train_x, train_y)


##### 4. 예측
predict=model.predict(test_x)

predict = (predict > 0.5).astype(int).ravel()
print('predict :', predict)


##### 제출
submit = pd.DataFrame({"PassengerId":test.PassengerId, 'Survived':predict})
submit.to_csv("final_submission.csv",index = False)
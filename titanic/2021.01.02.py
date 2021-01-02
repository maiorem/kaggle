from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


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


# model = LinearSVC()
# model = SVC()
# model = KNeighborsClassifier(n_neighbors = 3)
model = LogisticRegression()
model.fit(train2,train["Survived"])
result = model.predict(test2)

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

kfold = StratifiedKFold(n_splits=5, random_state=7)
results = cross_val_score(model,train2,train["Survived"], cv=kfold)
print("Accuracy: %.2f%%" % (results.mean()*100))


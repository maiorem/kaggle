import numpy as np # linear algebra
import pandas as pd

train=pd.read_csv('/kaggle/input/digit-recognizer/train.csv', header=0, index_col=None)
test=pd.read_csv('/kaggle/input/digit-recognizer/test.csv', header=0, index_col=None)

train_y=train['label']
train.drop(['label'], axis=1, inplace=True)

dataset=pd.concat([train, test])

from sklearn.decomposition import PCA
from tensorflow.keras.utils import to_categorical

pca=PCA()
pca.fit(dataset)
cumsum=np.cumsum(pca.explained_variance_ratio_) #누적된 합 표시

d=np.argmax(cumsum >= 1) + 1


pca1=PCA(n_components=d)
dataset=pca1.fit_transform(dataset)

dataset=dataset.reshape(dataset.shape[0], dataset.shape[1], 1)

train_x=dataset[:42000]
test=dataset[42000:]

train_y=to_categorical(train_y)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout, BatchNormalization, MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler

model=Sequential()
model.add(Conv1D(64, kernel_size=3, padding='same', activation='relu', input_shape=(train_x.shape[1], train_x.shape[2])))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.25))
model.add(Conv1D(128,  kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.25))
model.add(Conv1D(256,  kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.25))
model.add(Conv1D(512,  kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(10, activation='softmax'))


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

es=EarlyStopping(monitor='val_acc', patience=30, mode='auto', restore_best_weights=True)
reduce_lr=ReduceLROnPlateau(monitor='val_acc', patience=5, factor=np.sqrt(0.1))

model.fit(train_x, train_y, epochs=1000, batch_size=1000, verbose=1, validation_split=0.2, callbacks=[es, reduce_lr])

predict=model.predict(test)
predict=np.argmax(predict, axis=1)

imageid=[]
for i in range(len(predict)) :
    imageid.append(i+1)


submit=pd.DataFrame({"ImageId": imageid, "Label":predict})
submit.to_csv("submission.csv", index=False)

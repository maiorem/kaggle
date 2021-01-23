import numpy as np 
import pandas as pd 
import os
import zipfile

with zipfile.ZipFile("../input/aerial-cactus-identification/train.zip","r") as z:
    z.extractall("/kaggle/temp/")
with zipfile.ZipFile("../input/aerial-cactus-identification/test.zip","r") as z:
    z.extractall("/kaggle/temp/test/")
    


train_dir = "../temp/train"
test_dir = "../temp/test"
labels = pd.read_csv('../input/aerial-cactus-identification/train.csv')


labels.has_cactus = labels.has_cactus.astype(str) # Classes must be str and not int


from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout, Activation
from tensorflow.keras.layers import BatchNormalization, Reshape, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import VGG16, DenseNet121

validation_split = 0.8
idxs = np.random.permutation(range(len(labels))) < validation_split*len(labels)
train_labels = labels[idxs]
val_labels = labels[~idxs]

train_datagen = ImageDataGenerator(rescale=1/255, horizontal_flip=True, vertical_flip=True)
batch_size = 128

train_generator = train_datagen.flow_from_dataframe(train_labels,
                                                    directory=train_dir,
                                                    x_col='id',
                                                    y_col='has_cactus',
                                                    class_mode='binary',
                                                    batch_size=batch_size,
                                                    target_size=(32,32))

val_generator = train_datagen.flow_from_dataframe(val_labels,directory=train_dir,
                                                  x_col='id',
                                                  y_col='has_cactus',
                                                  class_mode='binary',
                                                  batch_size=batch_size,
                                                  target_size=(32,32))


dn=DenseNet121(weights='imagenet', include_top=False, input_shape = (32, 32, 3))
dn.trainable=False

model = Sequential()
model.add(dn)
model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))


# model = Sequential()

# model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape = (32, 32, 3)))
# model.add(MaxPooling2D((2, 2)))

# # model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
# # model.add(MaxPooling2D((2, 2)))

# model.add(Flatten())
# model.add(Dense(1, activation='sigmoid'))

model.summary()

model.compile(loss ='binary_crossentropy', optimizer = 'adam', metrics = ['acc'])

callbacks = [EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='auto', restore_best_weights=True), ReduceLROnPlateau(patience=10, verbose=1)]

epochs = 1000

model.fit(train_generator, epochs = epochs, verbose = 1, callbacks = callbacks, validation_data = val_generator)


test_datagen = ImageDataGenerator(rescale = 1/255)

test_generator = test_datagen.flow_from_directory(
    directory = test_dir,
    target_size = (32, 32),
    batch_size = 1,
    class_mode = None,
    shuffle = False)

probabilities = model.predict(test_generator)


sample_submission = pd.read_csv('../input/aerial-cactus-identification/sample_submission.csv')
df = pd.DataFrame({'id': sample_submission['id']})
df['has_cactus'] = probabilities
df.to_csv("submission.csv", index=False)

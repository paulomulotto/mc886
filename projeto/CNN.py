import os; os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
 

# Initialize the CNN
classifier = Sequential()

# Convolution
# Create 32 feature detector of 3x3 dimentions
num_features_detectors = 32
input_shape = (50,50,3)
nClasses = 120
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(input_shape)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(nClasses, activation='softmax'))# Full Connection
classifier.add(Dense(units= 120, activation='relu'))
classifier.add(Dense(units= 120, activation='sigmoid'))

classifier.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics=['accuracy'])


from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./2)

training_set = train_datagen.flow_from_directory('treino',
                                                   target_size=(150, 150),
                                                    batch_size=32,
                                                    class_mode='categorical')

teste_set = test_datagen.flow_from_directory(
                                        'validacao',
                                        target_size=(150, 150),
                                        batch_size=32,
                                        class_mode='categorical')

classifier.fit_generator(
                        training_set,
                        steps_per_epoch=260,
                        epochs=20,
                        validation_data=teste_set,
                        validation_steps=78,
                        verbose=1)
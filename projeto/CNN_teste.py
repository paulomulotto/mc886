from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialize the CNN
classifier = Sequential()

# Convolution
# Create 32 feature detector of 3x3 dimentions
num_features_detectors = 32
classifier.add(Convolution2D(num_features_detectors, 3, 3, input_shape=(150,150, 3), activation='relu'))

#Pooling
classifier.add(MaxPooling2D(pool_size=(3,3)))

# Flattening
classifier.add(Flatten())
# Full Connection
classifier.add(Dense(units= 120, activation='relu'))
classifier.add(Dense(units= 120, activation='sigmoid'))

classifier.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics=['accuracy'])


from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

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
                        epochs=300,
                        validation_data=teste_set,
                        validation_steps=20444,
                        verbose=1)

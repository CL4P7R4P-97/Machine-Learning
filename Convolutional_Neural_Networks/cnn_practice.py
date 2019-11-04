#Part 1 - Building the convolutional Neural Network
from keras.models import Sequential
from keras.layers import Convolution2D#2
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#Initialising the CNN
classifier = Sequential()

#Step -1 : Convolutioning - applying the feature detector to give feature map
#this will give number of feature maps giving the convolutional layer!

classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))#(Tensorflow backend )no of rows and columns in #2

#Step -2 : Pooling : taking the max of the 2x2
#max pooling half of original +1 and for even divide by 2
classifier.add(MaxPooling2D(pool_size = (2,2)) )

#Step - 3 : Flattening converting the feature maps obtained from maxPooilng the into single vector having high feature maps (special structures of all imaages)
classifier.add(Flatten())

#Step - 4 : Full connnetion
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

#Compiling the model
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics =[ 'accuracy'])

# Part -2 fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(   'dataset/training_set',
                                                    target_size=(64, 64),
                                                    batch_size=32,
                                                    class_mode='binary')

test_set = test_datagen.flow_from_directory(   'dataset/test_set',
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='binary')

classifier.fit_generator(   training_set,
                            steps_per_epoch=8000,
                            epochs=25,
                            validation_data=test_set,
                            validation_steps =2000)
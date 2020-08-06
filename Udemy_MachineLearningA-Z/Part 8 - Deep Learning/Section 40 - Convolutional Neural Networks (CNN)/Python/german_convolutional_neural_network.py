# Convolutional Neural Network

#%% Importing the libraries
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
print(tf.__version__)

#%% Part 1 - Data Preprocessing        


#%% Preprocessing the Training set
# execute image augmentation strategy to prevent overfitting. we do zoom, horizontal flip and shear range. We also execute feature scaling to normzalize de values of the pixels.
# code snippet copied from keras/data preprocessing/imaga data preprocessing: https://keras.io/api/preprocessing/image/
train_datagen = ImageDataGenerator(   
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# read the images from the source directory
# The dimensions to which all images found will be resized will be 64x64
# size of batches of the data = 32
# binary because we are going to classify between 2 labels
training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

#%% Preprocessing the Test set
 # we have to execute feature scaling only. there is no need to augment the number of test images
test_datagen = ImageDataGenerator(rescale=1./255)

# read the images from the source directory
# The dimensions to which all images found will be resized will be 64x64
# size of batches of the data = 32
# binary because we are going to classify between 2 labels
test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

#%% Part 2 - Building the CNN

#%% Initialising the CNN
cnn = tf.keras.models.Sequential()            # create an cnn

#%% Step 1 - Convolution
# choose 32 feature detectors, each detector with size 3x3, rectifier activation function, and asking to reshape the images to 64x64 (same size as read the images) x3 (RGB) 
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))  

#%% Step 2 - Pooling
# size of pool set to 2 and stride (equal to step size) set to 2
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

#%% Adding a second convolutional layer
# choose 32 feature detectors, each detector with size 3x3, rectifier activation function, and asking to reshape the images to 64x64 (same size as read the images) x3 (RGB)
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))   
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

#%% Step 3 - Flattening
cnn.add(tf.keras.layers.Flatten())

#%% Step 4 - Full Connection
# creating a fully connected (Dense class) hidden layer with 128 neurons and each neuron using rectifier activation function
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))  

#%% Step 5 - Output Layer
# creating the output layer with 1 neuron (since it is a binary output 0 or 1) and each neuron using sigmoid activation function
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))  

#%% Part 3 - Training the CNN

#%% Compiling the CNN
cnn.compile(optimizer='adam', loss='binary_crossentropy' , metrics=['accuracy'])

#%% Training the CNN on the Training set and evaluating it on the Test set
cnn.fit(x=training_set, validation_data=test_set, epochs=25, steps_per_epoch=8000/32, validation_steps=2000/32)

#%% Part 4 - Making a single prediction
import numpy as np
from keras.preprocessing import image
test_image = image.load_img(path='dataset/single_prediction/cat_or_dog_1.jpg', target_size=(64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = cnn.predict(test_image)
print(training_set.class_indices)
if (result[0][0] == 1):
    prediction = 'dog'
else:
    prediction = 'cat'

print(prediction)
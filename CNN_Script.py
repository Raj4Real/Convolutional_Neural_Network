# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a CNN Classification script file.
"""

# Importing the Keras libraries and packages
from keras.models import Sequential

from keras.layers import Conv2D

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.layers import Dense

from keras.layers import Dropout

import matplotlib.pyplot as plt

import numpy

# PART 1 - Initialising the CNN
classifier = Sequential()
# Convolution Layer 1
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
# Pooling Layer 1 
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Convolutional Layer 2
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
# Pooling Layer 2
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Convolutional Layer 3
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
# Pooling Layer 3
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Convolutional Layer 4
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
# Pooling Layer 4
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Flattening
classifier.add(Flatten())

# Fully Connected Layer
classifier.add(Dense(units = 64, activation = 'relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(units = 1, activation = 'sigmoid'))


# Compiling the CNN Model
classifier.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])

# PART 2 - Initializing Training Parameters
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,
shear_range = 0.2,
zoom_range = 0.2,
horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory('dataset//training_set',
target_size = (64, 64),
batch_size = 64,
class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset//test_set',
target_size = (64, 64),
batch_size = 64,
class_mode = 'binary')

# Fitting the Model
history = classifier.fit_generator(training_set,
steps_per_epoch = 800,
epochs = 22,
validation_data = test_set,
validation_steps = 300)

# Save trained model
#classifier.save('experiment-01-05-2020-80-per.h5') #22 epoch
# Load trined model
from keras.models import load_model 

#classifier = load_model('experiment-22-04-2020-70per.h5') #20 epoch
#classifier = load_model('experiment-01-05-2020-80-per.h5') #22 eoch
#classifier = load_model('experiment-29-04-2020.h5') #blurred images 
#classifier = load_model('mymodel.h5') #20 epoch less layes #40 percent accuracy  


print(history.history.keys())
#visualizing results 

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# PART 3 - Making predictions
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('image.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)

if result[0][0] == 1:
    prediction = 'Intoxicated'
else:
    prediction = 'Sober'
    
####################
## Extra ##

#### storing prediction values in variable
import os.path    
    
prediction_array = []
counter = 0
while counter < 100 :
    counter += 1    
    path = os.path.join('image_name/',str(counter))
    #path = os.path.join('image_name/',str(counter))
    #path = os.path.join('image_name/',str(counter))
    extension = '.jpg'
    path = path+extension
    test_image = image.load_img(path, target_size = (64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = classifier.predict(test_image)
    training_set.class_indices

    if result[0][0] == 1:
        prediction = 'stoned'
    else:
        prediction = 'sober'
    prediction_array.append(prediction)
    prediction_array


# Calculating intoxicated accuracy for test images
i = len(prediction_array)
n = 0
count = 0
while n < i:
    if prediction_array[n] == 'Intoxicated':
        n += 1
        count += 1
    else:
        n += 1  
test_accuracy = (count*100)/i

# Calculating sober accuracy for test images
i = len(prediction_array)
n = 0
count = 0
while n < i:
    if prediction_array[n] == 'Sober':
        n += 1
        count += 1
    else:
        n += 1  
test_accuracy = (count*100)/i



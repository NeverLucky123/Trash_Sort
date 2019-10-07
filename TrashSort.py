import os

base_dir = '/Users/henry/Desktop/Trash_Dataset/'
# Directory with our training horse pictures
trash_dir = os.path.join('/Users/henry/Desktop/Trash_Dataset/Trash')
# Directory with our training horse pictures
recycle_dir = os.path.join('/Users/henry/Desktop/Trash_Dataset/Recycling')
# Directory with our training horse pictures
compost_dir = os.path.join('/Users/henry/Desktop/Trash_Dataset/Compost')

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.applications.xception import Xception, preprocess_input

CLASSES = 3
base_model = Xception(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D(name='avg_pool')(x)
x = Dropout(0.1)(x)
predictions = Dense(CLASSES, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# transfer learning
for layer in base_model.layers:
    layer.trainable = False

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1/255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   validation_split=0.2
                                   )
# Flow training images in batches of 128 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        base_dir,  # This is the source directory for training images
        target_size=(299, 299),   # All images will be resized to 299x299 to match the NN
        batch_size=128,
        class_mode='categorical',
        subset='training')

# Flow training images in batches of 32 using train_datagen generator
validation_generator = train_datagen.flow_from_directory(
        base_dir,  # This is the source directory for validation images
        target_size=(299, 299),  # All images will be resized to 299x299 to match the NN
        batch_size=32,
        class_mode='categorical',
        subset='validation')

history = model.fit_generator(
      train_generator,
      steps_per_epoch=20,
      epochs=100,
      verbose=1,
      validation_data=validation_generator,
      validation_steps=8)

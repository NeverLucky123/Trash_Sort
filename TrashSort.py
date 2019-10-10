import os
import tensorflow as tf

base_dir = '/Users/henry/Desktop/DATASET/'
# Directory with our training horse pictures
trash_dir = os.path.join('/Users/henry/Desktop/DATASET/L/')
# Directory with our training horse pictures
recycle_dir = os.path.join('/Users/henry/Desktop/DATASET/R/')
# Directory with our training horse pictures
compost_dir = os.path.join('/Users/henry/Desktop/DATASET/O/')

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications.xception import Xception, preprocess_input, decode_predictions
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix

CLASSES = 3
base_model = Xception(weights='imagenet', include_top=False)
x = base_model.output
x = Dropout(0.2)(x)
x = GlobalAveragePooling2D(name='avg_pool')(x)
predictions = Dense(CLASSES, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# transfer learning
for layer in model.layers[:125]:
    layer.trainable = False
for layer in model.layers[125:]:
    layer.trainable = True
# for layer in base_model.layers:
# layer.trainable = False
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False),
              metrics=['acc'])

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=40,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    validation_split=0
)
# Flow training images in batches of 128 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
    base_dir,  # This is the source directory for training images
    target_size=(299, 299),  # All images will be resized to 299x299 to match the NN
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
    epochs=5,
    verbose=1)
    #validation_data=validation_generator,
   # validation_steps=8)

MODEL_FILE = 'filename.model'
model.save(MODEL_FILE)

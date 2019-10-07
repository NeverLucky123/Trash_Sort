import os
# Directory with our training horse pictures
trash_dir = os.path.join('/Users/henry/Desktop/Trash_Dataset/Trash')
# Directory with our training horse pictures
recycle_dir = os.path.join('/Users/henry/Desktop/Trash_Dataset/Recycling')
# Directory with our training horse pictures
compost_dir = os.path.join('/Users/henry/Desktop/Trash_Dataset/Compost')

import tensorflow as tf

model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 300x300 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('horses') and 1 for the other ('humans')
    tf.keras.layers.Dense(3, activation='softmax')
])
adam=tf.train.AdamOptimizer(learning_rate=0.0003)
model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['acc'])

# All images will be rescaled by 1./255
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255,
                                                                shear_range=0.2,
                                                                zoom_range=0.2,
                                                                horizontal_flip=True,
                                                                validation_split=0.2
                                                                )
# Flow training images in batches of 128 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        '/Users/henry/Desktop/Trash_Dataset/',  # This is the source directory for training images
        target_size=(300, 300),  # All images will be resized to 150x150
        batch_size=128,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='categorical',
        subset='training')

# Flow training images in batches of 128 using train_datagen generator
validation_generator = train_datagen.flow_from_directory(
        '/Users/henry/Desktop/Trash_Dataset/',  # This is the source directory for training images
        target_size=(300, 300),  # All images will be resized to 150x150
        batch_size=32,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='categorical',
        subset='validation')

history = model.fit_generator(
      train_generator,
      steps_per_epoch=20,
      epochs=100,
      verbose=1,
      validation_data=validation_generator,
      validation_steps=8)
# #"C:\Users\henry\Desktop\hoh training\horses\horse01-0.png"
# import numpy as np
# img = tf.keras.preprocessing.image.load_img('/Users/henry/Desktop/hoh training/horses/horse01-0.png', target_size=(300, 300))
# x = tf.keras.preprocessing.image.img_to_array(img)
# x = np.expand_dims(x, axis=0)
#
# images = np.vstack([x])
# classes = model.predict(images, batch_size=10)
# print(classes[0])
# if classes[0] > 0.5:
#     print(" is a human")
# else:
#     print(" is a horse")
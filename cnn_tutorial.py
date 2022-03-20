"""The thermal image dataset for object classification is comprised of total 6414 images were captured by using Seek Thermal and a total of 1014 images are captured by using FLIR. The resolution of the images captured by the FLIR camera is 1080 X 1440 and for Seek Thermal is 300 x 400. There are three classes cat, car and man. The resolution of the images is higher in comparison with the handheld thermal cameras. The texture of the object is not clear but the temperature of the object is distinguishable, edges and contours are also clear. A thermal image dataset for object classification is used for the training and testing of deep learning models.
"""
import tensorflow as tf
import os

from keras import datasets, layers, models, utils
import matplotlib.pyplot as plt
import pathlib
from PIL import Image
import numpy as np

path = r"/Users/amelyaavdeyev/capstone_data/new_images/"
images_dir = os.listdir(path)

# tot_imgs = 6414
batch_size = 4
img_height = 40
img_width = 30
num_channels = 3  # later, 1

# item = r"074.jpg"
# im = Image.open(path+item)
# print(im.mode)
# np_im = np.asarray(im)
# print(np_im.shape)

def probability_model():
    train_ds = tf.keras.utils.image_dataset_from_directory(
                path+"Train/",
                validation_split=0.2,
                subset="training",
                seed=123,
                image_size=(img_height, img_width),
                batch_size=batch_size)
    val_ds = tf.keras.utils.image_dataset_from_directory(
                path+"Validation/",
                validation_split=0.2,
                subset="validation",
                seed=123,
                image_size=(img_height, img_width),
                batch_size=batch_size)

    class_names = train_ds.class_names
    num_classes = len(class_names)
    print(f"\nClass names: {class_names}")
    print(val_ds.cardinality().numpy())


    # Preprocess data, bring out hottest areas
    # Convert np data 

    # Shuffle datasets
    train_ds.shuffle(1000)

    # plt.figure(figsize=(10, 10))
    # for images, labels in train_ds.take(1):
    #   for i in range(9):
    #     ax = plt.subplot(3, 3, i + 1)
    #     plt.imshow(images[i].numpy().astype("uint8"))
    #     plt.title(class_names[labels[i]])
    #     plt.axis("off")

    # Put in cache after loaded from mem and prefetch data while processing
    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # model = tf.keras.Sequential([
    #   tf.keras.layers.Rescaling(1./255),
    #   tf.keras.layers.Conv2D(128, 3, activation='relu'),
    #   tf.keras.layers.MaxPooling2D(),
    #   tf.keras.layers.Conv2D(128, 3, activation='relu'),
    #   tf.keras.layers.MaxPooling2D(),
    #   tf.keras.layers.Conv2D(128, 3, activation='relu'),
    #   tf.keras.layers.MaxPooling2D(),
    #   tf.keras.layers.Flatten(),
    #   tf.keras.layers.Dense(128, activation='relu'),
    #   tf.keras.layers.Dense(num_classes)
    # ])

    model = models.Sequential()
    model.add(layers.Conv2D(30, (3, 3), activation='relu', input_shape=(img_height, img_width, num_channels)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(60, (3, 3), activation='relu'))
    # model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Conv2D(60, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(60, activation='relu'))
    model.add(layers.Dense(num_classes))

    # To fix, same val_accuracy for many epochs error
    opt = tf.keras.optimizers.SGD(learning_rate=0.001) 

    model.compile(
    optimizer=opt,
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

    history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
    )

    # Plot eval info
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')

    # which img to check and plot
    i = 3

    test_loss, test_acc = model.evaluate(val_ds, verbose=2)

    probability_model = tf.keras.Sequential([model, 
                                            tf.keras.layers.Softmax()])

    predictions = probability_model.predict(val_ds)

    expected_value = class_names[np.argmax(predictions[i])]

    print(f"\nPrediction for img {i}: {expected_value}\n")

    plt.figure()
    for images, labels in val_ds.take(1):
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.xlabel("expected: " + expected_value)
        plt.title(class_names[labels[i]])
        plt.axis("off")

    plt.show()

    return probability_model
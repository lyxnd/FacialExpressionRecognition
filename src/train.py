import os
import argparse

from keras.src.callbacks import ModelCheckpoint
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.src.optimizers import SGD, Adam
from keras.src.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
from data import Fer2013, Jaffe, CK
from model import CNN1, CNN2, CNN3
from visualize import plot_loss, plot_acc

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="fer2013", help="dataset to train, fer2013 or jaffe or ck+")
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--plot_history", type=bool, default=True)
opt = parser.parse_args()
his = None
print(opt)

if opt.dataset == "fer2013":
    expressions, x_train, y_train = Fer2013().gen_train()
    _, x_valid, y_valid = Fer2013().gen_valid()
    _, x_test, y_test = Fer2013().gen_test()
    y_train = to_categorical(y_train).reshape(y_train.shape[0], -1)
    y_valid = to_categorical(y_valid).reshape(y_valid.shape[0], -1)
    y_train = np.hstack((y_train, np.zeros((y_train.shape[0], 1))))
    y_valid = np.hstack((y_valid, np.zeros((y_valid.shape[0], 1))))
    print("load fer2013 dataset successfully, it has {} train images and {} valid images".format(y_train.shape[0], y_valid.shape[0]))

    model = CNN3(input_shape=(48, 48, 1), n_classes=8)
    sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    callback = [
        ModelCheckpoint('./models/fer2013/cnn3_best.weights.h5', monitor='val_accuracy', verbose=True, save_best_only=True,
                        save_weights_only=True)
    ]

    train_generator = ImageDataGenerator(rotation_range=10, width_shift_range=0.05, height_shift_range=0.05, horizontal_flip=True, shear_range=0.2, zoom_range=0.2).flow(x_train, y_train, batch_size=opt.batch_size)
    valid_generator = ImageDataGenerator().flow(x_valid, y_valid, batch_size=opt.batch_size)

    history_fer2013 = model.fit(
        train_generator,
        steps_per_epoch=len(y_train) // opt.batch_size,
        epochs=opt.epochs,
        validation_data=valid_generator,
        validation_steps=len(y_valid) // opt.batch_size,
        callbacks=callback
    )
    his = history_fer2013

    pred = model.predict(x_test)
    pred = np.argmax(pred, axis=1)
    print("test accuracy", np.sum(pred.reshape(-1) == y_test.reshape(-1)) / y_test.shape[0])

elif opt.dataset == "jaffe":
    expressions, x, y = Jaffe().gen_train()
    y = to_categorical(y).reshape(y.shape[0], -1)
    y = np.hstack((y, np.zeros((y.shape[0], 1))))
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, random_state=2019)
    print("load jaffe dataset successfully, it has {} train images and {} valid images".format(y_train.shape[0], y_valid.shape[0]))

    train_generator = ImageDataGenerator(rotation_range=5, width_shift_range=0.01, height_shift_range=0.01, horizontal_flip=True, shear_range=0.1, zoom_range=0.1).flow(x_train, y_train, batch_size=opt.batch_size)
    valid_generator = ImageDataGenerator().flow(x_valid, y_valid, batch_size=opt.batch_size)

    model = CNN3()
    sgd = Adam(learning_rate=0.0001)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    callback = [
        ModelCheckpoint('./models/jaffe/cnn3_best.weights.h5', monitor='val_accuracy', verbose=True, save_best_only=True,
                        save_weights_only=True)
    ]

    history_jaffe = model.fit(
        train_generator,
        steps_per_epoch=len(y_train) // opt.batch_size,
        epochs=opt.epochs,
        validation_data=valid_generator,
        validation_steps=len(y_valid) // opt.batch_size,
        callbacks=callback
    )
    his = history_jaffe

else:
    expr, x, y = CK().gen_train()
    y = to_categorical(y).reshape(y.shape[0], -1)
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, random_state=2019)
    print("load CK+ dataset successfully, it has {} train images and {} valid images".format(y_train.shape[0], y_valid.shape[0]))

    train_generator = ImageDataGenerator(rotation_range=10, width_shift_range=0.05, height_shift_range=0.05, horizontal_flip=True, shear_range=0.2, zoom_range=0.2).flow(x_train, y_train, batch_size=opt.batch_size)
    valid_generator = ImageDataGenerator().flow(x_valid, y_valid, batch_size=opt.batch_size)

    model = CNN3()
    sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    callback = [
        ModelCheckpoint('./models/ck+/cnn3_best.weights.h5', monitor='val_accuracy', verbose=True, save_best_only=True,
                        save_weights_only=True)
    ]

    history_ck = model.fit(
        train_generator,
        steps_per_epoch=len(y_train) // opt.batch_size,
        epochs=opt.epochs,
        validation_data=valid_generator,
        validation_steps=len(y_valid) // opt.batch_size,
        callbacks=callback
    )
    his = history_ck

if opt.plot_history:
    plot_loss(his.history, opt.dataset)
    plot_acc(his.history, opt.dataset)
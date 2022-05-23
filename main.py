from os import path
from os import mkdir
from datetime import datetime
import pickle
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.initializers import Initializer
from tensorflow.keras.initializers import RandomUniform
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
from models import *

IMAGE_SHAPE = (None, 224, 224, 3)
BATCH_SIZE = 8
DATA_SET_DIR = "C:\\Users\\csaba\\Downloads\\birds"   # download full dataset from https://www.kaggle.com/gpiosenka/100-bird-species
TEST_SET_DIR = "test"
PREDICTIONS_OUTPUT_DIR = "predictions"
MODEL_SAVE_DIR_PREFIX = "models"
CHECKPOINTS_DIR_PREFIX = "checkpoints"
BEST_REFERENCE_LOSS = 0.0044
MODEL_SELECTION_EPOCHS_PER_TRIAL = 40



# Plotting, visualization and process data for report
def visualize_nearest_neighbour_reference(validation_iterator):
    """
    Plots validation samples after compression
    """
    model = NearestReference(IMAGE_SHAPE)
    model.compile(optimizer = "adam", loss = "mse", metrics = ["mean_squared_error"])
    model.build(IMAGE_SHAPE)
    X, y = next(validation_iterator)
    X_predict = model.__call__(X)
    for i in range(X_predict.shape[0]):
        plt.imshow(X_predict[i])
        plt.show()
        import pdb; pdb.set_trace()
        plt.imshow(y[i])
        plt.show()
        import pdb; pdb.set_trace()

def visualize_samples_after_compression(validation_iterator):
    """
    Plots validation samples after compression
    """
    model = DownsizeOnly(IMAGE_SHAPE)
    model.compile(optimizer = "adam", loss = "mse", metrics = ["mean_squared_error"])
    model.build(IMAGE_SHAPE)
    X, y = next(validation_iterator)
    X_predict = model.__call__(X)
    for i in range(X_predict.shape[0]):
        plt.imshow(X_predict[i])
        plt.show()
        import pdb; pdb.set_trace()
        plt.imshow(y[i])
        plt.show()
        import pdb; pdb.set_trace()

def visualize_samples(data_iterator):
    """
    Plots the samples using pyplot
    """
    (X, y) = next(data_generator)
    for i in range(X.shape[0]):
        plt.imshow(X[i])
        plt.show()
        plt.imshow(y[i])
        plt.show()

def plot_model_selection_data(training_losses, validation_losses):
    import pdb; pdb.set_trace()
    for model_name in list(training_losses.keys()):
        if len(training_losses[model_name]) == 1:
            training_losses[model_name] = np.repeat(training_losses[model_name], MODEL_SELECTION_EPOCHS_PER_TRIAL) # refernce models are not trainable so they only have 1 value
    for model_name in list(validation_losses.keys()):
        if len(validation_losses[model_name]) == 1:
            validation_losses[model_name] = np.repeat(validation_losses[model_name], MODEL_SELECTION_EPOCHS_PER_TRIAL) # refernce models are not trainable so they only have 1 value
    for model_name in list(training_losses.keys()):
        plt.plot(training_losses[model_name])
    plt.title("Training loss against epochs")
    plt.ylabel("MSE")
    plt.xlabel(str(MODEL_SELECTION_EPOCHS_PER_TRIAL) + " epochs")
    plt.legend(list(training_losses.keys()), loc='upper left')
    plt.show()
    for model_name in list(validation_losses.keys()):
        plt.plot(validation_losses[model_name])
    plt.title("Validation loss against epochs")
    plt.ylabel("MSE")
    plt.xlabel(str(MODEL_SELECTION_EPOCHS_PER_TRIAL) + " epochs")
    plt.legend(list(validation_losses.keys()), loc='upper left')
    plt.show()


# Main runs
# helper functions
def crop(X):
    X_out = tf.image.random_crop(X, size=[IMAGE_SHAPE[1], IMAGE_SHAPE[2], 3])
    return X_out
save_training_weights = tf.keras.callbacks.ModelCheckpoint(filepath = CHECKPOINTS_DIR_PREFIX + "/" +  "weights.{epoch:02d}-{val_loss:.4f}.hdf5", save_weights_only = True, mode = "min", save_best_only = True, monitor = "val_loss", verbose = 1)
save_weights = tf.keras.callbacks.ModelCheckpoint(filepath = CHECKPOINTS_DIR_PREFIX + "/" +  "weights.{epoch:02d}-{val_loss:.4f}.hdf5", save_weights_only = True, monitor = "val_loss", verbose = 1)

def learning_rate_fn(epoch, lr):
    return 10. ** (-1 * (3 + (epoch // 50)))


def model_selection(train_iterator, validation_iterator, load = False, checkpoint_name = None):
    """
    Trains and evaluates all the competing models
    """
    np.random.seed(0)
    tf.random.set_seed(0)
    training_losses = {}
    validation_losses = {}
    testing_losses = {}
    reference_models = [NearestReference(IMAGE_SHAPE), BilinearReference(IMAGE_SHAPE), BicubicReference(IMAGE_SHAPE), GaussianReference(IMAGE_SHAPE)]
    test_models = [AutoencoderA(IMAGE_SHAPE), AutoencoderB(IMAGE_SHAPE), AutoencoderC(IMAGE_SHAPE), AutoencoderD(IMAGE_SHAPE)]
    #import pdb; pdb.set_trace()
    for model in reference_models:
        if load and path.exists(MODEL_SAVE_DIR_PREFIX + "/" + type(model).__name__):
            model = keras.models.load_model(MODEL_SAVE_DIR_PREFIX + "/" + type(model).__name__)
        else:
            model.compile(optimizer = "adam", loss = "mse", metrics = ["mean_squared_error"])
            model.build(IMAGE_SHAPE)
        model.summary()
        history = model.fit(train_iterator, validation_data = validation_iterator, epochs = 1) #these refernce models are not NNs so we do not need to train them
        test_loss = model.evaluate(validation_iterator)
        model.save(MODEL_SAVE_DIR_PREFIX + "/" + type(model).__name__)
        training_losses[type(model).__name__] = history.history["loss"]
        validation_losses[type(model).__name__] = history.history["val_loss"]
        testing_losses[type(model).__name__] = test_loss
    for model in test_models:
        if load and checkpoint_name is None and path.exists(MODEL_SAVE_DIR_PREFIX + "/" + model.name):
            model = keras.models.load_model(MODEL_SAVE_DIR_PREFIX + "/" + model.name)
        else:
            model.compile(optimizer = tf.keras.optimizers.SGD(learning_rate = 0.001, momentum = 0.9999), loss = "mse", metrics = ["mean_squared_error"])
            model.build(IMAGE_SHAPE)
        if load and checkpoint_name is not None and path.exists(CHECKPOINTS_DIR_PREFIX + "/" +  checkpoint_name):
            model.load_weights(CHECKPOINTS_DIR_PREFIX + "/" +  checkpoint_name)
        model.summary()
        history = model.fit(train_iterator, validation_data = validation_iterator, epochs = MODEL_SELECTION_EPOCHS_PER_TRIAL, shuffle = True, callbacks = [save_weights])
        test_loss = model.evaluate(validation_iterator)
        model.save(MODEL_SAVE_DIR_PREFIX + "/" + model.name)
        training_losses[model.name] = history.history["mean_squared_error"]
        validation_losses[model.name] = history.history["val_mean_squared_error"]
        testing_losses[model.name] = test_loss
    pickle.dump(training_losses, open("training_losses", "wb"))
    pickle.dump(validation_losses, open("validation_losses", "wb"))
    pickle.dump(testing_losses, open("testing_losses", "wb"))
    plot_model_selection_data(training_losses, validation_losses)

def model_training(train_iterator, test_iterator, model = AutoencoderD(IMAGE_SHAPE), load = None):
    """
    Trains the chosen model
    """
    training_losses = []
    validation_losses = []
    psnr = []
    test_loss = 0
    if load is not None:
        model = keras.models.load(MODEL_SAVE_DIR_PREFIX + "/" + model.name)
    else:
        model.compile(optimizer = tf.keras.optimizers.Adam(), loss = "mse", metrics = ["mean_squared_error", psnr])
        model.build(IMAGE_SHAPE)
    model.summary()
    history = model.fit(train_iterator, validation_data = test_iterator, epochs = 200, shuffle = True, callbacks = [save_training_weights, tf.keras.callbacks.LearningRateScheduler(learning_rate_fn)])
    test_loss = model.evaluate(test_iterator)
    model.save(MODEL_SAVE_DIR_PREFIX + "/" + model.name)
    training_losses = history.history["mean_squared_error"]
    validation_losses = history.history["val_mean_squared_error"]
    psnr = history.history["psnr"]

def main():
    data_generator = ImageDataGenerator(validation_split = 0.05, rescale = 1./255, horizontal_flip = True)
    test_data_generator = ImageDataGenerator(rescale = 1./255, horizontal_flip = False)
    train_iterator = data_generator.flow_from_directory(DATA_SET_DIR, batch_size = BATCH_SIZE, target_size = (IMAGE_SHAPE[1], IMAGE_SHAPE[2]), color_mode = "rgb", class_mode = "input", subset = "training")
    validation_iterator = data_generator.flow_from_directory(DATA_SET_DIR, batch_size = BATCH_SIZE, target_size = (IMAGE_SHAPE[1], IMAGE_SHAPE[2]), color_mode = "rgb", class_mode = "input", subset = "validation")
    test_iterator = test_data_generator.flow_from_directory(TEST_SET_DIR, batch_size = BATCH_SIZE, target_size = (IMAGE_SHAPE[1], IMAGE_SHAPE[2]), color_mode = "rgb", class_mode = "input")
    visualize_samples(train_iterator)
    visualize_samples_after_compression(validation_iterator)
    visualize_nearest_neighbour_reference(validation_iterator)
    model_selection(train_iterator, validation_iterator, True, "weights.00-0.0032.hdf5")
    model_training(train_iterator, validation_iterator)

def predict_and_evaluate(model = AutoencoderD(IMAGE_SHAPE), load = True):
    """
    This loads the saved model and evaluates it on the test set and outputs predictions to ./predicitons
    """
    test_data_generator = ImageDataGenerator(rescale = 1./255, horizontal_flip = False)
    test_iterator = test_data_generator.flow_from_directory(TEST_SET_DIR, batch_size = BATCH_SIZE, target_size = (IMAGE_SHAPE[1], IMAGE_SHAPE[2]), color_mode = "rgb", class_mode = "input")
    trained_model = model
    if load and path.exists(MODEL_SAVE_DIR_PREFIX + "/" + model.name):
        trained_model = keras.models.load_model(MODEL_SAVE_DIR_PREFIX + "/" + model.name)
    trained_model.evaluate(test_iterator)
    # now output predictions
    low_resolution_model = DownsizeOnly(IMAGE_SHAPE)
    bilinear_model = BilinearReference(IMAGE_SHAPE)
    batch_offset = 0
    import pdb; pdb.set_trace()
    for _ in range(len(test_iterator)):
        X, y = next(test_iterator)
        X_low_resolution = low_resolution_model.__call__(X)
        X_bilinear = bilinear_model.__call__(X)
        X_predict = trained_model.__call__(X)
        for i in range(X.shape[0]):
            out_path = PREDICTIONS_OUTPUT_DIR + "/" + str(batch_offset + i + 1)
            mkdir(out_path)
            tf.keras.preprocessing.image.save_img(out_path + "/low_resolution.png", X_low_resolution[i], data_format = "channels_last", scale = [0, 255])
            tf.keras.preprocessing.image.save_img(out_path + "/bilinear.png", X_bilinear[i], data_format = "channels_last", scale = [0, 255])
            tf.keras.preprocessing.image.save_img(out_path + "/prediction.png", X_predict[i], data_format = "channels_last", scale = [0, 255])
            tf.keras.preprocessing.image.save_img(out_path + "/ground_truth.png", y[i], data_format = "channels_last", scale = [0, 255])
        batch_offset += X.shape[0]

if __name__ == "__main__":
    predict_and_evaluate()
    #main()

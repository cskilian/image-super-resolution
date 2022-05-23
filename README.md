This is an image super resolution project.

The entire test set has been included in the directory "test". It contains 1300 images of birds (224x224x3 in JPEG format).
If you use other data set for predictions it should be in the same format: (224x224x3) images in the "test" directory. 

Generated predictions are found under the directory "sample_predictions/{sample}". Each sample's directory contains a low resolution,
a binlinear 4x upscaled, our model's prediction and the ground truth (high resolution) sample.

To evaluate the best model on the included test set and generate predictions:

Run from the console: 
1. python main.py 

or from google colab: 
1. from main import predict_and_evaluate
   predict_and_evaluate()

This function loads fully trained model and evaluates it on the test set that has been included. For each image it
saves the low_resolution version, a bilinear interpolation, our model prediction and the ground truth under 
the directory "./predictions"

To complete model selection, training, and plotting:

1. download the training data: https://www.kaggle.com/gpiosenka/100-bird-species
2. set the constant: DATA_SET_DIR to the path containing the training data
3. from main import main
   main()
   
Prerequisites (installed using Anaconda):

python                    3.7.9
numpy                     1.18.5
mkl                       2020.2
scipy                     1.6.0
tensorflow                2.1.0
tensorflow-gpu            2.3.1
cudatoolkit               10.1.243
cudnn                     7.6.5
hdf5                      1.10.4
keras                     2.3.1
keras-applications        1.0.8
keras-base                2.3.1
keras-preprocessing       1.1.2

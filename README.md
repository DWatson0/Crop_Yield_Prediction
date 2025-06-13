# Crop Yield Prediction using Neural Networks

This project compares two implementations of a neural network that predicts crop yield based on soil and weather data:

- **From Scratch**: Implemented using NumPy, pandas, with data preprocessing and evaluation techniques from scikit-learn
- **TensorFlow/Keras**: Built using high-level deep learning APIs

## Dataset

**Source**: [Kaggle - Crop Yield Prediction using Soil and Weather Data](https://www.kaggle.com/datasets/gurudathg/crop-yield-prediction-using-soil-and-weather/data)
### Features include:
- Fertilizer
- temp
- N (Nitrogen)
- P (Phosphorus)
- K (Potassium)

### Target:
- Crop Yield

## Techniques Used On Scratch Neural Network 
- **Feature Scaling**
- **Architecture**: Feedforward neural network with two hidden layer (same as TensorFlow/Keras)
- **Activation Function**: Sigmoid for hidden layer, Linear for output
- **Optimizer**: Gradient Descent with computed derivatives (backpropagation)
- **Regularization**
- **Early Stopping**

## Model Performance

| Implementation   | Mean Squared Error (MSE) | R-squared Score |
|------------------|---------------------------|----------|
| TensorFlow/Keras | 0.0826                    | 0.978    |
| From Scratch     | 0.1248                    | 0.965    |

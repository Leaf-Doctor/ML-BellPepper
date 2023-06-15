# Product Based Capstone Bangkit 2023

C23-PR584

We took the case that is closest to us as an agricultural country, where there are many problems faced by farmers, especially vegetable farmers to cultivate their vegetable crops. The background information reveals that plant disease is a significant problem faced by farmers and agricultural experts worldwide. Existing methods for identifying plant diseases are manual, time-consuming, and expensive. The use of image processing and artificial intelligence in plant disease detection can help improve efficiency and accuracy in identifying plant diseases. For this project, we focused on three types of vegetable crops like, tomatoes, potatoes, and bell peppers.

### Project Scope for Machine Learning

| Week 1 | Week 2 | Week 3 | Week 4 |
| ------ | ------ | ------ | ------ |
| Research, collect, and preprocessing the dataset | Build the ML model and training the dataset | Testing and evaluate the model | Deploy model|

### Dataset
We utilize a plant dataset from Kaggle

[PlantifyDr Dataset](https://www.kaggle.com/datasets/lavaman151/plantifydr-dataset)

### Model
We utilize the TensorFlow framework to develop and train our machine learning models. Specifically, we employ the Convolutional Neural Network (CNN) architecture in TensorFlow to recognize patterns in the image data we have. By using CNN, we can extract important features from the images, allowing us to understand the underlying patterns and characteristics present.

Through an intensive training process, the model learns from labeled training data, adjusting itself to minimize the difference between predictions and actual labels. TensorFlow provides a comprehensive set of tools and functionalities for model training, including optimization algorithms and automatic differentiation. Once the training is complete, our model is ready to predict images.

When presented with new, unseen images in the testing data, our trained CNN model, implemented using TensorFlow, can accurately identify and classify objects or patterns present in the images. This enables us to leverage the power of TensorFlow's deep learning capabilities in automating image recognition tasks and providing accurate predictions based on the patterns learned during training.

![Prediction Images](https://github.com/Leaf-Doctor/ML-BellPepper/blob/main/Prediciton.png)

### Technology and Tools
There are several tools and technologies we utilize to build this model:
* Kaggle
* Visual Studio Code
* Python
* TensorFlow
* Convolutional Neural Network (CNN)

### How to Build The Model
These are the steps to build the model :
1. Prepare the dataset
2. Preprocess the dataset
3. Build the layers
4. Train the model
5. Evaluate the model
6. Make predictions on the testing set
7. Save the model

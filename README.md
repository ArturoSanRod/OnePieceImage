# Instituto Tecnológico y de Estudios Superiores de Monterrey Campus Querétaro
## Arturo Sánchez Rodríguez | A01275427



# OnePiece character image classification 

## Table of Contents


1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Libraries that were Used](#libraries-that-were-used)
4. [Dataset Path](#dataset-path)
5. [Image Loading Function](#image-loading-function)
6. [Verifying the Dataset](#verifying-the-dataset)
7. [Train-Test Split](#train-test-split)
8. [Image Scaling](#image-scaling)
9. [Applying and Verifying the Scaling Process](#applying-and-verifying-the-scaling-process)
10. [Saving the Processed Dataset](#saving-the-processed-dataset)
11. [Dataset Visualization](#dataset-visualization)
12. [Second Preview | 31 / 05 / 2026](#second-preview--31--05--2026)
13. [Library](#library)
14. [Define how many Classes there are in our Dataset](#define-how-many-classes-there-are-in-our-dataset)
15. [We Created the Traditional MLP Model](#we-created-the-traditional-mlp-multi-layer-perceptron-neural-network-model)
16. [Then we Compile the Model](#then-we-compile-the-model)
17. [Training the Model](#training-the-model)
18. [Evaluating the Test Set](#evaluating-the-test-set)
19. [Model 1 MLP Results](#model-1-mlp-results)
20. [Model 2 CNN](#model-2-cnn)
21. [Model 2 Results](#model-2-results)
22. [Very Deep Convolutional Networks for Large-Scale Image Recognition](#very-deep-convolutional-networks-for-large-scale-image-recognition)
23. [Model 3 CNN + Data Augmentation](#model-3-cnn--data-augmentation)
24. [Building Model 3](#building-model-3-cnn--data-augmentation)
25. [Compiling and Training Model 3](#compiling-and-training-model-3)
26. [Evaluating Model 3](#evaluating-the-model-with-data-augmentation-with-the-test-set)
27. [Compare the Accuracy in all Three Models](#compare-the-accuracy-in-all-three-models)
28. [Model Comparison](#model-comparison)
29. [Selected Paper](#selected-paper)
30. [Result Comparison](#result-comparison)
31. [Final Model](#final-model)
32. [Data Augmentation](#data-augmentation)
33. [Defining the Model](#defining-the-model)
34. [Compile the Model](#compile-the-model)
35. [Checkpoint Callback](#checkpoint-callback)
36. [Training the Model](#training-the-model)
37. [Evaluating Test Set](#evaluating-test-set)
38. [Evaluation Metrics](#evaluation-metrics)
39. [Training Curves](#training-curves)
40. [Confusion Matrix](#confusion-matrix)
41. [Saving the Model](#saving-the-model)
42. [Image Classification Query System](#image-classification-query-system)
43. [References](#references)

## Introduction
In this project we have an objective to prepare a dataset full of images of OnePiece characters for a future model image classification. The means to this project is to transform the original images into numerical data that can be used by a Machine Learning model or Deep Learning (both are good with dealing with images).
In this stage of the project we are working primarly with the selection of the dataset, image loading, train and test splitting, and the necessary preprocessing steps needed before building a classification model.

## Dataset
The dataset that we used in this project contains images of One Piece characters that are organized into folders and each folder represents a different class, which means that its a different character.
### The general structure of the dataset is as follows:
- OnePiece/Data/Data
-   Ace/
-   Akainu/
-   Brook/
-   Chopper/
-   Crocodile/
-   Franky/
-   Jinbei/
-   Kurohige/
-   Law/
-   Luffy/
-   Mihawk/
-   Nami/
-   Rayleigh/
-   Robin/
-   Sanji/
-   Shanks/
-   Usopp/
-   Zoro/

<br>The dataset contains a total of 11,737 images distributed across 18 classes (characters).
This dataset was selected because it contains a large number of labeled images that belong to different classes, and the objective of this project is to classify images of different characters organized into separate folders makes it possible to train a supervised learning model.
<br>You can download the Dataset here: [One Piece Image Classifier](https://www.kaggle.com/datasets/ibrahimserouis99/one-piece-image-classifier).

## Libraries that were used
```
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
```

- We used matplotlib to visualize images from the dataset, since we are working with images it is important to display sample images and verify that they are loading and processing correctly.
- We use numpy to store and manipulate the image data as a multidimensional arrays, since Machine Learning cannot work with image files like the human being we have to convert it into numerical arrays before we process it.
- The OS library was primarily used to navigate through the dataset folders and so we can access the image files.
- TensorFlow was used to load images, resize them and convert them into numerical arrays, TensorFlow provides us the tool that will be used later to build and train the image classification model.
- ImageDataGenerator is commonly used for image preprocessing and data augemntation that will be used in a future stage.
- ```from sklearn.model_selection import train_test_split``` was used to divide the dataset into training and testing sets, it allows us to help the model learn from one portion of the dataset and the test the training with unseen data.

## Dataset Path

```
onePiece_data = "OnePiece/Data/Data" 
print(os.listdir("OnePiece/Data/Data/")[::]) 
```
This path allows the program to locate the dataset and access all characters folders.
Before loading the images, I used ```os``` to verify the dataset structure, we did it to confirm that all the classes were available and that the dataset path was correct.


## Image Loading Function

```
def cargar_img(path, img_size=(64, 64)):
    images = []
    labels = []

    class_names = sorted([
        clase for clase in os.listdir(path)
        if os.path.isdir(os.path.join(path, clase))
    ])

    for label_index, class_name in enumerate(class_names):
        class_path = os.path.join(path, class_name)
        for file_name in os.listdir(class_path):
            if file_name.lower().endswith((".png", ".jpg", "jpeg")):
                img_path = os.path.join(class_path, file_name)
                img = tf.keras.utils.load_img(img_path, target_size=img_size)
                img_array = tf.keras.utils.img_to_array(img)
                images.append(img_array)
                labels.append(label_index)
    images = np.array(images)
    labels = np.array(labels)

    return images, labels, class_names
```

Since the One Piece dataset is stored as image files inside the folders, it cannot be loaded directly as an image, the computer can only see numbers so we have to change that. 
<br>The function receives two parameters, the location of the dataset and the size that every image will be resized we used (64x64), converting them into numerical arrays and assigning a numercial label to each character class.
<br>The 64x64 pixels was selected to make sure that every image have the same dimensions while keeping memory usage and time manageable, this preprocessing step prepares the dataset for later stages.

## Verifying the Dataset

```
print(images.shape)
print(labels.shape)
print(class_names)
print(len(class_names))
```
We verified that the dataset was loaded correctly and it displays the dimensions of the image dataset ```(11737, 64, 64, 3)``` are the images that were loaded, each image has a height of 64 pixels and a width of 64 pixels, and finally each image contains 3 RGB color channels. Then we confirm that there is one label for every image in the dataset. This displays the names of all the character that were detected in the dataset and it confirms that the dataset has 18 different characters.
<br>These verifications seem unecesary but are important because they confirm that all of the images were loaded successfully and organized before moving on to the train-test split and preprocessing stages.

## Train-Test Split

```
train_img, test_img, train_labels, test_labels = train_test_split(
    images,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

print("Train imágenes:", train_img.shape)
print("Test imágenes:", test_img.shape)
print("Train labels:", train_labels.shape)
print("Test labels:", test_labels.shape)
```

Once all of the images and labels were loaded, the dataset was divided into training and testing sets, the purpose of this step is so the model can learn from one portion of the dataset and then be tested on data that was never been seen before.
<br>If a model is trained and tested with the same information it is considered cheating because it memorizes the data instead of learning patterns, that is why we seperate train and test sets, we obtain more realistic measurments of the models performance. The parameters specifies that 20% of the dataset will be used for testing and the remaining 80% will be used for training.
<br>Parameters Used: Images that contain all image data that were loaded from the dataser, labels that contain the numerical labels associated with each image.


## Image Scaling

```
def scale_img(train_img, test_img):
    train_img = train_img / 255.
    test_img = test_img / 255.
    return train_img, test_img
```
After we split the dataset into train and test we have to scale the images, to normalize the pixel values of the images. We have to do this due to the fact that every pixel is represented by an value between o and 255.
Machine Learning models generally perform better whe the inputs are small and consistent and having large values can make the training process slower and unstable. After scaling, all of the pixels are transformed into a range of 0-1 so having large values wont affect the model.
<br>We scale train and test because the model has to receive data in the same format during both train and test. The function also returns the normalized train and test dataset so they can be used in later stages of the project.


## Applying and Verifying the Scaling Process

```
train_img, test_img = scale_img(train_img, test_img)

print("train_img.shape:", train_img.shape)
print("test_img.shape:", test_img.shape)
print("train_img.min():", train_img.min(), "train_img.max():", train_img.max())
print("test_img.min():", test_img.min(), "test_img.max():", test_img.max())
```

This function replaces the original dataset with the normalized versions, the returned dataset now has pixel values between 0-1 instead of the original 0-255. 
<br>Then we verify that the dimensions of the dataset stayed unchanged after scaling.
```
train_img.shape: (9389, 64, 64, 3)
test_img.shape: (2348, 64, 64, 3)
```
This tells us that the train set has 9,389 images, the test set has 2,348 images and both of them still have dimensions of 64x64 pixels and each image stilla contains 3 RGB color channels.

```
train_img.min(): 0.0 train_img.max(): 1.0
test_img.min(): 0.0 test_img.max(): 1.0
```
This output confirms that all of the pixels are now in the range of 0-1 pixels.


## Saving thr Processed Dataset

```
np.save("train_img.npy", train_img)
np.save("test_img.npy", test_img)

np.save("train_labels.npy", train_labels)
np.save("test_labels.npy", test_labels)
```

After having to load, split and scale the dataset, the processed data was saved with NumPy´s ```.npy``` format. We saved the dataset for multiple reasons, one of the being that loading and preprocessing more than 11,000 images requires time and resources.
If the notebook is restarted all of the variables stored in memory are lost, instead the processed data can be loaded directly in future stages of the project. 
We used .npy because its a binary file format very good for storing multidimensional arrays.


## Dataset Visualization

```
import random

plt.figure(figsize=(10,10))

for i in range(9):
    idx = random.randint(0, len(train_img)-1)

    plt.subplot(3,3,i+1)
    plt.imshow(train_img[idx])
    plt.title(class_names[train_labels[idx]])
    plt.axis("off")

plt.show()
```

Here we display images not only to see them but to verify that the dataset was loaded correctly before training the model.
We randomly select images from the training set so we can confirm that the images were loaded, the resizing process did not corrupt the images, the labels correspond to the correct characters and that differents are represented in the dataset.
<br>This verification step helps us detect possible problems in the dataset before moving to the training phase.

# Second Preview | 31 / 05 / 2026


## Library

```
from tensorflow.keras import models, layers
```
We implemented these new tools because we are not going to be loading images, now we are going to be constructing the model, the model allows us to create a CNN (Convolutional Neural Network) with Sequential.
- ```Layers``` allows us to add layers like: Conv2D, MaxPooling2D, Flatten, Dense.


## Define how many layers there are in our dataset
```
num_classes = len(class_names)
print("Número de clases:", num_classes)
print("Clases:", class_names)
```
Since there are 18 classes in our dataset then we have to have 18 neurons in the last layer, one neuron for each character.


## We created the traditional MLP (Multi-layer Perceptron) neural network model

````
def get_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Flatten(input_shape=input_shape),
        layers.Dense(128, activation="relu"),
        layers.Dense(num_classes, activation="softmax")
    ])
    return model

model = get_model(input_shape=(64, 64, 3), num_classes=num_classes)
````

Here we defined a function to create the model that takes the ```input_shape``` to know the size of the image when we loaded it with the img_size(64x64), then it also takes the ```num_classes``` because it can adapt to when the dataset increases or decreases in class numbers. 
<br>We start constructing the neural netowrk layer by layer with ```models.Sequential``` then with ```layers.flatten(input_shape=input_shape)``` we transform the structur of an image (64 x 64 x 3) and make it into a one dimension vector that contains `12,288`values, this is necesary because the Dense layers can only process vectors in one dimensiolnal structure. 
<br>Now going on to the `layers.Dense(128, activation="relu"), it basically means a "fully connected layer" that receives the previous value that flatten generated, its usually something like this `[0.34, 0.12, 0.89, 0.01...]` and Dense(128) creates 128 neurons (thats our hiperparameter that we can change to train the model) and each neuron receibes 12,288 values of input. 
<br>We use `relu` because if the result of the operation is a negative it "transforms" is to a 0 and if the result is positive it maintains the results outcome.
<br>Finally going with one of the most important layers of our model `layers.Dense(num_classes, activation="softmax")`, this layer is the one that gives us the final prediction, `softmax` turns the neurons results in probabilities for each class and the model then selects the class with higher probability of showing.


## Then we compile the Model 
````
def compile_model(model):
    model.compile(
        optimizer="adam", #Adam es el algoritmo que modifica los pesos y los ajusta automáticamente
        loss="sparse_categorical_crossentropy", #Función de pérdida para clasificación multiclase
        metrics=["accuracy"] #Métrica para evaluar el rendimiento del modelo
    )

compile_model(model)
````
We use "Adam" that tells the model how it will learn, its an algorithm that modifies the weights, when the model responds with an incorrect answer Adama analizes the mistake ans modifies the weights so it does not make the same mistake next time.
<br>```loss="sparse_categorical_crossentropy"``` tells the model "How will you know you were wrong?", we opted in using this one because our labels are like follows [0, 1, 2, 3, 4, 5, 6, ... 17], they are whole numbers.
<br>```metrics=["accuracy"]``` is the metric we want to show to know how accurate the model was.


## Training the model 
````
def train_model(model, train_img, train_labels):
    history = model.fit(train_img, train_labels, validation_split=0.2, epochs=10)
    return history

history = train_model(model, train_img, train_labels)
````
After creating and compiling the model, our next step is to train it with the train set we previously splitted.
The model receives three important parameters, the model with the nerual network we previously created and compiled, `train_img` that are the training images and finally the `train_labels` that are the correct labels associated with each image.
<br>The training process is performed usinf `.fit()`, it basically teaches the model how to recognize patterns in the images. Here in training is where the model analyzes each image, compares its prediction against the correct label and calculates the error with the selected loss function and updates the internal weight with the Adam optimizer. This is not a one time thing, it repeats multiple times so the model can gradually improve its predictions. 
<br>With `validation_split=0.2` we "save" a 20% of the training dataset for validation. So the training set trains with 80% of the available train images and the remaining 20% are used to evaluate how well the model reacts to unseen data during training, this also helps to detect problems like `overfitting`.
<br>And finally with the variable `History` is where we store information that was generated during training phase (accuracy, loss, validation loss, etc).



## Evaluating the Test set
````
test_loss, test_accuracy = model.evaluate(test_img, test_labels)

print("Test loss:", test_loss)
print("Test accuracy:", test_accuracy)
````
These were the results after training the first model.

74/74 ━━━━━━━━━━━━━━━━━━━━ 0s 713us/step - accuracy: 0.0643 - loss: 2.8846
Test loss: 2.8845694065093994
Test accuracy: 0.06431005150079727



## Model 1 MLP Results
Our model after being trained for 10 epochs obtined a test accuracy of aproximately 6.43% and a test loss of 2.88. Even though the loss decreased during training we can see that the final accuracy remained low, ands this tells us that the model was not able to learn enough patterns from the images. This will be our baseline for future comparisons.
<br>This model was adapted from the TensorFlow MNIST example provided by the professor. The original example was designed for handwritten digit classification, but it was modified to work with the One Piece character dataset and its 18 classes.



# Model 2 CNN

For our second model we opted to use a CNN (Convolutional Neural Network). Not like the MLP model, CNNs are specifically designed for image processing because they can learn visual patterns such as edges, textures and shapes directly from the images.

````
def get_modelo_CNN(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(6, kernel_size=3, padding="same", activation="relu", input_shape=input_shape),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(num_classes, activation="softmax")
    ])
    return model 

model_cnn = get_modelo_CNN(input_shape=(64, 64, 3), num_classes=num_classes)
model_cnn.summary()
````

To explain some of the parts of this code we begin with `layers.Conv2D(6, kernel_size=3, padding="same", activation="relu")`, this layer applies 6 filters to the images and these filters help the model identify patterns like edges, colors or even textures that will help to identify different One Piece characters.
<br>`kernel_size=3` tells us that the model will analyze the images using smaleer 3x3 windows/spaces. `padding="same"` will keep the image size unchanged after applying previous filters, `activation="relu"` we also used relu here because its a very useful tool to help the model learn more complex patterns, `layers.Flatten()` are going to transform the information into a one dimension vector so the computer can be processed.
<br>Finally we use `layers.Dense(128, activation="relu")` that contains 128 neurons  so the model makes a better prediction. Since we have 18 characters in the dataset we have to generate probabilities and selet the class with the highest probability for the final prediction and for that we used `layers.Dense(num_classes, activation="softmax")`.


## We compiled and trained the CNN model

````
compile_model(model_cnn)
history_cnn = train_model(model_cnn, train_img, train_labels)
````
After we create de CNN model we compile it using the same function that we used for the MLP model. We dont have to rewrite the full `model.compile()` because the CNN uses the same configuration, reusing the same function we keep and maintain order in our code and not only that but we make the comparison between MLP and CNN more fair. 
The we train the CNN model with the same training function.
<br>This is very useful to compare both models and cerify if adding a convolutional layer (helps us find important patterns inside an image) improves the classification results.


## Evaluating the CNN model with the test set 
````
test_loss_cnn, test_accuracy_cnn = model_cnn.evaluate(test_img, test_labels)
print("Test loss CNN:", test_loss_cnn)
print("Test accuracy CNN:", test_accuracy_cnn)
````

The goal is to compare these results with the first MLP model that we did, with this we can see if using a convolutional layer was the right thing to do.

## Model 2 Results
The CNN model got better results than the MLP model with a test accuracy of approximately 22.87%, which represents an improvement compared to the MLP model that only got 6.43%. This tells us that the convolutional layer was able to learn useful visual patterns from the images before classification, when the model still has room for improvement, the results confirm that CNNs are what we can say "better" for image classification tasks. During training of the model it reached almost 100% training accuracy, while the validation accuracy stayed around at 25%, this also tells us that the model may be experiencing overfitting, this can be interpreted as the machine may be learning (or memorizing) the training images extremely well but is having dificulty generalizing to new images. 


## Very Deep Convolutional Networks for Large-Scale Image Recognition 
Model 2 CNN was inspired/compared by the VGG architecture proposed by Simonyan and Zisserman, where it tells us where convolutional nerual networks with small 3x3 filters are used for image classification, this is why we had to add Conv2D layer with `kernel_size=3` before we flatten the image. 



# Model 3 CNN + Data Augmentation
The Data Augmentation strategy used in Model 3 was inspired by the work of Mikołajczyk and Grochowski (2018).

Before creating this model a good way to go was implementing data augmentation, this technique is to generate slightly modified versions of the training images that are used in the train process.
<br>The main reason for implementing Data Augmentation was because the Model 2 shows signs of overfitting.
For the third model we reused the CNN architecture from Model 2 and added the data augmentation layer befor the convolutional layer.

````
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
])
````
`RandomFlip("horizontal")` has the purpose to flip some of the images horizontally, this allows the model to see different versions of the same character and learn more visual patterns. `RandomRotation(0.1)` randomly rotate images by a small amount, `RandomZoom(0.1)`, these transformations are applied only during the training phase, the objective of this step is to expose our model to more image variations and reducing overfitting so it learns more visual patterns.

## Building Model 3 (CNN + Data Augmentation)

````
def get_modelo_CNN_aug(input_shape, num_classes):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        data_augmentation,
        layers.Conv2D(6, kernel_size=3, padding="same", activation="relu"),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(num_classes, activation="softmax")
    ])
    return model 

model_cnn_aug = get_modelo_CNN_aug(input_shape=(64, 64, 3), num_classes=num_classes)
model_cnn_aug.summary()
````
We have to define the shape of the images that will enter the network, each image has dimensions of 64x64 pixels with 3 RGB channels.
<br>The main difference compared to model 2 is the addition of the data augmentation block, before the images reach the convolutional layer Tensorflow randomly applies transformations to some of the images, this helps the model look at new images during training.
<br>After the augmentation step the CNN keeps the same architecture as the CNN model. 


## Compiling and training Model 3

````
compile_model(model_cnn_aug)
history_cnn_aug = train_model(model_cnn_aug, train_img, train_labels)
````

We are reusing the same compilations configurations as the previous models.


## Evaluating the model with Data Augmentation with the test set
````
test_loss_cnn_aug, test_accuracy_cnn_aug = model_cnn_aug.evaluate(test_img, test_labels)
print("Test loss CNN con data augmentation:", test_loss_cnn_aug)
print("Test accuracy CNN con data augmentation:", test_accuracy_cnn_aug)
````

These images were never used during training so it makes it reliable way to measure how well the model performs with new images.
- Test loss: tells us how far the models predictions are from the correct labels.
- Test Accuracy: measures the percentage of images that were correct.

<br>Compared to Model 2 that has a test accuracy of 22.87%, using Data Augmentation improves the classifications answer. This tells us that exposing the model to new images helps it learn more visual patterns. 


## Compare the Accuracy in all three models 

````
#MLP
print(history.history["accuracy"][-1])
print(history.history["val_accuracy"][-1])

#CNN
print(history_cnn.history["accuracy"][-1])
print(history_cnn.history["val_accuracy"][-1])

#CNN con data augmentation
print(history_cnn_aug.history["accuracy"][-1])
print(history_cnn_aug.history["val_accuracy"][-1])
````

With this we can analyze the final training and validation accuracy of each model.


## Model Comparison

| Model | Train Accuracy | Validation Accuracy | Test Accuracy |
|--------|---------------:|--------------------:|--------------:|
| Model 1 - MLP | 8.00% | 7.03% | 7.58% |
| Model 2 - CNN | 100.00% | 23.06% | 22.87% |
| Model 3 - CNN + Data Augmentation | 36.35% | 29.66% | 29.55% |

<br> The results show that each model improved upon the previous version.
<br> MLP achieved a very low performance because it was unable to effectively capture the images visual patterns.
<br> The CNN model improves  the classification performance by using convolutional filters that can analyze better image features, but the bad thing is that it shows signs of overfitting.
<br> Finally the CNN with augmentation achieved the best validation and testa accuracy.


# Selected Paper
The selected paper evaluates the performance of different deep learning architectures for image classification jut like our case og image classification with 18 classes.
This paper was selected because it follows a very strategic methodology and it is very simmilar to the one used in this project:
- Image classification
- Convolutional Neural Networks
- Accuracy 
- Recall
- F1-score
- Confusión matrix
Even though the datasets are different, this paper gives us a very useful reference of comparison and evaluating and improvising CNN-based image classification models.

## Result comparison
- Accuracy: 61.88%
- Precision: 63.86%
- Recall: 61.88%
- F1 Score: 61.86%
<br>Even though the performance is lower than some results reported, these values are considered reasonable considering

- Different dataset
- Different image resolution
- Different number of classes
- Simple architecture
- No transfer learning used (yet)

<br>The paper reports that deeper CNN architectures generally have better outcomes and obviously better performance.
<br>We had an increase of of good results in this project and are still working on getting even better results.

- Alem, A., & Kumar, S. (2022). Deep Learning Models Performance Evaluations for Remote Sensed Image Classification. IEEE Access, 10, 111784–111793. https://doi.org/10.1109/ACCESS.2022.3215264



# Final Model

## Data Augmentation
````
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.15),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomContrast(0.1),
    layers.RandomFlip("vertical")
])
````
To improve generalization and reduce overfitting we applied more hiperparameters during training like:
- Horizontal Flio
- Vertical Flip
- Random Rotation
- Random Zoom
- Random Translation
- Random Contrast

## Defining the model
````
def get_model_CNN_V2(input_shape, num_classes):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        data_augmentation,

        layers.Conv2D(32, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),

        layers.Conv2D(128, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),
        
        layers.Conv2D(256, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),

        layers.Flatten(),

        layers.Dense(128, activation="relu"),

        layers.Dense(num_classes, activation="softmax")
    ])

    return model
model_cnn_v2 = get_model_CNN_V2(input_shape=(64, 64, 3), num_classes=num_classes)
model_cnn_v2.summary()
````
This "final" model was created as an improvment of the previous CNN architecture, with this one we increased the models ability tp exytact visual features from our dataset.
`data_augmentation` is defined before entering the convolutional layer because the images are randomly transformed through data augmentation (helps reduce overfitting).
<br>First convolutional layer `layers.Conv2D(32, 3, activation="relu", padding="same"),
layers.MaxPooling2D()`applies 32 filters of size 3 by 3 so it helps it learn visual patterns and it continues from 32, then 54, then 128 and finally 256 filters and with this one it learns even more complex visual representations.
<br>`layers.Flatten()` after the convolutional layers the features are STILL stored as multidimensional features, the flatten layer turns these features into a one-dimensional vector that can be processed by the connected layers.
<br>`layers.Dense(128, activation="relu")` with this layer we combine all of the previous features and learns from the connections between them before it makes the final prediction.
<br>`layers.Dense(num_classes, activation="softmax")` Here we have 1 neuron for each class, and since we have 18 classes the layer produces 18 outputs, with "softmax" makes the outputs into probabilities, the character with the highest probability will be the final prediction.
<br> 

## Compile the Model
[Compile the Model](#then-we-compile-the-model)


## Checkpoint Callback
After we defined the model we implemented a `callback` to automatically save the best version of the model during training.

```
from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint_best = ModelCheckpoint(
    filepath="best_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=0
)
```
Basically here we save the best models accuracy during training and store it `filepath="best_model.keras"`, with this we can later try the model and make predictions with the best part of the model and when a better model is found during training it´s automatically updated. 
<br>We opted to `monitor="val_accuracy"` because it measures the model´s performance on the data that was not used for learning during that specific epoch.


## Training the Model
````
def train_model(model, train_img, train_labels):
    history = model.fit(
        train_img,
        train_labels,
        validation_split=0.2,
        epochs=25,
        batch_size=64,
        callbacks=[checkpoint_best]
    )
    return history

history = train_model(model_cnn_v2, train_img, train_labels)
````
This training function works like previous versions, the main thing to look at here is the `callbacks=[checkpoint_best]`, during the training this function monitored "val_accuracy" during every epoch and automatically saved the best performing model.


## Evaluating Test set

````
#Evaluamos el modelo con el set de prueba
test_loss, test_accuracy = model_cnn_v2.evaluate(test_img, test_labels)

print("Test loss:", test_loss)
print("Test accuracy:", test_accuracy)
````
74/74 ━━━━━━━━━━━━━━━━━━━━ 1s 8ms/step - accuracy: 0.6188 - loss: 1.3175
Test loss: 1.3174875974655151
Test accuracy: 0.6188245415687561

<br> Here we test the model with totally new images, images that were not seen during the training phase to see how it will react, and looking at the result it had a "good" result.


## Evaluation Metrics
```
y_pred_probs = model_cnn_v2.predict(test_img)
y_pred_classes = np.argmax(y_pred_probs, axis=1)
y_true = test_labels
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score)

accuracy = accuracy_score(test_labels, y_pred_classes)
precision = precision_score(test_labels, y_pred_classes, average="weighted")
recall = recall_score(test_labels, y_pred_classes, average="weighted")
f1 = f1_score(test_labels, y_pred_classes, average="weighted")

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)
```

Although accuracy is one of the most common metric, it does not always provide a complete picture of how the model is performing. To obtain a more valid and detailed evaluation we added more metrics:
- Accuraacy
- Precision
- Recall
- F1-Score

The model first generates a probability distribution for every image. the network provides 18 outputs data corresponds to the 18 classes in the dataset and the class that has the highest probability is chosen as the final prediction. 
<br>
```
Ace     → 0.92
Luffy   → 0.03
Zoro    → 0.02
```
The models prediction would be "Ace" because it has the highest probability.

- Accuracy tells us the percentage of the correctly classified images.
- Precision measures how many of the model´s positive predictions are actualy correct.
- Recall calculates how many of the actual class instances were correct.
- F1-Score is the combination of Precision and Recall into on powerful metric, it´s more useful when we have a multiclass classification.
<br>

We selected these metrics because we can relate them with the paper we chose, the same evaluation strategy was implemented to compare and ensure every model was understood.

## Training Curves
````
frame = pd.DataFrame(history.history)

train_acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]

train_loss = history.history["loss"]
val_loss = history.history["val_loss"]

plt.figure(figsize=(12,4))

# Accuracy
plt.subplot(1,2,1)
plt.plot(train_acc, label="Training", linewidth=2)
plt.plot(val_acc, label="Validation", linewidth=2)
plt.title("Accuracy vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.grid(True, alpha=0.3)
plt.legend()

# Loss
plt.subplot(1,2,2)
plt.plot(train_loss, label="Training", linewidth=2)
plt.plot(val_loss, label="Validation", linewidth=2)
plt.title("Loss vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.grid(True, alpha=0.3)
plt.legend()
````

To make sure we visualize and understand the behaviour of the model during the training process we made these graphics. 
- Accuracy vs Epochs
<br>Shows us how the models accuracy evolved during training, the training accuracy represents the performance on the images used for learning. The objective is for both curves to increase and stay close to each other, when the gap is big it usualy means there is overfitting involved.

- Loss vs Epochs
<br>Here it shows the prediction error during training, a good training process usualy shows a decrease in training loss, decrease of validation loss if validation loss starts to increase during training while training loss decreases it means overfitting may be ocurring.


## Confusion Matrix

````
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(test_labels, y_pred_classes)
fig , ax = plt.subplots(figsize=(8,8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(ax=ax, cmap="Blues", xticks_rotation=90, colorbar=True)
plt.title("Matriz de Confusión")
plt.tight_layout()
plt.show()
````
With this Confusion Matrix we can see that the main diagonal representes the correctly classified images, if it has a strong diagonal it tells us that the model learned to distinguish between the different One Piece characters and cells outside the diagonal represente classification errors.
<br>This is important because while having accuracy it´s only a single performance value and with the confusion matrix it provides us a detailed view of model behaviour and allows us to answer several important questions like:

- Which classes are classified correctly most often?
- Which classes cause confusion?
- Are there specific characters that are harder to recognize?
<br> This is what makes it very valuable. 


## Saving the Model
````
model_cnn_v2.save("onepiece_model.keras")
print("Modelo guardado correctamente!")


"""
loaded_model = tf.keras.models.load_model(
    "onepiece_model.keras"
)
"""
````

Here is where we save the model, we save it because training a CNN can take time depending on the dataset, with this we save time and only has to be performed once.


## Image Classification Query System

````
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Cargar el modelo guardado
model_query = tf.keras.models.load_model("best_model.keras")

# Función para preparar una imagen nueva
def cargar_imagen_query(image_path, img_size=(64, 64)):
    img = tf.keras.utils.load_img(image_path, target_size=img_size)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img, img_array

# Función para predecir una imagen
def predecir_imagen(image_path):
    img, img_array = cargar_imagen_query(image_path)

    predictions = model_query.predict(img_array)

    predicted_index = np.argmax(predictions[0])
    confidence = np.max(predictions[0])

    predicted_class = class_names[predicted_index]

    print("Predicción:", predicted_class)
    print("Confianza:", confidence)

    plt.imshow(img)
    plt.title(f"Predicción: {predicted_class} | Confianza: {confidence:.2f}")
    plt.axis("off")
    plt.show()

    return predicted_class, confidence

image_path = "/Users/arturosr/Desktop/OnePiece/OnePieceImage/Imagenes/OnePieceImagnesPrueba/05.png"

predecir_imagen(image_path)
````

After everything that was done the only thing left to do is to try the model, we load images outside the dataset and see how it performs. `model_query = tf.keras.models.load_model(
    "best_model.keras"
)` we load the best model that we saved so we dont have to retrain the whole CNN everytime we need to classify an image. 
<br>We have to process the images because images that are not from the dataset may have different dimensions, thats why we use: ` def cargar_imagen_query(image_path, img_size=(64, 64)):`, we have to prepare the images so the CNN can process them correctly.
<br>Then we load the new image, process them and generate a prediction with our CNN, it then displays the image with the prediction of the class and the "trust" it has on that prediction.






# References
- Alem, A., & Kumar, S. (2022). Deep Learning Models Performance Evaluations for Remote Sensed Image Classification. IEEE Access, 10, 111784–111793. https://doi.org/10.1109/ACCESS.2022.3215264
- Simonyan, K., & Zisserman, A. (2015).
Very Deep Convolutional Networks for Large-Scale Image Recognition.
International Conference on Learning Representations (ICLR).
- Mikołajczyk, A., & Grochowski, M. (2018).
Data augmentation for improving deep learning in image classification problem.
2018 International Interdisciplinary PhD Workshop (IIPhDW).
IEEE.


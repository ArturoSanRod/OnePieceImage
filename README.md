# Instituto Tecnológico y de Estudios Superiores de Monterrey Campus Querétaro
## Arturo Sánchez Rodríguez | A01275427



# OnePiece character image classification 

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

<br>The dataset contains a total of 11,737 images distributed across 18 clases (characters).
This dataset was selected because it contains a large number of labeled images that belong to different classes, and the objective of this project is to classify images of different characters organized into separate folders makes it possible to train a supervised learning model.


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
- TensroFlow was used to load images, resize them and convert them into numerical arrays, TensorFlow provides us the tool that will be used later to build and train the image classification model.
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
Since there are 18 clases in our dataset then we have to have 18 neurons in the last layer, one neuron for each character.


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


## Evualating the CNN model with the test set 
````
test_loss_cnn, test_accuracy_cnn = model_cnn.evaluate(test_img, test_labels)
print("Test loss CNN:", test_loss_cnn)
print("Test accuracy CNN:", test_accuracy_cnn)
````

The goal is to compare these results with the first MLP model that we did, with this we can see if using a convolutional layer was the right thing to do.

## Model 2 Results
The CNN model got better results than the MLP model with a test accuracy of approximately 25.76%, which represents an improvement compared to the MLP model that only got 6.43%. This tells us that the convolutional layer was able to learn useful visual patterns from the images before classification, when the model still has room for improvement, the results confirm that CNNs are what we can say "better" for image classification tasks. During training of the model it reached almost 100% training accuracy, while the validation accuracy stayed around at 25%, this also tells us that the model may be experiencing overfitting, this can be interpreted as the machine may be learning (or memorizing) the training images extremely well but is having dificulty generalizing to new images. 


## Very Deep Convolutional Networks for Large-Scale Image Recognition 
Model 2 CNN was inspired/compared by the VGG architecture proposed by Simonyan and Zisserman, where it tells us where convolutional nerual networks with small 3x3 filters are used for image classification, this is why we had to add Conv2D layer with `kernel_size=3` before we flatten the image. 

## References

- Simonyan, K., & Zisserman, A. (2015).
Very Deep Convolutional Networks for Large-Scale Image Recognition.
International Conference on Learning Representations (ICLR).








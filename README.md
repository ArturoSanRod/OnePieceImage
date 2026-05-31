# Instituto Tecnológico y de Estuidos Superiores de Monterrey Campus Querétaro
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
The dataset contains a total of 11,737 images distributed across 18 clases (characters).
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
plt.imshow(train_img[0])
plt.title(class_names[train_labels[0]])
plt.axis("off")
plt.show()
```

We went ahead to verify the images were loaded and processed correctly by displying a sample from the train set. This helps us to confirm that all the images were loaded successfully, the labels were assigned correctly and the preprocessing did not corrupt the image data.


































![](https://i.imgur.com/iywjz8s.png)


# Day 2, Afternoon, Collaborative Document 'Introduction to deep learning'
28th & 29th of January

Welcome to The Workshop Collaborative Document.

This Document is synchronized as you type, so that everyone viewing this page sees the same text. This allows you to collaborate seamlessly on documents.

----------------------------------------------------------------------------


This is the document for this afternoon (day 2): https://edu.nl/gf86m

This is the document for this morning (day 2): https://edu.nl/bym3j

Collaborative Document day 1: https://edu.nl/uuayc

##  🫱🏽‍🫲🏻 Code of Conduct

Participants are expected to follow these guidelines:
* Use welcoming and inclusive language.
* Be respectful of different viewpoints and experiences.
* Gracefully accept constructive criticism.
* Focus on what is best for the community.
* Show courtesy and respect towards other community members.
 
## ⚖️ License

All content is publicly available under the Creative Commons Attribution License: [creativecommons.org/licenses/by/4.0/](https://creativecommons.org/licenses/by/4.0/).

## 🙋Getting help

To ask a question, just raise your hand.

If you need help from a helper, place a pink post-it note on your laptop lid. A helper will come to assist you as soon as possible.

## 🖥 Workshop website & data
https://esciencecenter-digital-skills.github.io/2025-01-28-ds-dl-intro/

🛠 Setup

https://esciencecenter-digital-skills.github.io/2025-01-28-ds-dl-intro/#setup

Download files

[Weather dataset](https://zenodo.org/record/5071376/files/weather_prediction_dataset_light.csv?download=1)
[Dollar street dataset (4 files in total)](https://zenodo.org/api/records/10970014/files-archive)

## 👩‍🏫👩‍💻🎓 Instructors
Sven van der Burg, Carsten Schnober

## 🧑‍🙋 Helpers
Maurice de Kleijn, Laura Ootes

## 🗓️ Agenda
09:30	Welcome and recap
09:45	Monitor the training processs
10:30	Break
10:40	Monitor the training process
11:30	Break
11:40	Advanced layer types
12:30	Lunch Break
13:30	Advanced layer types
14:30	Break
14:40	Transfer learning
15:30	Break
15:40	Outlook
16:15	Post-workshop Survey
16:30	Drinks


## 🏢 Location logistics
* Coffee and toilets are in the hallway, just outside of the classroom.
* If you leave the building, 
  be sure to be accompanied by someone from the escience center to let you back in through the groundfloor door
* For access to this floor you might need to ring the doorbell so someone can let you in
* In case of an emergency, you can exit our floor using the main staircase.
  Or follow green light signs at the ceiling to the emergency staircase.
* **Wifi**: Eduroam should work. Otherwise use the 'matrixbuilding' network, password should be printed out and available somewhere in the room.

## 🎓 Certificate of attendance
If you attend the full workshop you can request a certificate of attendance by emailing to training@esciencecenter.nl.
Please request your certificate within 8 months after the workshop, as we will delete all personal identifyable information after this period.

## 🔧 Exercises

### Exercise 3.1 Number of features in Dollar Street 10

How many features does one image in the Dollar Street 10 dataset have?

- A. 64
- B. 4096
- C. 12288
- D. 878



**Answer:** The correct solution is C: 12288

There are 4096 pixels in one image (64 * 64), each pixel has 3 channels (RGB). So 4096 * 3 = 12288.

## Exercise 3.2 Number of parameters 
Suppose we create a single Dense (fully connected) layer with 100 hidden units that connect to the input pixels, how many parameters does this layer have?

- A. 1228800
- B. 1228900
- C. 100
- D. 12288


The correct answer is **B**: Each entry of the input dimensions, i.e. the shape of one single data point, is connected with 100 neurons of our hidden layer, and each of these neurons has a bias term associated to it. So we have 1228900 parameters to learn.

## Exercise 3.3 Number of model parameters
Suppose we apply a convolutional layer with 100 kernels of size 3 * 3 * 3 (the last dimension applies to the rgb channels) to our images of 32 * 32 * 3 pixels (mismatch with our data 64 * 64 * 3 pixels, but stick to 32). How many parameters do we have? Assume, for simplicity, that the kernels do not use bias terms. Compare this to the answer of the exercise 3.2.



## Exercise 3.4 Convolutional Neural Network

Inspect the network above:

1. What do you think is the function of the Flatten layer?
2. Which layer has the most parameters? Do you find this intuitive?
3. (optional) This dataset is similar to the often used CIFAR-10 dataset. We can get inspiration for neural network architectures that could work on our dataset here: https://paperswithcode.com/sota/image-classification-on-cifar-10 . Pick a model and try to understand how it works.




## Exercise 3.5 Network depth
What, do you think, will be the effect of adding a convolutional layer to your model? Will this model have more or fewer parameters?
Try it out. Create a `model` that has an additional `Conv2d` layer with 50 filters and another MaxPooling2D layer after the last MaxPooling2D layer. Train it for 10 epochs and plot the results.

**HINT**:
The model definition that we used previously needs to be adjusted as follows:
```python
inputs = keras.Input(shape=train_images.shape[1:])
x = keras.layers.Conv2D(50, (3, 3), activation='relu')(inputs)
x = keras.layers.MaxPooling2D((2, 2))(x)
x = keras.layers.Conv2D(50, (3, 3), activation='relu')(x)
x = keras.layers.MaxPooling2D((2, 2))(x)
# Add your extra layers here
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(50, activation='relu')(x)
outputs = keras.layers.Dense(10)(x)
```

```python
def create_nn_extra_layer():
    inputs = keras.Input(shape=train_images.shape[1:])
    x = keras.layers.Conv2D(50, (3, 3), activation='relu')(inputs)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Conv2D(50, (3, 3), activation='relu')(x)
    x = keras.layers.MaxPooling2D((2, 2))(x) #
    x = keras.layers.Conv2D(50, (3, 3), activation='relu')(x) # extra layer
    x = keras.layers.MaxPooling2D((2, 2))(x) # extra layer
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(50, activation='relu')(x)
    outputs = keras.layers.Dense(10)(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="dollar_street_model")
    return model

model = create_nn_extra_layer()
```
The number of parameters has decreased by adding this layer. We can see that the extra layers decrease the resolution from 14x14 to 6x6, as a result, the input of the Dense layer is smaller than in the previous network. To train the network and plot the results:

## Exercise 4.1
## Inspect the DenseNet121 network
Have a look at the network architecture with `model.summary()`.
It is indeed a deep network, so expect a long summary!

### 1.Trainable parameters
How many parameters are there? How many of them are trainable?

Why is this and how does it effect the time it takes to train the model?

### 2. Head and base
Can you see in the model summary which part is the base network and which part is the head network?

### 3. Max pooling
Which layer is added because we provided `pooling='max'` as argument for `DenseNet121()`?

 
**ANSWER**


1. Trainable parameters

Total number of parameters: 7093360, out of which only 53808 are trainable.

The 53808 trainable parameters are the weights of the head network. All other parameters are ‘frozen’ because we set base_model.trainable=False. Because only a small proportion of the parameters have to be updated at each training step, this will greatly speed up training time.
2. Head and base

The head network starts at the flatten layer, 5 layers before the final layer.
3. Max pooling

The max_pool layer right before the flatten layer is added because we provided pooling='max'.



## Exercise 4.2

## Training and evaluating the pre-trained model

### 1. Compile the model
Compile the model:
- Use the `adam` optimizer
- Use the `SparseCategoricalCrossentropy` loss with `from_logits=True`.
- Use 'accuracy' as a metric.

### 2. Train the model
Train the model on the training dataset:
- Use a batch size of 32
- Train for 30 epochs, but use an earlystopper with a patience of 5
- Pass the validation dataset as validation data so we can monitor performance on the validation data during training
- Store the result of training in a variable called `history`
- Training can take a while, it is a much larger model than what we have seen so far.

### 3. Inspect the results
Plot the training history and evaluate the trained model. What do you think of the results?

### 4. (Optional) Try out other pre-trained neural networks
Train and evaluate another pre-trained model from https://keras.io/api/applications/. How does it compare to DenseNet121?


Carsten's solution:
```python
model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
early_stopper = keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                              patience=5)
history = model.fit(x=train_images,
                    y=train_labels,
                    batch_size=32,
                    epochs=30,
                    callbacks=[early_stopper],
                    validation_data=(val_images, val_labels))
```

# Episode 6-outlook.Rmd 

## Exercise: A real-world deep learning application

We will have a look at [this notebook](https://github.com/matchms/ms2deepscore/blob/0.4.0/notebooks/MS2DeepScore_tutorial.ipynb).
It is part of the codebase for [this paper](https://doi.org/10.1186/s13321-021-00558-4).

In short, the deep learning problem is that of finding out how similar two molecules are in terms of their molecular properties,
based on their mass spectrum.
You can compare this to comparing two pictures of animals, and predicting how similar they are.

A Siamese neural network is used to solve the problem.
In a Siamese neural network you have two input vectors, let's say two images of animals or two mass spectra.
They pass through a base network. Instead of outputting a class or number with one or a few output neurons, the output layer
of the base network is a whole vector of for example 100 neurons. After passing through the base network, you end up with two of these
vectors representing the two inputs. The goal of the base network is to output a meaningful representation of the input (this is called an embedding).
The next step is to compute the cosine similarity between these two output vectors,
cosine similarity is a measure for how similar two vectors are to each other, ranging from 0 (completely different) to 1 (identical).
This cosine similarity is compared to the actual similarity between the two inputs and this error is used to update the weights in the network.

Don't worry if you do not fully understand the deep learning problem and the approach that is taken here.
We just want you to appreciate that you already learned enough to be able to do this yourself in your own domain.


1. Looking at the 'Model training' section of the notebook, what do you recognize from what you learned in this course?
2. Can you identify the different steps of the deep learning workflow in this notebook?
3. (Optional): Try to understand the neural network architecture from the first figure of [the paper](https://doi.org/10.1186/s13321-021-00558-4).
a. Why are there 10.000 neurons in the input layer?
b. What do you think would happen if you would decrease the size of spectral embedding layer drastically, to for example 5 neurons?






## 🧠 Collaborative Notes

# EPISODE 4 ~ Advanced layer types

### 1 First step of the "deep learning recipe"
It will be a classification task. We will look at images which are classified in different categories (e.g. tile roof, street sign, etc.)

| Category        | label |
| --------------- | ----- |
| day bed         | 0     |
| dishrag         | 1     |
| plate           | 2     |
| running shoe    | 3     |
| soap dispenser  | 4     |
| street sign     | 5     |
| table lamp      | 6     |
| tile roof       | 7     |
| toilet seat     | 8     |
| washing machine | 9     |
|                 |       |

Download the data:
[Dollar street dataset (4 files in total)](https://zenodo.org/api/records/10970014/files-archive)


```python
import pathlib
import numpy as np
```

```python
DATA_FOLDER = pathlib.Path('data/dataset_dollarstreet/')
train_images = np.load(DATA_FOLDER / 'train_images.py')
val_images = np. load(DATA_FOLDER / 'test_images.py')
train_labels = np.load(DATA_FOLDER / 'train_labels.py')
val_labels = np.load(DATA_FOLDER /'test_labels.py')

```
### 2: explore the data


```python
train_images.shape
#(878,64,64,3)

```
Explore the values of the images
```python
train_images.min(), train_images.max()
```



```python
train_labels.shape
```

To check the number of different labels
```python
train_labels.min(), train_labels.max()
```

### 3: Prepare the data

The training set consists of 878 images of 64x64 pixels and 3 channels (RGB values). The RGB values are between 0 and 255. For input of neural networks, it is better to have small input values. So we normalize our data between 0 and 1:

```python
train_images = train_images / 255.0
val_images = val_images / 255.0

```

### 4 Choose a pretrained model or train from scratch
We are now going to focus at convolutional layers, which work best for images. T


![](https://codimd.carpentries.org/uploads/upload_3c519e70312481cee99af01dcd2772ea.png)





#### Playing with convolutions

Convolutions applied to images can be hard to grasp at first. Fortunately there are resources out there that enable users to interactively play around with images and convolutions:

- [Image kernels](https://setosa.io/ev/image-kernels/) explained shows how different convolutions can achieve certain effects on an image, like sharpening and blurring.
- [The convolutional neural network](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks#) cheat sheet shows animated examples of the different components of convolutional neural nets




```python
from tensorflow import keras

inputs = keras.Input(shape=train_images[1:])
x= keras.layers.Conv2D(50, (3,3), activation = "relu")(inputs)
x= keras.layers.Conv2D(50, (3,3), activation = "relu")(x)
x= keras.layers.Flatten()(x)

outputs = keras.layers.Dense(10)(x)

model = keras.Model(inputs=inputs, outputs=outputs, name="dollar_street_model_small")

model.summary()
```

#### Pooling layers

Often in convolutional neural networks, the convolutional layers are intertwined with Pooling layers. As opposed to the convolutional layer, the pooling layer actually alters the dimensions of the image and reduces it by a scaling factor. It is basically decreasing the resolution of your picture. The rationale behind this is that higher layers of the network should focus on higher-level features of the image. By introducing a pooling layer, the subsequent convolutional layer has a broader ‘view’ on the original image.

Let’s put it into practice. We compose a Convolutional network with two convolutional layers and two pooling layers.


```python
def create_nn():
    inputs = keras.Input(shape=train_images.shape[1:])
    
    x= keras.layers.Conv2D(50, (3,3), activation = "relu")(inputs)
    x= keras.MaxPooling2D((2,2))(x)
    x= keras.layers.Conv2D(50, (3,3), activation = "relu")(x)
    x= keras.MaxPooling2D((2,2))(x)
    x= keras.layers.Flatten()(x)

    outputs = keras.layers.Dense(10)(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="dollar_street_model_small")
    return model

```

Create the model using the function

```python
model = create_nn()
model.summary()
```
### 5. Choose a loss function and optimizer


We create a function to compile the model. We are doing classification so we use SpareCategorialCrossentropy. 

```python
def compile_model(model):
    model.compile(optimizer='adam',
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
compile_model(model)

```

### 6. Train the model
Let us train the model for let us say 10 epochs.

```python
history = model.fit(train_images, train_labels, epochs=10, validation_data=(val_images, val_labels))
```

In order to say something we should compare it with the baseline. 

### 7. Perform a Prediction/Classification
(we are skipping this one for the moment)

### 8. Measure performance

First we are importing the libraries we need.
```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

```
We create a function for our plot to measure the performance.


```python
def plot_history(history, metrics):
    history_df = pd.DataFrame.from_dict(history.history)
    sns.lineplot(data=history_df[metrics])
    plt.xlabel("epochs")
    plt.ylabel("metric")
   
```
Let us plot the performance

```python
plot_history(history, ['accuracy', 'val_accuracy'])

```
We see a sign for overfitting. 

Next, lets look at the loss.

```python
plot_history(history, ['loss', 'val_loss'])

```
Loss is going down, but when we look at the validation set it is increasing. So it is a strong sign for overfitting.

### 9. Refine the model

See exercise above.

#### Dropout layers
Another layer for your toolbox! For countering overfitting dropout is a strong tool. You randomly shut down some of the neurons.

Note that the training loss continues to decrease, while the validation loss stagnates, and even starts to increase over the course of the epochs. Similarly, the accuracy for the validation set does not improve anymore after some epochs. This means we are overfitting on our training data set.

Techniques to avoid overfitting, or to improve model generalization, are termed regularization techniques. One of the most versatile regularization technique is dropout (Srivastava et al., 2014). Dropout means that during each training cycle (one forward pass of the data through the model) a random fraction of neurons in a dense layer are turned off. This is described with the dropout rate between 0 and 1 which determines the fraction of nodes to silence at a time.

![](https://codimd.carpentries.org/uploads/upload_574d305694658ecb52c65c5659c47884.png)

Create a dropout function

```python
def create_nn_with_dropout():
    inputs = keras.Input(shape=train_images.shape[1:])
    x = keras.layers.Conv2D(50, (3, 3), activation='relu')(inputs)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Dropout(0.8)(x) # This is new!

    x = keras.layers.Conv2D(50, (3, 3), activation='relu')(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Dropout(0.8)(x) # This is new!

    x = keras.layers.Conv2D(50, (3, 3), activation='relu')(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Dropout(0.8)(x) # This is new!

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(50, activation='relu')(x)
    outputs = keras.layers.Dense(10)(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="dropout_model")
    return model

```

Create the model


```python

model_dropout = create_nn_with_dropout()
model_dropout.summary()

```
Compile the model and train it again

```python
compile_model(model)

```


```python
history = model_dropout.fit(train_images, train_labels, epochs=25,
                    validation_data=(val_images, val_labels))
```

Plot the result.



```python
plot_history(history, ['accuracy', 'val_accuracy'])
```
We would expect the accuracy to increase.


[keras_tuner](https://carpentries-incubator.github.io/deep-learning-intro/4-advanced-layer-types.html#hyperparameter-tuning)


### 10. Save/ share model

```python
model.save('cnn_model.keras')
````

## EPISODE 5: [Transfer learning](https://carpentries-incubator.github.io/deep-learning-intro/5-transfer-learning.html)

![](https://codimd.carpentries.org/uploads/upload_48ce146a9d2602952ce2f74a8daaba2b.png)


### 1. Formulate / Outline the problem

```python
import pathlib
import numpy as np

DATA_FOLDER = pathlib.Path('data/dataset_dollarstreet/') # change to location where you stored the data
train_images = np.load(DATA_FOLDER / 'train_images.npy')
val_images = np.load(DATA_FOLDER / 'test_images.npy')
train_labels = np.load(DATA_FOLDER / 'train_labels.npy')
val_labels = np.load(DATA_FOLDER / 'test_labels.npy')

```

### 2. Identify inputs and outputs

We skip this step and continue to step 3.

### 3. Prepare the data

We prepare the data as before, scaling the values between 0 and 1. (255 pixel values for an image)

```python
train_images = train_images / 255.0
val_images = val_images / 255.0

```


### 4. Choose a pre-trained model or start building architecture from scratch

Let’s define our model input layer using the shape of our training images:

```python
from tensorflow import keras
inputs = keras.Input(train_images.shape[1:])

```
We use some helper function from tensorflow.

Our images are 64 x 64 pixels, whereas the pre-trained model that we will use was trained on images of 160 x 160 pixels. To adapt our data accordingly, we add an upscale layer that resizes the images to 160 x 160 pixels during training and prediction.

```python
import tensorflow as tf

method = tf.image.ResizeMethod.BILINEAR

upscale = keras.layers.Lambda(
    lambda x: tf.image.resize_with_pad(x, 160,160, method=method))(inputs)

```

From the keras.applications module we use the DenseNet121 architecture. This architecture was proposed by the paper: Densely Connected Convolutional Networks (CVPR 2017). It is trained on the Imagenet dataset, which contains 14,197,122 annotated images according to the WordNet hierarchy with over 20,000 classes.

We will have a look at the architecture later, for now it is enough to know that it is a convolutional neural network with 121 layers that was designed to work well on image classification tasks.

Let’s configure the DenseNet121:


```python
base_model = keras.applications.DenseNet121(include_top=False,
                                            pooling='max',
                                            weights='imagenet',
                                            input_tensor=upscale,
                                            input_shape=(160, 160, 3)
                                            )
```
It will download the data now. In case you have SSL certificate error: certificate verify failed: unable to get local issuer certificate, you can download the weights of the model manually and then load in the weights from the downloaded file:

```python base_model = keras.applications.DenseNet121(
    include_top=False,
    pooling='max',
    weights='densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5', # this should refer to the weights file you downloaded
    input_tensor=upscale,
    input_shape=(160,160,3)
)
```

```python
base_model.trainable = False
```

```python
out = base_model.output # as out initial layer
out = keras.layers.Flatten()(out)
out = keras.layers.BatchNormalization()(out)
out = keras.layers.Dense(50, activation='relu')(out)
out = keras.layers.Dropout(0.5)(out)
out = keras.layers.Dense(10)(out)

```

Transfer learning is very powerful.

```python
model = keras.models.Model(inputs=inputs, outputs=out)
```







## 💬 Feedback (MORNING)
Can you tell us one thing you thought went well and one thing that can be improved in the workshop this morning?
Think about the pace, the contect, the divison between theory and practice, the instructors, the facilities.

### 👍 What went well?
- Nice recap of the first day before continuing
- Nice to play around with models
- Great still to keep it hands-on
- Great to get to play around with the models to see what works/doesn't
- The exercise of improving the weather model with your own ideas and the discussion afterwards was nice!
- All good!
- Good build up into more advanced topics. Nice to play around a bit more. Breaks.
- Nice followup on yesterday!
- Good examples to learn different strategies for nn architecture design
- Probably already mentioned but calling the line numbers works well.
### 🤔 What can be improved?
- More examples to build intuitive understanding
- Perhaps given too long to play (can do this in our own time afterwards)
- Concepts start to pile up and, just as the notebooks become messy, sometimes difficult to keep all stuff in mind and make sense of when to use what (early stop, normalization, architecture). Also, would've been nice to have data where we see improvements (could never really make good predictions on weather)
- Playing around ourselves is nice but it also feels like it slows down the usefulness of the course a bit because this is something we could do at home while we won't get to cover more ground on our own so easily. Not sure where the balance is.
- I would appreciate more time for individual assignments where we're free to freely explore the tools.
- maybe compile/share a list of resources?
- just an idea: maybe have a poster (potentially showing the two day program).
- Clarify earlier the distinction with time series analysis where the order is imnportant; stress that the network treats the obs as independent   

### Sven's summary
- Balance between reinforcing what you learned so you have a firm understanding and covering many new topics
- list of resources
- we will build furter on concepts we introduced so far, so hopefully they will sink in deeper


## Post-workshop survey
Please fill in the post-workshop survey. It really helps us improve this workshop for next time and others that use the lesson material!
[post-workshop survey link](https://www.surveymonkey.com/r/68KYWWF)
-



-
## 📚 Resources

[Next steps](https://carpentries-incubator.github.io/deep-learning-intro/6-outlook.html)

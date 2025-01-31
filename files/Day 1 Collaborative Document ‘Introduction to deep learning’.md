
![](https://i.imgur.com/iywjz8s.png)


# Day 1 Collaborative Document 'Introduction to deep learning'
28th & 29th of January

Welcome to The Workshop Collaborative Document.

This Document is synchronized as you type, so that everyone viewing this page sees the same text. This allows you to collaborate seamlessly on documents.

----------------------------------------------------------------------------


This is the Document for today: https://edu.nl/uuayc


Collaborative Document day 1: https://edu.nl/uuayc

This is the document for day 2, morning: https://edu.nl/bym3j

This is the document for day 2, afternoon: https://edu.nl/gf86m

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

If you need help from a helper, place an orange post-it note on your laptop lid. A helper will come to assist you as soon as possible. We will use yellow to vote, indicate that you are done etcetera.

## 🖥 Workshop website
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

## WiFi
SSID: escience
password: U7t4Bdpkd

## 👩‍💻👩‍💼👨‍🔬🧑‍🔬🧑‍🚀🧙‍♂️🔧 Roll Call
Name/ pronouns (optional) / job, role / social media (bluesky!, github, ...) / background or interests (optional) / city


## 🗓️ Agenda
* 09:30	Welcome and icebreaker
    1. Setting the stage
    2. Icebreaker
    3. Introduction eScience Center
    4. Workshop logistics
* 09:45	Introduction to Deep Learning
* 10:30	Break
* 10:40	Introduction to Deep Learning
* 11:30	Break
* 11:40	Classification by a Neural Network using Keras
* 12:30	Lunch Break
* 13:30	Classification by a Neural Network using Keras
* 14:30	Break
* 14:40	Monitor the training process
* 15:30	Break
* 15:40	Monitor the training process
* 16:15	Wrap-up
* 16:30	END

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
## 1. Activation functions
Look at the following activation functions:

**A. Sigmoid activation function**
The sigmoid activation function is given by:
$f(x) = \frac{1}{1 + e^{-x}}$

![](https://codimd.carpentries.org/uploads/upload_9281fd700679c2b1e57e031807fec3bd.png)

**B. ReLU activation function**
The Rectified Linear Unit (ReLU) activation function is defined as:
$f(x) = \max(0, x)$

This involves a simple comparison and maximum calculation, which are basic operations that are computationally inexpensive.
It is also simple to compute the gradient: 1 for positive inputs and 0 for negative inputs.

![](https://codimd.carpentries.org/uploads/upload_2baf23fa1b9dcb163e05916d960b2016.png)


**C. Linear (or identity) activation function (output=input)**
The linear activation function is simply the identity function:
$f(x) = x$ 


![](https://codimd.carpentries.org/uploads/upload_ff695064fbb02e321132575b47e27246.png)



Combine the following statements to the correct activation function:

1. This function enforces the activation of a neuron to be between 0 and 1
2. This function is useful in regression tasks when applied to an output neuron
3. This function is the most popular activation function in hidden layers, since it introduces non-linearity in a computationally efficient way.
4. This function is useful in classification tasks when applied to an output neuron
5. (optional) For positive values this function results in the same activations as the identity function.
6. (optional) This function is not differentiable at 0
7. (optional) This function is the default for Dense layers (search the Keras documentation)




Answers:
1: A
2: C
3: B
4: A
5: B
6: B
7: C

## 2. Neural network calculations

#### 2.1. Calculate the output for one neuron
Suppose we have:

- Input: X = (0, 0.5, 1)
- Weights: W = (-1, -0.5, 0.5)
- Bias: b = 1
- Activation function _relu_: `f(x) = max(x, 0)`

What is the output of the neuron?

_Note: You can use whatever you like: brain only, pen&paper, Python, Excel..._


**Answer:**
You can calculate the output as follows:

Weighted sum of input: 0 * (-1) + 0.5 * (-0.5) + 1 * 0.5 = 0.25
Add the bias: 0.25 + 1 = 1.25
Apply activation function: max(1.25, 0) = 1.25
So, the neuron’s output is 1.25

#### 2.2. (optional) Calculate outputs for a network

Have a look at the following network where:

* $X_1$ and $X_2$ denote the two inputs of the network.
* $h_1$ and $h_2$ denote the two neurons in the hidden layer. They both have ReLU activation functions.
* $h_1$ and $h_2$ denotes the output neuron. It has a ReLU activation function.
* The value on the arrows represent the weight associated to that input to the neuron.
* $b_i$ denotes the bias term of that specific neuron
![](https://codimd.carpentries.org/uploads/upload_f6ae9b8af7a4b4c7ac47c07ec334dd70.png)


a. Calculate the output of the network for the following combinations of inputs:

| x1 | x2 | y |
|----|----|---|
| 0  | 0  | ..|
| 0  | 1  | ..|
| 1  | 0  | ..|
| 1  | 1  | ..|

b. What logical problem does this network solve?





**Answer:**

2: Calculate outputs for a network

| x1  | x2  | y   |
| --- | --- | --- |
| 0   | 0   | 0   |
| 0   | 1   | 1   |
| 1   | 1   | 0   |
| 1   | 0   | 1   |
This solves the XOR logical problem, the output is 1 if only one of the two inputs is 1.

## 3. Exercise: Loss function
#### 3.1. Compute the Mean Squared Error
One of the simplest loss functions is the Mean Squared Error. MSE = $\frac{1}{n} \Sigma_{i=1}^n({y}-\hat{y})^2$
It is the mean of all squared errors, where the error is the difference between the predicted and expected value.
In the following table, fill in the missing values in the 'squared error' column. What is the MSE loss
for the predictions on these 4 samples?

| **Prediction** | **Expected value** | **Squared error** |
|----------------|--------------------|-------------------|
| 1              | -1                 | 4                 |
| 2              | -1                 | ..                |
| 0              | 0                  | ..                |
| 3              | 2                  | ..                |
|                | **MSE:**           | ..                |


**Answer:**
| Prediction | Expected value | Squared error |
| ---------- | -------------- | ------------- |
| 1          | -1             | 4             |
| 2          | -1             | 9             |
| 0          | 0              | 0             |
| 3          | 2              | 1             |
|            | MSE:           | 3.5           |

#### 2. (optional) Huber loss
A more complicated and less used loss function for regression is the [Huber loss](https://keras.io/api/losses/regression_losses/#huber-class).

Below you see the Huber loss (green, delta = 1) and Squared error loss (blue)
as a function of `y_true - y_pred`.

![](https://codimd.carpentries.org/uploads/upload_b10fb8078ea86399740612ee03979cc4.png)


Which loss function is more sensitive to outliers?


**Answer:**
The squared error loss is more sensitive to outliers. Errors between -1 and 1 result in the same loss value for both loss functions. But, larger errors (in other words: outliers) result in quadratically larger losses for the Mean Squared Error, while for the Huber loss they only increase linearly.

## 4. Deep learning workflow exercise

Think about a problem you would like to use deep learning to solve.

1. What do you want a deep learning system to be able to tell you?
2. What data inputs and outputs will you have?
3. Do you think you will need to train the network or will a pre-trained network be suitable?
4. What data do you have to train with? What preparation will your data need? Consider both the data you are going to predict/classify from and the data you will use to train the network.

## 5. Create the neural network
With the code snippets above, we defined a Keras model with 1 hidden layer with
10 neurons and an output layer with 3 neurons.

1. How many parameters does the resulting model have?
2. What happens to the number of parameters if we increase or decrease the number of neurons
in the hidden layer?

#### (optional) Visualizing the model
Optionally, you can also visualize the same information as model.summary() in graph form. This step requires the command-line tool dot from Graphviz installed, you installed it by following the setup instructions. You can check that the installation was successful by executing dot -V in the command line.

3. Provided you have dot installed, execute the plot_model function as shown below.

```python
keras.utils.plot_model(
    model,
    show_shapes=True,
    show_layer_names=True,
    show_layer_activations=True,
    show_trainable=True
)
```

#### (optional) Keras Sequential vs Functional API
So far we have used the [Functional API](https://keras.io/guides/functional_api/) of Keras.
You can also implement neural networks using [the Sequential model](https://keras.io/guides/sequential_model/).
As you can read in the documentation, the Sequential model is appropriate for **a plain stack of layers**
where each layer has **exactly one input tensor and one output tensor**.

4. Use the Sequential model to implement the same network.



+ 3 

**Answer:**
The model has 83 trainable parameters. Each of the 10 neurons in the in the dense hidden layer is connected to each of the 4 inputs in the input layer resulting in 40 weights that can be trained. The 10 neurons in the hidden layer are also connected to each of the 3 outputs in the dense_1 output layer, resulting in a further 30 weights that can be trained. By default Dense layers in Keras also contain 1 bias term for each neuron, resulting in a further 10 bias values for the hidden layer and 3 bias terms for the output layer. 40+30+10+3=83 trainable parameters.

The value (332.00 B) next to it describes the memory footprint for model weights and this depends on their data type. Take a look at what model.dtype is.

The model weights are represented using float32 data type, which consumes 32 bits or 4 bytes for each weight. We have 83 parameters, and therefore in total, the model requires 83*4=332 bytes of memory to load into the computer’s memory.

If you increase the number of neurons in the hidden layer the number of trainable parameters in both the hidden and output layer increases or decreases in accordance with the number of neurons added. Each extra neuron has 4 weights connected to the input layer, 1 bias term, and 3 weights connected to the output layer. So in total 8 extra parameters.

optional
3.
Upon executing the plot_model function, you should see the following image:
![](https://codimd.carpentries.org/uploads/upload_bb0f89bb0d57639ae2a41f958e6b3e95.png)

optional
4.
This implements the same model using the Sequential API:
```python
model = keras.Sequential(
    [
        keras.Input(shape=(X_train.shape[1],)),
        keras.layers.Dense(10, activation="relu"),
        keras.layers.Dense(3, activation="softmax"),
    ]
)
```

## 6. The Training Curve
Looking at the training curve we have just made.

1. How does the training progress?
* Does the training loss increase or decrease?
* Does it change quickly or slowly?
* Does the graph look very jittery?
2. Do you think the resulting trained network will work well on the test set?

When the training process does not go well:

3. (optional) Something went wrong here during training. What could be the problem, and how do you see that in the training curve?
Also compare the range on the y-axis with the previous training curve.
![](https://codimd.carpentries.org/uploads/upload_243a34957398e72c728601b7d828e5d3.png)

                                    |


## 7. Confusion Matrix

Measure the performance of the neural network you trained and visualize a confusion matrix.

- Did the neural network perform well on the test set?
- Did you expect this from the training loss you saw?
- What could we do to improve the performance?




## 8. Exercise: Architecture of the network

As we want to design a neural network architecture for a regression task,
see if you can first come up with the answers to the following questions:

1 What must be the dimension of our input layer?
2 We want to output the prediction of a single number. The output layer of the NN hence cannot be the same as for the classification task earlier. This is because the softmax activation being used had a concrete meaning with respect to the class labels which is not needed here. What output layer design would you choose for regression?
**Hint**: A layer with relu activation, with sigmoid activation or no activation at all?
3 (Optional) How would we change the model if we would like to output a prediction of the precipitation in Basel in addition to the sunshine hours?


## 9. Exercise: Gradient descent

Answer the following questions:
1. What is the goal of optimization?

    A. To find the weights that maximize the loss function
    B. To find the weights that minimize the loss function

2. What happens in one gradient descent step?

    A. The weights are adjusted so that we move in the direction of the gradient, so up the slope of the loss function
    B. The weights are adjusted so that we move in the direction of the gradient, so down the slope of the loss function
    C. The weights are adjusted so that we move in the direction of the negative gradient, so up the slope of the loss function
    D. The weights are adjusted so that we move in the direction of the negative gradient, so down the slope of the loss function

3. When the batch size is increased: (multiple answers might apply)

    A. The number of samples in an epoch also increases
    B. The number of batches in an epoch goes down
    C. The training progress is more jumpy, because more samples are consulted in each update step (one batch).
    D. The memory load (memory as in computer hardware) of the training process is increased




## 10. Exercise: Reflecting on our results
1. Is the performance of the model as you expected (or better/worse)?
2.  Is there a noteable difference between training set and test set? And if so, any idea why?
3. (Optional) When developing a model, you will often vary different aspects of your model like
which features you use, model parameters and architecture. It is important to settle on a
single-number evaluation metric to compare your models.
4. What single-number evaluation metric would you choose here and why?

                                               |


## 🧠 Collaborative Notes


### 1. Formulate/outline the problem: penguin classification

In this episode we will be using the penguin dataset. This is a dataset that was published in 2020 by Allison Horst and contains data on three different species of the penguins.

We will use the penguin dataset to train a neural network which can classify which species a penguin belongs to, based on their physical characteristics.


### 2. Identify inputs and outputs

To identify the inputs and outputs that we will use to design the neural network we need to familiarize ourselves with the dataset. This step is sometimes also called data exploration.

We will start by importing the Seaborn library that will help us get the dataset and visualize it. Seaborn is a powerful library with many visualizations. Keep in mind it requires the data to be in a pandas dataframe, luckily the datasets available in seaborn are already in a pandas dataframe.


```python
import seaborn as sns
import tensorflow
import pandas as pd
```

We are loading the penguin data as a pandas dataframe

```python
penguins = sns.load_dataset('penguins')
```
explore the data

```python
penguins.head()
```

```python
penguins.shape
```

![](https://codimd.carpentries.org/uploads/upload_cef6b7c5bb1cde2f9bd943181d4f8167.png)

Based on characteristics of the penguins you can predict the penguin species.

![](https://codimd.carpentries.org/uploads/upload_eaf0cd12d6b4cd557b1d4b7fb50475ad.png)


```python
sns.pairplot(penguins, hue='species')
```

#### Input and Output Selection

Now that we have familiarized ourselves with the dataset we can select the data attributes to use as input for the neural network and the target that we want to predict.

In the rest of this episode we will use the **bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g attributes**. The target for the classification task will be the species.

### 3. Prepare data

The input data and target data are not yet in a format that is suitable to use for training a neural network.

For now we will only use the numerical features **bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g only**, so let’s drop the categorical columns:

Drop the columns that represent features that we will not use in our model.

```python
penguins_filtered = penguins.drop(columns=['island', 'sex'])
```

Drop the rows that contain NaN values
```python
penguins_filtered = penguins_filtered.dropna()
```

Define a parameter that holds the features we will use for our models, dropping the column species, because that is our target column.
```python
features = penguins_filtered.drop(columns=['species'])
```

Import the pandas library and convert the species into a dummy variable.

```
import pandas as pd

target = pd.get_dummies(penguins_filtered['species'])
```

```python
target.head()
```

### Split data into training and test set

Finally, we will split the dataset into a training set and a test set. As the names imply we will use the training set to train the neural network, while the test set is kept separate. We will use the test set to assess the performance of the trained neural network on unseen samples. In many cases a validation set is also kept separate from the training and test sets (i.e. the dataset is split into 3 parts). This validation set is then used to select the values of the parameters of the neural network and the training methods. For this episode we will keep it at just a training and test set however.


To split the cleaned dataset into a training and test set we will use a very convenient function from sklearn called train_test_split (see https://scikit-learn.org/1.6/modules/generated/sklearn.model_selection.train_test_split.html).

```python
from sklearn.model_selection import train_test_split
```

Split the data into a training and test set, using 20% of the data for testing, and 80% for the training. 
- Random_state controls the shuffling of the dataset, setting this value will reproduce the same results (assuming you give the same integer) every time it is called.
- Shuffle which can be either True or False, it controls whether the order of the rows of the dataset is shuffled before splitting. It defaults to True.
- Stratify is a more advanced parameter that controls how the split is done. By setting it to target the train and test sets the function will return will have roughly the same proportions (with regards to the number of penguins of a certain species) as the dataset.

```python
X_train, X_test, y_train, y_test = train_test_split(
    features,
    target,
    test_size=0.2,
    random_state=0,
    shuffle=True,
    stratify=target
)
```

## 4. Build an architecture from scratch or choose a pretrained model
Keras for neural networks

Keras is a machine learning framework with ease of use as one of its main features. It is part of the tensorflow python package and can be imported using from tensorflow import keras.

Keras includes functions, classes and definitions to define deep learning models, cost functions and optimizers (optimizers are used to train a model).

Before we move on to the next section of the workflow we need to make sure we have Keras imported. We do this as follows:


```python
from tensorflow import keras
```

Set the random state
```python
from numpy.random import seed

seed(1)
keras.utils.set_random_seed(2)
```

Define the neural network layers
```python
# The input layer of the network
inputs = keras.Input(shape=(X_train.shape[1],))
```

We will use a network with one hidden layer, a dense layer type (we will discuss different layer types later), with 10 neurons. We will use the relu activate function for this layer.

[More on dense layers](https://keras.io/api/layers/core_layers/dense/).

```python
hidden_layer = keras.layers.Dense(10, activation="relu")(inputs)
```
(in case you get a CUDA error here, just ignore it for now)

Define the output layer, also a dense layer type. For this layer we are using the softmax activating function. 

The softmax activation ensures that the three output neurons produce values in the range (0, 1) and they sum to 1. We can interpret this as a kind of ‘probability’ that the sample belongs to a certain species.

```python
output_layer = keras.layers.Dense(3, activation="softmax")(hidden_layer)
```


```python
model = keras.Model(inputs=inputs, outputs=output_layer)
model.summary()
```

## 5. Choose a loss function and optimizer

We have now designed a neural network that in theory we should be able to train to classify Penguins. However, we first need to select an appropriate loss function that we will use during training. This loss function tells the training algorithm how wrong, or how ‘far away’ from the true value the predicted value is.

For the one-hot encoding that we selected earlier a suitable loss function is the Categorical Crossentropy loss. In Keras this is implemented in the keras.losses.CategoricalCrossentropy class. This loss function works well in combination with the softmax activation function we chose earlier. The Categorical Crossentropy works by comparing the probabilities that the neural network predicts with ‘true’ probabilities that we generated using the one-hot encoding. This is a measure for how close the distribution of the three neural network outputs corresponds to the distribution of the three values in the one-hot encoding. It is lower if the distributions are more similar.

For more information on the available loss functions in Keras you can check the documentation.

Next we need to choose which optimizer to use and, if this optimizer has parameters, what values to use for those. Furthermore, we need to specify how many times to show the training samples to the optimizer.

Once more, Keras gives us plenty of choices all of which have their own pros and cons, but for now let us go with the widely used Adam optimizer. Adam has a number of parameters, but the default values work well for most problems. So we will use it with its default parameters.

Combining this with the loss function we decided on earlier we can now compile the model using model.compile. Compiling the model prepares it to start the training.


```python
model.compile(
    optimizer='adam',
    loss=keras.losses.CategoricalCrossentropy()
    )
```


### 6. Train model

We are now ready to train the model.

Training the model is done using the fit method, it takes the input data and target data as inputs and it has several other parameters for certain options of the training. Here we only set a different number of epochs. One training epoch means that every sample in the training data has been shown to the neural network and used to update its parameters.


```python
history = model.fit(X_train, y_train, epochs=100)
```

The fit method returns a history object that has a history attribute with the training loss and potentially other metrics per training epoch. It can be very insightful to plot the training loss to see how the training progresses. Using seaborn we can do this as follows:

[documentation sns.lineplot](https://seaborn.pydata.org/generated/seaborn.lineplot.html)

```python
sns.lineplot(x=history.epoch, y=history.history['loss'])
```
![](https://codimd.carpentries.org/uploads/upload_7bca0c238c3a54d6d2c65fd5f924ff1d.png)


!!! **I get a different plot** !!!

It could be that you get a different plot than the one shown here. This could be because of a different random initialization of the model or a different split of the data. This difference can be avoided by setting **random_state** and random seed in the same way like we discussed in When to use random seeds?


### 7. Perform a prediction/classification [AFTER LUNCH]

We finished with the trained model. We went through the different steps. Now that we have a trained neural network, we can use it to predict new samples of penguin using the predict function.

We will use the neural network to predict the species of the test set using the predict function. We will be using this prediction in the next step to measure the performance of our trained network. This will return a numpy matrix, which we convert to a pandas dataframe to easily see the labels.


```python
y_pred = model.predict(X_text)
#now lets convert it to a pandas df
prediction = pd.DataFrame(y_pred, columns= target.columns)
prediciton
```

We see three neurons the predict the penguin species. Valuues between 0 - 1

We want the model to provide an actual answer. 

```python
prediction_species = prediction.idmax(axis="columns")
prediction_species
```
provides the names of the predictions (i.e. Adelie, Gentoo etc.)

### 8. Measuring performance

Now that we have a trained neural network it is important to assess how well it performs. We want to know how well it will perform in a realistic prediction scenario, measuring performance will also come back when refining the model.

We have created a test set (i.e. y_test) during the data preparation stage which we will use now to create a confusion matrix.
Confusion matrix

With the predicted species we can now create a confusion matrix and display it using seaborn.

A confusion matrix is an N x N matrix used for evaluating the performance of a classification model, where N is the number of target classes. The matrix compares the actual target values with those predicted from the classification model, which gives a holistic view of how well the classification model is performing.

To create a confusion matrix we will use another convenience function from sklearn called confusion_matrix. This function takes as a first parameter the true labels of the test set. We can get these by using the idxmax method on the y_test dataframe. The second parameter is the predicted labels which we did above.

Let us import the confusion_matrix library from sklearn

```python
from sklearn.metrics import confusion_matrix

true_species = y_test.idxmax(axis="columns")

```
Now we create the confusion matrix

```python
matrix = confusion_matrix(true_species, predicted_species)
```
Next we are converting the matrix to a pandas df

```python
confusion_df = pd.DataFrame(matrix, index=y_test.columns.values, column=y_test.columns.values)

confusion_df.index.name = 'True Label'
confusion_df.columns.name = 'Predicted Label'
confusion_df.head()

```

Now we creat a heatmp (sns has a nice function for it)

```python
sns.heatmap(confusion_df, annot=True)
```

Now we used the dense layer. There are other options as well.


### 9.Refine the model

As we discussed before the design and training of a neural network comes with many hyperparameter and model architecture choices. We will go into more depth of these choices in later episodes. For now it is important to realize that the parameters we chose were somewhat arbitrary and more careful consideration needs to be taken to pick hyperparameter values.

### 10. Share/Save model



```python
model.save('my_first_model.keras')
```

In case you want to load your model


```python
pretrained_model = keras.models.load_model('my_first_model.keras')

```

This loaded model can be used as before to predict.

```python
y_pretrained_pred = pretrained_model.predict(X_text)
pretrained_prediction = pd.DataFrame(y_pretrained_pred, columns=target.columns.values)
pretrained_predicted_species = pretrained_prediction.idxmax(axis="columns")
pretrained_predicted_species

```
We went through the 10 steps of the deep learning recipe. 


Key Points

- The deep learning workflow is a useful tool to structure your approach, it helps to make sure you do not forget any important steps.
- Exploring the data is an important step to familiarize yourself with the problem and to help you determine the relavent inputs and outputs.
- One-hot encoding is a preprocessing step to prepare labels for classification in Keras.
- A fully connected layer is a layer which has connections to all neurons in the previous and subsequent layers.
- keras.layers.Dense is an implementation of a fully connected layer, you can set the number of neurons in the layer and the activation function used.
- To train a neural network with Keras we need to first define the network using layers and the Model class. Then we can train it using the model.fit function.
- Plotting the loss curve can be used to identify and troubleshoot the training process.
- The loss curve on the training set does not provide any information on how well a network performs in a real setting.
- Creating a confusion matrix with results from a test set gives better insight into the network’s performance.


# CHAPTER 2 Monitor the training process
We start a **NEW NOTEBOOK**. 

### 1. Formulate / Outline the problem: weather prediction

We are going to predict the number of sunshine hours in Basel

We are using this data for it: https://zenodo.org/records/5071376 (which you downloaded before)

### 2. Identify inputs and outputs

Let us load the data

```python
import pandas as pd
data = pd.read_csv("https://zenodo.org/record/5071376/files/weather_prediction_dataset_light.csv?download=1") 
```

```python
data_head() #to explore the data
```


```python
data.shape
#gives (3654, 91) 10 year data 91 features

```

### 3. Prepare data

The full dataset comprises of 10 years (3654 days) from which we will select only the first 3 years. The present dataset is sorted by “DATE”, so for each row i in the table we can pick a corresponding feature and location from row i+1 that we later want to predict with our model. As outlined in step 1, we would like to predict the sunshine hours for the location: BASEL.

```python
nr_rows = 365*3 #is 3 yrs
X_data = data.loc[:nr_rows] # Select first 3 years
X_data = X_data.drop(columns=['DATE', 'MONTH'])
#now we want ot have our sunshine data
y_data = data.loc[1:(nr_rows + 1)]["BASEL_sunshine"]

```

Next step is to split data and label into training, validation and test set


```python
import sklearn
from sklearn.model_selection import train_test_split
X_train, X_holdout, y_train, y_holdout = train_test_split(X_data, y_data, test_size=0.3, random_state=0)

```
We want to cut the holdout set in two


```python
X_val, X_test, y_val, y_test = train_test_split(X_holdout, y_holdout, test_size=0.5, random_state=0)

```
Let us have a look at the data

```python
X_train.shape, X_test.shape, X_val.shape
#((767,89),(165,89), (164,89))

```
Let us go to the next step

### 4. Choose a pretrained model or start building architecture from scratch


Import the library
```python
from tensorflow import keras
```

define input
```python
inputs = keras.Input(shape=(input_shape[1],), name='input')

layers_dense = keras.layers.Dense(100, 'relu')(inputs)
layers_dense = keras.layers.Dense(50, 'relu')(layers_dense)

outputs = keras.layers.Dense(1)(layers_dense)
```

```python
keras.Model(inputs=inputs, outputs=outputs, name="weather_prediction_first_attempt")
```
Lets have a look at the model.

Why Relu? it is a good way to introduce how it works. In other cases you would go for other options.

```python
model.summary()

```
Lets change this so it becomes a function, since we are going to use it multiple times.

```python
def create_nn(input_shape, name):
    inputs = keras.Input(shape=input_shape, name='input')
    layers_dense = keras.layers.Dense(100, 'relu')(inputs)
    layers_dense = keras.layers.Dense(50, 'relu')(layers_dense)
    outputs = keras.layers.Dense(1)(layers_dense)
    model = keras.Model(inputs=inputs, outputs=outputs, name=name)
    return model

```

```python
model = create_nn(input_shape=(X_data.shape[1],), name="first_attempt")
model.summary()
```

## Intermezzo: How do neural networks learn?

In the introduction we learned about the loss function: it quantifies the total error of the predictions made by the model. During model training we aim to find the model parameters that minimize the loss. This is called optimization, but how does optimization actually work?
Gradient descent

Gradient descent is a widely used optimization algorithm, most other optimization algorithms are based on it. It works as follows: Imagine a neural network with only one neuron. Take a look at the figure below. The plot shows the loss as a function of the weight of the neuron. As you can see there is a global loss minimum, we would like to find the weight at this point in the parabola. To do this, we initialize the model weight with some random value. Then we compute the gradient of the loss function with respect to the weight. This tells us how much the loss function will change if we change the weight by a small amount. Then, we update the weight by taking a small step in the direction of the negative gradient, so down the slope. This will slightly decrease the loss. This process is repeated until the loss function reaches a minimum. The size of the step that is taken in each iteration is called the ‘learning rate’.


![](https://codimd.carpentries.org/uploads/upload_77ae6285e07d06df26d1ec8311a55d2e.png)

# Glossary BinGO!

- Sample = 1 observation = 1 row in a 2D dataframe
- Batch = a set of observations that you feed to the model in a learning step
- Epoch = one iteration of going through the data
- Feature = one variable in the input data = input
- Learning step = one iteration of learing (using gradient descent)
    1. Predicition on one batch of data (e.g. 32 samples)
    2. Calculate the loss
    3. We adjust the weights according to the loss
- Learning rate = controls the magnitue of weight adjustment in gradient descent

### 5. Choose a loss function and optimizer

**Hint** In Keras this is implemented in the keras.losses.MeanSquaredError class (see Keras documentation: https://keras.io/api/losses/). This can be provided into the model.compile method with the loss parameter and setting it the loss function)

It is a regression task and not a classification task. We will use the mean square error. 


```python
model.compile(loss='mse') #mse is mean square error
model.compile(loss='mse', optimizer='adam')
```

We are now introducing the metrics parameter. 


```python
model.compile(loss='mse', optimizer='adam', metrics= [keras.metrics.RootMeanSquareError()])
```

Let us create a function for this
```python
def compile_model(model):
    model.compile(loss='mse', optimizer='adam', metrics=[keras.metrics.RootMeanSquaredError()])
```



```python
compile_model(model)
```

### 6. Train the model



We are going to start the training process


```python
history = model.fit(X_train, y_train, batch_size= 32, epochs=200,
                  verbose=2)
```
batch_size of 32 is a bit arbitrary, normally you would want to allign it with you GPU, but since the data is relatively small 32 is fine.

We will use the history we created before
```python
history_df = pd.DataFrame.from_dict(history.history)

```
We are using sns to create plots


```python
import seaborn as sns
sns.lineplot(data=history_df['root_mean_squared_error'])

```

```python
import matplotlib.pyplot as plt
def plot_history(history, metrics):
    history_df = pd.DataFrame.from_dict(history.history)
    sns.lineplot(data=history_df[metrics])
    plt.xlabel("epochs")
    plt.ylabel("metric")

```

```python
plot_history(history, 'root_mean_squared_error')
plot_history(history, 'loss')

```

### 7. Perform a Prediction/Classification

```python
y_train_predicted = model.predict(X_train)
y_test_predicted = model.predict(X_test)

```


### 8. Measure performance

We make a function so we can easily reuse it. 


```python
def plot_predictions(y_pred, y_true, title):
    plt.style.use("ggplot")
    plt.scatter(y_pred,y_true, s=10, alpha=0.5)
    plt.xlabel('predicted sunshine hours')
    plt.ylabel('True sunshine hours')
    plt.title(title)
    
plot_predications(y_train_predicted, y_train, title='Predictions on the train set')

```


Evaluate the training data
```python
loss, rmse = model.evaluate(X_train, y_train)
loss, rmse
```

Evaluate the test data
```python
loss, rmse = model.evaluate(X_test, y_test)
loss, rmse
```

Lets do a baseline prediction.
```python
y_baseline_prediction = X_test['BASEL_sunshine']
```


```python
plot_predictions(y_baseline_prediction, y_test, title= 'Baseline predictions')
```
Lets compute the root mean error.


```python
from sklearn.metrics import root_mean_squared_error
rmse_baseline = root_mean_squared_error(y_test, y_baseline_prediction)
rmse_baseline

```




## 💬 Feedback (MORNING)
Can you tell us one thing you thought went well and one thing that can be improved in the workshop this morning?
Think about the pace, the contect, the divison between theory and practice, the instructors, the facilities.

### 👍 What went well?
- Soft hands-on intro that doesn't assume too much previous knowledge but still introduces enough new material. Lekker food and coffee. Regular breaks. Follow-along coding is a very nice way to learn.
- Kind people well, prepared session, direct hands on
- I like the 'learning by doing' and then only later diving under the hood to see how things actually work.
- Efficiently straight to the practical part
- teaching with interesting and  practical examples
- Good introduction, followed by practical (always nice to learn-by-doing)
- Nice pace of the course, with frequent brakes and check-in's
- Relaxed atmosphere

### 🤔 What can be improved?
- Nice way of working with questions and filling them in durign the presentation. Might be helpfull to structure the answering sections (where to put what)
- Slightly faster demonstrations would be nice
- Prepare notebooks with code, then execute code and discuss, more advanced level discussions and sharing practices, without background theory concepts are probably difficult to 'absorb', create a beginnner and advanced DL course at some time?
- Switching between different parts of the collaborative document can be difficult and confusing at times
- The coding part could have gone a bit faster and the theory part a bit more in detail.
- Some more time at transistions would be nice (showing URL of collaborative notebook, starting up Jupiter notebook...)
- Sometimes wait a sec before executing cell, as it can create output and advance the scroll before we type stuff

### Sven's summary
- We can move through the live coding parts a bit faster, and should take more time (or more instructions) for transitions and theoretical parts



## 💬 Feedback (AFTERNOON)

_ good pac

### 👍 What went well?
- Faster pace in coding now that we have covered the basics. Regular breaks. The collaborative doc works well (calling the lines helps navigate!). Coding by hand is nice to get used to the syntax.
- Better pace and like the short, sharp, regular breaks.
- Great break discipline! Really helps to keep the focus.
- Nice build-up in content throughout the day
- More hands-on; good pace
- Faster pace, good energy, flow,  learned something today
- Friendly and accessible instructors 
### 🤔 What can be improved?
- More theory and intuition would be nice
- Could be nice to complement the explanations by implementing some things "by hand" to understand how they work, but this could take a lot of time and explaining that maybe is not there or needed.
- Would be nice to have the code ready for copy-pasting. I can't/don't want to keep up with the typing, it just distracts from following the course.

### Sven's summary
- Balance between theory/intuition and practice
- We encourage you to not copy paste, but instead type along, and have us pause for a moment if it goes too fast. You learn much less from copy-pasting. But if you really want to, this is the material that we use: https://carpentries-incubator.github.io/deep-learning-intro/ Disclaimer: we deviate from the material a little bit.
- 2 collaborative documents today



## 📚 Resources
*Activation function plots by Laughsinthestocks - Own work, CC BY-SA 4.0, https://commons.wikimedia.org/w/index.php?curid=44920411,
https://commons.wikimedia.org/w/index.php?curid=44920600, https://commons.wikimedia.org/w/index.php?curid=44920533*

https://playground.tensorflow.org/

Coursera course "Introduction to Machine Learning": https://www.coursera.org/learn/machine-learning-duke

Some popular Python packages for machine learning:
[TensorFlow](https://www.tensorflow.org/)
[PyTorch](https://pytorch.org/)
[Keras](https://keras.io/)

**Exaplainable AI (XAI)**
DIANNA [(eScience Center project)](https://research-software-directory.org/projects/dianna): https://pypi.org/project/dianna/
Lime: https://pypi.org/project/lime/
Rise: http://bmvc2018.org/contents/papers/1064.pdf
Shap: https://pypi.org/project/shap/
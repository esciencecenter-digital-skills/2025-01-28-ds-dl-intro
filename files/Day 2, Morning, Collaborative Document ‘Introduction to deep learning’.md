
![](https://i.imgur.com/iywjz8s.png)


# Day 2, Morning, Collaborative Document 'Introduction to deep learning'
28th & 29th of January

Welcome to The Workshop Collaborative Document.

This Document is synchronized as you type, so that everyone viewing this page sees the same text. This allows you to collaborate seamlessly on documents.

----------------------------------------------------------------------------


This is the document for this morning (day 2): https://edu.nl/bym3j

This is the document for this afternoon (day 2): https://edu.nl/gf86m

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
    * Yesterday's feedback
    * Deep learning 30 seconds
09:45	Monitor the training process
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

## Notebook snapshot yesterday
If you did not save yesterday's final state of the notebook:
https://github.com/esciencecenter-digital-skills/2025-01-28-ds-dl-intro/blob/main/files/02-monitor-the-training-progress.ipynb


## ⏱ Deep learning 30 seconds
|        | Round 1 | Round 2 | Round 3 | **Total** |
| ------ | ------- | ------- | ------- | --------- |
| Team 1 | 4       | 3       | 3       | 10        |
| Team 2 | 2       | 2       | 4       | 8         |
| Team 3 | 1       | 1       | 13      | 5         |
| Team 4 | 3       | 3       | 2       | 8         |

Team 1 wins!


### Round 1 concepts:
- learning rate
- artificial intelligence
- Python
- neural network
- input layer

### Round 2 concepts:
- hidden layer
- batch size
- pandas
- activation function
- loss function

### Round 3 concepts:
- evaluation metric
- gradient descent
- output layer
- keras
- epoch

## 🔧 Exercises

### 2.1 Exercise: Try to reduce the degree of overfitting by lowering the number of parameters

We can keep the network architecture unchanged (2 dense layers + a one-node output layer) and only play with the number of nodes per layer. Try to lower the number of nodes in one or both of the two dense layers and observe the changes to the training and validation losses. If time is short: Suggestion is to run one network with only 10 and 5 nodes in the first and second layer.

- Is it possible to get rid of overfitting this way?
- Does the overall performance suffer or does it mostly stay the same?
- (optional) How low can you go with the number of parameters without notable effect on the performance on the validation set?



Copy the function and update and tweak it. The "best" result would be 10 and 5 or 3 and 3.

```python
def create_nn(input_shape, name="weather_prediction_model"):
    inputs = keras.Input(shape=input_shape, name='input')

    layers_dense = keras.layers.Dense(100, 'relu')(inputs)
    layers_dense = keras.layers.Dense(50, 'relu')(layers_dense)

    outputs = keras.layers.Dense(1)(layers_dense)

    model = keras.Model(inputs=inputs, outputs=outputs, name=name)
	return model
```

```python
model = create_nn(input_shape=(X_data.shape[1],), name='smaller_model')
```



## 2.2 Exercise: Simplify the model and add data
You may have been wondering why we are including weather observations from
multiple cities to predict sunshine hours only in Basel. The weather is
a complex phenomenon with correlations over large distances and time scales,
but what happens if we limit ourselves to only one city?

1. Since we will be reducing the number of features quite significantly,
we could afford to include more data. Instead of using only 3 years, use
8 or 9 years!
2. Only use the features in the dataset that are for Basel, remove the data for other cities.
You can use something like:
```python
cols = [c for c in X_data.columns if c[:5] == 'BASEL']
X_data = X_data[cols]
```
3. Now rerun the last model we defined which included the BatchNorm layer.
Recreate the scatter plot comparing your predictions with the true values,
and evaluate the model by computing the RMSE on the test score.
Note that even though we will use many more observations than previously,
the network should still train quickly because we reduce the number of
features (columns).
Is the prediction better compared to what we had before?
4. (Optional) Try to train a model on all years that are available,
and all features from all cities. How does it perform?




## 🧠 Collaborative Notes

### 9. Refine the model
We continue where we were left yesterday. We are now going to refine the model, which actually entails going through the whole cycle (steps 1-8) again.

We are going to train a new model. And will validate it.

```python
model = create_nn(input_shape=(X_data.shape[1],),name='first_approach_with_validation') #we are using a function we created yesterday

```

```python
compile_model(model)
```
Now we fit the model

```python
history = model.fit(X_train, y_train,
                   batch_size=32,
                   epochs=200,
                   validation_data=(X_val, y_val))

```
Let's plot it in order to understand what is going on.

```python
plot_history(history, ['root_mean_squared_error', 'val_root_mean_squared_error'])
```

![](https://codimd.carpentries.org/uploads/upload_c46f8f2d5e7a1bb558629d4888d64a5e.png)


#### Exercise 2.1 see ^^

Early stopping. Stop when things are looking best. 

```python
model = create_nn(input_shape=(X_data.shape[1],), name='model_with_early_stopping')
compile_model(model)
```


```python
from tensorflow.keras.callbacks import EarlyStopping
earlystopper = EarlyStopping(monitor='val_loss', patience=10)
```


```python
history = model.fit(X_train, y_train, batch_size=32,epochs=200, validation_data=(X_val,y_val), ccallbacks=[earlystopper])

```
Let us plot the history

```python
plot_history(history, ['root_mean_squared_error', 'val_root_mean_squared_error'])
```

Rule of thumb, the more you evaluate the better (do early stopping after you plotted the it with more epochs)


```python
model.evaluate(X_test,y_test)
```

**BREAK**

We are leaving the topic of overfitting for the moment. In the afternoon we will pay additional attention to this. 

As a next topic we are focussing at batch normalization

### Add a Batch Norm layer (standard scaler for deep learning)


```python
def create_nn(input_shape, name="weather_prediction_model"):
    inputs = keras.Input(shape=input_shape, name='input')
    batch_norm = keras.layers.BatchNormalization()(inputs)
    layers_dense = keras.layers.Dense(100, 'relu')(batch_norm)
    layers_dense = keras.layers.Dense(50, 'relu')(layers_dense)

    outputs = keras.layers.Dense(1)(layers_dense)

    model = keras.Model(inputs=inputs, outputs=outputs, name=name)
    return model
```


```python
model = create_nn(input_shape=(X_data.shape[1],), name='model_with_batch_norm')
```


```python
compile_model(model)
```


```python
model.summary()
```


```python
history = model.fit(X_train, y_train,
                    batch_size=32,
                    epochs=200,
                    validation_data=(X_val, y_val),
                    callbacks=[earlystopper])
```


```python
model.evaluate(X_test,y_test)
```

Exercise 2.2 see ^^


### 10. Save the model

```python
model.save('my_amazing_model_keras_v01')
```




#### (Extra) Adding data from the past week
Here is a solution to add the data from the past $n$ days as features to train on.
```python

# load raw data
data = pd.read_csv("../weather_prediction_dataset_light.csv")

# keep only month & Basel features
data1 = data.drop(columns=['DATE'])
basel_only = (data1.columns.str[:5] == 'BASEL')
data1 = data1.iloc[:, basel_only]
data1.insert(0, 'MONTH', data['MONTH'])

# combining data from the past 7 days
upto = 7
df = pd.DataFrame(index=data1.index, data={'MONTH': data1['MONTH']})
for i in range(upto):
    # add data from n=0 days ago
    if i == 0:
        df[[f'{c}_{i}daysago' for c in data1.columns[1:]]] = data1.to_numpy()[:, 1:]
        continue
    
    # add data from n>=1 by shifting dataframe by n rows
    df[[f'{c}_{i}daysago' for c in data1.columns[1:]]] = np.vstack([
        np.full((i, data1.shape[1] - 1), np.nan),
        data1.to_numpy()[:-i, 1:]
    ])
    
df = df.dropna()

# remove first few entries
df = df.dropna()


# split features and target
X_data = df.drop(columns=['BASEL_sunshine_0daysago'])
y_data = df['BASEL_sunshine_0daysago']
```


## 📚 Resources
- [McFly: a simple deep learning tool for time series classification and regression](https://github.com/NLeSC/mcfly)

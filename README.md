# BeyondTheBuzz-Task-Submission  
### Objective:  
To train a binary classifier model that given the parameters, identifies if a transaction is fraudulent or not.  
### Working of the code:  
**1. Reading and preprocessing data:**    
We first read the csv file using the `pandas` library and then check if any null values are present or not. Here, there are no null values present. Then we separate our dataframe into our independent variable, X conataining the parameters and dependent variable y containing the verdict. Then we use the `train_test_split()` method of `sklearn.model_selection` to  randomly split our dataset into training and testing dataset. They have been aptly named as `X_train`, `X_test`, `y_train` and `y_test`. On studying the dataset, we can find that all parameters are not properly scaled, which can wrongly influence the machine while it's learning, leading it to make biased predictions. 

Thus, to prevent that Standardization technique has been implemented that scales down `X_train` and `X_test` using `StandardScaler` to bring uniformity in the dataset.

**2. Creating model using Neural Network:**   
To create the Neural Network I've used the `tensorflow` library and `keras`. We use the `Sequential` class to group our layers in to a linear stack. Then using the `Dense` we're implementing the layers of neural network. `Dense` takes the matrix product of input and weight matrix, adds the bias matrix and performs the specified *activation function* on the result. The activation function for the hidden layers has been chosen to be *relu* while output layer has a activation function of *sigmoid* to scale down values of activation between 0 and 1. *Relu* has been chosen because it has been found to be most optimal in this case. The neural network model created in code has 9 neurons in input layer, 64 and 32 respectively in the hidden layers and 1 neuron in the output layer.
During compiling, we use the loss function as ***binary_crossentropy*** . It is found to be optimal when we have to classify into 2 categories. We also use the `adam` optimizer which stands for **Adaptive Moment Estimation** based on stochastic gradient descent method where instead of training the model on the whole dataset, we train it on small parts of it, to optimize memory and time. To check the accuracy of training, we used the `metric` as `'accuracy'`.  
Then we use the `fit()` function to train our model for 50 epochs and batchsize of 32.   

**3. Predicting using model:**   
We use the `predict()` function to make a prediction of our X_test data and then convert it into binary on whether the probability is > 0.5 or not. This we store in `binary_predictions`. We use `Accuracy` class of `tf.keras.metrics` to compare `y_test` and `binary_predictions` and give the accuracy. Here, the accuracy comes out to be about 94.25%.

**4. Prediction of test.csv using model:**   
We read our test.csv file into `X_submit` after dropping the 'Id' column. After rewriting the `binary_predictions`, we create a dictionary with our `binary_predictions` and `Id` columns. Then we convert the dictionary to a `pandas` Dataframe and then convert it to a `predictions.csv` file which has been uploaded here. 

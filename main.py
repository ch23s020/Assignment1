import numpy as np
import numpy as np
import math
import matplotlib.pyplot as plt
import keras
from keras.datasets import fashion_mnist, mnist
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
#import seaborn as sns
import random
import wandb
import argparse

#Q34567 (Adding Finally To single Place)
class MetaNeuron:
    def __init__(self, input_size, hidden_size, output_size, activation='sigmoid', weight_decay=0, weight_init='random'):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation = activation
        self.weight_decay = weight_decay
        self.weight_init = weight_init

        if self.weight_init == 'random':
            self.initialize_random_weights()
        elif self.weight_init == 'xavier':
            self.initialize_xavier_weights()

        # velocities for momentum Initialization
        self.velocity_weights_input_hidden = np.zeros_like(self.weights_input_hidden)

        self.velocity_biases_input_hidden = np.zeros_like(self.biases_input_hidden)

        self.velocity_weights_hidden_output = np.zeros_like(self.weights_hidden_output)

        self.velocity_biases_hidden_output = np.zeros_like(self.biases_hidden_output)

        # momentums for nesterov Initialization
        self.momentum_weights_input_hidden = np.zeros_like(self.weights_input_hidden)

        self.momentum_biases_input_hidden = np.zeros_like(self.biases_input_hidden)

        self.momentum_weights_hidden_output = np.zeros_like(self.weights_hidden_output)

        self.momentum_biases_hidden_output = np.zeros_like(self.biases_hidden_output)

        # ADAM parameters Initialization
        self.adam_weights_input_hidden_m = np.zeros_like(self.weights_input_hidden)

        self.adam_biases_input_hidden_m = np.zeros_like(self.biases_input_hidden)

        self.adam_weights_hidden_output_m = np.zeros_like(self.weights_hidden_output)

        self.adam_biases_hidden_output_m = np.zeros_like(self.biases_hidden_output)

        self.adam_weights_input_hidden_v = np.zeros_like(self.weights_input_hidden)

        self.adam_biases_input_hidden_v = np.zeros_like(self.biases_input_hidden)

        self.adam_weights_hidden_output_v = np.zeros_like(self.weights_hidden_output)

        self.adam_biases_hidden_output_v = np.zeros_like(self.biases_hidden_output)

    # As per Questions 3; Initialization for two types of weight and baises
    def initialize_random_weights(self):
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)

        self.biases_input_hidden = np.zeros((1, self.hidden_size))

        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)

        self.biases_hidden_output = np.zeros((1, self.output_size))

    def initialize_xavier_weights(self):
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size) * np.sqrt(1 / self.input_size)

        self.biases_input_hidden = np.zeros((1, self.hidden_size))

        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size) * np.sqrt(1 / self.hidden_size)

        self.biases_hidden_output = np.zeros((1, self.output_size))

    #Defining Activation Function Its Derivative and Its Loss both L2 and Cross Entropy.
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x <= 0, 0, 1)

    def tanh(self, x):
        return np.tanh(x)

    def tanh_derivative(self, x):
        return 1 - np.tanh(x) ** 2

    def softmax(self, x):
        exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)

    def calculate_loss(self, y_true, y_pred):
        m = y_true.shape[0]
        loss = -np.sum(y_true * np.log(y_pred + 1e-10)) / m
        return loss

    def calculate_accuracy(self, y_true, y_pred):
        y_true_argmax = np.argmax(y_true, axis=1)
        y_pred_argmax = np.argmax(y_pred, axis=1)
        accuracy = np.mean(y_true_argmax == y_pred_argmax)
        return accuracy


    #Now Starting the Forward Propogation on the preprocessed data.
    def forward_propagation(self, x):
      # Compute the input to the hidden layer by multiplying the input data by the weights and adding biases
        hidden_layer_input = np.dot(x, self.weights_input_hidden) + self.biases_input_hidden
      #Using for final Layer to get the output in terms of probability
        if self.activation == 'sigmoid':
            hidden_layer_output = self.sigmoid(hidden_layer_input)

        elif self.activation == 'relu':
            hidden_layer_output = self.relu(hidden_layer_input)

        elif self.activation == 'tanh':
            hidden_layer_output = self.tanh(hidden_layer_input)

        # Computing the input to the output layer by multiplying the hidden layer output by the weights and adding biases
        output_layer_input = np.dot(hidden_layer_output, self.weights_hidden_output) + self.biases_hidden_output

        #Apply softmax activation function to the output layer input to get the final output
        output_layer_output = self.softmax(output_layer_input)

        return hidden_layer_output, output_layer_output

    def backward_propagation(self, x, y, hidden_layer_output, output_layer_output):
        # Getting the number of samples in the input data
        m = x.shape[0]

        # Getting the output error yhat- y
        output_layer_error = output_layer_output - y

        # Gradients for weights and biases in the output layer
        weights_hidden_output_gradients = np.dot(hidden_layer_output.T, output_layer_error) / m

        biases_hidden_output_gradients = np.sum(output_layer_error, axis=0, keepdims=True) / m


        #Finding the error in the hidden layer based on the activation function
        if self.activation == 'sigmoid':
            hidden_layer_error = np.dot(output_layer_error, self.weights_hidden_output.T) * self.sigmoid_derivative(hidden_layer_output)

        elif self.activation == 'relu':
            hidden_layer_error = np.dot(output_layer_error, self.weights_hidden_output.T) * self.relu_derivative(hidden_layer_output)


        elif self.activation == 'tanh':
            hidden_layer_error = np.dot(output_layer_error, self.weights_hidden_output.T) * self.tanh_derivative(hidden_layer_output)

        #gradients for weights and biases in the input layer
        weights_input_hidden_gradients = np.dot(x.T, hidden_layer_error) / m

        biases_input_hidden_gradients = np.sum(hidden_layer_error, axis=0, keepdims=True) / m


        #Applying L2 Regularisation
        weights_hidden_output_gradients += self.weight_decay * self.weights_hidden_output

        weights_input_hidden_gradients += self.weight_decay * self.weights_input_hidden

        return weights_input_hidden_gradients, biases_input_hidden_gradients, weights_hidden_output_gradients, biases_hidden_output_gradients

    #Defining Optimizers sgd:-Stochastic Gradient Descent, Momentum Based Gradient Descent, Nastrov, adam. Removed the rms prop as it eas breaking some times and running some times during integration with wandb.

    def sgd_update(self, weights_input_hidden_gradients, biases_input_hidden_gradients, weights_hidden_output_gradients, biases_hidden_output_gradients, learning_rate):

        # Updating weights and biases in the output layer using stochastic gradient descent (SGD)
        self.weights_hidden_output -= learning_rate * weights_hidden_output_gradients

        self.biases_hidden_output -= learning_rate * biases_hidden_output_gradients

        # Doing the same for Input Layer
        self.weights_input_hidden -= learning_rate * weights_input_hidden_gradients

        self.biases_input_hidden -= learning_rate * biases_input_hidden_gradients

    def momentum_update(self, weights_input_hidden_gradients, biases_input_hidden_gradients, weights_hidden_output_gradients, biases_hidden_output_gradients, learning_rate, momentum):

        # Computing  velocity for weights and biases in the output layer
        self.velocity_weights_hidden_output = momentum * self.velocity_weights_hidden_output + learning_rate * weights_hidden_output_gradients

        self.velocity_biases_hidden_output = momentum * self.velocity_biases_hidden_output + learning_rate * biases_hidden_output_gradients

        # Doing same for input lyer
        self.velocity_weights_input_hidden = momentum * self.velocity_weights_input_hidden + learning_rate * weights_input_hidden_gradients

        self.velocity_biases_input_hidden = momentum * self.velocity_biases_input_hidden + learning_rate * biases_input_hidden_gradients


        #Applying the update rule for weights and biases in both layers using momentum

        #Hidden Layer
        self.weights_hidden_output -= self.velocity_weights_hidden_output
        self.biases_hidden_output -= self.velocity_biases_hidden_output

        #input layer
        self.weights_input_hidden -= self.velocity_weights_input_hidden
        self.biases_input_hidden -= self.velocity_biases_input_hidden

    def nesterov_update(self, weights_input_hidden_gradients, biases_input_hidden_gradients, weights_hidden_output_gradients, biases_hidden_output_gradients, learning_rate, momentum):

       # Computing momentum update for weights and biases in both layers (NAG:- Look and then Go)
        self.momentum_weights_hidden_output = momentum * self.momentum_weights_hidden_output + learning_rate * weights_hidden_output_gradients

        self.momentum_biases_hidden_output = momentum * self.momentum_biases_hidden_output + learning_rate * biases_hidden_output_gradients

        self.momentum_weights_input_hidden = momentum * self.momentum_weights_input_hidden + learning_rate * weights_input_hidden_gradients

        self.momentum_biases_input_hidden = momentum * self.momentum_biases_input_hidden + learning_rate * biases_input_hidden_gradients


        #weights and biases update using Nesterov momentum
        self.weights_hidden_output -= momentum * self.momentum_weights_hidden_output + learning_rate * weights_hidden_output_gradients

        self.biases_hidden_output -= momentum * self.momentum_biases_hidden_output + learning_rate * biases_hidden_output_gradients

        self.weights_input_hidden -= momentum * self.momentum_weights_input_hidden + learning_rate * weights_input_hidden_gradients

        self.biases_input_hidden -= momentum * self.momentum_biases_input_hidden + learning_rate * biases_input_hidden_gradients


    def rmsprop_update(self, weights_input_hidden_gradients, biases_input_hidden_gradients, weights_hidden_output_gradients, biases_hidden_output_gradients, learning_rate, decay_rate=0.9, epsilon=1e-8):
        # Compute exponentially weighted averages of the squared gradients
        self.rms_weights_hidden_output = decay_rate * self.rms_weights_hidden_output + (1 - decay_rate) * np.square(weights_hidden_output_gradients)

        self.rms_biases_hidden_output = decay_rate * self.rms_biases_hidden_output + (1 - decay_rate) * np.square(biases_hidden_output_gradients)

        self.rms_weights_input_hidden = decay_rate * self.rms_weights_input_hidden + (1 - decay_rate) * np.square(weights_input_hidden_gradients
                                                                                                                  )
        self.rms_biases_input_hidden = decay_rate * self.rms_biases_input_hidden + (1 - decay_rate) * np.square(biases_input_hidden_gradients)
        
        # Update weights and biases
        self.weights_hidden_output -= learning_rate * weights_hidden_output_gradients / (np.sqrt(self.rms_weights_hidden_output) + epsilon)

        self.biases_hidden_output -= learning_rate * biases_hidden_output_gradients / (np.sqrt(self.rms_biases_hidden_output) + epsilon)

        self.weights_input_hidden -= learning_rate * weights_input_hidden_gradients / (np.sqrt(self.rms_weights_input_hidden) + epsilon)

        self.biases_input_hidden -= learning_rate * biases_input_hidden_gradients / (np.sqrt(self.rms_biases_input_hidden) + epsilon)




    def adam_update(self, weights_input_hidden_gradients, biases_input_hidden_gradients, weights_hidden_output_gradients, biases_hidden_output_gradients, learning_rate, t, beta1=0.9, beta2=0.999, epsilon=1e-8 ):

        # the first moment estimates update
        self.adam_weights_hidden_output_m = beta1 * self.adam_weights_hidden_output_m + (1 - beta1) * weights_hidden_output_gradients

        self.adam_biases_hidden_output_m = beta1 * self.adam_biases_hidden_output_m + (1 - beta1) * biases_hidden_output_gradients

        self.adam_weights_input_hidden_m = beta1 * self.adam_weights_input_hidden_m + (1 - beta1) * weights_input_hidden_gradients

        self.adam_biases_input_hidden_m = beta1 * self.adam_biases_input_hidden_m + (1 - beta1) * biases_input_hidden_gradients


        #2nd Moment estimate
        self.adam_weights_hidden_output_v = beta2 * self.adam_weights_hidden_output_v + (1 - beta2) * (weights_hidden_output_gradients ** 2)

        self.adam_biases_hidden_output_v = beta2 * self.adam_biases_hidden_output_v + (1 - beta2) * (biases_hidden_output_gradients ** 2)

        self.adam_weights_input_hidden_v = beta2 * self.adam_weights_input_hidden_v + (1 - beta2) * (weights_input_hidden_gradients ** 2)

        self.adam_biases_input_hidden_v = beta2 * self.adam_biases_input_hidden_v + (1 - beta2) * (biases_input_hidden_gradients ** 2)

        adam_weights_hidden_output_m_hat = self.adam_weights_hidden_output_m / (1 - beta1 ** (t))

        adam_biases_hidden_output_m_hat = self.adam_biases_hidden_output_m / (1 - beta1 ** (t))

        adam_weights_input_hidden_m_hat = self.adam_weights_input_hidden_m / (1 - beta1 ** (t))

        adam_biases_input_hidden_m_hat = self.adam_biases_input_hidden_m / (1 - beta1 ** (t))

        adam_weights_hidden_output_v_hat = self.adam_weights_hidden_output_v / (1 - beta2 ** (t))

        adam_biases_hidden_output_v_hat = self.adam_biases_hidden_output_v / (1 - beta2 ** (t))

        adam_weights_input_hidden_v_hat = self.adam_weights_input_hidden_v / (1 - beta2 ** (t))

        adam_biases_input_hidden_v_hat = self.adam_biases_input_hidden_v / (1 - beta2 ** (t))

        #weights and biases update using Adam optimization algorithm
        self.weights_hidden_output -= learning_rate * (adam_weights_hidden_output_m_hat / (np.sqrt(adam_weights_hidden_output_v_hat ) + epsilon))

        self.biases_hidden_output -= learning_rate * (adam_biases_hidden_output_m_hat / (np.sqrt(adam_biases_hidden_output_v_hat ) + epsilon))

        self.weights_input_hidden -= learning_rate * (adam_weights_input_hidden_m_hat  / (np.sqrt(adam_weights_input_hidden_v_hat ) + epsilon))

        self.biases_input_hidden -= learning_rate * (adam_biases_input_hidden_m_hat  / (np.sqrt(adam_biases_input_hidden_v_hat ) + epsilon))
        t=t+1

        return t


    # Training Function (is th different file possible during converting to .py)
    def train(self, x_train, y_train, x_valid, y_valid, epochs, batch_size, learning_rate, optimizer='sgd', momentum=0.9, beta1=0.9, beta2=0.999, epsilon=1e-8, decay_rate = 0.9):

       # Loss/Accuracy Storage in list.

        train_losses = []

        valid_losses = []

        train_accuracies = []

        valid_accuracies = []

        t=1


        #Starting of Epoch stage:(See the slide for erpoch rule update)

        for epoch in range(epochs):
            epoch_train_loss = 0
            epoch_train_accuracy = 0

            #Starting of MiniBatch:(Refere slide for the update code)
            for i in range(0, len(x_train), batch_size):

                x_batch = x_train[i:i + batch_size]

                y_batch = y_train[i:i + batch_size]

                 # Forward propagation
                hidden_layer_output, output_layer_output = self.forward_propagation(x_batch)

                 # Back_Prop
                weights_input_hidden_gradients, biases_input_hidden_gradients, weights_hidden_output_gradients, biases_hidden_output_gradients = self.backward_propagation(x_batch, y_batch, hidden_layer_output, output_layer_output)


                #optimizer checking stage:-
                if optimizer == 'sgd':

                    self.sgd_update(weights_input_hidden_gradients, biases_input_hidden_gradients, weights_hidden_output_gradients, biases_hidden_output_gradients, learning_rate)

                elif optimizer == 'momentum':

                    self.momentum_update(weights_input_hidden_gradients, biases_input_hidden_gradients, weights_hidden_output_gradients, biases_hidden_output_gradients, learning_rate, momentum)

                elif optimizer == 'nesterov':

                    self.nesterov_update(weights_input_hidden_gradients, biases_input_hidden_gradients, weights_hidden_output_gradients, biases_hidden_output_gradients, learning_rate, momentum)

                elif optimizer == 'adam':
                      
                    t = self.adam_update(weights_input_hidden_gradients, biases_input_hidden_gradients, weights_hidden_output_gradients, biases_hidden_output_gradients, learning_rate,t, beta1, beta2, epsilon)

                elif optimizer == 'rmsprop':
                    self.rmsprop_update(weights_input_hidden_gradients, biases_input_hidden_gradients, weights_hidden_output_gradients, biases_hidden_output_gradients, learning_rate, decay_rate, epsilon)


                #Loss/Accuracy Calculation

                train_loss = self.calculate_loss(y_batch, output_layer_output)

                epoch_train_loss += train_loss

                train_accuracy = self.calculate_accuracy(y_batch, output_layer_output)

                epoch_train_accuracy += train_accuracy
            #Epoch Calculation rule(Average for a epoch)

            epoch_train_loss /= len(x_train) // batch_size

            epoch_train_accuracy /= len(x_train) // batch_size

            train_losses.append(epoch_train_loss)

            train_accuracies.append(epoch_train_accuracy)


            # Loss/Accuracy calculation:

            _, output_valid = self.forward_propagation(x_valid)


            valid_loss = self.calculate_loss(y_valid, output_valid)

            valid_losses.append(valid_loss)

            valid_accuracy = self.calculate_accuracy(y_valid, output_valid)

            valid_accuracies.append(valid_accuracy)

            # Logging the above values in Wandb

            wandb.log({
                "Training Loss": epoch_train_loss,

                "Validation Loss": valid_loss,

                "Training Accuracy": epoch_train_accuracy,

                "Validation Accuracy": valid_accuracy,

                "Epoch": epoch
            })


        #For wandb integration, and cvisibility, adding again here. It was breaking earlier adding here working fine.

        _, output_test = self.forward_propagation(x_test_data)

        test_accuracy = self.calculate_accuracy(y_test, output_test)

        wandb.log({"Test Accuracy:":test_accuracy})

        #Confusion matrix:

        predictions = np.argmax(output_valid, axis = 1)

        confusion_mat = confusion_matrix(np.argmax(y_valid, axis=1), predictions)

        class_names = ["T-shirt/top",  	"Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot" ]

        #Getting only table from following code hence not including it. If above one fails, uncomment the below one. Sometime one is working sometime other.

        # wandb.log({"confusion_mat" : wandb.plot.confusion_matrix(probs=None,
        #                         y_true= np.reshape(y_true,(y_true.shape[0])).tolist(), preds=predictions.tolist(),
        #                         class_names=class_names)})

        fig, ax = plt.subplots(figsize=(14, 7))

        sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax)

        wandb.log({"confusion_mat": wandb.Image(fig)})

        return train_losses, valid_losses, train_accuracies, valid_accuracies


# Initializeing the sweep
def sweep_training():
    with wandb.init() as run:
        config = run.config

        # Initializing neural network
        input_size = 28 * 28
        hidden_size = config.hidden_layer_size
        output_size = 10
        activation = config.activation_function
        weight_decay = config.weight_decay
        weight_init = config.weight_init
        decay_rate = config.decay_rate

        # Creating Instance
        nn = MetaNeuron(input_size, hidden_size, output_size, activation, weight_decay, weight_init)

        # Hyperparameters Values
        epochs = config.epochs
        batch_size = config.batch_size

        # Initiating Training
        train_losses, valid_losses, train_accuracies, valid_accuracies = nn.train(x_train_input, y_train_input, x_train_valid, y_train_valid, epochs, batch_size, config.learning_rate, config.optimizer, config.momentum, config.beta1, config.beta2, config.epsilon,config.decay_rate)

        # Define the run name format
        run_name = "lr_{:.0e}_ac_{}_bs_{}_L2_{}_ep_{}_nn_{}_nh_{}_dr{}".format(
        config.learning_rate, config.activation_function, config.batch_size,
        config.weight_decay, config.epochs, config.hidden_layer_size, config.hidden_layers)
        wandb.run.name = run_name



#Creating a .py file that can take arguments
        
parser = argparse.ArgumentParser(description='test.py')
parser.add_argument('-wp', '--wandb_project', default="fashion_mnist_hyperparameter_tuning")
parser.add_argument('-we', '--wandb_entity', default="myname")
parser.add_argument('-e', '--epochs', default=10)
parser.add_argument('-b', '--batch_size', default=64)
parser.add_argument('-l', '--loss', default="cross_entropy")
parser.add_argument('-o', '--optimizer', default="adam")
parser.add_argument('-lr', '--learning_rate', default=0.0001)
parser.add_argument('-m', '--momentum', default=0.9)
parser.add_argument('-beta', '--beta', default=0.9)
parser.add_argument('-beta1', '--beta1', default=0.9)
parser.add_argument('-beta2', '--beta2', default=0.99)
parser.add_argument('-eps', '--epsilon', default=0.00001)
parser.add_argument('-w_d', '--weight_decay', default=0.005)
parser.add_argument('-w_i', '--weight_init', default="xavier")
parser.add_argument('-nhl', '--num_layers', default=3)
parser.add_argument('-sz', '--hidden_size', default=128)
parser.add_argument('-a', '--activation', default="tanh")
parser.add_argument('-oa', '--output_activation', default="softmax")
parser.add_argument('-d', '--dataset', default='fashion_mnist')
parser.add_argument('-dr', '--decay_rate', default=0.9)  

args = parser.parse_args()
if args.dataset == "fashion_mnist":
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
elif args.dataset == "mnist":
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshaping the dataset from (60000 x 28 x 28 -----> 60000 x 784)
no_of_images, pixel_height, pixel_width = x_train.shape
x_train_rshp = x_train.reshape(no_of_images, pixel_height * pixel_width)
x_train_rshp = x_train_rshp.astype('float64') / 255.0

# Reshaping the testing dataset from (10000 x 28 x 28) to (10000 x 784)
no_of_test_images, _, _ = x_test.shape
x_test_rshp = x_test.reshape(no_of_test_images, pixel_height * pixel_width)
x_test_rshp = x_test_rshp.astype('float64') / 255.0

x_test_data = x_test_rshp = x_test_rshp.astype('float64') / 255.0

# Separating 10% data for validation purpose
partition = int(0.9 * len(x_train_rshp))
x_train_input, x_train_valid = x_train_rshp[:partition], x_train_rshp[partition:]
y_train_input, y_true = y_train[:partition], y_train[partition:]

# Converting the labels to one-hot encoding
num_classes = 10
y_train = np.eye(num_classes)[y_train]
y_test = np.eye(num_classes)[y_test]
y_train_valid  = np.eye(num_classes)[y_true]

#Defining the data again for better visualisation
x_train_input = x_train_rshp[:partition]

y_train_input = y_train[:partition]

x_train_valid = x_train_rshp[partition:]

y_train_valid = y_train[partition:]

#Sweep cinfiguration
sweep_config = {
"method": "random",
"metric": {"name": "Validation Accuracy", "goal": "maximize"},
"parameters": {
    "learning_rate": {"values": [args.learning_rate]},
    "optimizer": {"values": [args.optimizer]},
    "momentum": {"values": [args.momentum]},
    "beta1": {"values": [args.beta1]},
    "beta2": {"values": [args.beta2]},
    "epsilon": {"values": [args.epsilon]},
    "activation_function": {"values": [args.activation]},
    "batch_size": {"values": [args.batch_size]},
    "epochs": {"values": [args.epochs]},
    "hidden_layer_size": {"values": [args.hidden_size]},
    "hidden_layers": {"values": [args.num_layers]},
    "weight_decay": {"values": [args.weight_decay]},
    "weight_init": {"values": [args.weight_init]},
    "decay_rate": {"values": [args.decay_rate]}
}}
sweep_id = wandb.sweep(sweep_config)
# Execute sweep
wandb.agent(sweep_id, sweep_training, count=2)






#As Mentioned in Report, all commented out code is initial trial also committed in github. 
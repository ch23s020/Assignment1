
**There is separate file for Q2.
The main.py file contains all the code from Q3 and Onwards**
**Please refer main.Py and Q2.py files for passing the arguments.****

**There were multiple files of initial commits. The Copy_of_Copy_of_Assignment1 is one among such file. If the main.py fails to take argument, Please refer google colab file Copy_of_Copy_of_Assignment1.

**Code Structure**
Code is in following parts:- 
1) Pip Install wandb
2) Data Processing
3) Question 2 to View The Images
4) An Single class MetaNeuron containing all the stages as follows:
   a) Initialsing Variables/Hyperparameters to Class
   b) Defining Activation Function and Its derivative and Loss Functions
   c) Forward  and Back Propogation.
   d) Defining Optimizers
   e) Training Function.
Training Function contains minibatch approach. Wandb Log

At the final Wandb initialisation is done.

It is possible that the converted .py file may not run properly as I am not aware of it completely. Hence, I request to run the google colab file to check the final output and to test for any added functionality.

During the Run I used two Wandb account for testing purposes as it is creating error
1) One:- Username:- CH23S020(My Roll No) from Smail
2) username:- rrjadhao27(Personal Mail id)

Hence you might see reports from both the account. All are for the same above mentioned stages in the same order.


**Preperations Stage **
# Assignment1
1.prepare data for n-dimensional input
2. passing this data to a sigmoid function---funct1
3. passing the ouyput of sigmoid into a softmax function to get the probability distri output---funct2
4. this will be y_hat
5. compare this y_hat with y-original(one hot vector)
6. for comaparison use of cross entropy(negative log-likelihood function)
7. calculate loss

Back Prop:-
1. calculate derivative of loss wrt y-hat
2. derivative of y-hat wrt funct2 (detailing of weight and preactivation at each layer use sirs slide)
3.funct2 wrt funct1
4. funct1 wrt to input weights

update:
1. use of gradient descent algo to update the w using the calculated dw(dl/dw)
2. pass the new w into forward prop again.

repeat:
1.repeat the same process for all data points (53999) for training purpose.

2.validate over validation data (54000 60000)

Set Up Wandb.

modification:-

1.Once done use other algorith for adam namdam sgd into main algorithm.
2. repaeat all procedure of running forward and back with added algo.

Finally push to wandb and see for correlation graph and other mentioned stuff in assignment.






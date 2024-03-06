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




## The Pretrained Models

There are currently two models I have already trained: v1 and v2.

### Version 1 - `97.5%` Accuracy

ReLu activation for the hidden layers, softmax activation for the output layer. Cross-Entropy Loss as loss function.

Hyperparameters: 

* 1800 Epochs
* 0.05 Learning Rate
* 1 input 
* 3 hidden layers - 128 nodes per hidden layer
* 10 output nodes - one per each possible digit

### Version 2 -  `98.0%` Accuracy 

Second attempt at training a model, now added a linearly decaying learning rate. The accuracy didn't improve very much since the first attempt was already very good if I can say so myself.
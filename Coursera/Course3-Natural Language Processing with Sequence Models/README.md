#### Word Embedding 


import trax.fastmath.numpy as np. If you see this line, remember that when calling np you are really calling Trax’s version of numpy that is compatible with JAX. s a result of this, where you used to encounter the type numpy.ndarray now you will find the type jax.interpreters.xla.DeviceArray.




```python
#-----------------------------------------------Install Trax-----------------------------------------------------
!pip install trax

import numpy as np  # regular ol' numpy

from trax import layers as tl  # core building block
from trax import shapes  # data signatures: dimensionality and type
from trax import fastmath  # uses jax, offers numpy on steroids

#-----------------------------------------------Relu Layer-----------------------------------------------------

relu = tl.Relu()

# Inspect properties
print("name :", relu.name) #name : Relu
print("expected inputs :", relu.n_in)  # expected inputs : 1
print("promised outputs :", relu.n_out, "\n") # promised outputs : 1 


x = np.array([-2, -1, 0, 1, 2])
print("x :", x, "\n") #x : [-2 -1  0  1  2] 
y = relu(x)
print("y :", y) y : [0 0 0 1 2]

#-----------------------------------------------Concatenate Layer-----------------------------------------------------

concat = tl.Concatenate()
print("name :", concat.name) #name : Concatenate
print("expected inputs :", concat.n_in) # expected inputs : 2
print("promised outputs :", concat.n_out, "\n") # promised outputs : 1 

# Inputs
x1 = np.array([-10, -20, -30])
x2 = x1 / -10
y = concat([x1, x2]) 
print("y :", y) # y : [-10. -20. -30.   1.   2.   3.]


#For example, you can change the expected inputs for a concatenate layer from 2 to 3 using the optional parameter n_items.

concat_3 = tl.Concatenate(n_items=3)  # configure the layer's expected inputs
print("name :", concat_3.name) # name : Concatenate
print("expected inputs :", concat_3.n_in) # expected inputs : 3
print("promised outputs :", concat_3.n_out, "\n") # promised outputs : 1 

# Inputs
x1 = np.array([-10, -20, -30])
x2 = x1 / -10
x3 = x2 * 0.99
y = concat_3([x1, x2, x3])
print("y :", y) # y : [-10.   -20.   -30.     1.     2.     3.     0.99   1.98   2.97]

help(tl.Concatenate) # see the function docstring with explaination


#-----------------------------------------------LayerNorm & shapes.signature -----------------------------------------------------
help(tl.LayerNorm)
help(shapes.signature)
# Layer initialization
norm = tl.LayerNorm()
# You first must know what the input data will look like
x = np.array([0, 1, 2, 3], dtype="float")

# Use the input data signature to get shape and type for initializing weights and biases
norm.init(shapes.signature(x)) # We need to convert the input datatype from usual tuple to trax ShapeDtype

print("Normal shape:",x.shape, "Data Type:",type(x.shape)) #Normal shape: (4,) Data Type: <class 'tuple'>
print("Shapes Trax:",shapes.signature(x),"Data Type:",type(shapes.signature(x))) #Shapes Trax: ShapeDtype{shape:(4,), dtype:float64} Data Type: <class 'trax.shapes.ShapeDtype'>

print("name :", norm.name) # name : LayerNorm
print("expected inputs :", norm.n_in) # expected inputs : 1
print("promised outputs :", norm.n_out) # promised outputs : 1
 
# Weights and biases
print("weights :", norm.weights[0]) # weights : [1. 1. 1. 1.]
print("biases :", norm.weights[1], "\n") # biases : [0. 0. 0. 0.] 

x = np.array([0, 1, 2, 3], dtype="float")
y = norm(x)
print("y :", y) # y : [-1.3416404  -0.44721344  0.44721344  1.3416404 ]

#----------------------------------------------- Custom Layers -----------------------------------------------------
#This is where things start getting more interesting! 
# You can create your own custom layers too and define custom functions for computations by using tl.Fn.
help(tl.Fn)
#Fn(name, f, n_out=1) Returns a layer with no weights that applies the function `f`.
# e.g.         `Fn('SumAndMax', lambda x0, x1: (x0 + x1, jnp.maximum(x0, x1)), n_out=2)`
# you must explicitly set the number of outputs (`n_out`) whenever it's not the default value 1.


def TimesTwo():
    layer_name = "TimesTwo" #don't forget to give your custom layer a name to identify
    # Custom function for the custom layer
    def func(x):
        return x * 2
    return tl.Fn(layer_name, func)


# Test it
times_two = TimesTwo()

# Inspect properties
print("name :", times_two.name) # name : TimesTwo
print("expected inputs :", times_two.n_in) # expected inputs : 1
print("promised outputs :", times_two.n_out, "\n") # promised outputs : 1 

x = np.array([1, 2, 3])
y = times_two(x) # y : [2 4 6]
print("y :", y)

#----------------------------------------------- Combinators -----------------------------------------------------
# combine layers to build more complex layers. Trax provides a set of objects named combinator layers to make this happen.
#  Combinators are themselves layers, so behavior commutes.
 help(tl.Serial)
 help(tl.Parallel)

"""
Serial Combinator:

This is the most common and easiest to use. For example could build a simple neural network by combining layers into a single layer using the Serial combinator. This new layer then acts just like a single layer, so you can inspect inputs, outputs and weights. Or even combine it into another layer! Combinators can then be used as trainable models. Try adding more layers

Note:As you must have guessed, if there is serial combinator, there must be a parallel combinator as well. 
"""

# Serial combinator
serial = tl.Serial(
    tl.LayerNorm(),         # normalize input
    tl.Relu(),              # convert negative values to zero
    times_two,              # the custom layer you created above, multiplies the input recieved from above by 2
)

# Initialization
x = np.array([-2, -1, 0, 1, 2]) #input
serial.init(shapes.signature(x)) #initialising serial instance
print(serial,"\n")
"""
Serial[
  LayerNorm
  Relu
  TimesTwo
] 
"""
print("name :", serial.name) # name : Serial
print("sublayers :", serial.sublayers) #  sublayers : [LayerNorm, Relu, TimesTwo]
print("expected inputs :", serial.n_in) # expected inputs : 1
print("promised outputs :", serial.n_out) # promised outputs : 1
print("weights & biases:", serial.weights, "\n")  
# weights & biases: [(DeviceArray([1, 1, 1, 1, 1], dtype=int32), DeviceArray([0, 0, 0, 0, 0], dtype=int32)), (), ()] 

# Inputs
y = serial(x)
print("y :", y)  # y : [0.        0.        0.        1.4142132 2.8284264]


serial = tl.Serial(
    tl.LayerNorm(),         # normalize input
    tl.Relu(),              # convert negative values to zero
    times_two,              # the custom layer you created above, multiplies the input recieved from above by 2
    
     tl.Dense(n_units=2),  # try adding more layers. eg uncomment these lines
     tl.Dense(n_units=1),  # Binary classification, maybe? uncomment at your own peril
     tl.LogSoftmax()       # Yes, LogSoftmax is also a layer
)
print(serial,"\n")
"""
Serial[
  LayerNorm
  Relu
  TimesTwo
  Dense_2
  Dense_1
  LogSoftmax
] 
""" 
print("name :", serial.name)  # name : Serial
print("sublayers :", serial.sublayers) # sublayers : [LayerNorm, Relu, TimesTwo, Dense_2, Dense_1, LogSoftmax]
print("expected inputs :", serial.n_in) # expected inputs : 1
print("promised outputs :", serial.n_out) # promised outputs : 1
print("weights & biases:", serial.weights, "\n")
"""
weights & biases: [(DeviceArray([1, 1, 1, 1, 1], dtype=int32) #LayerNorm weight, DeviceArray([0, 0, 0, 0, 0] #LayerNorm bias, dtype=int32)), () #ReLu , () #TimesTwo, (DeviceArray([[-0.31837648, -0.8734224 ],
             [-0.20758191, -0.7621965 ],  #这个是Dense_2 weight
             [-0.16000253, -0.9212737 ],
             [ 0.7888365 , -0.5979883 ],
             [-0.2151707 ,  0.44562957]], dtype=float32), DeviceArray([ 4.3718984e-07, -1.1996210e-06], dtype=float32)) ##这个是Dense_2 bias, (DeviceArray([[-0.8521533],
             [-1.0344262]], dtype=float32) #Dense_2 weight, DeviceArray([1.5953614e-06] #Dense_2 bias, dtype=float32)), () # softmax] 


"""

serial = tl.Serial(
    tl.Relu()            # convert negative values to zero
)
print("sublayers :", serial.sublayers) # sublayers : [Relu]
print("weights & biases:", serial.weights, "\n") # weights & biases: [()] 


#----------------------------------------------- JAX -----------------------------------------------------
# Note:There are certain things which are still not possible in fastmath.numpy which can be done in numpy, 
# so you sometimes switch between them to get our work done.
x_numpy = np.array([1, 2, 3])
print("good old numpy : ", type(x_numpy), "\n") #good old numpy :  <class 'numpy.ndarray'> 

# Fastmath and jax numpy
x_jax = fastmath.numpy.array([1, 2, 3])
print("jax trax numpy : ", type(x_jax)) # jax trax numpy :  <class 'jax.interpreters.xla.DeviceArray'>


#----------------------------------------------- Training & Evaluation -----------------------------------------------------
# trax.supervised.training.TrainTask which packages the train data, loss and optimizer (among other things) together into an object.
help(trax.supervised.training.TrainTask)

# evaluate a model, Trax defines trax.supervised.training.EvalTask which packages the eval data and metrics (among other things) into another object.
help(trax.supervised.training.EvalTask)

#The final piece tying things together is the trax.supervised.training.Loop abstraction that is a very simple 
# and flexible way to put everything together and train the model, all the while evaluating it and saving checkpoints. 
# Using Loop will save you a lot of code compared to always writing the training loop by hand.
#  More importantly, you are less likely to have a bug in that code that would ruin your training.
help(trax.supervised.training.Loop)

help(trax.optimizers) #include adam, momentum, rms_prop

from trax.supervised import training

batch_size = 16
rnd.seed(271)

#注意下面np 不是numpy 而是 import trax.fastmath.numpy as np
inputs, targets, example_weights = np.array([[2005, 4451, 3201,    9,    0,    0,    0,    0,    0,    0,    0]
                                            [4954,  567, 2000, 1454, 5174, 3499,  141, 3499,  130,  459,    9],
                                            [3761,  109,  136,  583, 2930, 3969,    0,    0,    0,    0,    0],
                                           [ 250,  3761,    0,    0,    0,    0,    0,    0,    0,    0,    0]),
                                   np.array([1, 1, 0, 0]), np.array([1,1,1,1])
train_generator = inputs, targets, example_weights 

#This defines a model trained using tl.CrossEntropyLoss optimized with the trax.optimizers.Adam optimizer, 
# all the while tracking the accuracy using tl.Accuracy metric. We also track tl.CrossEntropyLoss on the validation set.
train_task = training.TrainTask(
    labeled_data=train_generator,
    loss_layer=tl.CrossEntropyLoss(),
    optimizer=trax.optimizers.Adam(0.01),
    n_steps_per_checkpoint=10,
)

eval_task = training.EvalTask(
    labeled_data=val_generator(batch_size=batch_size, shuffle=True),
    metrics=[tl.CrossEntropyLoss(), tl.Accuracy()],
)

def classifier(vocab_size=len(Vocab), embedding_dim=256, output_dim=2, mode='train'):
    embed_layer = tl.Embedding(
        vocab_size=vocab_size, # Size of the vocabulary
        d_feature=embedding_dim)  # Embedding dimension
    
    # Create a mean layer, to create an "average" word embedding
    mean_layer = tl.Mean(axis=1)
    
    # Create a dense layer, one unit for each output
    dense_output_layer = tl.Dense(n_units = output_dim)

    
    # Create the log softmax layer (no parameters needed)
    log_softmax_layer = tl.LogSoftmax()
    
    model = tl.Serial(
      embed_layer, # embedding layer
      mean_layer, # mean layer
      dense_output_layer, # dense output layer 
      log_softmax_layer # log softmax layer
    )
    return model

model = classifier()

training_loop = training.Loop(
                                classifier, # The learning model
                                train_task, # The training task
                                eval_task = eval_task, # The evaluation task
                                output_dir = "/home/model/") # The output directory

training_loop.run(n_steps = 100)
tmp_inputs = np.array([[   3,    4,    5,    6,    7,    8,    9,    0,    0,    0,
                 0,    0,    0,    0,    0],
             [  10,   11,   12,   13,   14,   15,   16,   17,   18,   19,
                20,    9,   21,   22,    0],) , np.array([1,1]), np.array([1,1])
tmp_pred = training_loop.eval_model(tmp_inputs)

tmp_pred: 
"""
[[-4.9417334e+00, -7.1678162e-03], # log probabilities
 [-6.5846415e+00, -1.3823509e-03],

 Compare the probabilities in each column.
If column 1 has a value greater than column 0, classify that as a positive tweet.
Otherwise if column 1 is less than or equal to column 0, classify that example as a negative tweet.
"""


tmp_is_positive = tmp_pred[:,1] > tmp_pred[:,0] #The result of calculation is_positive is a boolean.
#The target is a type trax.fastmath.numpy.int32
tmp_is_positive_int = tmp_is_positive.astype(np.int32)  # DeviceArray([1, 1], dtype=int32)
tmp_is_positive_float = tmp_is_positive.astype(np.float32) # DeviceArray([1., 1.], dtype=float32)
```


"""
Custom ANN definition using TensorFlow.

This module contains the definition of an ANN comptable
with the SMT Toolbox
"""
from collections import defaultdict
##import datetime
##import tempfile

import matplotlib.pyplot as plt
import numpy as np
from smt.surrogate_models.surrogate_model import SurrogateModel
from smt.utils.checks import check_2d_array
import tensorflow as tf
from tensorflow_model_optimization.sparsity import keras as sparsity
        
class ANN(SurrogateModel):
    """
    ANN class.

    See also:
        Tensorflow documentation
    """

    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        # Initialize model
        in_dim, out_dim = self.options["dims"]
        no_layers, no_neurons = self.options["no_layers"], self.options["no_neurons"]
        self.options.declare("architecture", [in_dim]+[no_neurons]*no_layers+[out_dim], types=list, desc="default rchitecture")
        architecture = self.options["architecture"]
        activation = self.options["activation"]
        init = self.options["init"]
        bias_init = self.options["bias_init"]
        optimizer = self.options["optimizer"]
        loss = self.options["loss"]
        self.model = tf.keras.Sequential()
        kernel_regularizer = tf.keras.regularizers.l2(self.options["kernel_regularizer"])
        
        # Add layers        
        self.model.add(tf.keras.layers.Dense(architecture[1], activation=activation,
                                             kernel_initializer=init,
                                             bias_initializer=bias_init,
                                             kernel_regularizer=kernel_regularizer,
                                             input_shape=(architecture[0],)))
        for neurons in architecture[2:-1]:
            self.model.add(tf.keras.layers.Dense(neurons, activation=activation,
                                                 kernel_initializer=init,
                                                 bias_initializer=bias_init,
                                                 kernel_regularizer=kernel_regularizer))
        self.model.add(tf.keras.layers.Dense(architecture[-1]))

        # Calculate for pruning
        batch_size = self.options["batch_size"]
        no_epochs = self.options["no_epochs"]
        no_points = self.options["no_points"]
        final_sparsity = self.options["sparsity"]
        pruning_frequency = self.options["pruning_frequency"]

        if self.options["prune"]:
            num_train_samples = no_points
            end_step = np.ceil(1.0 * num_train_samples / batch_size).astype(np.int32) * no_epochs
            print('End step: ' + str(end_step))

            # Add pruning layers
            new_pruning_params = {
                  'pruning_schedule': sparsity.PolynomialDecay(initial_sparsity=0.0,
                                                               final_sparsity=final_sparsity,
                                                               begin_step=0,
                                                               end_step=end_step,
                                                               frequency=pruning_frequency)
            }

            self.model = sparsity.prune_low_magnitude(self.model, **new_pruning_params)

##        self.model.summary()
        self.model.compile(optimizer,loss)

        self._is_trained = False
        
    def _initialize(self):
        """
        Initialize the default hyperparameter settings.

        Parameters:
            no_layers: number of layers
            no_neurons: number of neurons per layer
            activation: activation functoin
            batch_size: batch size
            no_epochs: nu,ber of training epochs
            init: weight initialization strategy
            bias_init:  bias initialization strategy
            optimizer: optiimzer type
            loss: loss function
            kernel_regularizer: regularization paremeres
            dims: number of input and output dimension
            no_points: nu,ber of sample points
            prune: whether to use pruning
            sparsity: target network sparsity (fraction of zero weights)
            pruning_frequency: frequency of pruning
            tensorboard: whether to make tensorboard output
            stopping: use early stopping
            stopping_delta: required error delta threshold to stop training
            stopping_patience: number of iterations to wait before stopping
            plot_history: whether to plot the training history
        """

        # Set default values
        declare = self.options.declare
        declare("no_layers", 3, types=int, desc="number of layers")
        declare("no_neurons", 30, types=int, desc="neurons per layer")
        declare("activation", "swish", types=str, desc="activation function")
        declare("batch_size", 32, types=int, desc="batch size")
        declare("no_epochs", 30, types=int, desc="no epochs")
        declare("init", 'he_normal', types=str, desc="weight initialization")
        declare("bias_init", 'ones', types=str, desc="bias initialization")
        declare("optimizer", 'adam', types=str, desc="optimizer")
        declare("loss", 'mse', types=str, desc="loss function")
        declare("kernel_regularizer", 0.000, types=float, desc="regularization") # 000
        declare("dims", (None,None), types=tuple, desc="in and out dimensions")
        declare("no_points", None, types=int, desc="in and out dimensions")
        declare("prune", False, types=bool, desc="pruning")
        declare("sparsity", 0.50, types=float, desc="target sparsity")
        declare("pruning_frequency", 5, types=int, desc="pruning frequency")
        declare("tensorboard", False, types=bool, desc="tensorboard") ## not tested
        declare("stopping", False, types=bool, desc="early stopping")
        declare("stopping_delta", 0.001, types=float, desc="stopping delta")
        declare("stopping_patience", 5, types=int, desc="stopping patience")
        declare("plot_history", False, types=bool, desc="plot training history")

##        self.supports["derivatives"] = True
        self.name = "ANN"

        self.validation_points = defaultdict(dict)

    def _train(self):
        """
        Train the ANN.

        API function: train the neural net
        """
        stopping_delta = self.options["stopping_delta"]
        stopping_patience = self.options["stopping_patience"]
        
        callbacks = []

        if self.options["prune"]:
            callbacks.append(sparsity.UpdatePruningStep())
        if self.options["tensorboard"]:
            log_dir = "logs\\fit\\"#+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=20))
        if self.options["stopping"]:
            callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=stopping_delta, patience=stopping_patience, verbose=1))
##            callback = tf.keras.callbacks.MyStopping(monitor='val_loss', target=5, patience=3, verbose=1)

        train_in, train_out = self.training_points[None][0]
        test_in, test_out = self.validation_points[None][0]

        batch_size = self.options["batch_size"]
        no_epochs = self.options["no_epochs"]

        histor = self.model.fit(train_in, train_out, validation_data = (test_in, test_out), epochs=no_epochs, batch_size=batch_size, verbose=0, callbacks=callbacks)
        self.metric = self.evaluation()

        self._is_trained = True
        if self.options["plot_history"]:
            self.plot_training_history(histor)
        
        

    def _predict_values(self, x):
        """
        API method: predict values using appropriate methods from the neural_network.py module

        :param  x: np.ndarray[n, nx] -- Input values for the prediction points
        :return y: np.ndarray[n, ny] -- Output values at the prediction points
        """
        return self.model.predict(x)

    def set_validation_values(self, xt, yt, name=None):
        """
        Set training data (values).

        Parameters
        ----------
        xt : np.ndarray[nt, nx] or np.ndarray[nt]
            The input values for the nt training points.
        yt : np.ndarray[nt, ny] or np.ndarray[nt]
            The output values for the nt training points.
        name : str or None
            An optional label for the group of training points being set.
            This is only used in special situations (e.g., multi-fidelity applications).
        """
        xt = check_2d_array(xt, "xt")
        yt = check_2d_array(yt, "yt")

        if xt.shape[0] != yt.shape[0]:
            raise ValueError(
                "the first dimension of xt and yt must have the same length"
            )

        self.nt = xt.shape[0]
        self.nx = xt.shape[1]
        self.ny = yt.shape[1]
        kx = 0
        self.validation_points[name][kx] = [np.array(xt), np.array(yt)]

    def plot_training_history(self,history):
        """
        Plot the evolution of the training and testing error.

        Arguments:
            history: training history object

        Raises:
            ValueError: if the surrogate has not been trained yet
            
        """
        if self._is_trained:
            plt.title('Learning Curves')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
    ##        plt.plot(self.model.history.history['loss'], "k-", label='train')
    ##        plt.plot(self.model.history.history['val_loss'], "r--", label='val')
            plt.plot(history.history['loss'], "k-", label='train')
            plt.plot(history.history['val_loss'], "r--", label='val')
##            plt.ylim([0,.1])
            plt.legend()
            plt.ylim(bottom=0)
            plt.show()
        else:
            ValueError("Not trained yet")

    def evaluation(self):
        """
        Evaluate the generalization error.

        Returns:
            error: validation metric
        """
        test_in, test_out = self.validation_points[None][0]
        
        error = self.model.evaluate(test_in, test_out, verbose=0)
        print('MSE: %.3f, RMSE: %.3f' % (error, np.sqrt(error)))

        return error

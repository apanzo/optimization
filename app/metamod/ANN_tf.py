"""
Custom ANN definition using TensorFlow.

This module contains the definition of an ANN comptable
with the SMT Toolbox
"""
# Import native packages
from datetime import datetime
import os
from packaging import version

# Import pypi packages
from kerastuner.engine.hyperparameters import HyperParameters
import tensorflow as tf   
from tensorflow import keras
from tensorflow_model_optimization.sparsity import keras as sparsity

# Import custom packages
from core.settings import settings
from .kerastuner.bayesian_cv import BayesianOptimizationCV
from .kerastuner.randomsearch_cv import RandomSearchCV
from metamod.ANN import ANN_base
from visumod import plot_training_history

class ANN(ANN_base):
    """
    ANN class.

    """
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.name = "ANN"

    def build_sparse_model(self,neurons_hyp,activation_hyp,kernel_regularizer,in_dim,layers_hyp,out_dim):
        """
        Learn outpus one by one.

        """
        # Initialize input layer
        inputs = keras.Input(shape=(in_dim,))

        # Make output layer for each input
        output_layer_list = []
        for _ in range(out_dim):
            dense = keras.layers.Dense(neurons_hyp, activation=activation_hyp,
                                     kernel_initializer=self["init"],
                                     bias_initializer=self["bias_init"],
                                     kernel_regularizer=kernel_regularizer)(inputs)
            for _ in range(layers_hyp-1):
                dense = keras.layers.Dense(neurons_hyp, activation=activation_hyp,
                                                     kernel_initializer=self["init"],
                                                     bias_initializer=self["bias_init"],
                                                     kernel_regularizer=kernel_regularizer)(dense)
            outputs = keras.layers.Dense(1,activation="linear",use_bias=True)(dense)
            output_layer_list.append(outputs)

        # Get overall output layer
        if len(output_layer_list)>1:
            outputs = keras.layers.concatenate(output_layer_list)
        else:
            outputs = output_layer_list[0]

        #Make a model
        model = keras.Model(inputs=inputs, outputs=outputs)
##        model.summary()

        return model        

    def build_dense_model(self,neurons_hyp,activation_hyp,kernel_regularizer,in_dim,layers_hyp,out_dim):
        """
        A fully connected model.

        """
        model = keras.Sequential()
        # Add layers        
        model.add(keras.layers.Dense(neurons_hyp, activation=activation_hyp,
                                             kernel_initializer=self["init"],
                                             bias_initializer=self["bias_init"],
                                             kernel_regularizer=kernel_regularizer,
                                             input_shape=(in_dim,)))
        for _ in range(layers_hyp-1):
            model.add(keras.layers.Dense(neurons_hyp, activation=activation_hyp,
                                                 kernel_initializer=self["init"],
                                                 bias_initializer=self["bias_init"],
                                                 kernel_regularizer=kernel_regularizer))
        model.add(keras.layers.Dense(out_dim,activation="linear",use_bias=True))

        return model

    def build_hypermodel(self,hp):
        """
        General claass to build the ANN using tensorflow with autokeras hyperparameters defined

        Notes:
            * hyperparameters initialized using default values in config file
            * bias deactivated in output layer as it is centered around 0
        """

        model_type = self["type"]
        
        # Initialize hyperparameters as fixed    
        neurons_hyp = hp.Fixed("no_neurons",self["no_neurons"])
        layers_hyp = hp.Fixed("no_hid_layers",self["no_layers"])
        lr_hyp = hp.Fixed("learning_rate",self["learning_rate"])
        activation_hyp = hp.Fixed("activation_function", self["activation"])
        regularization_hyp = hp.Fixed("regularization",self["kernel_regularizer_param"])
        sparsity_hyp = hp.Fixed("sparsity",self["sparsity"])

        # Load regularizer
        if self["kernel_regularizer"] == "l1":
            kernel_regularizer = keras.regularizers.l1(regularization_hyp)
        elif self["kernel_regularizer"] == "l2":
            kernel_regularizer = keras.regularizers.l2(regularization_hyp)
        else:
            raise Exception("Invalid regularized specified")
        
        # Initiallze model
        in_dim, out_dim = self["dims"]        

        if model_type == "dense":
            model = self.build_dense_model(neurons_hyp,activation_hyp,kernel_regularizer,in_dim,layers_hyp,out_dim)
        elif model_type == "sparse":
            model = self.build_sparse_model(neurons_hyp,activation_hyp,kernel_regularizer,in_dim,layers_hyp,out_dim)
        else:
            raise Exception("Invalid model type")

        # Train
        optimizer = keras.optimizers.get(self["optimizer"])
        optimizer._hyper["learning_rate"] = lr_hyp

        # Prune the model
        if self["prune"]:
            model = self.prune_model(model,sparsity_hyp)

        model.compile(optimizer,self["loss"],metrics=["mse","mae",keras.metrics.RootMeanSquaredError()])

        return model

    def prune_model(self,model,target_sparsity):
        """
        Notes:
        * Only constant sparsity active
        """
        if True:
            model = sparsity.prune_low_magnitude(model,sparsity.ConstantSparsity(target_sparsity,0,frequency=10))
        else:
            model = sparsity.prune_low_magnitude(model,sparsity.PolynomialSparsity(0,target_sparsity,0,self["epochs"], frequency=10))

        return model

    def get_callbacks(self):
        """
        Docstring

        Notes:
            * MyStopping never used
        """
        callbacks = []

        if self["prune"]:
            callbacks.append(sparsity.UpdatePruningStep())
        if self["tensorboard"]:
            raise ValueError("Tensorboard not implemented yet")
            callbacks.append(keras.callbacks.TensorBoard(log_dir=os.path.join(self.log_dir,"tensorboard"), histogram_freq=20))
        if self["stopping"]:
            if True:
                callbacks.append(keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=self["stopping_delta"],patience=self["stopping_patience"],verbose=1,restore_best_weights=True))
            else:
                callbacks.append(keras.callbacks.MyStopping(monitor='val_loss',target=5,patience=stopping_patience,verbose=1,restore_best_weights=True))

        return callbacks

    def pretrain(self,inputs,outputs,iteration):
        """
        Notes:
            - optimization objective val_loss
        """
        print("### Performing Keras Tuner optimization of the ANN###")
        # Select hyperparameters to tune
        hp = HyperParameters()
        valid_entries = ["activation","neurons","layers","learning_rate","regularization","sparsity"]
        if not all([entry in valid_entries for entry in self["optimize"]]):
            raise Exception("Invalid hyperparameters specified for optimization")
        
        if "activation" in self["optimize"]:
            hp.Choice("activation_function",["sigmoid","relu","swish","tanh"],default=self["activation"])
        if "neurons" in self["optimize"]:
            hp.Int("no_neurons",3,20,sampling="log",default=self["no_neurons"])
        if "layers" in self["optimize"]:
            hp.Int("no_hid_layers",1,6,default=self["no_layers"])
        if "learning_rate" in self["optimize"]:
            hp.Float("learning_rate",0.001,0.1,sampling="log",default=self["learning_rate"])
        if "regularization" in self["optimize"]:
            hp.Float("regularization",0.0001,0.01,sampling="log",default=self["kernel_regularizer"])
        if "sparsity" in self["optimize"]:
            hp.Float("sparsity",0.3,0.9,default=self["sparsity"])

        no_hps = len(hp._space)

        # In case none are chosen, only 1 run for fixed setting
        if no_hps == 0:
            hp.Fixed("no_neurons",self["no_neurons"])
        
        # Load tuner
        max_trials = self["max_trials"]*no_hps
        path_tf_format = "logs"

        time = datetime.now().strftime("%Y%m%d_%H%M")

        tuner_args = {"objective":"val_loss","hyperparameters":hp,"max_trials":max_trials,
                      "executions_per_trial":self["executions_per_trial"],"directory":path_tf_format,
                      "overwrite":True,"tune_new_entries":False,"project_name":f"opt"}
##                      "overwrite":True,"tune_new_entries":False,"project_name":f"opt_{time}"}

        if self["tuner"] == "random" or no_hps==0:
            tuner = RandomSearchCV(self.build_hypermodel,**tuner_args)            
        elif self["tuner"] == "bayesian":
            tuner = BayesianOptimizationCV(self.build_hypermodel,num_initial_points=3*no_hps,**tuner_args)

        # Load callbacks and remove early stopping
        callbacks_all = self.get_callbacks()
        callbacks = [call for call in callbacks_all if not isinstance(call,keras.callbacks.EarlyStopping)]

        # Optimize
        tuner.search(inputs,outputs,epochs=self["tuner_epochs"],verbose=0,shuffle=True,
                     callbacks=callbacks,iteration_no=iteration)

        # Retrieve and save best model
        best_hp = tuner.get_best_hyperparameters()[0]
        self.write_stats([best_hp.values],"ann_best_models")
        untrained_model = tuner.hypermodel.build(best_hp)
        untrained_model.save(self.log_dir+"_untrained")

        # Make a table of tuner stats
        scores = [tuner.oracle.trials[trial].score for trial in tuner.oracle.trials]
        hps = [tuner.oracle.trials[trial].hyperparameters.values for trial in tuner.oracle.trials]
        for idx,entry in enumerate(hps):
            entry["score"] = scores[idx]
        self.write_stats(hps,"ann_tuner_stats")

    def _train(self):
        """
        Train the ANN.

        API function: train the neural net
        """
        # Load untrained optimized model
        self.model = keras.models.load_model(self.log_dir+"_untrained")
        
        # Load data and callbacks
        train_in, train_out = self.training_points[None][0]
        callbacks = self.get_callbacks()

        # Train the ANN
        if self.CV:
            test_in, test_out = self.validation_points[None][0]
            histor = self.model.fit(train_in, train_out, batch_size = train_in.shape[0],
                epochs=self["no_epochs"], validation_data = (test_in, test_out), verbose=0, shuffle=True, callbacks=callbacks)
            plot_training_history(histor,train_in,train_out,test_in,test_out,self.model.predict,self.progress)
        else:
            callbacks = [call for call in callbacks if not isinstance(call,keras.callbacks.EarlyStopping)]
            histor = self.model.fit(train_in, train_out, batch_size = train_in.shape[0],
                epochs=settings["surrogate"]["early_stop"], verbose=0, shuffle=True, callbacks=callbacks)

        self._is_trained = True
        self.early_stop = histor.epoch[-1]+1

    def _predict_values(self, x):
        """
        API method: predict values using appropriate methods from the neural_network.py module

        :param  x: np.ndarray[n, nx] -- Input values for the prediction points
        :return y: np.ndarray[n, ny] -- Output values at the prediction points
        """
        return self.model.predict(x)

# Turn off warnings
if version.parse(tf.__version__) >= version.parse("2.2"):
    tf.get_logger().setLevel('ERROR')

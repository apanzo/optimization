"""
Custom ANN definition using TensorFlow.

This module contains the definition of an ANN comptable
with the SMT Toolbox
"""
# Import native packages
from datetime import datetime
import os

# Import pypi packages
from kerastuner.engine.hyperparameters import HyperParameters
import numpy as np
import torch
import torch.nn as nn

# Import custom packages
from core.settings import settings
from .kerastuner.bayesian_pytorch_cv import BayesianPyTorchCV
from .kerastuner.randomsearch_pytorch_cv import RandomSearchPyTorchCV
from metamod.ANN import ANN_base
from visumod import plot_training_history

class TrainHistory:
    def __init__(self):
        self.history = {"loss":[],"val_loss":[]}

    def store(self,loss,loss_eval=None):
        self.history["loss"].append(loss.item())
        if loss_eval is not None:
            self.history["val_loss"].append(loss_eval.item())

class SparseModel(nn.Module):
    def __init__(self,neurons_hyp,activation_hyp,kernel_regularizer,in_dim,layers_hyp,out_dim,init,bias_init):
        super().__init__()

        nonlinearity = "relu" if activation_hyp == "swish" else activation_hyp
        
        self.activation = activations[activation_hyp]
        subnetworks = []
        for i in range(out_dim):
            layers = [nn.Linear(in_dim,neurons_hyp)]
            for _ in range(layers_hyp-1):
                layers.append(nn.Linear(neurons_hyp,neurons_hyp))
            for layer in layers:
                initializers[init](layer.weight,nonlinearity=nonlinearity)
                initializers[bias_init](layer.bias)
            layers.append(nn.Linear(neurons_hyp,1))
            subnetworks.append(nn.ModuleList(layers))

        self.subnetworks = nn.ModuleList(subnetworks)
        
    def forward(self,x):
        outputs = []
        for subnet in self.subnetworks:
            xx = x
            for layer in subnet[:-1]:
                xx = self.activation(layer(xx))
            xx = subnet[-1](xx)
            outputs.append(xx)

        out = torch.cat(outputs,1)
        return out
    
    def fit(self,epochs,train_in,train_out,test_in=None,test_out=None,optimizing=False):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(),lr=self.learning_rate,eps=1e-8)

        history = TrainHistory()
        
        for epochs_actual in range(epochs):
            self.eval()
            y_pred = self(train_in)
            loss = loss_fn(y_pred,train_out)
            self.train()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if test_in is not None:
                y_eval = self(test_in)
                loss_eval = loss_fn(y_eval,test_out)
##                print(t, loss.item(), loss_eval.item())
                history.store(loss,loss_eval)
                if not optimizing:
                    stop = self.early_stopping(history.history,"val_loss",self.tolerance,self.patience)
                    if stop:
                        print(f"Early stopping after {epochs_actual} epochs")
                        self.load_state_dict(self.best_state)
                        break
            else:
                history.store(loss)

        return history

    def early_stopping(self,history,metric,tolerance,patience):
        values = np.array(history[metric])[-patience:]
##        diffs = np.diff(history[metric])
##        improved = diffs[-patience:] < -tolerance

        if len(values) < patience:
            self.es_best_val_stored = np.min(values)
            self.es_best_iter_stored = np.argmin(values)
            self.best_state = self.state_dict()
            stop = False
        else:
            new_best = np.min(values)
            if new_best < self.es_best_val_stored:
                self.es_best_val_stored = np.min(values)
                self.es_best_iter_stored = np.argmin(values)
                self.best_state = self.state_dict()
                stop = False
            else:
                diffs = values - self.es_best_val_stored
                if np.all(diffs > tolerance):
                    stop = True
                else:
                    stop = False
        
        return stop

    
class ANN_pt(ANN_base):
    """
    ANN class.

    """
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.name = "ANN_pt"

    def build_hypermodel(self,hp):
        model_type = self["type"]
        in_dim, out_dim = self["dims"]

        # Initialize hyperparameters as fixed    
        neurons_hyp = hp.Fixed("no_neurons",self["no_neurons"])
        layers_hyp = hp.Fixed("no_hid_layers",self["no_layers"])
        lr_hyp = hp.Fixed("learning_rate",self["learning_rate"])
        activation_hyp = hp.Fixed("activation_function", self["activation"])
        regularization_hyp = hp.Fixed("regularization",self["kernel_regularizer_param"])
        sparsity_hyp = hp.Fixed("sparsity",self["sparsity"])

        kernel_regularizer = None

        if model_type == "dense":
            raise Exception("Dense model not supported for PyTorch")
        elif model_type == "sparse":
            model = SparseModel(neurons_hyp,activation_hyp,kernel_regularizer,in_dim,layers_hyp,out_dim,self["init"],self["bias_init"])
        else:
            raise Exception("Invalid model type")

        model.learning_rate = self["learning_rate"]
        model.tolerance = self["stopping_delta"]
        model.patience = self["stopping_patience"]
        
        return model
        
    def pretrain(self,inputs,outputs,iteration):
        """
        Notes:
            - optimization objective val_loss
        """
        print("### Performing Keras Tuner optimization of the ANN ###")
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
            tuner = RandomSearchPyTorchCV(self.build_hypermodel,**tuner_args)          
        elif self["tuner"] == "bayesian":
            tuner = BayesianPyTorchCV(self.build_hypermodel,num_initial_points=3*no_hps,**tuner_args)
        # Optimize
        tuner.search(inputs,outputs,epochs=self["tuner_epochs"],verbose=0,shuffle=True,
                     iteration_no=iteration)

        # Retrieve and save best model
        best_hp = tuner.get_best_hyperparameters()[0]
        self.write_stats([best_hp.values],"ann_best_models")

        # Make a table of tuner stats
        scores = [tuner.oracle.trials[trial].score for trial in tuner.oracle.trials]
        hps = [tuner.oracle.trials[trial].hyperparameters.values for trial in tuner.oracle.trials]
        for idx,entry in enumerate(hps):
            entry["score"] = scores[idx]
        self.write_stats(hps,"ann_tuner_stats")

        return best_hp

    def _train(self):
        """
        Train the ANN.

        API function: train the neural net
        """
        self.model = self.build_hypermodel(self.best_hp)

        # Load data and callbacks
        train_in, train_out = self.training_points[None][0]
        train_in = torch.Tensor(train_in)
        train_out = torch.Tensor(train_out)

        if self.CV:
            test_in, test_out = self.validation_points[None][0]
            test_in = torch.Tensor(test_in)
            test_out = torch.Tensor(test_out)
            histor = self.model.fit(self["no_epochs"],train_in,train_out,test_in,test_out)
            plot_training_history(histor,train_in,train_out,test_in,test_out,self._predict_values,self.progress)
            self.early_stop = len(histor.history["loss"])
        else:
            histor = self.model.fit(settings["surrogate"]["early_stop"],train_in,train_out)

        self._is_trained = True

    def _predict_values(self, x):
        """
        API method: predict values using appropriate methods from the neural_network.py module

        :param  x: np.ndarray[n, nx] -- Input values for the prediction points
        :return y: np.ndarray[n, ny] -- Output values at the prediction points
        """
        return self.model(torch.Tensor(x)).detach().numpy()

    def save(self):
        torch.save(self.model.state_dict(),os.path.join(settings["folder"],"logs","ann"))

def swish(x):
    return x*torch.sigmoid(x)

activations = {"relu":nn.functional.relu,"swish":swish}
initializers = {"he_normal":nn.init.kaiming_normal_,"zeros":nn.init.zeros_,"ones":nn.init.ones_}

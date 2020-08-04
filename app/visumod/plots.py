"""
This is the visualization module.

visual - scatter 2D/3D, curve, surface tri (not quad)
"""
# Import native packages
import os

# Import pypi packages
##from matplotlib import cm
import matplotlib.pyplot as plt
from pymoo.factory import get_visualization

# Import custom packages
from core.settings import settings

def scatter_pymoo(data,name,label=None,**kwargs):
    """
    Plot either a scatter, curve or surface plot.

    Arguments:
        data: data object
        name: visualization type
        normalized: whether the data is normalized

    Raises:
        ValueError: if the visualization method is not supported

    Notes:
        * surface plot not used
    """
    # Add data to plot
    plot_args = get_plot_args(data,label)
##    plot_args["title"] = "Raw data"

    angles = 4 if data.shape[1] == 3 else 1

    for angle in range(angles):
        plot_args["angle"] = (30,-135+angle*90)
        
        plot = get_visualization("scatter",**plot_args)
        plot.add(data,plot_type=None,**kwargs)
        plot.do()
##        plot.apply(lambda ax: ax.set_xlim([0,1]))

        # Save figure
        save_figure(name+f"_{angle+1}",plot)

def scatter(data,name,lower_bound=None,compare=False):
    fig = plt.figure()

    n_dim = data.shape[1]

    if n_dim == 2:
        plt.scatter(data[:,0],data[:,1])
    else:
        breakpoint()
    if lower_bound:
        plt.ylim(ymin=0)
    if compare:
        plt.plot([-1,1],[-1,1])
    save_figure(name)

def curve(data,name,lower_bound=None):
    fig = plt.figure()
    plt.plot(data)
    if lower_bound:
        plt.ylim(ymin=0)
    save_figure(name)

def heatmap(correlation):
    plot = get_visualization("heatmap",labels="x",cmap="BrBG",reverse=False)
    plot.add(correlation,vmin=-1, vmax=1)
    save_figure("heatmap",plot)

def pcp(data,name):
    plot = get_visualization("pcp", labels="f")
    plot.add(data,color="k")
    plot.do()

    save_figure(name,plot)

def surface_pymoo(data):
    """
    Docstring
    """
    # Add data to plot
    plot_args = get_plot_args(data,"x")

    angles = 1 if data.shape[1] == 3 else 1 ###

    for angle in range(angles):
        plot_args["angle"] = (30,-135+angle*90)
            
        plot = get_visualization("scatter",**plot_args)
        kwargs = {"cmap":"gist_gray"} if data.shape[1] == 3 else {"color":"k"}
        plot.add(data,plot_type="surface",**kwargs)
        plot.do()

        # Save figure
##        save_figure(name+f"_{angle+1}",plot)
        plot.show()

def learning_curves(training_loss,validation_loss,data_train,prediction_train,data_test,prediction_test,progress,trial_id):
    plt.figure()
    if len(progress) == 3:
        plt.suptitle(f"Iteration {progress[0]}, model {progress[1]}/{progress[2]}")
    else:
        plt.suptitle(f"Iteration {progress[0]}, model {progress[1]}/{progress[2]}, run {progress[3]}/{progress[4]}")
    plt.subplot(1, 2, 1)
    plt.title('Learning Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(training_loss, "k-", label='train')
    plt.plot(validation_loss, "r--", label='val')
    plt.ylim(bottom=0)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.scatter(data_train.flatten(),prediction_train.flatten(),c="k")
    plt.scatter(data_test.flatten(),prediction_test.flatten(),c="r")
    plt.plot([-1,1],[-1,1])
    plt.title('Prediction correletation')
    plt.xlabel('Data')
    plt.ylabel('Prediction')
    plt.xlim([-1,1])
    plt.ylim([-1,1])
    if len(progress) == 3:
        name = os.path.join("ann",f"model_{progress[0]}_{progress[1]}")
    else:
        name = os.path.join("ann",f"model_{trial_id[:8]}_{progress[0]}_{progress[1]}_{progress[3]}")
    save_figure(name)

def get_plot_args(data,label):
    n_dim = data.shape[1]
    plot_args = {}
    if n_dim > 3:
        plot_args["figsize"] = (1.5*n_dim,1.3*n_dim)
        plot_args["labels"] = [None]*n_dim
    else:
        plot_args["labels"] = label

    return plot_args

def save_figure(name,plot=None,iteration=None):
    """
    Docstring
    """
    # Set plot name
    fname = os.path.join(settings["folder"],"figures",name)
        
    # Save the plot
    if plot:
        plot.save(fname)
    else:
        plt.savefig(fname)

    # Close the plot
    plt.close()

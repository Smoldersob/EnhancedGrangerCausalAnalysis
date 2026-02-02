
import numpy as np
import pandas as pd

class Callback():
    """
    Template for callbacks used in MultiTaskConstrainedLinearRegression2D training.
    Provides hooks at different stages of the training process.
    """
    
    def __init__(self, parameters=None):
        """
        Initialize the callback with optional parameters.

        Parameters
        --------------
        parameters : any, optional
            Configuration parameters for the callback
        """
        self.parameters = parameters
        
    def on_train_begining(self):
        """
        Reset callback state before training. No return value.

        Parameters
        --------------
        None

        Return
        --------------
        None
        """
        pass

    def on_epoch_begining(self, prev_epoch_loss, curr_epoch_loss):
        """
        Decide whether to continue training at epoch start.

        Parameters
        --------------
        prev_epoch_loss : float
            Loss value from two epochs prior
        curr_epoch_loss : float
            Loss value from previous epoch

        Return
        --------------
        bool
            True to continue training, False to stop
        """
        return True

    def on_epoch_end(self, model=None):
        """
        To save model parameter from training actions.

        Parameters
        --------------
        model : object, optional
            Trained model object to save

        Return
        --------------
        None
        """
        pass
    
    def on_train_end(self, model=None):
        """
        Execute post-training actions (e.g., model restoration).

        Parameters
        --------------
        model : object, optional
            Trained model object for modification

        Return
        --------------
        None
        """
        pass



class ProcentageChange(Callback):
    """
    Implementation of MTCLR_Callback that stops training if relative change in loss function (betweed epochs) is to small. 
    """

    def __init__(self,proc_change=1e-4):
        self.proc_change=proc_change

    def on_epoch_begining(self,prev_epoch_loss,curr_epoch_loss):
        return ((np.abs(prev_epoch_loss - curr_epoch_loss) > self.proc_change*curr_epoch_loss))


class EarlyStopping(Callback):
    """
    Implementation of MTCLR_Callback that stops training if relative change in loss function (betweed epochs) is to small. 
    """

    def __init__(self,patience=20,river_to_best=True):
        self.patience=patience
        self.current_patience=patience
        self.current_best=np.inf
        self.best_model=None
        self.current_model=None
        self.river_to_best=river_to_best

    def on_epoch_begining(self,prev_epoch_loss,curr_epoch_loss):
        if (curr_epoch_loss - self.current_best) < 0:
            self.current_patience=self.current_patience-1
        else:
            self.current_best=self.patience
            self.current_patience=self.patience
            if self.river_to_best: 
                self.best_model=self.current_model
        if self.current_patience<=0:
            return False
        else:
            return True
        
    def on_epoch_end(self, model):
        if self.river_to_best:
            if self.best_model==None:
                self.best_model=model.get_weights()
            self.current_model=model.get_weights()

    def on_train_end(self, model):
        if self.river_to_best and self.best_model is not None:
            model.set_weights(self.best_model[0],self.best_model[1])
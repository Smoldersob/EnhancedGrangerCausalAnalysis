from sklearn.linear_model._base import _preprocess_data,LinearModel
from sklearn.base import RegressorMixin
import importlib
if importlib.util.find_spec("tensorflow") is not None:
    from tensorflow import summary

from sklearn.utils.validation import (
    check_consistent_length,
    validate_data,
)
import numpy as np


class MultiTaskConstrainedLinearRegression(LinearModel, RegressorMixin):
    """
    This class defines a version of Linear Regression that includes constraints on the coefficients and solves multitask problems.
    It extends the LinearModel class. Problem is solved by matrix gradinet descent with regulatization and contraints.

    Methods
    --------------
    fit(self, X, y, min_coef=None, max_coef=None, initial_beta=None, abs_sum_min=None, batch=None, tensoboard_writer=None, callbacks=None)
        This method is used to fit the ConstrainedLinearRegression model.

    Parameters
    --------------
    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and n_features is the number of features.

    y : array-like, shape (n_samples,)
        Target vector relative to X.

    min_coef : array-like, shape (n_features,), optional
        Lower constraint for coefficients. Defaults to negative infinity for each coefficient.

    max_coef : array-like, shape (n_features,), optional
        Upper constraint for coefficients. Defaults to positive infinity for each coefficient.

    initial_beta : array-like, shape (n_features,), optional
        Initial coefficients to start the optimization. Defaults to zeros.

    abs_sum_min : list-like, [int,list/range-like,float], optional
        List containing task number (int), coefficients positions (list/range-like) and value (flaot). The value is
        the smallest acceptable sum of absolute values of chosen coefficents of chosen task.
    
    batch : 
        Size of packeges of data in which model is trained. If None all given training data is used in one step.
    
    tensoboard_writer : (_TrackableResourceSummaryWriter | _ResourceSummaryWriter), optional
        Can be given write to tensorboard if training proces is suposed to be recorded.

    callback : list-like, [MTCLR_Callback], optional
        List of callbacks. Used call backs have to fit/derive from MTCLR_Callback class in order to work properly.

    Return
    --------------
    self : object
        Returns the instance itself.
    """
    cycle=np.linspace(1.0, 3.0, 20)
    
    def __init__(
        self,
        fit_intercept=True,
        copy_X=True,
        nonnegative=False,
        ridge=1e-12,
        lasso=0,
        tol=1e-15,
        learning_rate=1.0,
        max_iter=10000,
        cycle_indexes=False
    ):
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X
        self.nonnegative = nonnegative
        self.ridge = ridge
        self.lasso = lasso
        self.tol = tol
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.cycle_indexes=cycle_indexes

    def _set_coef(self, beta):
        self.coef_ = beta

    def _verify_coef2D(self, coef, value, feature_count, target_count):
        if coef is not None:
            coef_ = coef
            assert (
                coef_.shape[0] == target_count and coef_.shape[1] == feature_count
            ), f"Incorrect shape for coef_, the dimensions must be {target_count,feature_count}"
        else:
            coef_ = np.ones((target_count,feature_count)) * value
        return coef_

    def _verify_initial_beta2D(self, feature_count, target_count, initial_beta):
        if initial_beta is not None:
            beta = initial_beta
            assert beta.shape[0] == target_count and beta.shape[1] == feature_count, "Incorrect shape for initial_beta"
        else:
            beta = np.zeros((target_count,feature_count)).astype(float)
        return beta

    def fit(self, X, y, min_coef=None, max_coef=None, initial_beta=None, abs_sum_min=None,batch=None,tensoboard_writer=None, callbacks=[]):
        self.writer = tensoboard_writer

        check_X_params = dict(
            dtype=[np.float64, np.float32],
            order="F",
            force_writeable=True,
            copy=self.copy_X and self.fit_intercept,
        )
        check_y_params = dict(ensure_2d=False, order="F")
        X, y = validate_data(self,
            X, y, validate_separately=(check_X_params, check_y_params)
        )
        check_consistent_length(X, y)
        y = y.astype(X.dtype)

        n_samples, n_features = X.shape
        if y.ndim>1:
            n_targets = y.shape[1]
        else:
            n_targets =1

        X, y, X_offset, y_offset, X_scale, y_scale = _preprocess_data(
            X, y, fit_intercept=self.fit_intercept, copy=False
        )

        
        min_coef_ = self._verify_coef2D(
            min_coef,
            -np.inf,
            n_features,
            n_targets,
        )
        max_coef_ = self._verify_coef2D(
            max_coef,
            np.inf,
            n_features,
            n_targets,
        )

        beta = self._verify_initial_beta2D(n_features, n_targets,initial_beta)
        
        if self.nonnegative:
            min_coef_ = np.clip(min_coef_, 0, None)

        prev_epoch_loss = -1
        epoch_loss= np.inf
        hessian = self._calculate_hessian(X)
        loss_scale = len(y)
        step = 0
        loss_history = []  # Lista of losses
        
        for callback in callbacks:
            if hasattr(callback, 'on_train_begining'):
                callback.on_train_begining()
        
        if_break=False
        
        cycling=1
        if self.cycle_indexes is not False:
            positions = np.concatenate([np.arange(l) for l in np.diff(self.cycle_indexes)])
            cycling=self.cycle[positions]

        while prev_epoch_loss<0 or (np.abs(prev_epoch_loss - epoch_loss) > self.tol):
            if (step>0): 
                for callback in callbacks:
                    if hasattr(callback, 'on_epoch_begining') and not callback.on_epoch_begining(prev_epoch_loss,epoch_loss,step):
                        if_break=True
                        break
            if if_break:
                break

            if step > self.max_iter:
                print("THE MODEL DID NOT CONVERGE")
                break

            step += 1
            prev_epoch_loss = epoch_loss

            if batch is None:
                grad = self._calculate_gradient(X, beta, y)
                beta = self._update_beta(
                    beta, grad, hessian, loss_scale, min_coef_, max_coef_,abs_sum_min, n_features, cycling
                )
            else:
                for batch_beggining in range(0,n_samples,batch):
                    batch_end = min(batch_beggining+batch,n_samples-1)
                    grad = self._calculate_gradient(X[batch_beggining:batch_end,:], beta, y[batch_beggining:batch_end,:])
                    beta = self._update_beta(
                        beta, grad, hessian, loss_scale, min_coef_, max_coef_,abs_sum_min, n_features, cycling
                    )
            for callback in callbacks:
                if hasattr(callback, 'on_epoch_end'):
                    callback.on_epoch_end(self)
            # Current loss
            epoch_loss = np.mean((np.dot(X, beta.T) - y)**2)
            loss_history.append((step, epoch_loss))  # Save to listii
        
        
        if self.writer is not None and self.writer != False: 
            if importlib.util.find_spec("tensorflow") is not None:
                with self.writer.as_default():
                    for step, loss in loss_history:
                        summary.scalar('epoch_loss', loss, step=step)
                    self.writer.flush()  # Save to file
            elif importlib.util.find_spec("tensorflow") is not None:
                self.writer.add_scalar('epoch_loss', epoch_loss, step)
            else:
                print("WRITING TO TENSORBOARD UNAVAILABLE. PLEASE USE TENSORFLOW 2.0 OR HIGHER OR PYTORCH 1.2 OR HIGHER")
                self.writer==False
                
        
        for callback in callbacks:
            if hasattr(callback, 'on_train_end'):
                callback.on_train_end(self)
        
        self._set_coef(beta)
        self._set_intercept(X_offset, y_offset, X_scale)
        return self

    
    def _calculate_hessian(self, X):
        hessian = np.dot(X.transpose(), X)
        if self.ridge:
            hessian += np.eye(X.shape[1]) * self.ridge
        return hessian

    def _calculate_gradient(self, X, beta, y):
        grad = np.dot((np.dot(X, beta.transpose()) - y).transpose(), X)
        if self.ridge:
            grad += beta * self.ridge
        return grad

    def _update_beta(self, beta, grad, hessian, loss_scale, min_coef, max_coef,abs_sum_min, n_features, cycling):
        prev_value = beta
        new_value = beta - grad / hessian.diagonal() * self.learning_rate/n_features
        
        if self.lasso:
            new_value = self._apply_cycling_lasso(beta, grad, hessian, loss_scale, prev_value, new_value, n_features, cycling)

        if abs_sum_min is not None:
            for r,w,value in abs_sum_min:
                if sum(abs(new_value[r,w]))<value and sum(abs(beta[r,w]))!=0:
                    new_value[r,w]+=new_value[r,w]/sum(new_value[r,w])*abs(sum(abs(new_value[r,w]))-value)


        return np.clip(new_value, min_coef, max_coef)

    def _apply_cycling_lasso(self, beta, grad, hessian, loss_scale, prev_value, new_value, n_features, cycling):
        sign=np.sign([[prev_value[i][j] for j in range(len(prev_value[i]))] for i in range(len(prev_value))])
        new_value2 = (
            beta
            - (grad + cycling * sign * self.lasso * loss_scale)
            / hessian.diagonal()/n_features
            * self.learning_rate
        )

        lasso_change=np.zeros(beta.shape)
        cond=new_value2*new_value>=0
        lasso_change[cond]+=new_value2[cond]
        return  lasso_change
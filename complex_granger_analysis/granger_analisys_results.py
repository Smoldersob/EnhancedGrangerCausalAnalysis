
import numpy as np
import pandas as pd
from scipy.stats import f
import matplotlib.pyplot as plt

def RSS(model,X,Y):
    return np.sum((Y-model.predict(X))**2,axis=0)

def F_test_value(error1,error2,lag,rang,n):
    return (error1-error2)*(n-rang)/(error2*lag)

def sort_rows_desc_abs(array):
    return np.array([row[np.argsort(-np.abs(row))] for row in array])

class GrangerAnalisysResults():

    def __init__(self,effects,causes):
        nrows=len(effects)
        ncols=len(causes)
        self.base_error=pd.DataFrame(np.zeros((nrows,ncols)),effects,causes)
        self.ref_error=pd.DataFrame(np.zeros((nrows,ncols)),effects,causes)
        self.F_test=pd.DataFrame(np.ones((nrows,ncols)),effects,causes)
        self.p_value=pd.DataFrame(np.ones((nrows,ncols)),effects,causes)
        self.sign=pd.DataFrame(np.ones((nrows,ncols)),effects,causes)
        self.base_weights=None
        self.ref_weights=dict()
    
    def update_column(self,column,column_id,base_model,ref_model,x,y,column_indexes,model_type=0):
        #Models errors
        RSS_base=RSS(base_model,x,y)
        RSS_ref=RSS(ref_model,x,y)
        self.base_error.loc[:,column]=RSS_base/y.shape[0]/y.shape[1]
        self.ref_error.loc[:,column]=RSS_ref/y.shape[0]/y.shape[1]
        
        #Sign of relation check
        start = column_indexes[column_id]
        end = column_indexes[column_id+1]
        
        max_coef = 0
        min_coef = 0
        
        match model_type:
            case 0:
                #Scikit model
                max_coef = np.array([max(_) for _ in base_model.coef_[:,int(start):int(end)]])
                min_coef = np.array([min(_) for _ in base_model.coef_[:,int(start):int(end)]])
                self.ref_weights.update({column:ref_model.coef_})
                self.base_weights = base_model.coef_
            case 2:
                #Pytorch model
                max_coef = np.array([max(_) for _ in base_model.linear.weight.data.cpu().numpy()[:,int(start):int(end)]])
                min_coef = np.array([min(_) for _ in base_model.linear.weight.data.cpu().numpy()[:,int(start):int(end)]])
                self.ref_weights.update({column:ref_model.linear.weight.data.cpu().numpy()})
                self.base_weights = base_model.linear.weight.data.cpu().numpy()
            case 1:
                #Tensorflow model
                max_coef=np.array([max(_) for _ in base_model.get_weights()[-2].transpose()[:,int(start):int(end)]])
                min_coef=np.array([min(_) for _ in base_model.get_weights()[-2].transpose()[:,int(start):int(end)]])
                self.ref_weights.update({column:ref_model.get_weights()[-2].transpose()})
                self.base_weights = base_model.get_weights()[-2].transpose()
            case _:
                TypeError("Undefinef type of")
        self.sign.loc[:,column]=np.sign(max_coef+min_coef)
        
        #Causation check
        lag_order=end-start
        
        n,p = x.shape
        p=p/lag_order
        self.F_test.loc[:,column]=F_test_value(RSS_ref,RSS_base,lag_order,p,n)
        self.p_value.loc[:,column] = 1 - f.cdf(self.F_test[column]*(self.F_test[column]>=0), lag_order, (n -p),)
        
    def result(self,treshold=0.01,sign=False):
        ans=(self.p_value<np.ones(self.p_value.shape)*treshold)*1
        if sign:
            ans[ans==1]=ans[ans==1]*self.sign[ans==1]
        return ans

    def plot_significant_weights(self,i=None,referece_model=False,model_nr=0):
        if referece_model:
            if model_nr>=len(self.ref_weights)-1 or model_nr <-len(self.ref_weights):
                raise ValueError(f'i value was to big. Max value {len(self.ref_weights-1)}, min value {-len(self.ref_weights)}')
            sorted_arr = sort_rows_desc_abs(self.ref_weights[model_nr])
        else:
            sorted_arr = sort_rows_desc_abs(self.base_weights)
        
        rows = sorted_arr.shape[0]  # Number of necessry rows
        fig, axes = plt.subplots(rows, 1, figsize=(6, rows * 2))
        if rows == 1:
            axes = [axes]
        
        if i is None or i>sorted_arr.shape[1] or i<0:
            i=sorted_arr.shape[1]

        for j, row in enumerate(sorted_arr):
            axes[j].plot(row[:i], marker='o', linestyle='-')
            axes[j].set_title(f'Graph of sorted weights for {j+1} output ({self.base_error.columns[j]})')
            axes[j].grid(True)
        plt.tight_layout()
        plt.show()
import numpy as np
import pprint

def performance_metrics(y_pred , y_target):
    mape = np.mean(np.abs(y_pred - y_target)/np.abs(y_target))
    mae = np.mean(np.abs(y_pred - y_target))
    mpe = np.mean((y_pred - y_target)/y_target)
    rmse = np.mean((y_pred - y_target)**2)**0.5
    corr = np.corrcoef(y_pred ,y_target)[0,1]

    mins = np.amin(np.hstack([y_pred[:,None],y_target[:,None]]), axis=1)
    maxs = np.amax(np.hstack([y_pred[:,None],y_target[:,None]]), axis=1)
    minmax = 1-np.mean(mins/maxs)

    pprint.pprint({'mape':mape,'mae':mae,'mpe':mpe,'rmse':rmse,'corr':corr,'minmax':minmax})

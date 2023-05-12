from statsmodels.tsa.stattools import acf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
plt.rcParams.update({'figure.figsize':(9,3), 'figure.dpi':120})

def get_model(data, order):

    model = ARIMA(data, order=(1,1,1))
    model_fit = model.fit(disp=0)
    return model_fit

def get_residuals(model_fit):
    residuals = pd.DataFrame(model_fit.resid)
    fig, ax = plt.subplots(1,2)
    residuals.plot(title="Residuals", ax=ax[0])
    residuals.plot(kind='kde', title='Density', ax=ax[1])
    plt.show()

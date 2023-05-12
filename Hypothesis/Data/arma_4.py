import numpy as np
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.arima_process import ArmaProcess
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd


amount_timeseries = 2000
length_timeseries = 1000
timeseries_mat = np.zeros([length_timeseries+1, amount_timeseries])

#ARMA(2,0)
for i in range(amount_timeseries):
    choice = np.random.randint(4)
    if choice == 0:
        #1st component
        ar2 = np.array([1, 0.25, -0.52])
        ma2 = np.array([1, 0, 0])
        simulated_ARMA2_data = ArmaProcess(ar2, ma2).generate_sample(nsample=length_timeseries)
        #noise
        et = np.random.normal(0, 0.27, simulated_ARMA2_data.shape)
        simulated_ARMA2_data += et
        cluster = 1
    elif choice == 1:
        #1st component
        ar2 = np.array([1, 0.2, 0.09])
        ma2 = np.array([1, 0, 0])
        simulated_ARMA2_data = ArmaProcess(ar2, ma2).generate_sample(nsample=length_timeseries)
        #noise
        et = np.random.normal(0, 0.27, simulated_ARMA2_data.shape)
        simulated_ARMA2_data += et
        cluster = 4
    # elif choice == 2:
    #     #1st component
    #     ar2 = np.array([1, -0.01, 0.333])
    #     ma2 = np.array([1, 0, 0])
    #     simulated_ARMA2_data = ArmaProcess(ar2, ma2).generate_sample(nsample=length_timeseries)
    #     #noise
    #     et = np.random.normal(0, 0.27, simulated_ARMA2_data.shape)
    #     simulated_ARMA2_data += et
    #     cluster = 5
    # elif choice == 3:
    #     #1st component
    #     ar2 = np.array([1, -0.5, 0.52])
    #     ma2 = np.array([1, 0, 0])
    #     simulated_ARMA2_data = ArmaProcess(ar2, ma2).generate_sample(nsample=length_timeseries)
    #     #noise
    #     et = np.random.normal(0, 0.27, simulated_ARMA2_data.shape)
    #     simulated_ARMA2_data += et
    #     cluster = 6
    elif choice == 3:
        #3rd component
        ar2 = np.array([1, -1.5, 0.75])
        ma2 = np.array([1, 0, 0])
        simulated_ARMA2_data = ArmaProcess(ar2, ma2).generate_sample(nsample=length_timeseries)
        #noise
        et = np.random.normal(0, 0.27, simulated_ARMA2_data.shape)
        simulated_ARMA2_data += et
        cluster = 3
    else:
        #2nd component
        ar2 = np.array([1, -0.34, -0.27])
        ma2 = np.array([1, 0, 0])
        simulated_ARMA2_data = ArmaProcess(ar2, ma2).generate_sample(nsample=length_timeseries)
        #noise
        et = np.random.normal(0, 0.27, simulated_ARMA2_data.shape)
        simulated_ARMA2_data += et
        cluster = 2

    timeseries_mat[1:, i] = simulated_ARMA2_data
    timeseries_mat = timeseries_mat[::-1]
    timeseries_mat[-1, i] = cluster
    timeseries_mat = timeseries_mat[::-1]

timeseries_mat = timeseries_mat[:, np.random.permutation(timeseries_mat.shape[1])]
timeseries_mat = timeseries_mat.T
#timeseries_df = pd.DataFrame(timeseries_mat, index=pd.date_range(start='1/1/2022', freq='1H', tz='UTC', periods=timeseries_mat.shape[0]))
timeseries_df = pd.DataFrame(timeseries_mat, index=np.arange(0,timeseries_mat.shape[0],1))
timeseries_df.index.name = 'Timestamp'
# timeseries_df = pd.DataFrame(timeseries_mat)
# timeseries_df.insert(0, "Timestamp", np.arange(0,timeseries_df.shape[0],1), True)

timeseries_df.to_csv('syntheticARMA_4.csv')

compare = pd.read_csv('filename.csv')

print('end')
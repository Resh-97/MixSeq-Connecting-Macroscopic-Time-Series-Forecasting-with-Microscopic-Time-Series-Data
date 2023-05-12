
from utils import get_data
from model import get_model, get_residuals
from evaluate import performance_metrics
import pandas as pd

train_macro, test_macro, train_micro_1, test_micro_1, train_micro_2, test_micro_2  = get_data('Data','syntheticARMA',n_component = 4)
print(train_macro.shape)
model_macro = get_model(train_macro,  order = (1,1,1))
print(model_macro.summary())
get_residuals(model_macro)
y_pred_macro = pd.Series(model_macro.forecast(200)[0], index = test_macro.index)

model_micro_1 = get_model(train_micro_1,  order = (1,1,1))
print(model_micro_1.summary())
get_residuals(model_micro_1)
y_pred_micro_1 = pd.Series(model_micro_1.forecast(200)[0], index = test_micro_1.index)

model_micro_2 = get_model(train_micro_2,  order = (1,1,1))
print(model_micro_2.summary())
get_residuals(model_micro_2)
y_pred_micro_2 = pd.Series(model_micro_2.forecast(200)[0], index = test_micro_2.index)
'''
model_micro_3 = get_model(train_micro_3,  order = (1,1,1))
print(model_micro_3.summary())
get_residuals(model_micro_3)
y_pred_micro_3 = pd.Series(model_micro_3.forecast(200)[0], index = test_micro_3.index)
'''
"""
model_micro_4 = get_model(train_micro_4,  order = (1,1,1))
print(model_micro_4.summary())
get_residuals(model_micro_4)
y_pred_micro_4 = pd.Series(model_micro_4.forecast(200)[0], index = test_micro_4.index)
"""

print(".........Performance Report on Macroscpic Data.........")
performance_metrics(y_pred_macro , test_macro)

y_pred_micro = y_pred_micro_1 + y_pred_micro_2
print(".........Performance Report on Microscpic Data.........")
performance_metrics(y_pred_micro , test_macro)

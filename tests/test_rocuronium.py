from python_anesthesia_simulator.pk_models import CompartmentModel
from python_anesthesia_simulator.pd_models import TOF_model
from python_anesthesia_simulator.simulator import Patient
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



model_list = ['Wierda', "Szenohradszky", "Cooper", "Alvarez-Gomez"]

dataframe_final = None

age = 30
weight = 65
height = 165
sex = 0

u = 0.02

for model in model_list:
    George = Patient([age, weight, height, sex], model_nmb=model, model_tof=model, ts=10)

    data_model = George.full_sim(u_nmb=np.hstack([np.ones(100*6)*u,np.zeros(100*6)]))
    data_model.columns = [col + f'_{model}' if col != 'Time' else col for col in data_model.columns]
    if dataframe_final is None:
        dataframe_final = data_model
    else:
        dataframe_final = pd.merge(dataframe_final, data_model, on='Time', how='inner')


def rocuronium_pk_coehrence():
    """Ensure that the default models give correct results"""
    for model in model_list:
        assert np.all(dataframe_final[f'x_nmb_4_{model}'] >= 0)
        assert np.all(dataframe_final[f'x_nmb_4_{model}'] <= 10)
        assert np.all(dataframe_final[f'x_nmb_4_{model}'] <= np.max([dataframe_final[f'TOF_{model}'].iloc[100*6+1],dataframe_final[f'TOF_{model}'].iloc[100*6+2]]))

def rocuronium_pd_coehrence():
    """Ensure that the default models give correct results"""
    for model in model_list:
        assert np.all(dataframe_final[f'TOF_{model}'] >= 0)
        assert np.all(dataframe_final[f'TOF_{model}'] <= 110)
        assert np.all(dataframe_final[f'TOF_{model}'] >= np.min([dataframe_final[f'TOF_{model}'].iloc[100*6+1],dataframe_final[f'TOF_{model}'].iloc[100*6+2]]))

# plot everything
if __name__ == '__main__':
    for model in model_list:
        plt.plot(dataframe_final['Time']/60, dataframe_final[f'x_nmb_4_{model}'], label=model)
    plt.legend()
    plt.grid()
    plt.ylabel('Concentration (µg/mL) ')
    plt.xlabel('Time (min)')
    plt.show()

    for model in model_list:
        plt.plot(dataframe_final['Time']/60, dataframe_final[f'TOF_{model}'], label=model)
    plt.legend()
    plt.grid()
    plt.ylabel('TOF ratio (%) ')
    plt.xlabel('Time (min)')
    plt.show()

    rocuronium_pk_coehrence()
    rocuronium_pd_coehrence()
    print('All tests passed successfully')
import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

def predict_and_evaluate(df, key, span, pred_len):

    truth = []
    ewma = []
    shifted = []

    df['EWMA'] = df[key].ewm(span=span, min_periods=8, adjust=True).mean()

    b_truth  = df[key].to_numpy()
    b_ewma = df['EWMA'].to_numpy()

    for i in range(len(b_truth) - span - pred_len):
        truth.append(b_truth[(i+span):(i+span+pred_len)])
        ewma.append(np.repeat(b_ewma[i+span-1], pred_len)) #correct?
        #ewma8.append(np.repeat(b_ewma8[i+span-0], pred_len)) #informer
        shifted.append(np.repeat(b_truth[i+span-1], pred_len))

    #EWMA
    rmse = root_mean_squared_error(truth, ewma)
    mae = mean_absolute_error(truth, ewma)
    print(f'EWMA8 rmse: {rmse}, mae: {mae}')

    #Shifted
    rmse = root_mean_squared_error(truth, shifted)
    mae = mean_absolute_error(truth, shifted)
    print(f'Shifted rmse: {rmse}, mae: {mae}')
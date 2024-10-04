from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error

# ARIMA model parameters (p, d, q)
p = 1  # AutoRegressive (AR) order
d = 1  # Differencing (I) order
q = 0  # Moving Average (MA) order

# Function to predict next value using ARIMA model
def predict_next_value(df, window_size=32, horizon=1):
    predictions = []

    for i in range(len(df) - window_size):
        window = df.iloc[i:i+window_size]  # Accessing by index using iloc

        model = ARIMA(window, order=(p, d, q))  # ARIMA model, you may need to adjust the parameters
        model_fit = model.fit()
        
        next_value = model_fit.forecast(steps=horizon)  # Extracting the forecasted value
        #print(type(next_value.values))
        #print((next_value.values))
        #print((next_value.iloc[0]))

        predictions.append(next_value.values)
        #predictions.append(next_value.iloc[0])

    return predictions

def evaluate(predictions, df, window_size, prediction_horizon):
    print(f'len(predictions): {len(predictions)}')
    print(f'len(predictions[0]): {len(predictions[0])}')
    #print(f'predictions[0]: {(predictions[0])}')

    # Evaluate the predictions
    actual_values = df['bandwidth']
    #actual_values = tail_arima['datarate'].iloc[window_size:]
    print(f'len(actual_values): {len(actual_values)}')

    arima_labels = []
    for i in range(len(actual_values) - window_size - prediction_horizon + 1):
        #print(i)
        arima_labels.append(actual_values[(i+window_size):(i+window_size+prediction_horizon)])

    print(f'len(arima_labels): {len(arima_labels)}')
    #print(f'len(arima_labels): {len(arima_labels[0])}')

    #selected_arima_labels = arima_labels[-(len(predictions)):]
    #print(f'len(selected_arima_labels): {len(selected_arima_labels)}')
    #print(f'len(selected_arima_labels): {len(selected_arima_labels[0])}')

    selected_predictions = predictions[:len(arima_labels)]
    print(f'len(selected_predictions): {len(selected_predictions)}')

    #ARIMA
    rmse = root_mean_squared_error(arima_labels, selected_predictions)
    mae = mean_absolute_error(arima_labels, selected_predictions)
    print(f'ARIMA rmse: {rmse}, mae: {mae}')
# BandSeer — Bandwidth Prediction for Cellular Networks

Welcome to the BandSeer GitHub repository!
BandSeer is an efficient stacked Bi-LSTM-based architecture for bandwidth prediction that outperforms state-of-the-art prediction baselines.

It will be presented at the 49th [IEEE Conference on Local Computer Networks (LCN)](https://www.ieeelcn.org/).

The code will be added before the LCN 2024.

This project is licensed under the terms of the MIT License.

## Abstract

In the context of cellular networks, such as with 5G and upcoming 6G networks, the available bandwidth of a connection is inherently dynamic. Accurate prediction of future bandwidth availability within a link is essential for latency-sensitive and mission-critical applications such as video streaming or remote driving. Bandwidth prediction ensures efficient utilization of a link and thus prevents delays. This paper introduces BandSeer, a stacked Bi-LSTM-based approach for bandwidth prediction in LTE and 5G cellular networks. BandSeer captures complex correlations in historical metrics better than prior work and outperforms SotA baselines. It achieves reductions of up to 18.32% in RMSE and 26.87% in MAE on the Berlin V2X dataset, and reductions of up to 12.43% in RMSE and 28.45% in MAE on the Beyond 5G dataset compared to the SotA Informer baseline. Furthermore, we argue that any bandwidth algorithm must be resource efficient to enable for development on various devices. Our evaluations show that BandSeer consumes one order of magnitude fewer resources and needs roughly a quarter to half the inference time of its closest competitor, the Informer model.

## Installation:

1. Clone this Git repository to your local machine using the following command:

```
git clone https://github.com/ds-kiel/bandseer.git
cd bandseer
```

2. Create a virtual Python environment (e.g. using pyenv)

```
curl https://pyenv.run | bash
pyenv install 3.11.7
pyenv virtualenv 3.11.7 bandseer
```

3. Install the necessary Python packages by running:

```
pip install -r requirements.txt -U
```

4. Get the datasets from their respective repos.


# Structure:

You should be running main.py in order to execute training, validation, testing and prediction stages.

1. main.py

	This file includes our training algorithm, parameter and config definitions, and hyperparameter search implementations. 
	The Bi-LSTM model is initated with specified configurations. 
    Currently set to run on a single GPU but Multi-GPU can also be enabled if necessary.
	This is the entry point for most configurations and actual training.	

2. models.py

	This file includes our Bi-LSTM implementations.

3. datamodule.py

	This file includes our data representation. We do preprocesing steps such as reading the values, feature scaling, and dataloader creation.

4. metrics.py

    This file includes function calls for calculating error metrics.

5. correlation.py

    This file includes functions to calculate Pearson, Spearman and Kendall correlations.

6. arima.py

    This file includes our ARIMA implementation to generate predictions.

7. ewma.py

	This file includes our EWMA implementation to generate predictions.

# Example run:

We used the following command line argument to train and evaluate our model:

```
python3 main.py
```
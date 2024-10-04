import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ':4096:8'

# External imports
import time
import torch

import lightning.pytorch as pl
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import wandb

# Profiling
from torchinfo import summary

# Internal imports
import datamodule
import models
from metrics import calculate_error

# Print CUDA details
#torch.set_float32_matmul_precision('“highest”') # “highest” (default), “high”, or “medium”
print(torch.get_float32_matmul_precision())
print('Torch version: {}'.format(torch.__version__))
print('CUDA available: {}'.format(torch.cuda.is_available()))
if torch.cuda.is_available():
    print(torch.cuda.device_count())
    print(torch.cuda.current_device())
    print(torch.cuda.device(0))
    print(torch.cuda.get_device_name(0))

ITERATIONS=1

# wandb sweep configuration for hyperparameter search
sweep_configuration = {
    'method': 'grid', # grid, random, bayes
    'name': 'sweep',
    'metric': {
        'goal': 'minimize',
        'name': 'val_loss'},
    'early_terminate': {
        'type': 'hyperband',
        'min_iter': 3},
    'parameters': {
        'pred_len': {'values': [1]},
        'hidden_size': {'values': [64]},
        'num_layers': {'values': [3]},
        'num_linear_layers': {'values': [3]},
        'seq_len': {'values': [32]},
        'bidirectional': {'values': [True]}, # True, False

        'dropout_rnn': {'values': [0.05]},
        'dropout_linear': {'values': [0.05]},
        'learning_rate': {'values': [0.0001]},
        
        'criterion': {'values': ['L1Loss']}, # 'L1Loss', 'MSELoss'
        'activation': {'values': ['GELU']}, # 'GELU', 'LeakyReLU', 'ReLU'
        'scaler': {'values': ['StandardScaler']}, # 'MinMaxScaler', 'StandardScaler'
        
        'optimizer': {'values': ['Adam']}, # Adam, AdamW
        'lr_scheduler': {'values': ['ReduceLROnPlateau']}, # 'ReduceLROnPlateau', 'StepLR'
        'full_out': {'values': [False]} # True, False
     },
    'run_cap' : 1
}
sweep_id = wandb.sweep(sweep=sweep_configuration, project='BandSeer')

def train(config=None):
    
    config = dict(
        # Setting common hyperparameters for model
        model_type='LSTM', # LSTM
        input_size=17, # 8 for NYU, 9 for Beyond5G, 17 for BerlinV2X
        pred_len=1,
        hidden_size=64,
        num_layers=3,
        num_linear_layers=3,
        dropout_rnn=0.05,
        dropout_linear=0.05,
        learning_rate=0.0001,
        seq_len=32,
        batch_size=32,
        criterion='L1Loss', # MSELoss or L1Loss
        activation='GELU', # ReLU, GELU or LeakyReLU
        scaler='StandardScaleR', # MinMaxScaler, StandardScaleR
        optimizer='Adam', # Adam or AdamW
        lr_scheduler='ReduceLROnPlateau', # StepLR or ReduceLROnPlateau
        bidirectional=True,
        batch_first=True,
        inverse=True,

        # Datasets
        #current_dataset='NYU-METS-MM15',
        #data_dir='preprocessed-data/NYU-METS/',

        #current_dataset='Beyond5G',
        #data_dir='preprocessed-data/5Gdataset/',
        
        current_dataset='BerlinV2X', #BerlinV2X
        data_dir='preprocessed-data/BerlinV2X/',

        # Datamodule Parameters
        num_workers=0,
        persistent_workers=False,
        PARQUET = False, # True, False
        
        # %60, %20, %20 Train-Val-Test Split 
        val_p=0.2,
        test_p=0.2,

        # Directories
        working_dir='./',
        log_dir="./data/logs/",
        plot_dir="./data/plots/",
        outputs_dir="./data/outputs/",
        checkpoints_dir="./data/checkpoints/",
        model_save='./data/model_save',
        # Logging
        logging_name='base',
        experiment_name='10-epochs-grid-batch-size',
        # Trainer
        max_epochs=1,
        iterations=ITERATIONS
    )
    pl.seed_everything(42, workers=True)

    run = wandb.init(config=config, allow_val_change=True)
    wandb_logger = WandbLogger(project='BandSeer', save_code=True)
    #wandb_logger.experiment.config.update(config)

    config['pred_len'] = wandb.config.pred_len
    config['hidden_size'] = wandb.config.hidden_size
    config['num_layers'] = wandb.config.num_layers
    config['num_linear_layers'] = wandb.config.num_linear_layers
    config['optimizer'] = wandb.config.optimizer

    config['dropout_rnn'] = wandb.config.dropout_rnn
    config['dropout_linear'] = wandb.config.dropout_linear
    config['learning_rate'] = wandb.config.learning_rate
    config['criterion'] = wandb.config.criterion
    config['activation'] = wandb.config.activation
    config['scaler'] = wandb.config.scaler
    config['bidirectional'] = wandb.config.bidirectional
    config['lr_scheduler'] = wandb.config.lr_scheduler
    config['seq_len'] = wandb.config.seq_len

    data = datamodule.MetricsDataModule(config)
    #data.setup(stage = 'fit')

    model = models.Stateless(config)
 
    if False: # Check efficiency numbers, model size etc
        # Define the sizes
        batch_size = 1
        input_seq_length = 32
        input_feature_dim = 17

        # Create dummy data tensors
        model_input_dummy = torch.randn(batch_size, input_seq_length, input_feature_dim)
        #print(summary(model, input_data=model_input_dummy, verbose=2))
        print('torchinfo summary')
        print(summary(model, input_size=(1, 32, 17), verbose=2))

        return
    
    else:
        callbacks = []
        callbacks.append(EarlyStopping(
            monitor='mean_val_loss', 
            patience=10,
            verbose=True, 
            strict=True))
        callbacks.append(ModelCheckpoint(
            filename='{epoch}-{step}-{val_loss:.3f}', 
            monitor='mean_val_loss', 
            mode='min',
            verbose=True))
        #callbacks.append(TQDMProgressBar())
        #callbacks.append(RichProgressBar())
        
        trainer = pl.Trainer(
            accelerator='auto', #'auto', 'gpu', 'cuda', 'cpu'
            devices='auto',  # 'auto', '4', -1
            max_epochs=config['max_epochs'],
            logger=wandb_logger,
            callbacks=callbacks,
            enable_progress_bar=True,
            precision=32, # 64, 32, 16-mixed, 'bf16-mixed'
            profiler=None, # None, "pytorch", "simple", 'advanced'
            deterministic=True)

        start = time.time()
        trainer.fit(model, data)
        print('Training time: {} s'.format(time.time()-start))

        # automatically restores model, epoch, step, LR schedulers, etc...
        #trainer.fit(model, ckpt_path="BandSeer/../my_checkpoint.ckpt")
        #model = models.Stateless.load_from_checkpoint("BandSeer/../my_checkpoint.ckpt")

        start = time.time()
        trainer.test(model, data)
        print('Testing time: {} s'.format(time.time()-start))

        start = time.time()
        labels_predictions = trainer.predict(model, data)
        print('Prediction time: {} s'.format(time.time()-start))
        
        return config, data, labels_predictions, wandb_logger
    
def calculate_errors(y_true, y_pred, model_name):
    mse, rmse, mae, std, ae = calculate_error(y_true, y_pred)

    evaluation = {
        model_name+" MAE": mae,
        model_name+" MSE": mse,
        model_name+" RMSE": rmse,
        model_name+" STD": std,
    }
    wandb.log(evaluation)

    model_error_results = f'{model_name} MAE: {mae}, MSE: {mse}, RMSE: {rmse}, STD: {std}'
    print(model_error_results)
    return model_error_results, mse, rmse, mae, std, ae

def evaluate(config, wandb_logger, data, labels_predictions, model_name, draw_plot):
    labels = []
    predictions = []

    print(type(labels_predictions))
    for item in labels_predictions:
            labels.append(item['label'])
            predictions.append(item['prediction'])

    print(f'before cat labels[0].size(): {labels[0].size()}, predictions[0].size(): {predictions[0].size()}')

    labels = torch.cat(labels)
    predictions = torch.cat(predictions)
    print(f'after cat labels.size(): {labels.size()}, predictions.size(): {predictions.size()}')
    print(f'after cat labels[0].size(): {labels[0].size()}, predictions[0].size(): {predictions[0].size()}')

    labels = labels.numpy()
    predictions = predictions.numpy()

    print('dtype:', predictions.dtype, labels.dtype)
    print('shape:', predictions.shape, labels.shape)
    print(f'labels[0].shape: {labels[0].shape}, predictions[0].shape: {predictions[0].shape}')

    # Inverse transform
    if config['inverse']:
        inverse_predictions = []
        inverse_labels = []

        for pred in predictions:
            inverse_predictions.append(data.inverse_transform(pred))

        for label in labels:
            inverse_labels.append(data.inverse_transform(label))

        inverse_predictions = np.array(inverse_predictions)
        inverse_labels = np.array(inverse_labels)

        print('inverse dtype:', inverse_predictions.dtype, inverse_labels.dtype)
        print('inverse shape:', inverse_predictions.shape, inverse_labels.shape)

        predictions = inverse_predictions
        labels = inverse_labels

        #labels = data.inverse_transform(labels)
        #predictions = data.inverse_transform(predictions)
    
    if config['pred_len'] == 1: # Only for single predictions
        labels = labels.flatten()
        predictions = predictions.flatten()

        shift_by = 1
        df = pd.DataFrame({'labels': labels, 'predictions': predictions})
        df['shifted'] = df['labels'].shift(shift_by, fill_value=0)
        df['ewma8'] = df['labels'].ewm(span=8, adjust=True).mean().shift(shift_by, fill_value=0)

        # setting record of experiments
        setting = '{}_in{}_pred{}_hidden{}_nl{}_nll{}_dr{}_dl{}_lr{}_ql{}_bs{}_c{}_a{}_s{}_o{}_lrs{}_b{}_bf{}i_{}c_{}me_{}'.format(
            config['model_type'], config['input_size'], config['pred_len'], config['hidden_size'], config['num_layers'], config['num_linear_layers'], 
            config['dropout_rnn'], config['dropout_linear'], config['learning_rate'], config['seq_len'], config['batch_size'],
            config['criterion'], config['activation'], config['scaler'], config['optimizer'], config['lr_scheduler'],
            config['bidirectional'], config['batch_first'], config['inverse'],
            config['current_dataset'], config['max_epochs'])

        directory = 'results/'
        if not os.path.exists(directory):
            os.makedirs(directory)
    
        # result save
        folder_path = 'results/' + setting +'/'
        # ./../../Data/BerlinV2X-preprocessed/
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        df.to_parquet(folder_path+'data.parquet.gzip', compression='gzip')
        df.to_csv(folder_path+'data.csv', encoding='utf-8', index=False)

        model_error_results, model_mse, model_rmse, model_mae, model_std, model_ae = calculate_errors(predictions, labels, model_name)
        shifted_error_results, shifted_mse, shifted_rmse, shifted_mae, shifted_std, shifted_ae = calculate_errors(df['shifted'].to_numpy(), df['labels'].to_numpy(), 'Shifted')
        ewma8_error_results, ewma8_mse, ewma8_rmse, ewma8_mae, ewma8_std, ewma8_ae = calculate_errors(df['ewma8'].to_numpy(), df['labels'].to_numpy(), 'EWMA8')

        if config['current_dataset'] == 'Beyond5G':
             divider = 1024
        elif config['current_dataset'] == 'BerlinV2X':
             divider = 1024*1024
        elif config['current_dataset'] == 'NYU-METS-MM15':
             divider = 1
        else:
             divider = 1
        
        ser_model = pd.Series(model_ae/(divider))
        ser_shifted = pd.Series(shifted_ae/(divider))
        ser_ewma8 = pd.Series(ewma8_ae/(divider))

        cdf = plt.figure(figsize=(16,9), dpi=200)
        #plt.hist([ser_model, ser_shifted, ser_ewma8], cumulative=True, density=1, bins=100, histtype=u'step', alpha=1, label=['LSTM', 'Shifted', 'EWMA8'])
        plt.ecdf(ser_model, label='ecdf LSTM')
        plt.ecdf(ser_shifted, label='ecdf Shifted')
        plt.ecdf(ser_ewma8, label='ecdf EWMA8')
        plt.ylabel('Percentage')
        plt.xlabel('Absolute Error (Mbit/s)')
        plt.xscale('log', base=10)
        plt.title('Cumulative Distribution Function of Absolute Errors')
        plt.grid(True)
        plt.legend()
        plt.show()
        wandb_logger.log_image(key='CDF log', images=[cdf])

        cdf = plt.figure(figsize=(16,9), dpi=200)
        #plt.hist([ser_model, ser_shifted, ser_ewma8], cumulative=True, density=1, bins=100, histtype=u'step', alpha=1, label=['LSTM', 'Shifted', 'EWMA8'])
        plt.ecdf(ser_model, label='ecdf LSTM')
        plt.ecdf(ser_shifted, label='ecdf Shifted')
        plt.ecdf(ser_ewma8, label='ecdf EWMA8')
        plt.ylabel('Percentage')
        plt.xlabel('Absolute Error (Mbit/s)')
        plt.title('Cumulative Distribution Function of Absolute Errors')
        plt.grid(True)
        plt.legend()
        plt.show()
        wandb_logger.log_image(key='CDF', images=[cdf])

        error_fig = model_error_results + '\n' + shifted_error_results + '\n' + ewma8_error_results

        print('Plotting now ...')    
        plt.figure(figsize=(30, 20), dpi=200)

        scope = 60
        plt.subplot(3, 1, 1)
        plt.plot(df['labels'][:scope].div(divider) , "-o", color="r", label="labels")
        plt.plot(df['predictions'][:scope].div(divider), "-o", color="g", label="predictions")
        plt.plot(df['predictions'][:scope].div(divider), "-o", color="g")
        plt.plot(df['shifted'][:scope].div(divider) , "-o", color="b", label="shifted")
        plt.plot(df['ewma8'][:scope].div(divider) , "-o", color="k", label="ewma8")
        plt.xlabel('timestamps')
        plt.ylabel('values')
        plt.figtext(0.5, 0.01, error_fig, wrap=True, horizontalalignment='center', fontsize=12)
        plt.legend()

        scope2 = 60 * 3
        plt.subplot(3, 1, 2)
        plt.plot(df['labels'][:scope2].div(divider) , "-o", color="r", label="labels")
        plt.plot(df['predictions'][:scope2].div(divider), "-o", color="g", label="predictions")
        plt.plot(df['shifted'][:scope2].div(divider) , "-o", color="b", label="shifted")
        plt.plot(df['ewma8'][:scope2].div(divider) , "-o", color="k", label="ewma8")
        plt.xlabel('timestamps')
        plt.ylabel('values')
        plt.figtext(0.5, 0.01, error_fig, wrap=True, horizontalalignment='center', fontsize=12)
        plt.legend()

        plt.subplot(3, 1, 3)
        plt.plot(df['labels'].div(divider) , "-o", color="r", label="labels")
        plt.plot(df['predictions'].div(divider), "-o", color="g", label="predictions")
        plt.plot(df['shifted'].div(divider) , "-o", color="b", label="shifted")
        plt.plot(df['ewma8'].div(divider) , "-o", color="k", label="ewma8")
        plt.xlabel('timestamps')
        plt.ylabel('values')
        plt.figtext(0.5, 0.01, error_fig, wrap=True, horizontalalignment='center', fontsize=12)
        plt.legend()

        wandb_logger.log_image(key=model_name, images=[plt])
    else:
         model_error_results = calculate_errors(predictions, labels, model_name)

def experiment(config=None):
    for _ in range(ITERATIONS):
        config, data, predictions_labels, wandb_logger = train()

        evaluate(config, wandb_logger, data, predictions_labels, 'LSTM', draw_plot=True)

        wandb.finish()

# Test run for flops
#train()

# Debug run
#config, data, predictions_labels, wandb_logger = train()

# Debug run
#evaluate(config, wandb_logger, data, predictions_labels, 'LSTM', draw_plot=True)
#wandb.finish()

# Normal run
experiment()

# wandb sweep
#wandb.agent(sweep_id, function=experiment)
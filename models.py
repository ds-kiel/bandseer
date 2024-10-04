import torch
from torch import nn
import torchmetrics
import lightning.pytorch as pl

# Stateless RNN implementation
class Stateless(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.input_size = config["input_size"]
        self.pred_len = config["pred_len"]
        self.hidden_size = config["hidden_size"]
        self.num_layers = config["num_layers"]
        self.batch_first = config["batch_first"]
        self.dropout_rnn = config["dropout_rnn"]
        self.dropout_linear = config["dropout_linear"]
        self.learning_rate = config["learning_rate"]
        self.bidirectional = config['bidirectional']
        self.optimizer = config['optimizer']
        self.lr_scheduler = config['lr_scheduler']
        self.num_linear_layers = config['num_linear_layers']

        if config["criterion"] == 'MSELoss':
            self.criterion = nn.MSELoss()
        else:
            self.criterion = nn.L1Loss()

        if config["activation"] == 'GELU':
            self.activation = nn.GELU()
        elif config["activation"] == 'ReLU':
            self.activation = nn.ReLU()
        else:
            self.activation = nn.LeakyReLU()

        self.MAE = torchmetrics.MeanAbsoluteError()
        self.MSE = torchmetrics.MeanSquaredError()
        self.RMSE = torchmetrics.MeanSquaredError(squared=False)
        #self.MAPE = torchmetrics.MeanAbsolutePercentageError()
        #self.sMAPE = torchmetrics.SymmetricMeanAbsolutePercentageError()

        self.stateless = nn.LSTM(
            self.input_size,
            self.hidden_size, 
            self.num_layers, 
            batch_first=self.batch_first, 
            dropout=self.dropout_rnn, 
            bidirectional=self.bidirectional)
        
        if self.bidirectional:
            self.in_features = self.hidden_size*2
        else:
            self.in_features = self.hidden_size

        linear_layers = []
        for layer in range(self.num_linear_layers):
            if layer == self.num_linear_layers - 1:
                linear_layers.append(self.activation)
                linear_layers.append(nn.Linear(
                    self.in_features, self.pred_len))
            else:
                linear_layers.append(self.activation)
                linear_layers.append(nn.Linear(
                    self.in_features, self.in_features))
                if self.dropout_linear:
                    linear_layers.append(nn.Dropout(
                        self.dropout_linear))
        self.linear = nn.Sequential(*linear_layers)

        self.validation_step_outputs = []
        self.test_step_outputs = []

        # save hyper-parameters to self.hparams (auto-logged by W&B)
        self.save_hyperparameters()

    def configure_optimizers(self):

        if self.optimizer == 'Adam':
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.learning_rate)
        else:
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=self.learning_rate, amsgrad=False)

        if self.lr_scheduler == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=1, gamma=0.9)
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=5, factor=0.5)

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'mean_val_loss'
        }

    def forward(self, input):
        #input is size (batch_size, seq_len, num_features)
        
        prediction, _ = self.stateless(input) 
        #prediction is size (batch_size, seq_len, hidden_size)

        # get only last output of LSTM
        prediction_out = self.linear(prediction[:, -1]) # equivalent to [:, -1, :]

        return prediction_out

    def training_step(self, batch, batch_idx, dataloader_idx=None):
        inputs, label = batch
        #print(f'inputs.shape: {inputs.shape}, label.shape: {label.shape}')

        prediction = self(inputs)
        
        loss = self.criterion(prediction, label)
        self.log('train_loss', loss)
        #self.log('train_loss', loss, on_step=True, on_epoch=True)
        
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        inputs, label = batch

        prediction = self(inputs)
        #print(f'prediction.shape: {prediction.shape}, label.shape: {label.shape}')

        loss = self.criterion(prediction, label)
        self.log('val_loss', loss)
        #self.log('val_loss', loss, on_step=True, on_epoch=True)
        self.validation_step_outputs.append(loss)

        self.log('val_loss_MAE', self.MAE(prediction, label))
        self.log('val_loss_MSE', self.MSE(prediction, label))
        self.log('val_loss_RMSE', self.RMSE(prediction, label))
        #self.log('val_loss_MAPE', self.MAPE(prediction, label))
        #self.log('val_loss_sMAPE', self.sMAPE(prediction, label))
        
        return loss
    
    def on_validation_epoch_end(self):
        mean_val_loss = torch.stack(self.validation_step_outputs).mean()
        self.log('mean_val_loss', mean_val_loss)
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        inputs, label = batch

        prediction = self(inputs)

        loss = self.criterion(prediction, label)
        self.log('test_loss', loss)
        self.test_step_outputs.append(loss)
        
        return loss
    
    def on_test_epoch_end(self):
        mean_test_loss = torch.stack(self.test_step_outputs).mean()
        self.log('mean_test_loss', mean_test_loss)
        self.test_step_outputs.clear()

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        inputs, label = batch

        prediction = self(inputs)

        return {"label": label, "prediction": prediction}
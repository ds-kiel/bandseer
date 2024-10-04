import os
import lightning.pytorch as pl
import numpy as np
import pandas
import torch

from torch.utils.data import Dataset, DataLoader
from lightning.pytorch.utilities.combined_loader import CombinedLoader

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

class MetricsDataModule(pl.LightningDataModule):

    def __init__(self, config):
        super().__init__()
        self.seq_len = config['seq_len']
        self.batch_size = config['batch_size']
        self.dataset = config['current_dataset']
        self.data_dir = config['data_dir']
        self.num_workers = config['num_workers']
        self.persistent_workers = config['persistent_workers']
        self.pred_len = config['pred_len']

        self.val_p = config['val_p']
        self.test_p = config['test_p']

        if config['scaler'] == 'MinMaxScaler':
            self.scaler_input = MinMaxScaler()
            self.scaler_label = MinMaxScaler()
        else:
            self.scaler_input = StandardScaler()
            self.scaler_label = StandardScaler()

        self.read_data = True

        self.train_inputs = {}
        self.train_labels = {}

        self.val_inputs = {}
        self.val_labels = {}

        self.test_inputs = {}
        self.test_labels = {}

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        
        if self.read_data:
            print('Reading data')

            num_files = len(os.listdir(self.data_dir))
            print('Number of files: {}'.format(num_files))

            file_count = 0
            for file in os.listdir(self.data_dir):
                # Skipping the files we're not using
                if file[-4:] != ".csv":
                    continue

                FILE_NAME = file[:-4]

                df = pandas.read_csv(self.data_dir+file)
                #print(df.info())
                #print(df.head())
                df = reduce_mem_usage(df)
                #print(df.info())

                if self.dataset == 'NYU-METS-MM15':
                    label = df['bandwidth'].to_numpy(dtype=np.float32)
                    input = df[['bandwidth', 'LTE-neighbors', 'RSSI', 'RSRQ','ENodeB-change', 
                                'time-advance', 'speed', 'band']].to_numpy(dtype=np.float32)

                elif self.dataset == 'Beyond5G':        
                    label = df['DL_bitrate'].to_numpy(dtype=np.float32)             
                    input = df[['Speed', 'CellID', 'RSRP', 'RSRQ', 'SNR', 'CQI', 
                                'RSSI', 'DL_bitrate', 'UL_bitrate']].to_numpy(dtype=np.float32)
                    

                elif self.dataset == 'BerlinV2X':
                    label = df['datarate'].to_numpy(dtype=np.float32)
                    input = df[['datarate', 
                                'PCell_Downlink_TB_Size', 'SCell_Downlink_TB_Size',
                                'PCell_SNR_1', 'PCell_SNR_2', 
                                'SCell_SNR_1', 'SCell_SNR_2',
                                'PCell_RSSI_1', 'PCell_RSSI_2', 'PCell_RSSI_max',
                                'PCell_RSRP_1', 'PCell_RSRP_2', 'PCell_RSRP_max',
                                'PCell_Downlink_Average_MCS', 'SCell_Downlink_Average_MCS',
                                'PCell_Downlink_Num_RBs', 'SCell_Downlink_Num_RBs'
                                ]].to_numpy(dtype=np.float32)
                
                # Every file gets split into train, val, test
                # Split into train+validation and test
                train_val_input, self.test_inputs[FILE_NAME] = train_test_split(
                    input, test_size=self.test_p, random_state=42, shuffle=False)
                train_val_label, self.test_labels[FILE_NAME] = train_test_split(
                    label, test_size=self.test_p, random_state=42, shuffle=False)

                # Split into train and validation
                self.train_inputs[FILE_NAME], self.val_inputs[FILE_NAME] = train_test_split(
                    train_val_input, test_size=self.val_p*(1.25), random_state=42, shuffle=False)
                self.train_labels[FILE_NAME], self.val_labels[FILE_NAME] = train_test_split(
                    train_val_label, test_size=self.val_p*(1.25), random_state=42, shuffle=False)

            self.read_data = False

        # Scaling the input data
        if stage == 'fit' or stage is None:
        #if stage == 'fit' or stage == 'test' or stage is None:
            print('fit stage')

            for key in self.train_inputs:
                print(key)
                # Get fitted scalers on training data
                self.scaler_input.partial_fit(self.train_inputs[key])
                self.scaler_label.partial_fit(self.train_labels[key].reshape(-1,1))

            for key in self.train_inputs:
                # Apply scalers on training data
                self.train_inputs[key] = self.scaler_input.transform(self.train_inputs[key])
                self.train_labels[key] = self.scaler_label.transform(self.train_labels[key].reshape(-1,1)).flatten()

            for key in self.val_inputs:
                # Apply scalers on validation data
                self.val_inputs[key] = self.scaler_input.transform(self.val_inputs[key])
                self.val_labels[key] = self.scaler_label.transform(self.val_labels[key].reshape(-1,1)).flatten()

        if stage == 'test' or stage is None:
            print('test stage')
            
            for key in self.test_inputs:
                # Apply scalers on test data
                self.test_inputs[key] = self.scaler_input.transform(self.test_inputs[key])
                self.test_labels[key] = self.scaler_label.transform(self.test_labels[key].reshape(-1,1)).flatten()

    def train_dataloader(self):

        train_loaders = []

        for key in self.train_inputs:

            train_set = TimeSeriesDataset(
                self.train_inputs[key],
                self.train_labels[key],
                self.seq_len,
                self.pred_len)

            train_loader = DataLoader(
                train_set, 
                shuffle=True, 
                batch_size=self.batch_size, 
                drop_last=True, 
                num_workers=self.num_workers,
                persistent_workers=self.persistent_workers, 
                pin_memory=True)
            
            print('len of train_loader: {}'.format(len(train_loader)))
            train_loaders.append(train_loader)
        
        # CombinedLoader does not work with training, 
        # see SequentialLoader class down below
        #return CombinedLoader(train_loaders, 'sequential') 
        return SequentialLoader(*train_loaders)

    def val_dataloader(self):

        val_loaders = []

        for key in self.val_inputs:

            val_set = TimeSeriesDataset(
                self.val_inputs[key],
                self.val_labels[key],
                self.seq_len,
                self.pred_len)

            val_loader = DataLoader(
                val_set, 
                shuffle=False, 
                batch_size=self.batch_size, 
                drop_last=True, 
                num_workers=self.num_workers,
                persistent_workers=self.persistent_workers, 
                pin_memory=True)
            
            val_loaders.append(val_loader)

        #return SequentialLoader(*val_loaders)
        return CombinedLoader(val_loaders, 'sequential')

    def test_dataloader(self):
        test_loaders = []
        for key in self.test_inputs:

            test_set = TimeSeriesDataset(
                self.test_inputs[key],
                self.test_labels[key],
                self.seq_len,
                self.pred_len)

            test_loader = DataLoader(
                test_set, 
                shuffle=False, 
                batch_size=self.batch_size, 
                drop_last=False, 
                num_workers=self.num_workers,
                persistent_workers=self.persistent_workers, 
                pin_memory=True)
            
            test_loaders.append(test_loader)

        #return SequentialLoader(*test_loaders)
        return CombinedLoader(test_loaders, 'sequential')

    def predict_dataloader(self):
        return self.test_dataloader()

    def inverse_transform(self, data):
        return self.scaler_label.inverse_transform(data.reshape(-1,1)).flatten()

class TimeSeriesDataset(Dataset):   
    '''
    Custom Dataset subclass. 
    Serves as input to DataLoader to transform input into sequence data using rolling window. 
    DataLoader using this dataset will output batches of `(batch_size, seq_len, n_features)` shape.
    '''
    def __init__(self, 
        inputs: np.ndarray,
        labels: np.ndarray,
        seq_len: int,
        pred_len: int):

        # Assertions to avoid problems in training
        assert (inputs.dtype == np.float32) or (inputs.dtype == np.float16) 
        assert (labels.dtype == np.float32) or (labels.dtype == np.float16)
        assert len(inputs) == len(labels)

        self.inputs = torch.from_numpy(inputs)
        self.labels = torch.from_numpy(labels)
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return self.inputs.__len__() - self.seq_len - self.pred_len + 1

    def __getitem__(self, index):
        # returns inputs for the model and labels for loss calculation
        return (
            self.inputs[index:index+self.seq_len],
            self.labels[index+self.seq_len:index+self.seq_len+self.pred_len])

# SequentialLoader taken from the following suggestion as 
# Pytorch Lightning does not support sequential loaders in training
# https://github.com/PyTorchLightning/pytorch-lightning/discussions/11024
class SequentialLoader:
    def __init__(self, *dataloaders: DataLoader):
        self.dataloaders = dataloaders

    def __len__(self):
        return sum(len(d) for d in self.dataloaders)

    def __iter__(self):
        for dataloader in self.dataloaders:
            yield from dataloader

# https://www.kaggle.com/code/gemartin/load-data-reduce-memory-usage/notebook
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df
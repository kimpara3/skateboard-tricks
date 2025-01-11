from copy import deepcopy
import numpy as np
from scipy.signal import hilbert

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset, Subset, random_split
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.preprocessing import LabelEncoder

import eeg_utils
   
class SignalPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, split_length=1, transform_type="hilbert", use_sub=True, use_stance=True, use_count=True):
        """
        Parameters:
        split_length (int): The length of the window for averaging. Default is 1 (no windowing).
        """
        self.split_length = split_length
        self.scaling_factors_ = None
        self.signal_channels = 72
        self.transform_type = transform_type
        self.use_sub = use_sub
        self.use_stance = use_stance
        self.use_count = use_count
    
    def _hilbert_and_windowing(self, X):
        """
        Apply Hilbert transform and windowing to the input data.
        Parameters:
        X (np.ndarray): Input data of shape (N, T, C), where
                        N is the number of samples,
                        T is the number of time steps,
                        C is the number of channels.      
        Returns:
        np.ndarray: Transformed data after Hilbert transform and windowing.
        """
        # Apply Hilbert transform to each channel
        X_hilbert = np.abs(hilbert(X, axis=1))
        if self.split_length > 1:
            X_averaged = self._window_and_average(X_hilbert)
        else:
            X_averaged = X_hilbert
        return X_averaged
    
    def _integrate_and_windowing(self, X):
        '''
        X (np.ndarray): Input data of shape (N, T, C), where
                        N is the number of samples,
                        T is the number of time steps,
                        C is the number of channels.      
        '''
        X_integrated = np.abs(X)
        if self.split_length > 1:
            X_averaged = self._window_and_average(X_integrated)
        else:
            X_averaged = X_integrated
        return X_averaged

    def _window_and_average(self, X_hilbert):
        """
        Apply windowing and averaging to the Hilbert transformed data.
        Parameters:
        X_hilbert (np.ndarray): Hilbert transformed data of shape (N, T, C).
        Returns:
        np.ndarray: Averaged data after windowing.
        """
        n_samples, n_timesteps, n_channels = X_hilbert.shape
        n_windows = n_timesteps // self.split_length
        X_windowed = X_hilbert[:, :n_windows * self.split_length, :].reshape(n_samples, n_windows, self.split_length, n_channels)
        X_averaged = X_windowed.mean(axis=2)
        return X_averaged
    
    def fit(self, X, y=None):
        """
        Fit the preprocessor to the data.
        
        Parameters:
        X (np.ndarray): Input data of shape (N, T, C+7), where
                        N is the number of samples,
                        T is the number of time steps,
                        C is the number of channels. 0-15:signals, 16-20:subs, 21:stance 22:trial
        y: Ignored.
        
        Returns:
        self
        """
        end_col = self.signal_channels
        if self.transform_type == "hilbert":
            X_averaged = self._hilbert_and_windowing(X[:,:,:end_col])
        elif self.transform_type == "integrate":
            X_averaged = self._integrate_and_windowing(X[:,:,:end_col])           
        
        # Calculate scaling factors
        self.scaling_factors_ = X_averaged.max(axis=(0,1), keepdims=True)
        
        # Replace zero scaling factors with one to avoid division by zero
        self.scaling_factors_[self.scaling_factors_ == 0] = 1        
        return self
    
    def transform(self, X):
        """
        Transform the data using the fitted preprocessor.
        
        Parameters:
        X (np.ndarray): Input data of shape (N, T, C+7), where
                        N is the number of samples,
                        T is the number of time steps,
                        C is the number of channels. 0-15:signals, 16-20:subs, 21:stance 22:trial
        
        Returns:
        np.ndarray: Transformed data of shape (N, T', C + 7), where
                    T' is the number of time steps after windowing,
                    7 is the number of additional features.
        """
        if self.scaling_factors_ is None:
            raise RuntimeError("The fit method must be called before transform.")
        
        end_col = self.signal_channels
        if self.transform_type == "hilbert":
            X_averaged = self._hilbert_and_windowing(X[:,:,:end_col])
        elif self.transform_type == "integrate":
            X_averaged = self._integrate_and_windowing(X[:,:,:end_col])           
        
        # Scaling each channel using the saved scaling factors
        X_scaled = X_averaged / self.scaling_factors_
        
        if self.use_sub:
            start_col = self.signal_channels
            end_col = end_col+5
            X_add = X[:,::self.split_length,start_col:end_col]
            X_scaled = np.concatenate((X_scaled, X_add), axis=2)

        if self.use_stance:
            start_col = self.signal_channels+5
            end_col = start_col+1
            X_add = X[:,::self.split_length,start_col:end_col]
            X_scaled = np.concatenate((X_scaled, X_add), axis=2)

        if self.use_count:
            start_col = self.signal_channels+6
            end_col = start_col+1
            X_add = X[:,::self.split_length,start_col:end_col]/320
            X_scaled = np.concatenate((X_scaled, X_add), axis=2)
        
        return X_scaled
    
    def fit_transform(self, X, y=None, additional_info=None):
        """
        Fit the preprocessor to the data and then transform it.
        
        Parameters:
        X (np.ndarray): Input data of shape (N, T, C+7), where
                        N is the number of samples,
                        T is the number of time steps,
                        C is the number of channels.
        y: Ignored.
       
        Returns:
        np.ndarray: Transformed data of shape (N, T', C + 7), where
                    T' is the number of time steps after windowing,
                    7 is the number of additional features.
        """
        self.fit(X, y)
        return self.transform(X)


class ConvLSTMModel(nn.Module):
    def __init__(self, num_classes, signal_channels=72, time_steps=None, conv_params=[], lstm_param={}, fc_params=[], dropout=0.0):
        super().__init__()
        
        self.signal_channels = signal_channels
        self.num_classes = num_classes
        
        # Initialize in_channels for Conv1d layers
        in_channels = self.signal_channels
        
        # Create Conv1d layers if conv_params is provided
        self.conv_layers = nn.ModuleList()
        for params in conv_params:
            layers=[
                nn.Conv1d(in_channels=in_channels, **params),
                nn.BatchNorm1d(params["out_channels"]),
                nn.ReLU(),
            ]
            if dropout:
                layers+=[nn.Dropout(dropout)]
            layers+=[nn.MaxPool1d(kernel_size=2, stride=2)]

            self.conv_layers.extend(layers)
            in_channels = params['out_channels']

        last_conv_channels=512
        self.conv_layers.append(nn.Conv1d(in_channels, last_conv_channels, kernel_size=3, stride=1))
        self.conv_layers.append(nn.BatchNorm1d(last_conv_channels))
        
        # Create LSTM layer if lstm_params is provided
        self.lstm = None
        if lstm_param:
            # Calculate LSTM input size based on Conv1d output channels or input size
            lstm_input_size = last_conv_channels

            self.lstm = nn.LSTM(input_size=lstm_input_size, batch_first=True, **lstm_param)
            self.hidden_size = lstm_param['hidden_size']
            self.num_layers = lstm_param.get('num_layers',1)
            in_features = self.hidden_size
        else:
            # Calculate in_features for the first fully connected layer
            assert time_steps is not None, "time_steps is required if lstm is not used."
            if conv_params:
                self.gap=nn.AdaptiveAvgPool1d(1)
                in_features = last_conv_channels
            else:
                in_features = self.signal_channels*time_steps
               
        # Create Linear layers if fc_params is provided
        self.fc_layers = nn.ModuleList()
        for params in fc_params:
            layers = [
                nn.Linear(in_features=in_features, **params),
                nn.ReLU()]
            if dropout:
                layers+=[nn.Dropout(dropout)]
            self.fc_layers.extend(layers)
            in_features = params['out_features']
        
        # Final output layer
        self.head = nn.Linear(in_features, self.num_classes)
    
    def forward(self, x):
        """
        x : torch.tensor(N, T, C)
        """
        # Split the input into signal data and additional input
        if x.shape[2] > self.signal_channels:
            signal_data = x[:, :, :self.signal_channels]  # (N, T, C)
            additional_input = x[:, 0, self.signal_channels:]  # (N, F)
        else:
            signal_data = x  # (N, T, C)
            additional_input = None

        if self.conv_layers:
            signal_data = torch.permute(signal_data, (0, 2, 1))  # (N, C, T)
            for layer in self.conv_layers:
                signal_data = layer(signal_data)
        
        if self.lstm:
            signal_data = torch.permute(signal_data, (0, 2, 1))  # (N, T, C)
            h0 = torch.zeros(self.num_layers, signal_data.size(0), self.hidden_size).to(signal_data.device)
            c0 = torch.zeros(self.num_layers, signal_data.size(0), self.hidden_size).to(signal_data.device)
            signal_data, _ = self.lstm(signal_data, (h0, c0))
            signal_data = signal_data[:, -1, :]  # Use the last time step output
        else:
            if self.conv_layers:
                signal_data = self.gap(signal_data)
                signal_data = signal_data.squeeze(2)
            else:
                signal_data = signal_data.reshape(signal_data.size(0), -1)
        
        # Concatenate additional input
        if x.shape[2] > self.signal_channels:
            x = torch.cat((signal_data, additional_input), dim=1)
        else:
            x = signal_data
        
        if self.fc_layers:
            for layer in self.fc_layers:
                x = layer(x)
        
        # Apply final output layer
        x = self.head(x)
        return x

class SignalDataset(Dataset):
    def __init__(self, signals, raw_label, signal_channels=72, device="cuda", transform=None):
        '''
        signals : (N, T, signal_channels)
        raw_label : (N,) 11,12,13,21,22,23
        '''
        self.signals=torch.tensor(signals[:,:,:signal_channels],dtype=torch.float32,device=device)
        self.label=torch.tensor((raw_label%10)-1 ,device=device)
        self.direction=torch.tensor((raw_label//10)-1, dtype=int, device=device)
        self.transform=transform

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        item={
            'signals': self.signals[idx],
            'label': self.label[idx],
            'direction': self.direction[idx],
            'index': idx
        }

        item["signals"]=self._random_crop(item["signals"], 250)
        if self.transform:
            item["signals"]=self.transform(item["signals"])
        return item
    
    def _random_crop(self, signal, size):
        '''
        sizeにランダムクロップ
        '''
        start_idx = np.random.randint(0, signal.size(0) - size + 1)
        return signal[start_idx:start_idx + size]
    

def transforms(x):
    '''
    x : tensor(T,C)
    '''
    x=add_noise(x)

    # if np.random.randint(2):
    #     return x.flip(dims=[0])
    return x

def add_noise(data, noise_level=0.01):
    noise = torch.normal(0, noise_level, size=data.shape, device=data.device)
    data_noisy = data + noise
    return data_noisy

class SignalEstimator(BaseEstimator, ClassifierMixin):
    def __init__(self, num_epochs, batch_size=64, lr=0.001, weight_decay=1e-4, 
                 f_scale=0.0, 
                 model_history_len=1,
                 conv_params=[], lstm_param={}, fc_params=[], dropout=0.1,
                 num_classes=3,
                 signal_channels=72,
                 seed=1,
                ):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay

        self.f_scale = f_scale
        
        self.seed = seed
        self.signal_channels = signal_channels

        self.conv_params = conv_params
        self.lstm_param = lstm_param
        self.fc_params = fc_params
        self.dropout = dropout

        self.model_history_len = model_history_len #最後からこのepoch数だけモデルを保存する
        self.model_history = []

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fix_seed(seed)

        self.num_classes=num_classes

        self.model = ConvLSTMModel(self.num_classes, signal_channels=self.signal_channels, time_steps=250, conv_params=self.conv_params, lstm_param=self.lstm_param, fc_params=self.fc_params, dropout=self.dropout)

        self.minimum_loss = float('inf')
        self.minimum_loss_weight = None


    @staticmethod
    def fix_seed(seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms = True

    def fit(self, in_arr, out_arr, verbose=0, val_X=None,val_y=None):
        '''
        in_arr : ndarray(N, T, C)
        out_arr : ndarray(N,)
        '''
        # モデルの初期化
        self.model.to(self.device)
        if type(self.lr)==float:
            optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            conv_lr,lstm_lr,fc_lr=self.lr
            param_groups=[
                dict(params=self.model.conv_layers.parameters(), lr=conv_lr),
                dict(params=self.model.fc_layers.parameters(), lr=fc_lr),
                dict(params=self.model.output_layer.parameters(), lr=fc_lr)
            ]
            if self.model.lstm:
                param_groups+=[
                    dict(params=self.model.lstm.parameters(), lr=lstm_lr)
                ]
            optimizer = optim.Adam(param_groups, weight_decay=self.weight_decay)
            
        criterion = nn.CrossEntropyLoss(reduction="sum")

        dataset = SignalDataset(in_arr, out_arr, signal_channels=self.signal_channels, device=self.device, transform=transforms)
        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        val_loader = None

        if val_X is not None:
            val_dataset = SignalDataset(val_X, val_y, signal_channels=self.signal_channels, device=self.device)
            val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)

        train_loss_history=[]
        val_loss_history=[]
        for epoch in range(self.num_epochs):
            self.model.train()

            train_loss = 0.0
            for b in train_loader:
                x=b['signals']
                targets=b['label']
                optimizer.zero_grad()
                outputs = self.model(x)

                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss = train_loss / len(train_loader.dataset)
            train_loss_history.append(train_loss)

            if val_loader:
                self.model.eval()
                with torch.no_grad():
                    for b in val_loader:
                        x = b["signals"]
                        targets = b["label"]
                        val_outputs = self.model(x)
                        val_loss = criterion(val_outputs, targets).item()

                        _,predicted = torch.max(val_outputs.data, 1)
                        data_correct = (predicted==targets).sum().item()

                val_loss /= len(val_dataset)
                val_loss_history.append(val_loss)
                accuracy = data_correct/len(val_dataset)

                if verbose and  (epoch+1)%10==0 or epoch+1==self.num_epochs:
                    print(f'Epoch [{epoch+1}/{self.num_epochs}], '
                          f'Loss: {train_loss:.4f}, '
                          f'Val Loss: {val_loss:.4f}, '
                          f'Acc: {accuracy:.3f}'
                    )
            else:
                if verbose and  (epoch+1)%10==0 or epoch+1==self.num_epochs:
                    print(f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {train_loss:.4f}')

            self.save_model_history(self.model, val_loss if val_loader else None)

        train_loss_history=np.array(train_loss_history)
        val_loss_history=np.array(val_loss_history)
        
        return train_loss_history, val_loss_history   
  
    def predict(self, X:np.ndarray)->np.ndarray:
        probabilities = self.predict_proba(X)
        categories = np.argmax(probabilities, axis=1)+1
        return categories

    def predict_proba(self, X:np.ndarray)->np.ndarray:
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        self.model.eval()
        with torch.no_grad():
            self.model.load_state_dict(self.minimum_loss_weight)
            self.model=self.model.to(self.device)
            logits = self.model(X_tensor[:,:,:self.signal_channels])

        probabilities = self.postprocess(logits)
        return probabilities
    
    def __call__(self, X:np.ndarray)->torch.Tensor:
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        self.model.eval()
        with torch.no_grad():
            self.model.load_state_dict(self.minimum_loss_weight)
            self.model=self.model.to(self.device)
            logits = self.model(X_tensor[:,:,:self.signal_channels])

        return logits
    
    
    def postprocess(self, logits):
        '''
        cnn_output : tensor(N, 6)
        '''
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        probabilities = probabilities.cpu().numpy()
        return probabilities
    
    def save_model_history(self,model:nn.Module, loss=None):
        '''
        save snapshot
        '''
        model.eval()
        self.model_history.append(deepcopy(model.state_dict()))
        self.model_history=self.model_history[-self.model_history_len:]

        if loss is not None and loss < self.minimum_loss:
            self.minimum_loss=loss
            self.minimum_loss_weight=deepcopy(model.state_dict())
        
# **LightNet+: A Dual-Source Lightning Forecasting Network with Bi-direction Spatiotemporal Transformation**

This is the origin Pytorch implementation of LightNet+.

## Requirements

- Python 3.6
- numpy == 1.18.4
- torch == 1.5.0+cu92
- DCN installation refer to:  https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch



## Directory description
**checkpoints**　　　　　<br>
　|-model.pkl　　　　　\#  Trained models.  <br>

**data_index**　　　　　\# train, validation, test set index files.  <br>
　|-TrainCase.txt　　　　\# The periods used for train.  <br>
　|-ValCase.txt　　　　\# The periods used for validation.  <br>
　|-TestCase.txt　　　　\# The periods used for test.  <br>

**deformable_convolution**	　　　　# You need to install it yourself according to https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch.  <br>

**layers**           　　　　        \# Define structure of all models.  <br>
　|-ablation.py　　　　\# The ablation study code.   <br>
　|-ConvLSTM.py　　　　\# The code of ConvLSTM.  <br>
　|-LightNet_plus.py　　　　\# The code of LightNet+.   <br>
　|-transformer_decoder.py　　　　\# The code of transformer decoder.   <br>

config.py					　　　　# The script to read configuration file.  <br>

config_train				　　　　# Configuration file.  <br>

generator.py				　　　　# Load data and format them.  <br>

main.py						　　　　# Train the neural network model.  <br>

scores.py					　　　　# Calculate performance scores for prediction results.  <br>

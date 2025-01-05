import torch
import torch.nn as nn
import torch.nn.init as init

class CLSTM(nn.Module):
    # 输入的维度顺序为 (batch_size, input_channels, sequence_length)
    def __init__(self, CLArgesDicts, LSTMArgesDict,lastSeqLength,num_classes=2, bias=True):
        super(CLSTM, self).__init__()
        self.LSTM_layers_num=LSTMArgesDict['LSTM_layers_num']
        self.Conv_layers_num=len(CLArgesDicts)
        # 卷积层+归一化+激活函数(input_size, output_size, kernel_size=ks, stride=stride, padding=padding, bias=bias)
        for i, argesItem in enumerate(CLArgesDicts):
            setattr(self, f'conv{i+1}', nn.Conv1d(argesItem['input_size'], argesItem['output_size'],
                                                  kernel_size=argesItem['kernel_size'], stride=argesItem['stride'], 
                                                  padding=argesItem['padding'], bias=True, dtype=torch.float32)) 
            if argesItem['batchnorm1D'] is not None :
                setattr(self, f'bn1d{i+1}', nn.BatchNorm1d(argesItem['output_size'], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True,dtype=torch.float32))
            else:
                setattr(self, f'bn1d{i+1}', None)
            if argesItem['activation'] == "swish":
                setattr(self, f'acti{i+1}', nn.SiLU())
            elif argesItem['activation'] is not None:
                setattr(self, f'acti{i+1}', nn.Tanh())
            else:
                setattr(self, f'acti{i+1}', None)
        
        for i in range(LSTMArgesDict['LSTM_layers_num']):
            lstm = nn.LSTM(input_size=LSTMArgesDict['LSTM_input_size'], hidden_size=LSTMArgesDict['LSTM_hidden_size'], num_layers=1, bidirectional=False, dtype=torch.float32) 
            for name, param in lstm.named_parameters():
                if 'weight_ih' in name or 'weight_hh' in name:
                    init.orthogonal_(param)
            setattr(self, f'LSTM{i+1}', lstm)
        # 全连接层
        self.fc = nn.Linear(in_features=lastSeqLength*LSTMArgesDict['LSTM_input_size'], out_features=int(num_classes), dtype=torch.float32)
        

    def forward(self, x):
        # conv部分 输入数据格式(batch_size, input_channels, sequence_length)
        for i in range(0,self.Conv_layers_num):
            x = getattr(self, f'conv{i+1}')(x)
            # 批归一化
            if getattr(self, f'bn1d{i+1}') is not None:
                x = getattr(self, f'bn1d{i+1}')(x)
            # 激活函数
            if getattr(self, f'acti{i+1}') is not None:
                x = getattr(self, f'acti{i+1}')(x)
        #维度变换
        x=x.permute([2, 0, 1])
        # LSTM部分 输入数据格式(sequence_length, batch_size, input_size)
        for i in range(0,self.LSTM_layers_num):
            reverse=True if (self.LSTM_layers_num - i) % 2== 1 else False
            if reverse : x = x.flip(0) 
            x, _ = getattr(self, f'LSTM{i+1}')(x)
            if reverse : x = x.flip(0)
        batch_size = x.size(1)
        x = x.permute(1, 0, 2).contiguous().view(batch_size, -1)  #`contiguous()`方法确保张量在内存中是连续的
        x = self.fc(x)
        return x

def get_pretrain_model():
    CLArgesList = [
        [1, 16, 5, 1, 2, "swish", True],
        [16, 64, 13, 6, 5, "tanh", True]
    ]
    CLArgesDict = []
    for args in CLArgesList:
        args_dict = {
            'input_size': args[0],
            'output_size': args[1],
            'kernel_size': args[2],
            'stride': args[3],
            'padding': args[4],
            'activation': args[5],
            'batchnorm1D': args[6]
        }
        CLArgesDict.append(args_dict)
    LSTMArgesDict={
        'LSTM_input_size':64,
        'LSTM_hidden_size':64,
        'LSTM_layers_num':2,
        'orthogonal_weight_init':True,
        }
    lastSeqLength=417 #每层[(输入长度-卷积核大小+2*填补padding长度)/步长+1]
    model = CLSTM(CLArgesDict,LSTMArgesDict,lastSeqLength)
    return model
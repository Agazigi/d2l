import torch
import math
from torch import nn
from torch.autograd import Variable
from torch.nn import functional, init
from torch.nn.init import kaiming_normal, kaiming_uniform, constant

#model

class SeparatedBatchNorm1d(nn.Module): # 批归一化层 （Time Batch Normalization）。

    """
    A batch normalization module which keeps its running mean
    and variance separately per timestep.
    一个批量归一化模块，它在每个时间步长分别保持其运行平均值和方差。
    """

    def __init__(self, num_features, max_length, eps=1e-3, momentum=0.1,
                 affine=True):
        """
        Most parts are copied from
        torch.nn.modules.batchnorm._BatchNorm.
        """

        super(SeparatedBatchNorm1d, self).__init__()
        self.num_features = num_features  #表示特征的数量
        self.max_length = max_length  #时间步的最大长度
        self.affine = affine  #是否使用可学习的缩放和偏移参数
        self.eps = eps   #数值稳定性
        self.momentum = momentum  #控制运行平均的更新速率
        if self.affine:
            self.weight = nn.Parameter(torch.FloatTensor(num_features)) # 创建可学习的缩放权重 `weight` 和偏移 `bias` 参数
            self.bias = nn.Parameter(torch.FloatTensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        for i in range(max_length):  #通过循环为每个时间步创建并注册缓冲区（buffer）用于存储每个时间步的运行均值和方差
            self.register_buffer(
                'running_mean_{}'.format(i), torch.zeros(num_features))
            self.register_buffer(
                'running_var_{}'.format(i), torch.ones(num_features))   
        self.reset_parameters()

    def reset_parameters(self): #初始化参数
        for i in range(self.max_length): 
            running_mean_i = getattr(self, 'running_mean_{}'.format(i))
            running_var_i = getattr(self, 'running_var_{}'.format(i))
            running_mean_i.zero_()
            running_var_i.fill_(1)
        if self.affine:
            # 将权重初始化为均匀分布，偏移初始化为零。
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def _check_input_dim(self, input_):
        # 检查输入张量的特征维度是否与模型期望的特征数匹配
        if input_.size(1) != self.running_mean_0.nelement():
            raise ValueError('got {}-feature tensor, expected {}'
                             .format(input_.size(1), self.num_features))

    def forward(self, input_, time): #前向传播过程，执行批归一化操作
        self._check_input_dim(input_)
        if time >= self.max_length:
            time = self.max_length - 1
        running_mean = getattr(self, 'running_mean_{}'.format(time))
        running_var = getattr(self, 'running_var_{}'.format(time))
        return functional.batch_norm( #批归一化
            input=input_, running_mean=running_mean, running_var=running_var,
            weight=self.weight, bias=self.bias, training=self.training,
            momentum=self.momentum, eps=self.eps)
    
    def __repr__(self): #输出
        return ('{name}({num_features}, eps={eps}, momentum={momentum},'
                ' max_length={max_length}, affine={affine})'
                .format(name=self.__class__.__name__, **self.__dict__))

class BNLSTMCell(nn.Module):

    """A BN-LSTM cell."""

    def __init__(self, input_size, hidden_size, max_length, use_bias=True):

        super(BNLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.use_bias = use_bias
         # 初始化输入到隐藏状态的权重和隐藏状态到隐藏状态的权重
        self.weight_ih = nn.Parameter(
            #映射到隐藏状态的四个门控单元（输入门、遗忘门、输出门和细胞状态）
            torch.FloatTensor(input_size, 4 * hidden_size))
        self.weight_hh = nn.Parameter(
            torch.FloatTensor(hidden_size, 4 * hidden_size))
        if use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(4 * hidden_size))
        else:
            self.register_parameter('bias', None)
        # 输入到隐藏状态、隐藏状态到隐藏状态和细胞状态的归一化。
        self.bn_ih = SeparatedBatchNorm1d(num_features=4 * hidden_size, max_length=max_length)
        self.bn_hh = SeparatedBatchNorm1d(num_features=4* hidden_size, max_length=max_length)
        self.bn_c = SeparatedBatchNorm1d(num_features=hidden_size, max_length=max_length)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize parameters following the way proposed in the paper.
        """

        init.orthogonal_(self.weight_ih.data) #权重矩阵初始化为一个正交矩阵：其转置矩阵与自身的逆矩阵相等，减少梯度消失或爆炸
        weight_hh_data = torch.eye(self.hidden_size) #单位矩阵，较好的信息流动性，有助于模型学习长期依赖关系
        weight_hh_data = weight_hh_data.repeat(1, 4) #沿着列复制了4次扩展，使其适应LSTM中四个门（遗忘门、输入门、输出门和更新门）
        with torch.no_grad():  #在 `torch.no_grad()` 环境中进行的操作不会影响梯度计算。
            self.weight_hh.set_(weight_hh_data)
        init.constant_(self.bias.data, val=0) # 偏置设置为0向量
        # Initialization of BN parameters.
        self.bn_ih.reset_parameters()
        self.bn_hh.reset_parameters()
        self.bn_c.reset_parameters() # 调用了 Batch Normalization 层对象中的 `reset_parameters()` 方法
        self.bn_ih.bias.data.fill_(0)
        self.bn_hh.bias.data.fill_(0)
        self.bn_ih.weight.data.fill_(0.1)
        self.bn_hh.weight.data.fill_(0.1)
        self.bn_c.weight.data.fill_(0.1)

    def forward(self, input_, hx, time):
        """
        Args:
            input_: A (batch, input_size) tensor containing input
                features
                .input_：包含输入的（batch，input_size）张量
            hx: A tuple (h_0, c_0), which contains the initial hidden
                and cell state, where the size of both states is
                (batch, hidden_size).
                一个元组（h_0，c_0），它包含初始隐藏和单元状态，其中两个状态的大小均为（批次，隐藏_大小）
            time: The current timestep value, which is used to
                get appropriate running statistics.
                当前时间步长值，用于获取适当的运行统计信息。
        Returns:
            h_1, c_1: Tensors containing the next hidden and cell state.
            包含下一个隐藏状态和单元格状态的张量。
        """

        # print(input_)
        h_0, c_0 = hx
        batch_size = h_0.size(0)  #从输入参数 `hx` 中解包出初始的隐藏状态 `h_0` 和细胞状态 `c_0`，并获取批量大小 `batch_size`
        #在第 0 维（即最外层维度）上增加一个维度，将偏置项在第 0 维（新增的维度）上复制扩展 `batch_size` 次，同时保持其他维度与原始偏置项的维度一致。
        bias_batch = (self.bias.unsqueeze(0).expand(batch_size, *self.bias.size())) 
        # 通过矩阵乘法计算隐藏状态的权重项 `wh`（`h_0` 乘以隐藏到隐藏状态的权重矩阵）和输入的权重项 `wi`（`input_` 乘以输入到隐藏状态的权重矩阵）
        wh = torch.mm(h_0, self.weight_hh)
        wi = torch.mm(input_, self.weight_ih)
        # 将隐藏状态的权重项和输入的权重项分别通过 Batch Normalization 层 `bn_hh` 和 `bn_ih` 进行处理

        bn_wh = self.bn_hh(wh, time=time)
        bn_wi = self.bn_ih(wi, time=time)
         # 将 `bn_wh`, `bn_wi` 和 `bias_batch` 沿着第 1 维度进行分割，每个子张量的大小为 `self.hidden_size`  # 遗忘门（f）、输入门（i）、输出门（o）、更新门（g）
        f, i, o, g = torch.split(bn_wh + bn_wi + bias_batch,split_size_or_sections=self.hidden_size, dim=1)
        # 使用门控机制更新细胞状态 `c_1`，然后计算新的隐藏状态 `h_1`。c1= sigmoid(f) ×c0 +sigmoid(i) x tanh(g)
        c_1 = torch.sigmoid(f)*c_0 + torch.sigmoid(i)*torch.tanh(g)
        # h1= sigmoid(o) x tanh(BN(c1))
        h_1 = torch.sigmoid(o) * torch.tanh(self.bn_c(c_1, time=time))
        # print(h_1)
        return h_1, c_1

class bnLSTM_32window(nn.Module):

    """A module that runs multiple steps of LSTM."""

    def __init__(self, input_size, hidden_size,max_length,  num_layers=1,
                 use_bias=True, batch_first=False, dropout=0.5,num_classes = 2):
        super(bnLSTM_32window, self).__init__()
        # self.cell_class = cell_class
        self.real_input_size = input_size
        self.input_size = input_size
        self.max_length = max_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_bias = use_bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.fc = nn.Linear(in_features=hidden_size, out_features=int(num_classes)) #全连接层（Linear）
        for layer in range(num_layers):
            layer_input_size = self.input_size  if layer == 0 else hidden_size  #计算当前层的输入大小
             # 创建 BNLSTMCell，其中包括输入特征大小为 `layer_input_size`，隐藏状态大小为 `hidden_size`，最大序列长度为 `max_length`，并且使用偏置项（bias）
            cell = BNLSTMCell(input_size=layer_input_size,hidden_size=hidden_size, max_length = max_length, use_bias=True)
            setattr(self, 'cell_{}'.format(layer), cell)  #将创建的 BNLSTMCell 设置为模型的属性
        self.dropout_layer = nn.Dropout(dropout)   #Dropout 层，在训练中随机失活以防止过拟合
        self.reset_parameters() 

    def get_cell(self, layer):
        return getattr(self, 'cell_{}'.format(layer))

    def reset_parameters(self):
        for layer in range(self.num_layers): 
            cell = self.get_cell(layer)   #调用get_cell(layer)方法获取当前循环层对应的单元（cell）
            cell.reset_parameters()   # 初始化该循环层的参数

    @staticmethod
    def _forward_rnn(cell, input_, length, hx): #  cell：RNN 单元如LSTM、GRU等，hx：RNN 单元初始隐藏状态
        max_time = input_.size(0) #获取输入序列的时间步数
        output = [] 
        for time in range(max_time): 
            h_next, c_next = cell(input_=input_[time], hx=hx, time=time) # 在 RNN 中计算当前时间步的输出隐藏状态h_next和细胞状态c_next
            # mask与隐藏状态h_next形状相同的张量,控制隐藏状态的更新。当前时间步小于序列长度，mask为1.0，表示需要更新隐藏状态；否则为0.0，表示保持隐藏状态不变。
            mask = (time < length).float().unsqueeze(1).expand_as(h_next)
            # 根据掩码更新隐藏状态和细胞状态。
            h_next = h_next*mask + hx[0]*(1 - mask)
            c_next = c_next*mask + hx[1]*(1 - mask)
            hx_next = (h_next, c_next) #更新隐藏状态和细胞状态。
            output.append(h_next)  #将当前时间步的隐藏状态添加到输出列表中。
            hx = hx_next 
        output = torch.stack(output, 0)  #将输出列表中的隐藏状态堆叠起来，形成一个张量
        # print( output, hx)
        return output, hx

    def forward(self, input_, length=None, hx=None):
        if self.batch_first:
            input_ = input_.transpose(0, 1)
        h0 = Variable(input_.data.new(input_.size(1), self.hidden_size)
                      .normal_(0, 0.1))
        c0 = Variable(input_.data.new(input_.size(1), self.hidden_size)
                      .normal_(0, 0.1))
        hx = (h0, c0)
        max_time, batch_size, _ = input_.size()
        if length is None:
            length = Variable(torch.LongTensor([max_time] * batch_size))
            if input_.is_cuda:
                device = input_.get_device()
                length = length.cuda(device)
        if hx is None:
            hx = Variable(input_.data.new(batch_size, self.hidden_size).zero_())
            hx = (hx, hx)
        h_n = []
        c_n = []
        layer_output = None
        for layer in range(self.num_layers):
            cell = self.get_cell(layer)
            if layer == 0:
                layer_output, (layer_h_n, layer_c_n) = bnLSTM_32window._forward_rnn(
                    cell=cell, input_=input_, length=length, hx=hx)
            else:
                layer_output, (layer_h_n, layer_c_n) = bnLSTM_32window._forward_rnn(
                    cell=cell, input_=layer_output, length=length, hx=hx)
            input_ = self.dropout_layer(layer_output)
            h_n.append(layer_h_n)
            c_n.append(layer_c_n)
        output = layer_output
        output = output[-1]
        output = functional.softmax(self.fc(output), dim=1)
        return output
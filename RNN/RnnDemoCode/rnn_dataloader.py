import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import numpy as np
import os
class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.signal_data = data
        self.labels = labels
        self.transform = transform
        if self.transform is not None:
            self.signal_data = np.array([self.transform(sample) for sample in self.signal_data])

    def __len__(self):
        return len(self.signal_data)

    def __getitem__(self, index):
        batch_signal_data = self.signal_data[index]
        batch_signal_data = batch_signal_data.unsqueeze(0)  # 添加channel维度为1
        batch_label = self.labels[index]
        # 输入的维度顺序为 (batch_size, input_channels, sequence_length)
        return batch_signal_data, batch_label
    
def get_dataset(positive_datapath,negative_datapath,data_slice_length=3000,data_suffix='.npz'):
    datadir=[positive_datapath,negative_datapath]
    merged_array=np.empty((0, data_slice_length),dtype="int16")
    for index in range(len(datadir)) :     
        file_list = [file for file in os.listdir(datadir[index]) if file.endswith(data_suffix)]
        print(datadir[index]," 下文件数量有 ",str(len(file_list)),"个")
        # 遍历文件列表
        for file in file_list:
            file_path = os.path.join(datadir[index], file)
            data = np.load(file_path, allow_pickle=True)
            #列出文件中的所有数组
            array_names = data.files
           # 访问并查看数组
            array=[]
            for array_name in array_names:
                array = data[array_name]
            #关闭文件
            data.close()
            merged_array = np.vstack((merged_array, array))
        if index == 0: 
            positive_datalen=len(merged_array)
        else :
            negative_datalen=len(merged_array)-positive_datalen
    print("positive_datalen:",positive_datalen,"   negative_datalen",negative_datalen)
    merged_array_label = np.concatenate((np.ones(positive_datalen,dtype="int16"), np.zeros(negative_datalen,dtype="int16")))
    # merged_array = np.hstack((merged_array,  merged_array_label.reshape(-1, 1))) 测试使用
    signal = torch.tensor(merged_array).float()
    label = torch.tensor(merged_array_label).long()
    return signal,label

def getDataLoader(dataArgesDict):
    if ('positive_datapath' not in dataArgesDict) or ('negative_datapath' not in dataArgesDict) or ('batch_size' not in dataArgesDict) or ('data_slice_length' not in dataArgesDict) :
        return print("The dictionary must include this keys: positive_datapath, negative_datapath, batch_size,data_slice_length")
    #获取数据集
    signal,label=get_dataset(dataArgesDict['positive_datapath'],dataArgesDict['negative_datapath'],
                            data_slice_length=dataArgesDict['data_slice_length'])
    #创建Dataset
    transforms = dataArgesDict.get('transforms', None)  # 如果传入值存在，则使用对应值，否则使用默认值
    dataset=CustomDataset(signal,label,transform=transforms)
    #创建DataLoader
    shuffleFlag = dataArgesDict.get('shuffleFlag', True)
    dropLastFlag = dataArgesDict.get('dropLastFlag', True)
    numWorkers = dataArgesDict.get('numWorkers', 0)
    usePinMemmory=dataArgesDict.get('usePinMemmory', False)
    dataLoader=DataLoader(dataset=dataset,batch_size=dataArgesDict['batch_size'], 
                          shuffle=shuffleFlag,drop_last =dropLastFlag,num_workers = numWorkers, pin_memory = usePinMemmory)
    return dataLoader
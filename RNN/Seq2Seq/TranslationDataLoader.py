import hashlib
import os
import requests
import torch as th
import zipfile
import tarfile
from torch.utils.data import DataLoader, TensorDataset
from RNN.utils.Loader import Vocab

DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/' # 数据集URL
DATA_HUB['fra-eng'] = (DATA_URL + 'fra-eng.zip', '94646ad1522d915e7b0f9296181140edcf86a4f5')

def download(name, cache_dir=os.path.join('.', 'data')):
    """
    下载数据集
    """
    assert name in DATA_HUB, f"{name} does not exist in {DATA_HUB}." # 断言，如果 name 不在 DATA_HUB 中，则报错
    url, sha1_hash = DATA_HUB[name] # 获取数据集的 URL 和 Hash 值
    os.makedirs(cache_dir, exist_ok=True) # 创建缓存目录
    fname = os.path.join(cache_dir, url.split('/')[-1]) # 获取缓存文件的路径
    if os.path.exists(fname):
        sha1 = hashlib.sha1() # 初始化哈希对象
        with open(fname, 'rb') as f: # 打开文件
            while True:
                data = f.read(1048576) # 读取 1MB 数据
                if not data:
                    break
                sha1.update(data) # 更新哈希值，就是将 data 加到哈希对象中
        if sha1.hexdigest() == sha1_hash: # 判断哈希值是否相等
            return fname # 如果相等，返回文件名
    print(f'Downloading {fname} from {url}...')
    r = requests.get(url, stream=True, verify=True) # 发送请求
    with open(fname, 'wb') as f:
        f.write(r.content) # 写入文件
    return fname

def download_extract(name, folder=None):
    """
    下载并解压数据集
    """
    fname = download(name) # 下载数据集
    base_dir = os.path.dirname(fname) # 获取数据集的目录
    data_dir, ext = os.path.splitext(fname) # 获取数据集的文件名和扩展名
    if ext == '.zip': # 如果是 zip 文件，则使用 zipfile 模块解压
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'): # 如果是 tar 文件，则使用 tarfile 模块解压
        fp = tarfile.open(fname, 'r')
    else: # 其他格式，则报错
        assert False, 'Only zip/tar files can be extracted.'
    fp.extractall(base_dir) # 解压文件
    return os.path.join(base_dir, folder) if folder else data_dir # 返回数据集的目录

def read_data_nmt():
    """
    返回数据集
    """
    data_dir = download_extract('fra-eng') # 下载并解压数据集
    with open(os.path.join(data_dir, 'fra.txt'), 'r', encoding='utf-8') as f: # 读取数据集
        return f.read() # 返回数据集


def preprocess_nmt(text):
    """
    文本预处理
    """
    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' ' # 判断是否需要删除空格，如果是标点符号且前一个字符不是空格，则返回 True
    # 使用空格替换不间断的空格；使用小写字母替换大写字母；
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower() # \u202f 是不间断空格，\xa0 是不可见空格
    # 对文本 text 的每一个字符进行处理
    # i: 字符的索引，char: 当前字符
    # 如果 i > 0 且当前字符为标点上一个字符不是空格的时候。将标点前边加上空格
    # 除此之外直接返回当前字符
    out = [' ' + char if i > 0 and no_space(char, text[i - 1])
           else char
           for i, char in enumerate(text)]
    return ''.join(out)

def tokenize_nmt(text, num_examples=None):
    """
    词元化
    """
    source, target = [], []
    for i, line in enumerate(text.split('\n')): # 遍历每一行
        if num_examples and i > num_examples:
            break
        parts = line.split('\t') # 将每行按照 tab 符号进行分割
        if len(parts) == 2: # 如果有 2 个部分，则继续处理
            source.append(parts[0].split(' ')) # 将第一个部分按照空格进行分割，并添加到 source 中
            target.append(parts[1].split(' ')) # 将第二个部分按照空格进行分割，并添加到 target 中
    return source, target

def truncate_pad(line, num_steps, padding_token):
    """
    截断或填充文本序列
    """
    if len(line) > num_steps:
        return line[:num_steps] # 如果长度大于 num_steps，则截断
    return line + [padding_token] * (num_steps - len(line)) # 如果长度小于 num_steps，则填充 padding_token

def build_array_nmt(lines, vocab, num_steps):
    """
    将机器翻译的文本序列转换成小批量
    """
    lines = [vocab[l] for l in lines]
    lines = [l + [vocab['<eos>']] for l in lines]
    array = th.tensor([truncate_pad(l, num_steps, vocab['<pad>']) for l in lines])
    valid_len = (array != vocab['<pad>']).type(th.int32).sum(1)
    return array, valid_len

def load_data_nmt(batch_size, num_steps, num_examples=600):
    """
    加载翻译数据集
    """
    text = preprocess_nmt(read_data_nmt()) # 读取数据集，之后进行预处理
    source, target = tokenize_nmt(text, num_examples) #　词元化
    # 构建词表
    src_vocab = Vocab(source, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])
    tgt_vocab = Vocab(target, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])
    # 构建数组和有效长度
    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)
    # 构建小批量
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    dataset = TensorDataset(*data_arrays)
    data_iter = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # 返回数据迭代器和词表
    return data_iter, src_vocab, tgt_vocab

train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size=2, num_steps=8)

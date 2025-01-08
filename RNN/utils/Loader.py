import hashlib
import collections
import re
import os
import requests
import torch as th
import random

# 文本预处理
'''
    转化成字符串
    拆分为词元
    建立一个词表，映射到数字索引
    将文本转化为数字索引
'''
DATA_HUB = dict() # 数据集存储位置
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/' # 数据集URL
DATA_HUB['time_machine'] = (DATA_URL + 'timemachine.txt', '090b5e7e70c295757f55df93cb0a180b9691891a') # 后边的是 Hash 值

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



'''
    读取数据集，进行正则、小写等
'''
def read_time_machine():
    """
    读取数据集
    """
    with open(download('time_machine') ,'r') as f:
        lines = f.readlines() # 读取所有行
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines] # 正则匹配，去掉非字母，并转换为小写，并且转化为列表

# lines = read_time_machine()
# print(f'# 文本总行数：{len(lines)}')
# print(lines[0])


'''
    将内容词元化
'''
def tokenize(lines, token='word'):
    """
    将整个文本进行词元化
    """
    if token == 'word':
        return [line.split() for line in lines] # 将文本拆分为单词
    elif token == 'char':
        return [list(line) for line in lines] # 将文本拆分为字符
    else:
        print('错误：未知词元类型：' + token)

# tokens = tokenize(lines)
# for i in range(20):
#     print(tokens[i])

'''
    制作词表，将字符串的单词和数字相对应
'''
def count_corpus(tokens):
    """
    统计词频
    """
    if len(tokens) == 0 or isinstance(tokens[0], list): # 如果 tokens 是一个列表，则将所有元素合并成一个列表
        tokens = [token for line in tokens for token in line] # 对 tokens 的每一 line，再对 line 中的每个单词，再对单词进行拆分
    return collections.Counter(tokens) # 统计词频。返回一个字典，键是单词，值是该单词出现的次数。

class Vocab:
    """
    制作词表的类。建立数字索引和单词的映射
    """
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        """
        tokens 是整个词元表
        min_freq 是一个整型，表示词元的最小出现频率，小于这个频率的就不统计
        reserved_tokens 是一个列表，表示需要保留的单词
        idx_to_token 这是一个列表，下标就是索引，其中存放的元素是单个词
        token_to_idx 是一个字典，其中单词为键，其值是该单词在 idx_to_token 列表中的索引
        """
        if tokens is None:
            tokens = [] # 词表初始为空
        if reserved_tokens is None:
            reserved_tokens = [] # 保留的词元初始为空
        counter = count_corpus(tokens) # 统计词频
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True) # 按词频排序。词频从大到小排序
        self.idx_to_token = ['<unk>'] + reserved_tokens # 词表初始化，其中 0 为 unk
        self.token_to_idx = { # 词表索引初始化
            token: idx for idx, token in enumerate(self.idx_to_token)
        }
        for token, freq in self._token_freqs: # 遍历词频列表
            if freq < min_freq: # 如果词频小于最小频率，则跳过
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1 # 这个词没有被添加过，添加进去，并且返回索引
    def __len__(self):
        return len(self.idx_to_token)
    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk) # 当单词不在词表中，返回unk索引
        return [self.__getitem__(token) for token in tokens] # 批量索引
    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]
    def unk(self):
        return 0
    def token_freqs(self):
        return self._token_freqs

# vocab = Vocab(tokens)
# print(list(vocab.token_to_idx.items())[:20])

def load_corpus_time_machine(max_tokens=-1):
    """
    整合上边的功能，读取数据，词元化，制作词表
    """
    lines = read_time_machine()
    tokens = tokenize(lines, 'char') # 将文本拆分为字符
    vocab = Vocab(tokens)
    corpus = [vocab[token] for line in tokens for token in line] # 对于 tokens 中的 每一 line，再对 line 中的每个单词，再对单词进行拆分
                                                                # 从而对原文本进行索引化
    if max_tokens > 0: # 如果最大词数大于 0，则只保留前 max_tokens 个词
        corpus = corpus[:max_tokens]
    return corpus, vocab

# corpus, vocab = load_corpus_time_machine()
# print(f'语料库大小：{len(corpus)}')
# print(f'词表大小：{len(vocab)}')


'''
    问题一：停用词过多
    问题二：词频衰减。符合齐普夫定律，对于一元、二元、三元词的词频，越低越少。
    问题三：拉普拉斯平滑不适合语言建模。
'''

'''
    如何读取长序列文本
    随机采样
'''
def seq_data_iter_random(corpus, batch_size, num_steps):
    """
    使用随机抽样生成一个小批量子序列
    corpus: 对原始文本的索引化
    batch_size: 批次大小
    num_steps: 子序列长度
    """
    corpus = corpus[random.randint(0, num_steps-1):] # 从随机偏移量开始对序列进行分区
    num_subseqs = (len(corpus) - 1) // num_steps # 减去 1 是因为我们要考虑标签
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps)) # 获取子序列的索引
    random.shuffle(initial_indices) # 随机打乱

    def data(pos):
        """
        给定开始的索引位置，返回长度为 num_steps 的子序列
        """
        return corpus[pos: pos + num_steps]

    num_batches = num_subseqs // batch_size # 总的子序列数
    for i in range(0, batch_size * num_batches, batch_size):
        initial_indices_per_batch = initial_indices[i: i + batch_size] # 获取子序列的索引
        X = [data(j) for j in initial_indices_per_batch] # 构造数据
        Y = [data(j + 1) for j in initial_indices_per_batch] # 构造标签
        yield th.tensor(X), th.tensor(Y) # 构造小批量

def test1():
    my_seq = list(range(35))
    for X, y in seq_data_iter_random(my_seq, batch_size=2, num_steps=5):
        print('X: ', X, '\nY:', y)

# test1()

'''
    顺序采样
'''
def seq_data_iter_sequential(corpus, batch_size, num_steps):
    offset = random.randint(0, num_steps) # 随机偏移量
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = th.tensor(corpus[offset: offset + num_tokens]) # 构造数据
    Ys = th.tensor(corpus[offset + 1: offset + 1 + num_tokens]) # 构造标签
    Xs, Ys = Xs.reshape((batch_size, -1)), Ys.reshape((batch_size, -1))
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        yield X, Y

def test2():
    my_seq = list(range(35))
    for X, Y in seq_data_iter_sequential(my_seq, batch_size=2, num_steps=5):
        print('X: ', X, '\nY:', Y)

# test2()

class SeqDataLoader:
    """
    数据加载器
    """
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        if use_random_iter:
            self.data_iter_fn = seq_data_iter_random
        else:
            self.data_iter_fn = seq_data_iter_sequential
        self.corpus, self.vocab = load_corpus_time_machine(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps
    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)

def load_data_time_machine(batch_size, num_steps, use_random_iter=False, max_tokens=10000):
    """
    加载数据，直接用于训练
    """
    data_iter = SeqDataLoader(batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab
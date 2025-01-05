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
DATA_HUB['time_machine'] = (DATA_URL + 'timemachine.txt', '090b5e7e70c295757f55df93cb0a180b9691891a')

def download(name, cache_dir=os.path.join('..', 'data')):
    """Download a file inserted into DATA_HUB, return the local filename.

    Defined in :numref:`sec_kaggle_house`"""
    assert name in DATA_HUB, f"{name} does not exist in {DATA_HUB}."
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1])
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname  # Hit cache
    print(f'Downloading {fname} from {url}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname



'''
    读取数据集，进行正则、小写等
'''
def read_time_machine():
    with open(download('time_machine') ,'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

# lines = read_time_machine()
# print(f'# 文本总行数：{len(lines)}')
# print(lines[0])


'''
    将内容词元化
'''
def tokenize(lines, token='word'):
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('错误：未知词元类型：' + token)

# tokens = tokenize(lines)
# for i in range(20):
#     print(tokens[i])

'''
    制作词表，将字符串的单词和数字相对应
'''
def count_corpus(tokens):
    if len(tokens) == 0 or isinstance(tokens[0], list):
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)

class Vocab:
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        """
        idx_to_token 是存放了单词，给定索引求单词
        token_to_idx 是一个字典，其中单词为键，其值是该单词在 idx_to_token 列表中的索引
        """
        if tokens is None:
            tokens = [] # 词表初始为空
        if reserved_tokens is None:
            reserved_tokens = [] # 保留的词元初始为空
        counter = count_corpus(tokens) # 统计词频
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True) # 按词频排序
        self.idx_to_token = ['<unk>'] + reserved_tokens # 词表初始化，其中0为unk
        self.token_to_idx = { # 词表索引初始化
            token: idx for idx, token in enumerate(self.idx_to_token)
        }
        for token, freq in self._token_freqs:
            if freq < min_freq:
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
    lines = read_time_machine()
    tokens = tokenize(lines, 'char')
    vocab = Vocab(tokens)
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
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
    """
    corpus = corpus[random.randint(0, num_steps-1):] # 从随机偏移量开始对序列进行分区
    num_subseqs = (len(corpus) - 1) // num_steps #
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    random.shuffle(initial_indices)

    def data(pos):
        return corpus[pos: pos + num_steps]

    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield th.tensor(X), th.tensor(Y)

def test1():
    my_seq = list(range(35))
    for X, y in seq_data_iter_random(my_seq, batch_size=2, num_steps=5):
        print('X: ', X, '\nY:', y)

# test1()

'''
    顺序采样
'''
def seq_data_iter_sequential(corpus, batch_size, num_steps):
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = th.tensor(corpus[offset: offset + num_tokens])
    Ys = th.tensor(corpus[offset + 1: offset + 1 + num_tokens])
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
    data_iter = SeqDataLoader(batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab
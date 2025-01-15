# 导入相关包
import os
import numpy as np
import jieba as jb
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchtext.data import Field, Dataset, Iterator, Example, BucketIterator
from torch.utils.tensorboard import SummaryWriter


class Net(nn.Module):
    def __init__(self, vocab_size):
        super(Net, self).__init__()
        pass

    def forward(self, x):
        """
        前向传播
        :param x: 模型输入
        :return: 模型输出
        """
        pass


def processing_data(data_path, split_ratio=0.7):
    """
    数据处理
    :data_path：数据集路径
    :validation_split：划分为验证集的比重
    :return：train_iter,val_iter,TEXT.vocab 训练集、验证集和词典
    """

    sentences = []  # 片段
    target = []  # 作者

    # 定义lebel到数字的映射关系
    labels = {'LX': 0, 'MY': 1, 'QZS': 2, 'WXB': 3, 'ZAL': 4}

    files = os.listdir(data_path)
    for file in files:
        if not os.path.isdir(file):
            f = open(data_path + "/" + file, 'r', encoding='UTF-8')  # 打开文件
            for index, line in enumerate(f.readlines()):
                sentences.append(line)
                target.append(labels[file[:-4]])

    mydata = list(zip(sentences, target))

    TEXT = Field(sequential=True, tokenize=lambda x: jb.lcut(x),
                 lower=True, use_vocab=True)
    LABEL = Field(sequential=False, use_vocab=False)

    FIELDS = [('text', TEXT), ('category', LABEL)]

    examples = list(map(lambda x: Example.fromlist(list(x), fields=FIELDS),
                        mydata))

    dataset = Dataset(examples, fields=FIELDS)

    TEXT.build_vocab(dataset, vectors='glove.6B.100d')

    train, val = dataset.split(split_ratio=split_ratio)

    # BucketIterator可以针对文本长度产生batch，有利于训练
    train_iter, val_iter = BucketIterator.splits(
        (train, val),  # 数据集
        batch_sizes=(16, 16),
        device=device,  # 如果使用gpu，此处将-1更换为GPU的编号
        sort_key=lambda x: len(x.text),
        sort_within_batch=False,
        repeat=False
    )

    return train_iter, val_iter, TEXT.vocab


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.lstm = torch.nn.LSTM(1, 64)
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 5)

    def forward(self, x):
        """
        前向传播
        :param x: 模型输入
        :return: 模型输出
        """
        output, hidden = self.lstm(x.unsqueeze(2).float())
        h_n = hidden[1]
        out = self.fc2(self.fc1(h_n.view(h_n.shape[1], -1)))
        return out


class BiLSTM_Attention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers):
        super(BiLSTM_Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim,
                           num_layers=n_layers, bidirectional=True, dropout=0.5)
        self.fc = nn.Linear(hidden_dim * 2, 5)
        self.dropout = nn.Dropout(0.5)

    def attention_net(self, x, query, mask=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, x.transpose(1, 2)) / math.sqrt(d_k)
        p_attn = F.softmax(scores, dim=-1)
        context = torch.matmul(p_attn, x).sum(1)
        return context, p_attn

    def forward(self, x):
        embedding = self.dropout(self.embedding(x))
        output, (final_hidden_state, final_cell_state) = self.rnn(embedding)
        output = output.permute(1, 0, 2)
        query = self.dropout(output)
        attn_output, attention = self.attention_net(output, query)
        logit = self.fc(attn_output)
        return logit


data_path = "./dataset"  # 数据集路径
save_model_path = "results/model.pth"  # 保存模型路径和名称
train_val_split = 0.7  # 验证集比重

# 自动选择设备：如果有 GPU 就用 GPU，否则使用 CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# 获取数据、并进行预处理
train_iter, val_iter, Text_vocab = processing_data(
    data_path, split_ratio=train_val_split)

vocab_path = "results/text_vocab.pth"
torch.save(Text_vocab, vocab_path)
print("词典已保存至:", vocab_path)

# 创建模型实例
# model = Net().to(device)
EMBEDDING_DIM = 100  # 词向量维度
len_vocab = len(Text_vocab)
model = BiLSTM_Attention(len_vocab, EMBEDDING_DIM,
                         hidden_dim=32, n_layers=2).to(device)
pretrained_embedding = Text_vocab.vectors
model.embedding.weight.data.copy_(pretrained_embedding)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(
    optimizer, mode='min', factor=0.3, patience=1, verbose=True)

# 创建一个 SummaryWriter 实例，指定日志目录
writer = SummaryWriter('runs/bi_32_5')
for epoch in range(5):
    train_acc, train_loss = 0, 0
    val_acc, val_loss = 0, 0
    for idx, batch in enumerate(train_iter):
        # 移动数据到同一个设备
        text, label = batch.text.to(device), batch.category.to(device)
        optimizer.zero_grad()
        out = model(text)
        loss = loss_fn(out, label.long())
        # loss.backward(retain_graph=True)
        loss.backward()
        optimizer.step()
        accuracy = np.mean((torch.argmax(out, 1) == label).cpu().numpy())
        # 计算每个样本的acc和loss之和
        train_acc += accuracy*len(batch)
        train_loss += loss.item()*len(batch)

        # 在 TensorBoard 中记录训练的 Loss 和 Accuracy
        writer.add_scalar('Train Loss', loss.item(),
                          epoch * len(train_iter) + idx)
        writer.add_scalar('Train Accuracy', accuracy,
                          epoch * len(train_iter) + idx)

    # 在验证集上预测
    with torch.no_grad():
        for idx, batch in enumerate(val_iter):
            text, label = batch.text, batch.category
            out = model(text)
            loss = loss_fn(out, label.long())
            accracy = np.mean((torch.argmax(out, 1) == label).cpu().numpy())
            # 计算一个batch内每个样本的acc和loss之和
            val_acc += accracy*len(batch)
            val_loss += loss.item()*len(batch)

    train_acc /= len(train_iter.dataset)
    train_loss /= len(train_iter.dataset)
    val_acc /= len(val_iter.dataset)
    val_loss /= len(val_iter.dataset)

    scheduler.step(train_loss)

    # 记录每个 epoch 的 Train 和 Validation 的 Loss 和 Accuracy
    writer.add_scalar('Epoch Train Loss', train_loss, epoch)
    writer.add_scalar('Epoch Train Accuracy', train_acc, epoch)
    writer.add_scalar('Epoch Val Loss', val_loss, epoch)
    writer.add_scalar('Epoch Val Accuracy', val_acc, epoch)

    print("epoch:{} loss:{}, val_acc:{}\n".format(
        epoch, train_loss, val_acc), end=" ")

# 保存模型
torch.save(model.state_dict(), 'results/temp.pth')

# 关闭 TensorBoard writer
writer.close()

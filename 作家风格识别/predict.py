import jieba as jb
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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


# 加载词典
vocab_path = "results/text_vocab.pth"
Text_vocab = torch.load(vocab_path)
print("词典加载完成...")

model = BiLSTM_Attention(
    len(Text_vocab), embedding_dim=100, hidden_dim=32, n_layers=2)
model_path = "results/temp.pth"  # 模型保存路径
model.load_state_dict(torch.load(model_path))  # 加载模型权重
model.eval()  # 设置模型为评估模式
print("模型加载完成...")


def predict(text):
    labels = {0: '鲁迅', 1: '莫言', 2: '钱钟书', 3: '王小波', 4: '张爱玲'}

    # 将句子做分词，然后使用词典将词语映射到他的编号
    text_tokens = jb.lcut(text)
    text2idx = [Text_vocab.stoi.get(
        word, Text_vocab.stoi["<unk>"]) for word in text_tokens]

    # 转化为Torch接收的Tensor类型
    text2idx = torch.Tensor(text2idx).long()

    # 模型预测部分
    with torch.no_grad():  # 在预测时禁用梯度计算
        results = model.forward(text2idx.view(-1, 1))
        prediction = labels[torch.argmax(results, 1).item()]

    return prediction


# 这是一个片段
text = "中国中流的家庭，教孩子大抵只有两种法。其一是任其跋扈，一点也不管，\
    骂人固可，打人亦无不可，在门内或门前是暴主，是霸王，但到外面便如失了网的蜘蛛一般，\
    立刻毫无能力。其二，是终日给以冷遇或呵斥，甚于打扑，使他畏葸退缩，彷佛一个奴才，\
    一个傀儡，然而父母却美其名曰“听话”，自以为是教育的成功，待到他们外面来，则如暂出樊笼的\
    小禽，他决不会飞鸣，也不会跳跃。"

print(predict(text))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from constant import EMBEDDING_LENGTH, LETTER_LIST, LETTER_MAP


class RNN1(nn.Module):
    '''
    输入是空格加单词的张量，输出是单词加空格的张量
    '''
    def __init__(self, hidden_units=64):
        super().__init__()
        self.hidden_units = hidden_units
        self.linear_a = nn.Linear(hidden_units + EMBEDDING_LENGTH,
                                  hidden_units)
        self.linear_y = nn.Linear(hidden_units, EMBEDDING_LENGTH)
        self.tanh = nn.Tanh()

    def forward(self, word: torch.Tensor):
        # word shape: [batch, max_word_length, embedding_length]
        batch, Tx = word.shape[0:2]

        # word shape: [max_word_length, batch,  embedding_length]
        word = torch.transpose(word, 0, 1)

        # output shape: [max_word_length, batch,  embedding_length]
        output = torch.empty_like(word)

        a = torch.zeros(batch, self.hidden_units, device=word.device) #(B（单词数量）, 32)
        x = torch.zeros(batch, EMBEDDING_LENGTH, device=word.device) #(B, 27) #! 严格来讲，这里应该输入空格的onehot向量
        for i in range(Tx): #(20)
            next_a = self.tanh(self.linear_a(torch.cat((a, x), 1)))
            hat_y = self.linear_y(next_a)
            output[i] = hat_y #(B, 27)
            x = word[i]
            a = next_a

        # output shape: [batch, max_word_length, embedding_length]
        return torch.transpose(output, 0, 1)

    @torch.no_grad() #! @torch.no_grad() 和 with torch.no_grad() 都是关闭梯度追踪，差别只是装饰器和语句块的用法
    def language_model(self, word: torch.Tensor): # 计算从零开始采样出某一单词的概率
        '''
        word = torch.tensor([
            [  # 单词0 abcd
                [1,0,0,0,0],
                [0,1,0,0,0],
                [0,0,1,0,0],
                [0,0,0,1,0],
            ],
            [  # 单词1 abce
                [1,0,0,0,0],
                [0,1,0,0,0],
                [0,0,1,0,0],
                [0,0,0,0,1],
            ],
            [  # 单词2 abde
                [1,0,0,0,0],
                [0,1,0,0,0],
                [0,0,0,1,0],
                [0,0,0,0,1],
            ],
        ])  # shape [3,4,5]

        word = torch.tensor([
            [  # time1
                [1,0,0,0,0],  # 单词0的第1个字母（编号0，对应a）
                [1,0,0,0,0],  # 单词1的第1个字母（编号0，对应a）
                [1,0,0,0,0],  # 单词2的第1个字母（编号0，对应a）
            ],
            [  # time2
                [0,1,0,0,0],  # 单词0的第2个字母（编号1，对应b）
                [0,1,0,0,0],  # 单词1的第2个字母（编号1，对应b）
                [0,1,0,0,0],  # 单词2的第2个字母（编号1，对应b）
            ],
            [  # time3
                [0,0,1,0,0],
                [0,0,1,0,0],
                [0,0,0,1,0],
            ],
            [
                # time4: dee
                # the same
            ],
        ])  # shape [4,3,5]

        word_label = torch.tensor([
            [0, 0, 0], # batch1的最大值的索引（1的索引）
            [1, 1, 1],
            [2, 2, 3],
            [3, 4, 4],
        ]) #shape [4, 3]
        
        '''
        # word shape: [batch, 20, embedding_length]
        batch, Tx = word.shape[0:2]
        # word_label shape: [max_word_length, batch]
        word = torch.transpose(word, 0, 1) # [20, batch,  embedding_length=27]
        word_label = torch.argmax(word, 2) #（20， B） 

        # output shape: [batch]
        output = torch.ones(batch, device=word.device) #! 输出概率初始化为1（因为要连乘）

        a = torch.zeros(batch, self.hidden_units, device=word.device)
        x = torch.zeros(batch, EMBEDDING_LENGTH, device=word.device) #( B, 27)
        for i in range(Tx): #(20) # B个单词一起算
            next_a = self.tanh(self.linear_a(torch.cat((a, x), 1))) 
            tmp = self.linear_y(next_a) 
            hat_y = F.softmax(tmp, 1) #（B， 27）
            # word_label[i]为索引为1的的字母的概率
            probs = hat_y[torch.arange(batch), word_label[i]] # shape [batch]，是第i步每个单词的真实字母编号
            # torch.arange(batch) 生成 [0, 1, ..., batch-1]，用于选取 batch 维度的每一行
            # word_label[i] 是一个长度为 batch 的标签数组，比如 [3, 0, 2]。
            # 这种索引方式叫 高级索引（fancy indexing），它会在每一行选取对应列的元素：
            # hat_y[[0,1,2], [0,0,0]]
            # 把这3个元素组合成一个一维张量 probs
            # 等价于：
            # probs = torch.tensor([hat_y[0, 0], hat_y[1, 0], hat_y[2, 0]])
            # probs = torch.tensor([0.5, 0.5, 0.6])
            output *= probs
            x = word[i]
            a = next_a

        return output # # [batch] 每个单词的概率

    @torch.no_grad()
    def sample_word(self, device='cuda:0'):
        #todo 将其改为向量化编程
        batch = 1
        output = ''

        a = torch.zeros(batch, self.hidden_units, device=device)
        x = torch.zeros(batch, EMBEDDING_LENGTH, device=device) #!这里的输入是空而非空格
        for i in range(5):
            next_a = self.tanh(self.linear_a(torch.cat((a, x), 1)))
            tmp = self.linear_y(next_a)
            hat_y = F.softmax(tmp, 1) # (B,27)=（1，27）

            # np_prob = hat_y[0].detach().cpu().numpy()
            np_prob = hat_y[0].cpu().numpy()
            letter = np.random.choice(LETTER_LIST, p=np_prob)
            output += letter

            if letter == ' ':
                break

            # 构建下一个字符的onehot向量
            x = torch.zeros(batch, EMBEDDING_LENGTH, device=device)
            x[0][LETTER_MAP[letter]] = 1
            a = next_a

        return output


class RNN2(torch.nn.Module):
    def __init__(self, hidden_units=64, embeding_dim=64, dropout_rate=0.2):
        super().__init__()
        self.drop = nn.Dropout(dropout_rate)
        self.encoder = nn.Embedding(EMBEDDING_LENGTH, embeding_dim)
        self.rnn = nn.GRU(embeding_dim, hidden_units, 1, batch_first=True)
        self.decoder = torch.nn.Linear(hidden_units, EMBEDDING_LENGTH)
        self.hidden_units = hidden_units

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, word: torch.Tensor):
        # word shape: [batch, max_word_length]
        batch, Tx = word.shape[0:2]
        first_letter = word.new_zeros(batch, 1)
        x = torch.cat((first_letter, word[:, 0:-1]), 1)
        hidden = torch.zeros(1, batch, self.hidden_units, device=word.device)
        emb = self.drop(self.encoder(x))
        output, hidden = self.rnn(emb, hidden)
        y = self.decoder(output.reshape(batch * Tx, -1))

        return y.reshape(batch, Tx, -1)

    @torch.no_grad()
    def language_model(self, word: torch.Tensor):
        batch, Tx = word.shape[0:2]
        hat_y = self.forward(word)
        hat_y = F.softmax(hat_y, 2)
        output = torch.ones(batch, device=word.device)
        for i in range(Tx):
            probs = hat_y[torch.arange(batch), i, word[:, i]]
            output *= probs

        return output

    @torch.no_grad()
    def sample_word(self, device='cuda:0'):
        batch = 1
        output = ''

        hidden = torch.zeros(1, batch, self.hidden_units, device=device)
        x = torch.zeros(batch, 1, device=device, dtype=torch.long)
        for _ in range(10):
            emb = self.drop(self.encoder(x))
            rnn_output, hidden = self.rnn(emb, hidden)
            hat_y = self.decoder(rnn_output)
            hat_y = F.softmax(hat_y, 2)

            np_prob = hat_y[0, 0].detach().cpu().numpy()
            letter = np.random.choice(LETTER_LIST, p=np_prob)
            output += letter

            if letter == ' ':
                break

            x = torch.zeros(batch, 1, device=device, dtype=torch.long)
            x[0] = LETTER_MAP[letter]

        return output

import torch  # 导入 PyTorch 库，用于张量操作和神经网络计算
import torch.nn as nn  # 导入 PyTorch 中的神经网络模块，方便定义网络层

# 定义一个变分自编码器（VAE）的类，该类继承自 nn.Module（PyTorch 构建神经网络的基类）
class VAE(nn.Module):
    # 构造函数 __init__ 定义网络的各个部分，这里构建了编码器和解码器两个主要模块
    # 参数说明：
    #   - hiddens: 一个列表，定义了编码器中每一层卷积输出的通道数（同时解码器会进行对称的信息还原）
    #   - latent_dim: 潜在空间（latent space）的维度，用于表示输入数据的低维表示（编码后的向量）
    def __init__(self, hiddens=[16, 32, 64, 128, 256], latent_dim=128) -> None:
        super().__init__()  # 调用父类（nn.Module）的构造函数，必须有
        
        # ===============================================================
        # 【编码器部分】
        # 宏观目的：将输入（例如一个64x64的RGB图像）通过连续的卷积操作逐步下采样，
        # 损失掉一些空间信息但提取出丰富的特征，最终将得到的高维特征展平，并通过两个线性层得到
        # 潜在空间分布的均值（mean）和对数方差（log variance），以便使用重参数化技术进行采样。
        # ===============================================================
        prev_channels = 3  # 初始输入的通道数。RGB图像有3个通道，类型为 int
        modules = []       # 用来存放多个卷积层模块的列表. 类型为 list，每个元素是 nn.Sequential 对象
        img_length = 64    # 初始图像的边长（假设为 64），类型为 int

        # 遍历 hiddens 列表，构造一系列的卷积模块。这些模块会依次将特征图的通道数增大，同时尺寸减半
        for cur_channels in hiddens:
            modules.append(
                nn.Sequential(
                    # nn.Conv2d: 二维卷积层，用于提取输入图像的局部空间特征
                    # 参数说明：
                    #   - in_channels: 前一层的通道数，这里为 prev_channels
                    #   - out_channels: 卷积产生的特征数，此处为 cur_channels（当前层的通道数）
                    #   - kernel_size=3: 卷积核大小为 3x3
                    #   - stride=2: 步长为 2，意味着输出的尺寸会减少到原来的一半
                    #   - padding=1: 每一侧补充 1 个像素，保证卷积后尺寸变化符合预期
                    nn.Conv2d(prev_channels, cur_channels, kernel_size=3, stride=2, padding=1),
                    # nn.BatchNorm2d: 批归一化，能稳定训练、加速收敛；参数为当前层的输出通道数
                    nn.BatchNorm2d(cur_channels),
                    # nn.ReLU: 激活函数，保证网络具有非线性表达能力
                    nn.ReLU()
                )
            )
            prev_channels = cur_channels  # 更新 prev_channels 为当前层的输出通道数，供下一层使用
            img_length //= 2  # 每经过一次 stride=2 操作，图像尺寸缩小一半（整除），更新 img_length

        assert img_length == 2, "img_length should be 64->2 at this point"
        # 将所有卷积块组合成一个顺序模型。nn.Sequential 接受多个层，并按照顺序依次执行
        self.encoder = nn.Sequential(*modules)  # *modules：将列表展开为多个参数

        # 下面使用两个线性（全连接）层将卷积输出扁平化后的特征映射到潜在空间
        # 注意，此时卷积输出的特征图形状为：(batch_size, 256, 2, 2)
        # 展平后形状为：(batch_size, prev_channels * img_length * img_length)
        self.mean_linear = nn.Linear(prev_channels * img_length * img_length, latent_dim) # (1024->128)
        # 生成潜在变量分布的对数方差（log variance）; 这里用对数表示数值更加稳定，方便后续计算标准差
        self.var_linear = nn.Linear(prev_channels * img_length * img_length, latent_dim)
        self.latent_dim = latent_dim  # 将 latent_dim 存为该类的一个属性，后面解码器和采样方法会用到

        # ===============================================================
        # 【解码器部分】
        # 宏观目的：将潜在向量（低维表示）转换回高维图像。先通过一个线性层“投影”到
        # 合适的张量形状，然后通过一系列转置卷积层（ConvTranspose2d）逐步上采样还原图像尺寸，
        # 最后生成和输入同样尺寸的RGB图像。
        # ===============================================================
        modules = []  # 重置列表，用于存放解码器的各层模块
        
        # 首先，用 nn.Linear 将潜在向量映射到相当于卷积层最后一层输出的特征图（反卷积的输入）
        self.decoder_projection = nn.Linear(
            latent_dim, prev_channels * img_length * img_length)  # 输出形状：(batch_size, prev_channels * img_length * img_length)
        
        # 保存解码器输入时张量的形状 (channels, height, width) 用于后续 reshape 操作
        self.decoder_input_chw = (prev_channels, img_length, img_length)

        # 构建解码器的转置卷积模块。这里的构建方式是对称于编码器，
        # 但是顺序是反着来的：将 hiddens 列表从后向前遍历，其作用就是逐步恢复图像的空间尺寸
        for i in range(len(hiddens) - 1, 0, -1):  # 从列表最后一个元素到第 1 个元素（不包含索引 0）
            modules.append(
                nn.Sequential(
                    # nn.ConvTranspose2d: 转置卷积层，用于上采样，扩大特征图尺寸
                    # 参数说明：
                    #   - in_channels: 当前转置卷积的输入通道数，这里为 hiddens[i]
                    #   - out_channels: 输出通道数，为 hiddens[i - 1] （减少通道数，逐步恢复特征图）
                    #   - kernel_size=3: 卷积核尺寸 3x3
                    #   - stride=2: 步长为 2，使输出尺寸变为输入的两倍
                    #   - padding=1: 填充 1 个像素，确保尺寸计算正确
                    #   - output_padding=1: 补充输出像素数，确保上采样结果达到预期尺寸
                    nn.ConvTranspose2d(hiddens[i], hiddens[i - 1],
                                       kernel_size=3, stride=2, padding=1, output_padding=1),
                    # 加入批归一化
                    nn.BatchNorm2d(hiddens[i - 1]),
                    # 使用 ReLU 激活函数
                    nn.ReLU()
                )
            )

        # 最后一个解码器模块，用来将特征图放大到最终图像大小，并进行最后的通道映射
        modules.append(
            nn.Sequential(
                # 再一次转置卷积，上采样特征图；这里仍旧保持通道数 hiddens[0]
                nn.ConvTranspose2d(hiddens[0], hiddens[0],
                                   kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(hiddens[0]),
                nn.ReLU(),
                # 使用普通卷积层将 hiddens[0] 通道映射为 3 通道（生成 RGB 图像）
                nn.Conv2d(hiddens[0], 3, kernel_size=3, stride=1, padding=1),
                nn.ReLU()  # 对最后结果再加一次 ReLU 激活，非线性输出
            )
        )
        # 将所有解码器块组合成一个顺序模块
        self.decoder = nn.Sequential(*modules)

    # 定义 forward 方法，用于模型的前向传播
    # 在 PyTorch 中，每个 nn.Module 子类都需要实现 forward 方法，它描述了数据如何流经网络
    def forward(self, x):
        # 输入参数 x: input tensor，形状一般为 (batch_size, 3, 64, 64)，即一批 RGB 图像

        # 1. 使用编码器将输入图像 x 逐步下采样并提取特征
        encoded = self.encoder(x)  # 输出形状：(batch_size, 256, img_length, img_length)
        
        # 2. 将多维的特征张量展平成二维张量。torch.flatten(tensor, start_dim) 从 start_dim 起将张量展平
        encoded = torch.flatten(encoded, 1)  # 结果形状：(batch_size, prev_channels * img_length * img_length)
        
        # 3. 生成潜在空间分布的两个参数：均值和对数方差
        #! 我们想要把每张图片都压缩成一个128维的向量，这个128维的向量采样自128的多维高斯分布
        mean = self.mean_linear(encoded)  # 输出均值，形状：(batch_size, latent_dim) #!batchsize中的每个样本都用一个128维的多维高斯分布来表示
        logvar = self.var_linear(encoded)  # 输出对数方差，形状：(batch_size, latent_dim)
        
        # 4. 使用重参数化技巧（reparameterization trick）进行采样
        # 生成一个与 logvar 同形状的随机噪声 eps，服从标准正态分布 N(0, 1)
        eps = torch.randn_like(logvar)  # eps 的形状：(batch_size, latent_dim)
        # 将对数方差转换为标准差：std = exp(logvar / 2)
        std = torch.exp(logvar / 2)  # 计算标准差，形状仍为 (batch_size, latent_dim)
        # 得到潜在向量 z，通过公式：z = mean + std * eps
        z = eps * std + mean  # z 的形状：(batch_size, latent_dim)
        
        # 5. 将潜在向量 z 通过一个全连接层映射到合适的初始形状，供解码器使用
        x = self.decoder_projection(z)  # 形状：(batch_size, prev_channels * img_length * img_length)
        # 6. 将上述线性输出重塑为三维格式：(batch_size, channels, height, width)
        # 这里 -1 自动推断 batch_size，*self.decoder_input_chw 用以解包 (channels, height, width) 三个数字
        x = torch.reshape(x, (-1, *self.decoder_input_chw))
        
        # 7. 用解码器将映射后的低维表示上采样还原为输出图像
        decoded = self.decoder(x)  # 输出形状一般为：(batch_size, 3, 64, 64)
        
        # 返回三个值：
        #   - decoded: 重构的图像
        #   - mean: 潜在分布的均值
        #   - logvar: 潜在分布的对数方差
        # 这两个参数在计算 VAE 的 KL 散度损失时会用到
        return decoded, mean, logvar

    # 定义 sample 方法，用于生成新的图像样本，通常用于模型训练后的生成任务
    def sample(self, device='cuda'):
        # 1. 随机采样一个潜在向量 z，服从标准正态分布
        # 生成的 z 形状为 (1, latent_dim)，表示我们只采样一个样本
        z = torch.randn(1, self.latent_dim).to(device)  # .to(device) 将张量移动到指定设备（如 GPU）
        
        # 2. 将随机采样的 z 通过解码器投影层映射到特征图形状
        x = self.decoder_projection(z)  # 形状：(1, prev_channels * img_length * img_length)
        x = torch.reshape(x, (-1, *self.decoder_input_chw))  # 重塑为 (1, channels, height, width)
        
        # 3. 通过解码器生成新的图像
        decoded = self.decoder(x)  # 输出生成图像，形状：(1, 3, 64, 64)
        
        # 返回生成的图像 decoded
        return decoded
    
if __name__ == '__main__':
    vae = VAE().cuda()
    input = torch.rand(2, 3, 64, 64).cuda()
    output = vae(input)
    print(output[0].shape)
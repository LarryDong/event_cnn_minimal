# 1. 网络
# 1.1. 基本架构：head、编码器、解码器、预测
E2VIDRecurrent(
  (unetrecurrent): UNetRecurrent(
    (head): ConvLayer( (conv2d): Conv2d(10, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))) # head

    (encoders): ModuleList( # 编码器
      (0): RecurrentConvLayer(
        (conv): ConvLayer((conv2d): Conv2d(32, 64, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)))
        (recurrent_block): ConvLSTM((Gates): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))))
      (1): RecurrentConvLayer(
        (conv): ConvLayer((conv2d): Conv2d(64, 128, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)))
        (recurrent_block): ConvLSTM( (Gates): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))))
      (2): RecurrentConvLayer(
        (conv): ConvLayer((conv2d): Conv2d(128, 256, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)))
        (recurrent_block): ConvLSTM( (Gates): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))))

    (resblocks): ModuleList( # 残差块
      (0): ResidualBlock(
        (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
      (1): ResidualBlock(
        (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))))

    (decoders): ModuleList( # 解码器
      (0): UpsampleConvLayer(
        (conv2d): Conv2d(256, 128, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)))
      (1): UpsampleConvLayer(
        (conv2d): Conv2d(128, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)))
      (2): UpsampleConvLayer(
        (conv2d): Conv2d(64, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))))

    (pred): ConvLayer((conv2d): Conv2d(32, 1, kernel_size=(1, 1), stride=(1, 1))) # 预测
  )
)
# 1.2. 带注释架构：
E2VIDRecurrent(
  (unetrecurrent): UNetRecurrent(
    (head): ConvLayer( (conv2d): Conv2d(10, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))) 
# 头部
#输入通道：num_bins=10；输出通道：base_num_channels=32	（data_loader json里的设置）kernel_size=5为默认参数
#实现代码
#    self.head = ConvLayer(self.num_bins, self.base_num_channels,
#                            kernel_size=self.kernel_size, stride=1, # kernel_size=5默认
#                            padding=self.kernel_size // 2)

    (encoders): ModuleList( # 编码器
      (0): RecurrentConvLayer(
        (conv): ConvLayer((conv2d): Conv2d(32, 64, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)))
        (recurrent_block): ConvLSTM((Gates): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))))
      (1): RecurrentConvLayer(
        (conv): ConvLayer((conv2d): Conv2d(64, 128, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)))
        (recurrent_block): ConvLSTM( (Gates): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))))
      (2): RecurrentConvLayer(
        (conv): ConvLayer((conv2d): Conv2d(128, 256, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)))
        (recurrent_block): ConvLSTM( (Gates): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))))
#编码器 
#encoder_input_sizes=[int(self.base_num_channels * pow(channel_multiplier, i)) for i in range(self.num_encoders)] 也就是[base_num_channels*2^0 base_num_channels*2^1 base_num_channels*2^2]=[32,64,128]
#encoder_output_sizes = [int(self.base_num_channels * pow(channel_multiplier, i + 1)) for i in range(self.num_encoders)]也就是[base_num_channels*2^1 base_num_channels*2^2 base_num_channels*2^3]=[63,128,256]
#实现代码
#     for input_size, output_size in zip(self.encoder_input_sizes, self.encoder_output_sizes):
#           self.encoders.append(RecurrentConvLayer(
#                input_size, output_size, kernel_size=self.kernel_size, stride=2,
#                padding=self.kernel_size // 2,
#                recurrent_block_type=self.recurrent_block_type, norm=self.norm))

    (resblocks): ModuleList( # 残差块
      (0): ResidualBlock(
        (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
      (1): ResidualBlock(
        (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))))
# 残差块
# 实现代码self.max_num_channels = self.encoder_output_sizes[-1]=256
#    self.resblocks = nn.ModuleList()
#        for i in range(self.num_residual_blocks):
#            self.resblocks.append(ResidualBlock(self.max_num_channels, self.max_num_channels, norm=self.norm))

    (decoders): ModuleList( # 解码器
      (0): UpsampleConvLayer(
        (conv2d): Conv2d(256, 128, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)))
      (1): UpsampleConvLayer(
        (conv2d): Conv2d(128, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)))
      (2): UpsampleConvLayer(
        (conv2d): Conv2d(64, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))))
# 解码器
# 实现，就是编码器的逆

    (pred): ConvLayer((conv2d): Conv2d(32, 1, kernel_size=(1, 1), stride=(1, 1))) 
# 预测
#实现
#ConvLayer(self.base_num_channels if self.skip_type == 'sum' else 2 * self.base_num_channels,
#                        num_output_channels, 1, activation=None, norm=norm)
  )
)



# 2. data_loader 和valid_data_loader打包数据集：

## 2.1. data_loader 将数据集打包为长度40的列表，列表元素为字典，每个字典的键为：dict_keys(['frame', 'flow', 'events', 'timestamp', 'data_source_idx', 'dt'])
训练只用了frame和event。
训练集模型的输入输出：
events, image, flow = self.to_device(item) # image就是frame
import pdb;pdb.set_trace()
pred = self.model(events)

events.shape ===》torch.Size([2, 10, 64, 64])
pred['image'].shape====》torch.Size([2, 1, 64, 64])

之后使用vgg进行image,和pred['image']的对比

## 2.2. valid_data_loader将验证集打包为长度320的列表，列表元素为字典，每个字典的键为：dict_keys(['frame', 'flow', 'events', 'timestamp', 'data_source_idx', 'dt'])
验证使用的函数和训练一样
events.shape===》torch.Size([1, 10, 64, 64])
image.shape===》torch.Size([1, 1, 64, 64])
pred['image'].shape====》torch.Size([1, 1, 64, 64])
注意：valid_data_loader打包后的通道数是2*num_bins 原因在于combined_voxel_channels  ===》self.channels = self.num_bins if combined_voxel_channels else self.num_bins*2


# 3. 问题：
# size由112改为64===>因为数据的大小就是64x64，因此size不能大于64，最大只能设置为64，valid_data_loader和data_loader都一样
#data_loader的num_bins改为10====>因为valid_data_loader的num_bins为5，且"combined_voxel_channels": false，因此由valid_data_loader打包的数据的通道数是10，需要将data_loader的num_bins改为10以匹配
#网络的输入尺寸基本是不受限制的

# 4. 训练问题
## 4.1 事件转化为图片时的问题：很多默认图片大小都是180x240，需要留意！
sensor_size=(180, 240)











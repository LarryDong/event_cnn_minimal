'''
calculate CNN tensor size
基本公式：
H_{out} = floor( ()/stride )
'''
import math

def cal(h_in,kernel_size=5, stride=1, padding=0, dilation=1):
    h_in = h_in if len(h_in)==2 else [h_in,h_in]
    return tuple([math.floor((h_in[0] + 2*padding - dilation*(kernel_size-1) -1)/stride+1),
            math.floor((h_in[1] + 2*padding - dilation*(kernel_size-1) -1)/stride+1)])
#

x = (128,128)
head_x = cal(x,kernel_size=5,stride=1,padding=2)
encoder1_conv = cal(head_x,kernel_size=5,stride=2,padding=2)
# 3,1,1这个组合正好不改变尺寸！！！之后可以不用考虑
gate_x = cal(encoder1_conv,kernel_size=3,stride=1,padding=1)
encoder2_conv = cal(gate_x,kernel_size=5,stride=2,padding=2)
encoder3_conv = cal(encoder2_conv,kernel_size=5,stride=2,padding=2)
# 残差块都是3，1，1，不改变尺寸，不考虑
# 5，1，2也不改变尺寸！


h_out = cal(x)
print(h_out)
# import pdb;pdb.set_trace()
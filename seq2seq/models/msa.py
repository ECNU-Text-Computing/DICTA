import torch
from baseSA import Mul_HeadSA


mutil_head_self_attention = Mul_HeadSA(16, 2, 0.1)  # input_dim = 16, num_heads = 2, dropout_rate = 0.1

# sequence_len = 5, batch_size = 4, input_dim = 16
inputs = torch.randint(1, 10, (5, 4, 16))
inputs = inputs.float()
print("inputs:", inputs.size(), "\n", inputs)

att_inputs = mutil_head_self_attention(inputs, inputs, inputs)[0]

# 残差连接
outputs = att_inputs + inputs
print("outputs:", outputs.size(), "\n", outputs)

# new_att_inputs = mutil_head_self_attention(new_inputs, new_inputs, new_inputs)




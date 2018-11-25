import numpy as np

timesteps = 100  # 输入序列的时间步数
input_features = 32  # 输入特征空间的都维度
output_features = 64  # 输出特征空间的维度

inputs = np.random.random((timesteps, input_features))

state_t = np.zeros((output_features,))  # 初始化状态

W = np.random.random((output_features, input_features))
U = np.random.random((output_features, output_features))
b = np.random.random((output_features,))

successive_outputs = []
for input_t in inputs:  # inputs的形状是(input_feature,)的向量
    output_t = np.tanh(np.dot(W, input_t) + np.dot(U, state_t) + b)

    successive_outputs.append(output_t)

    state_t = output_t  # 更新网络用语下个迭代

final_output_sequence = np.stack(successive_outputs, axis=0)  # 最终输出的形状为(timesteps,output_features)的二维张量

print(final_output_sequence)

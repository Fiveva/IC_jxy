import pandas as pd
import numpy as np
import os

# 1. 从您提供的文本文件中提取的刺激序列 (共 180 个元素)
stimulus_sequence = ['LC4', 'IC2', 'IC4', 'LC4', 'IC4', 'LC2', 'LC2', 'IC2', 'IC2', 'LC2', 'LC4', 'LC2', 'IC2', 'IC4', 'LC4', 'IC4', 'LC4', 'IC4', 'IC4', 'IC4', 'IC2', 'IC4', 'LC2', 'IC4', 'IC4', 'IC2', 'IC2', 'LC2', 'IC2', 'LC4', 'LC2', 'IC2', 'IC2', 'LC4', 'IC2', 'IC4', 'LC4', 'IC2', 'LC2', 'IC2', 'LC2', 'LC4', 'LC4', 'IC4', 'LC2', 'IC4', 'IC2', 'IC4', 'LC2', 'LC2', 'IC2', 'LC4', 'LC4', 'LC4', 'LC4', 'LC2', 'IC4', 'IC2', 'IC4', 'LC4', 'IC4', 'IC2', 'IC2', 'IC4', 'IC4', 'IC4', 'IC4', 'LC2', 'IC4', 'IC2', 'LC4', 'LC4', 'IC2', 'LC2', 'IC2', 'IC4', 'LC2', 'LC4', 'LC4', 'LC4', 'IC2', 'LC4', 'IC2', 'IC2', 'LC2', 'IC4', 'IC4', 'IC2', 'LC2', 'LC2', 'IC4', 'LC2', 'IC4', 'LC2', 'IC2', 'LC2', 'LC4', 'LC4', 'LC2', 'IC4', 'LC4', 'LC2', 'LC4', 'IC2', 'LC4', 'LC2', 'IC2', 'IC2', 'LC4', 'LC4', 'IC4', 'LC2', 'LC4', 'LC4', 'IC4', 'LC4', 'LC2', 'LC4', 'IC2', 'IC4', 'IC2', 'LC2', 'LC4', 'LC4', 'LC4', 'IC2', 'IC2', 'IC4', 'IC4', 'LC2', 'IC2', 'IC4', 'IC4', 'IC2', 'LC2', 'LC2', 'IC2', 'LC4', 'LC2', 'IC4', 'IC2', 'LC2', 'LC2', 'LC4', 'IC2', 'LC4', 'IC2', 'LC4', 'LC2', 'IC2', 'IC4', 'LC4', 'IC4', 'LC2', 'IC2', 'LC4', 'IC4', 'LC2', 'LC4', 'IC4', 'IC4', 'LC2', 'LC4', 'LC2', 'IC2', 'IC4', 'LC2', 'IC2', 'IC2', 'LC2', 'LC2', 'IC2', 'LC2', 'IC4', 'LC2', 'LC4', 'LC2', 'LC4', 'IC4', 'IC4']

# 2. 创建 DataFrame
df = pd.DataFrame(stimulus_sequence)

# 3. 保存为 CSV 文件
csv_output_filename = 'M79.csv'

# index=False: 不保存行索引
# header=False: 不保存列标题
df.to_csv(csv_output_filename, index=False, header=False)

print(f"成功创建文件: {csv_output_filename}")
print("请将此文件放置在您的数据目录中，并更新 loaddata.py 文件中的相关函数。")
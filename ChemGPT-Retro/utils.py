import numpy as np

def normalize_to_open_interval(data):
    # 检查数据是否有效
    if not data:
        raise ValueError("Input list is empty.")
    min_val = min(data)
    max_val = max(data)

    # 如果所有元素相等，返回长度相同的列表，所有元素为0.5
    if min_val == max_val:
        return [0.5] * len(data)

    epsilon = 1e-3  # 可以根据实际情况调整这个值
    normalized_data = [(x - min_val) / (max_val - min_val) * (1 - 2 * epsilon) + epsilon for x in data]
    normalized_data = np.round(normalized_data,4)
    return normalized_data.tolist()
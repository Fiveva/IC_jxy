import os
import numpy as np
import pandas as pd
import networkx as nx
import time
from four_class_false import ExpConfig # 假设 ExpConfig 类已定义在 four_class_false.py 中

# %% 配置参数
CORRELATION_THRESHOLD = 0.4  # 相关性阈值
HUB_PERCENTAGE = 0.2        # 枢纽节点比例 (2%)

# %% 1. 图分析功能函数
def perform_graph_analysis(corr_matrix, threshold, hub_percentage):
    """
    对相关性矩阵执行二值化、度计算和枢纽节点识别。
    
    :param corr_matrix: 形状为 (N, N) 的相关性矩阵。
    :param threshold: 用于二值化的相关性阈值。
    :param hub_percentage: 定义枢纽节点的比例 (例如 0.02 表示前 2%)。
    :return: degree_dict (所有节点的度), hub_indices (枢纽节点的绝对索引)
    """
    if corr_matrix is None or corr_matrix.ndim != 2 or corr_matrix.shape[0] != corr_matrix.shape[1]:
        print("错误：相关性矩阵无效。")
        return None, None
    
    N = corr_matrix.shape[0]
    print(f"-> 网络节点 (神经元) 总数 N: {N}")

    # 1. 阈值设定与二值化 (Thresholding and Binarization)
    # 使用绝对值 |R| > threshold，并将对角线元素设为 0（排除自连接）
    abs_corr_matrix = np.abs(corr_matrix)
    A = (abs_corr_matrix > threshold).astype(int)
    np.fill_diagonal(A, 0)
    
    print(f"-> 阈值设定为: {threshold}。网络边的数量: {np.sum(A) // 2} (无向图)")

    # 2. 构建 NetworkX 图
    # nx.from_numpy_array(A) 默认创建无向图
    G = nx.from_numpy_array(A)
    
    # 3. 度 (Degree) 分析
    degree_dict = dict(G.degree())
    
    # 将字典转换为 NumPy 数组以便排序
    degrees = np.array(list(degree_dict.values()))
    
    # 4. 识别枢纽节点 (Hubs)
    # 计算枢纽节点数量 (向上取整)
    n_hubs = int(np.ceil(N * hub_percentage))
    
    # 找到度值前 n_hubs 的阈值
    # np.partition(..., -n_hubs) 找到第 n_hubs 大的元素
    # 如果 N < 50，则至少取 1 个节点作为 Hub
    if n_hubs == 0 and N > 0:
        n_hubs = 1
    
    # 找到枢纽度的阈值 (th_degree 是第 n_hubs 大的度值)
    if N > 0 and n_hubs > 0:
        # np.partition 用于高效找到第 k 大的元素
        th_degree = np.partition(degrees, -n_hubs)[-n_hubs]
        
        # 筛选出度大于或等于 th_degree 的节点索引 (绝对索引 0 到 N-1)
        hub_indices = np.where(degrees >= th_degree)[0]
    else:
        hub_indices = np.array([], dtype=int)
        th_degree = 0

    print(f"-> 枢纽节点数量 (Hubs): {len(hub_indices)} 个 (占总数的 {len(hub_indices)/N*100:.2f}%)")
    print(f"-> 枢纽节点的最小度值 (Degree Threshold): {th_degree}")
    
    return degree_dict, hub_indices


# %% 2. 运行主程序
if __name__ == "__main__":
    # --- 用户配置选项 ---
    CONFIG_PATH = r"IC\M79\M79.json" 
    # -------------------
    
    t_start = time.time()
    
    try:
        cfg = ExpConfig(CONFIG_PATH)
        mouse_name = os.path.basename(os.path.dirname(CONFIG_PATH))
        
        # --------------------------------------------------------------------
        # 步骤 A: 加载相关性矩阵文件
        # --------------------------------------------------------------------
        
        CACHE_DIR = os.path.join(cfg.data_path, "network_cache")
        # 假设文件名为上次脚本保存的文件名
        corr_filename = "AllNeurons_CorrMatrix.npy"
        corr_file_path = os.path.join(CACHE_DIR, corr_filename)
        
        print("--- 步骤 A: 加载相关性矩阵 ---")
        if not os.path.exists(corr_file_path):
            raise FileNotFoundError(f"未找到相关性矩阵文件: {corr_file_path}")
            
        correlation_matrix = np.load(corr_file_path)
        print(f"-> 成功加载矩阵。维度: {correlation_matrix.shape}")

        # --------------------------------------------------------------------
        # 步骤 B: 执行图分析
        # --------------------------------------------------------------------
        print("\n--- 步骤 B: 执行图分析 (二值化 + 度计算) ---")
        degree_dict, hub_indices = perform_graph_analysis(
            correlation_matrix, CORRELATION_THRESHOLD, HUB_PERCENTAGE
        )

        if hub_indices is None:
            raise RuntimeError("图分析失败，未生成结果。")

        # --------------------------------------------------------------------
        # 步骤 C: 保存结果 (Hub 索引到 CSV)
        # --------------------------------------------------------------------
        
        GRAPH_DIR = os.path.join(cfg.data_path, "graph")
        os.makedirs(GRAPH_DIR, exist_ok=True)
        
        print("\n--- 步骤 C: 保存 Hub 节点索引 ---")
        
        # 将 Hub 索引转换为 DataFrame 并保存为 CSV
        hub_df = pd.DataFrame({
            'hub_absolute_index': hub_indices
        })
        
        csv_filename = f"{mouse_name}_Hubs_Indices_Th{int(CORRELATION_THRESHOLD*100)}_per{int(HUB_PERCENTAGE*100)}.csv"
        csv_save_path = os.path.join(GRAPH_DIR, csv_filename)
        
        hub_df.to_csv(csv_save_path, index=False)
        
        t_end = time.time()
        print(f"-> Hub 节点索引保存路径: {csv_save_path}")
        
        # --------------------------------------------------------------------
        # 额外：保存所有节点的度值 (可选，便于后续分析)
        # --------------------------------------------------------------------
        degree_df = pd.DataFrame({
            'absolute_index': list(degree_dict.keys()),
            'degree': list(degree_dict.values())
        })
        degree_filename = f"{mouse_name}_AllNeurons_Degree_Th{int(CORRELATION_THRESHOLD*100)}_per{int(HUB_PERCENTAGE*100)}.csv"
        degree_save_path = os.path.join(GRAPH_DIR, degree_filename)
        degree_df.to_csv(degree_save_path, index=False)
        print(f"-> 所有节点度值保存路径: {degree_save_path}")

        print(f"\n======================================================")
        print(f"✅ 图分析完成。")
        print(f"结果目录: {GRAPH_DIR}")
        print(f"总耗时: {t_end - t_start:.2f} 秒")
        print(f"======================================================")

    except Exception as e:
        print(f"\n❌ 运行过程中发生致命错误: {e}")
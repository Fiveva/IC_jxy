import os
import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats

# 假设 ExpConfig 类定义在 four_class.py 或其他可导入的位置。
# 如果不能导入，请将 ExpConfig 的定义复制到此处。
from four_class import ExpConfig 


        
cfg = ExpConfig("IC\M79\M79.json")
# ----------------------------------------


def analyze_network(corr_matrix, threshold=0.2):
    """
    接收相关性矩阵，构建二值化网络，并计算关键图特征。
    
    参数:
    corr_matrix: 相关性矩阵 (numpy array, N x N)。
    threshold: 用于构建二值化网络的阈值。
    
    返回:
    results: 包含所有图特征的字典。
    """
    
    N = corr_matrix.shape[0]
    if N < 2:
        return {'Error': '神经元数量少于2，无法构建网络。'}

    # 1. 二值化网络构建
    # 使用绝对值，大于阈值的连接设为 1，否则为 0
    # 注意: 自连接（对角线）应设为 0
    binary_matrix = (np.abs(corr_matrix) >= threshold).astype(int)
    np.fill_diagonal(binary_matrix, 0)
    
    # 2. 构建 NetworkX 图对象
    # from_numpy_array 默认创建无向图
    G = nx.from_numpy_array(binary_matrix)
    
    # 确保图是连接的，否则一些指标会出错
    if not nx.is_connected(G):
        # 如果不连接，只分析最大的连通分量 (GCC)
        largest_cc = max(nx.connected_components(G), key=len)
        G_cc = G.subgraph(largest_cc).copy()
        N_used = G_cc.number_of_nodes()
        print(f"警告: 网络不连通，正在分析最大的 {N_used}/{N} 个神经元构成的连通分量。")
        G = G_cc
        N = N_used # 更新节点数量
    
    if N < 2:
        return {'Error': '最大连通分量节点数少于2，无法分析。'}
        
    # 3. 特征计算
    
    # a. 平均度 (Degree)
    degrees = dict(G.degree())
    mean_degree = np.mean(list(degrees.values()))
    
    # b. 全局效率 (Global Efficiency)
    # 使用 NetworkX 的定义: 所有节点间最短路径长度倒数的平均值
    try:
        global_efficiency = nx.global_efficiency(G)
    except nx.NetworkXNoPath:
        global_efficiency = 0.0 # 无法计算，设为 0
        
    # c. 模块化 (Modularity)
    # 使用 Louvain 算法进行社区检测
    try:
        communities = nx.community.greedy_modularity_communities(G)
        modularity = nx.community.modularity(G, communities)
        num_communities = len(communities)
    except:
        modularity = 0.0
        num_communities = 1
        
    # d. Hubs 节点 (度大于 平均度 + 2*标准差)
    degree_values = np.array(list(degrees.values()))
    degree_mean = np.mean(degree_values)
    degree_std = np.std(degree_values)
    
    # 阈值计算
    hub_threshold = degree_mean + 2 * degree_std
    
    # 找到 Hub 节点
    hub_nodes = [node for node, degree in degrees.items() if degree > hub_threshold]
    
    # 4. 结果整理
    results = {
        'N_Neurons': N,
        'Threshold': threshold,
        'Mean Degree': mean_degree,
        'Global Efficiency': global_efficiency,
        'Modularity': modularity,
        'Num Communities': num_communities,
        'Num Hubs': len(hub_nodes),
        'Hub Degree Threshold': hub_threshold,
        'Hub Nodes (Relative Index)': hub_nodes # 节点在矩阵中的相对索引
    }
    
    return results


def main_analysis():
    """主分析流程：加载所有矩阵并进行网络分析。"""
    
    classes = ['IC2', 'IC4', 'LC2', 'LC4']
    all_results = []
    threshold = 0.4
    MODE = "25"
    
    print(f"开始图特征分析，数据模式：{MODE}，二值化阈值: {threshold}")
    
    for class_label in classes:
        # 构造缓存文件路径
        cache_filename = f"corr_matrix_{class_label}_Enhanced_RR_{MODE}.npy"
        cache_dir = os.path.join(cfg.data_path, "network_cache")

        cache_path = os.path.join(cache_dir, cache_filename)
        
        if not os.path.exists(cache_path):
            print(f"❌ 错误: 未找到缓存文件 {cache_path}，跳过 {class_label}")
            continue
            
        print(f"\n--- 正在处理类别: {class_label} ---")
        
        # 1. 读取缓存文件
        try:
            corr_matrix = np.load(cache_path)
            print(f"成功加载矩阵，形状: {corr_matrix.shape}")
        except Exception as e:
            print(f"加载 {cache_path} 失败: {e}，跳过。")
            continue
            
        # 2. 分析网络特征
        results = analyze_network(corr_matrix, threshold)
        results['Class'] = class_label
        
        # 3. 打印结果
        print(f"分析结果 ({class_label}):")
        for key, value in results.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        
        all_results.append(results)

    # 4. 汇总结果 (可选: 保存为 CSV)
    if all_results:
        df_results = pd.DataFrame(all_results)
        output_csv_path = os.path.join(cfg.data_path, f"network_analysis_T{threshold:.1f}_summary_{MODE}.csv")
        df_results.to_csv(output_csv_path, index=False)
        print(f"\n⭐ 总结结果已保存到: {output_csv_path}")
        print("\n所有类别图特征分析完成。")
    else:
        print("\n没有成功分析任何网络。")


if __name__ == "__main__":
    main_analysis()
import os
import numpy as np
import pandas as pd
import json
import networkx as nx
import time

# =============================================================================
# 1. 基础配置和辅助函数
# =============================================================================

class ExpConfig:
    """最小化配置类，用于定义数据路径和缓存路径"""
    def __init__(self, data_path="./M79_Data"):
        # **请务必修改此路径为你实际存放数据的目录**
        self.data_path = os.path.abspath(data_path) 
        self.network_cache_dir = os.path.join(self.data_path, "network_cache")

    def get_cache_path(self, filename):
        return os.path.join(self.network_cache_dir, filename)

def load_rr_neuron_indices_from_csv(file_path):
    """
    从 CSV 文件中读取 RR 神经元索引，假设文件只有一列且没有列标题。
    返回一个列表。
    """
    if not os.path.exists(file_path):
        return None
    
    try:
        df = pd.read_csv(file_path, header=None)
        column_index = 0
        
        if column_index not in df.columns or df.empty:
             print(f"加载失败: 文件 {file_path} 似乎是空的或没有数据列。")
             return None

        indices = df[column_index].dropna().astype(int).tolist()
        print(f"成功从 {file_path} 中加载 {len(indices)} 个兴奋性 RR 神经元索引。")
        return indices
    except Exception as e:
        print(f"加载 RR 神经元索引文件 {file_path} 失败: {e}")
        return None

def get_rr_group_indices(data_path, rr_full_list):
    """
    加载 IC-specific, LC-specific, Common 神经元索引。
    使用 rr_selection_boundaries.json 中的数量信息进行切片。
    """
    json_path = os.path.join(data_path, 'rr_selection_boundaries.json')
    
    if not os.path.exists(json_path):
        print(f"错误: RR 边界文件 {json_path} 未找到，无法划分神经元子集。")
        return [], [], []

    try:
        with open(json_path, 'r') as f:
            boundaries = json.load(f)
        
        N_IC = boundaries.get("N_IC_Selective", 0)
        N_LC = boundaries.get("N_LC_Selective", 0)
        N_Common = boundaries.get("N_Common_RR", 0)
        
        total_N_from_json = N_IC + N_LC + N_Common
        total_N_from_list = len(rr_full_list)
        
        if total_N_from_json > total_N_from_list:
            print(f"错误: JSON总数 ({total_N_from_json}) 大于列表总数 ({total_N_from_list})，切分失败。")
            return [], [], []

        # 按照约定顺序进行切片: IC 专属 -> LC 专属 -> 共同
        ic_specific = rr_full_list[0 : N_IC]
        lc_specific = rr_full_list[N_IC : N_IC + N_LC]
        common = rr_full_list[N_IC + N_LC : total_N_from_list] # 使用列表实际长度作为上限
        
        print(f"成功加载并划分 RR 子集: IC({len(ic_specific)}), LC({len(lc_specific)}), Common({len(common)})")
        return ic_specific, lc_specific, common
    except Exception as e:
        print(f"加载或处理 RR 边界信息失败: {e}，返回空列表。")
        return [], [], []

def load_correlation_matrix(cfg, filename):
    """从缓存中加载相关性矩阵 (.npy 文件)。"""
    file_path = cfg.get_cache_path(filename)
    if not os.path.exists(file_path):
        print(f"错误: 缓存文件未找到: {file_path}")
        return None
    try:
        matrix = np.load(file_path)
        return matrix
    except Exception as e:
        print(f"加载矩阵 {filename} 失败: {e}")
        return None

# =============================================================================
# 2. 核心图分析函数 (保持不变)
# =============================================================================

def calculate_graph_metrics(sub_adj_matrix, subset_name, matrix_label, threshold=0.3):
    """
    对相关性子矩阵进行阈值处理、二值化，并计算请求的图分析指标。
    """
    N = sub_adj_matrix.shape[0]
    if N < 2:
        return None

    # 1. 阈值处理/二值化
    A = (sub_adj_matrix > threshold).astype(int) 
    np.fill_diagonal(A, 0)
    G = nx.from_numpy_array(A, create_using=nx.Graph)

    print(f"--- 正在分析: {subset_name} (矩阵: {matrix_label}, 节点数: {N}) ---")
    
    # --- 全局指标 ---
    avg_degree = sum(dict(G.degree()).values()) / N
    E_glob = nx.global_efficiency(G) if G.number_of_edges() > 0 else 0.0
    
    # 模块化 Q 和 社区检测 (用于 P 和 Z)
    try:
        communities = list(nx.community.greedy_modularity_communities(G))
        Q = nx.community.modularity(G, communities)
        node_to_community = {node: i for i, comm in enumerate(communities) for node in comm}
    except Exception:
        communities = []
        Q = np.nan
    
    # --- 特定节点指标 ---
    k_centrality = pd.Series(dict(G.degree()), name='Degree_Centrality') 
    C_B = pd.Series(nx.betweenness_centrality(G), name='Betweenness_Centrality')
    C_C = pd.Series(nx.closeness_centrality(G), name='Closeness_Centrality')
    
    # 参与系数 (P) 和 模块内 Z-score (Z)
    P_list, Z_list = [], []
    if communities:
        for i in range(N):
            node_id = i
            k_i = G.degree(node_id) 
            
            # P - 参与系数
            sum_of_squares = 0
            for comm_id, comm in enumerate(communities):
                k_i_c = sum(1 for neighbor in G.neighbors(node_id) if neighbor in comm)
                sum_of_squares += (k_i_c / k_i)**2 if k_i > 0 else 0
            P_i = 1 - sum_of_squares
            P_list.append(P_i)
            
            # Z - 模块内 Z-score
            c_i = node_to_community.get(node_id)
            if c_i is not None and c_i < len(communities):
                k_i_c_i = sum(1 for neighbor in G.neighbors(node_id) if node_to_community.get(neighbor) == c_i)
                k_module_c = [sum(1 for neighbor in G.neighbors(node) if node_to_community.get(neighbor) == c_i) for node in communities[c_i]]
                
                avg_k_c = np.mean(k_module_c)
                std_k_c = np.std(k_module_c)
                
                Z_i = (k_i_c_i - avg_k_c) / std_k_c if std_k_c > 0 else 0.0
                Z_list.append(Z_i)
            else:
                Z_list.append(np.nan)
    else:
        P_list = [np.nan] * N
        Z_list = [np.nan] * N
    
    P_centrality = pd.Series(P_list, name='Participation_P')
    Z_centrality = pd.Series(Z_list, name='Within_Module_Z')

    metrics = {
        "N_Nodes": N,
        "Global_Avg_Degree": avg_degree,
        "Global_Efficiency": E_glob,
        "Modularity_Q": Q,
        "Node_Metrics": pd.concat([k_centrality, C_B, C_C, P_centrality, Z_centrality], axis=1)
    }
    
    return metrics


# =============================================================================
# 3. 主调度函数 (已修正)
# =============================================================================
def run_rr_group_analysis(cfg, whole_trials=True, threshold=0.3):
    """
    遍历 IC/LC/Common 神经元子集，并对 IC2/IC4/LC2/LC4 四个全量矩阵进行图分析。
    """
    # 确定试次模式后缀 (与您的 network_false.py 约定一致)
    trial_mode = "45" if whole_trials else "25"
    
    # 1. 加载全量 RR 索引 (有序列表)
    rr_full_csv_path = os.path.join(cfg.data_path, 'rr_enhanced_indices_ordered.csv') 
    rr_full_list = load_rr_neuron_indices_from_csv(rr_full_csv_path)
    
    if rr_full_list is None or not rr_full_list:
        print("错误: 无法加载全量 RR 神经元索引，终止分析。")
        return

    # 2. 加载 IC/LC/Common 神经元分组索引 (使用边界信息)
    ic_indices, lc_indices, common_indices = get_rr_group_indices(cfg.data_path, rr_full_list)
    
    # 3. 定义分析任务和矩阵列表
    neuron_subsets = [
        ("IC_Specific", ic_indices),
        ("LC_Specific", lc_indices),
        ("Common_Subset", common_indices),
    ]
    matrix_labels = ['IC2', 'IC4', 'LC2', 'LC4']
    
    all_global_metrics = {}
    all_node_metrics = pd.DataFrame()
    
    print(f"\n--- RR 神经元子集图分析 (模式: {trial_mode}, 阈值: {threshold}) ---")

    # 预计算 RR 索引到全量矩阵局部索引的映射
    full_rr_to_local_index = {rr_idx: i for i, rr_idx in enumerate(rr_full_list)}
    
    # 4. 矩阵迭代 (外层循环)
    for matrix_label in matrix_labels:
        full_matrix_filename = f"corr_matrix_{matrix_label}_Enhanced_RR_{trial_mode}.npy"
        
        # a. 加载当前全量相关性矩阵
        matrix_full = load_correlation_matrix(cfg, full_matrix_filename)
        if matrix_full is None:
            continue
        
        # b. 神经元子集迭代 (内层循环)
        for subset_name, subset_indices in neuron_subsets:
            
            # 组合唯一的任务名称
            task_name = f"{subset_name}_on_{matrix_label}_Matrix"
            
            if not subset_indices:
                print(f"跳过任务 {task_name}: 神经元子集为空。")
                continue
                
            # i. 提取子集神经元在 full_matrix 中的列/行索引
            local_indices_in_full_matrix = [full_rr_to_local_index[idx] for idx in subset_indices if idx in full_rr_to_local_index]
            
            if len(local_indices_in_full_matrix) < 2:
                 print(f"跳过任务 {task_name}: 有效神经元数量 ({len(local_indices_in_full_matrix)}) 少于 2。")
                 continue

            # ii. 提取子矩阵
            sub_adj_matrix = matrix_full[np.ix_(local_indices_in_full_matrix, local_indices_in_full_matrix)]
            
            # iii. 计算图指标
            metrics = calculate_graph_metrics(sub_adj_matrix, subset_name, matrix_label, threshold=threshold)
            
            if metrics:
                # iv. 存储全局指标
                all_global_metrics[task_name] = {k: v for k, v in metrics.items() if k != 'Node_Metrics'}
                all_global_metrics[task_name]['Matrix_Class'] = matrix_label
                all_global_metrics[task_name]['Subset_Type'] = subset_name

                # v. 格式化节点指标 DataFrame
                df_node = metrics['Node_Metrics'].copy()
                df_node['Task'] = task_name
                df_node['Matrix_Class'] = matrix_label
                df_node['Subset_Type'] = subset_name
                
                # 添加原始 RR 索引
                original_indices = [rr_full_list[i] for i in local_indices_in_full_matrix]
                df_node['Original_RR_Index'] = original_indices
                all_node_metrics = pd.concat([all_node_metrics, df_node], ignore_index=True)
                
                print(f"  {task_name} 分析完成。")

    # 5. 结果整理和保存
    
    # 全局指标保存
    df_global = pd.DataFrame.from_dict(all_global_metrics, orient='index')
    global_metrics_path = cfg.get_cache_path(f"global_metrics_full_iteration_{trial_mode}.csv")
    df_global.to_csv(global_metrics_path, index_label='Analysis_Task')
    print(f"\n✅ 全局图指标已保存至: {global_metrics_path}")
    
    # 节点指标保存
    node_metrics_path = cfg.get_cache_path(f"node_metrics_full_iteration_{trial_mode}.csv")
    all_node_metrics.to_csv(node_metrics_path, index=False)
    print(f"✅ 节点中心性指标已保存至: {node_metrics_path}")
    
    return all_global_metrics, all_node_metrics

# =============================================================================
# 4. 主程序入口
# =============================================================================

if __name__ == "__main__":
    
    # --- 用户配置 ---
    DATA_PATH = "IC/M79" 
    
    # 对应 network.py 生成矩阵时使用的试次数量：True=全部试次('45'), False=前25个试次('25')
    WHOLE_TRIALS = False 
    
    # 用于二值化图的阈值 (请根据实际稀疏度调整)
    THRESHOLD = 0.3    
    # ----------------

    cfg = ExpConfig(DATA_PATH)
    
    if not os.path.exists(cfg.network_cache_dir):
        print(f"错误: 缓存目录 {cfg.network_cache_dir} 不存在。请先运行 network.py 生成相关性矩阵。")
    else:
        run_rr_group_analysis(cfg, WHOLE_TRIALS, THRESHOLD)
        
    print("\n图分析脚本运行完成。")
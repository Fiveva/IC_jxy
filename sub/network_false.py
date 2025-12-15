import h5py
import os
import numpy as np
import scipy.io
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import ndimage
from scipy.signal import butter, filtfilt # 导入高通滤波
import time 

# 假设 four_class.py 的所有函数和变量（如 ExpConfig, cfg, load_data, 
# filter_and_segment_data, reclassify, rr_selection_by_class, 
# segment_neuron_data, save_rr_neuron_indices_to_csv, etc.）都已在这里可用。
from four_class_false import * # 重新定义配置（如果四分类代码未运行时需要）
# 确保 cfg 变量在整个脚本中是可用的。
cfg = ExpConfig("IC\M79\M79.json")

# ----------------------------------------------------------------------------
# 新功能函数：从 CSV 文件加载 RR 神经元索引
# ----------------------------------------------------------------------------
def load_rr_neuron_indices_from_csv(file_path):
    """
    从 CSV 文件中读取 RR 神经元索引，假设文件只有一列且没有列标题。
    """
    if not os.path.exists(file_path):
        return None # 返回 None 表示加载失败
    
    try:
        # 明确指定 header=None，告诉 pandas 文件没有标题行
        # pandas 会自动将列命名为 0, 1, 2, ...
        df = pd.read_csv(file_path, header=None)
        
        # 假设数据在第一列，即索引 0
        column_index = 0
        
        # 检查文件是否为空或者第一列是否存在
        if column_index not in df.columns:
             print(f"加载失败: 文件 {file_path} 似乎是空的或没有数据列。")
             return None

        # 读取数据，去除 NaN/None 值，然后转换为整数列表
        indices = df[column_index].dropna().astype(int).tolist()
        
        print(f"成功从 {file_path} 中加载 {len(indices)} 个兴奋性 RR 神经元索引。")
        return indices
    except Exception as e:
        print(f"加载 RR 神经元索引文件 {file_path} 失败: {e}")
        return None

# ----------------------------------------------------------------------------
# 新的预处理函数：高通滤波 + Z-Score (保持不变)
# ----------------------------------------------------------------------------
def high_pass_filter_and_zscore(neuron_data, high_pass_freq=0.05, fs=4.0):
    """
    对神经元数据进行高通滤波和Z-score标准化。
    """
    T, N = neuron_data.shape
    print(f"开始对 {N} 个神经元进行高通滤波和 Z-score 预处理...")
    nyquist = 0.5 * fs 
    Wn = high_pass_freq / nyquist 
    order = 3
    b, a = butter(order, Wn, btype='high', analog=False)

    filtered_data = np.zeros_like(neuron_data)
    for i in range(N):
        filtered_data[:, i] = filtfilt(b, a, neuron_data[:, i], axis=0)
        
    mean_val = np.mean(filtered_data, axis=0)
    std_val = np.std(filtered_data, axis=0)
    std_val[std_val == 0] = 1e-8 
    filtered_zscored_data = (filtered_data - mean_val) / std_val
    
    print("预处理完成。")
    return filtered_zscored_data

# ----------------------------------------------------------------------------
# 核心功能函数：相关性矩阵绘制 (保持不变)
# ----------------------------------------------------------------------------
def _plot_correlation_matrix(corr_matrix, class_label, plot_dir, suffix):
    """绘制相关性矩阵的热力图。"""
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        corr_matrix, 
        ax=ax, 
        cmap='coolwarm', 
        vmin=-1, vmax=1, 
        cbar_kws={'label': 'Pearson Correlation Coefficient'}
    )
    ax.set_title(f'RR Neuron Correlation Matrix ({class_label})', fontsize=14)
    ax.set_xlabel('RR Neuron Index', fontsize=11)
    ax.set_ylabel('RR Neuron Index', fontsize=11)
    fig.tight_layout()
    save_path = os.path.join(plot_dir, f"corr_matrix_{suffix}_{class_label}.png")
    fig.savefig(save_path, dpi=300)
    print(f"已保存相关性矩阵图: {save_path}")
    plt.close(fig)


def calculate_and_plot_rr_correlation(segments, labels, rr_enhanced_neurons, cfg, plot_dir, whole=True):
    """
    计算和绘制 RR 神经元的相关性矩阵，并将矩阵缓存到文件。
    新增 'whole' 参数控制试次使用数量：
    - whole=True: 使用所有可用试次。
    - whole=False: 仅使用每个分类的前25个试次。
    """
    print("\n开始计算和绘制 RR 神经元相关性矩阵...")

    if not rr_enhanced_neurons:
        print("警告: 未找到兴奋性 RR 神经元，跳过相关性分析。")
        return

    # 1. 提取兴奋性 RR 神经元的数据 (与之前保持一致)
    rr_enhanced_neurons_arr = np.array(rr_enhanced_neurons)
    max_idx = segments.shape[1] - 1
    valid_rr_indices = rr_enhanced_neurons_arr[rr_enhanced_neurons_arr <= max_idx]
    enhanced_segments = segments[:, valid_rr_indices, :] # (Trials, RR_Neurons, Timepoints)
    n_rr_neurons = enhanced_segments.shape[1]
    
    if n_rr_neurons < 2:
        print("警告: 有效 RR 神经元数量少于 2 个，无法计算相关性。")
        return

    # 定义时间窗口 (相对索引 12 到 19)
    t_stimulus = cfg.exp_info["t_stimulus"]
    l_stimulus = cfg.exp_info["l_stimulus"]
    stimulus_window = np.arange(t_stimulus, t_stimulus + l_stimulus)
    
    # 映射回原始字符串标签
    class_map = {1: 'IC2', 2: 'IC4', 3: 'LC2', 4: 'LC4'}
    valid_class_ids = sorted(np.unique(labels))
    
    # 设置保存文件的模式后缀和试次上限
    if whole:
        suffix_mode = "45"
    else:
        max_trials = 25
        suffix_mode = f"{max_trials}"
        
    # 检查并创建缓存目录 (修正后的逻辑)
    cache_dir = os.path.join(cfg.data_path, "network_cache")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True) 
        print(f"✅ 已创建缓存目录: {cache_dir}")
    
    
    for class_id in valid_class_ids:
        if class_id == 0: continue
        
        class_label_str = class_map.get(class_id, f'Class {int(class_id)}')
        class_mask = (labels == class_id)
        class_data = enhanced_segments[class_mask, :, :] # (N_trials_class, RR_Neurons, Timepoints)
        
        N_trials_class = class_data.shape[0]
        
        # --- 试次筛选逻辑 ---
        N_trials_used = N_trials_class
        if not whole:
            # 仅使用前 max_trials 个试次
            if N_trials_class > max_trials:
                class_data = class_data[:max_trials, :, :]
                N_trials_used = max_trials
                print(f"警告: 类别 {class_id} ({class_label_str}) 试次数量 {N_trials_class} > {max_trials}，仅使用前 {max_trials} 个试次。")
            else:
                print(f"提示: 类别 {class_id} ({class_label_str}) 试次数量 {N_trials_class} <= {max_trials}，使用全部试次。")
        else:
            print(f"提示: 类别 {class_id} ({class_label_str}) 使用全部 {N_trials_used} 个试次。")

        if N_trials_used < 1:
            print(f"类别 {class_id} ({class_label_str}) 无数据，跳过。")
            continue
            
        print(f"正在处理类别 {class_id} ({class_label_str})，使用的试次数量: {N_trials_used}")
        
        # a. 提取 stimulus_window 的数据
        stim_window_data = class_data[:, :, stimulus_window]
        
        # b. 拼接成一个长串 (Total_Timepoints, RR_Neurons)
        combined_trace = stim_window_data.transpose(1, 0, 2).reshape(n_rr_neurons, -1).T
        
        # c. 计算 Pearson 相关性矩阵
        corr_matrix = np.corrcoef(combined_trace, rowvar=False) 
        
        # ------------------- 缓存相关性矩阵 (文件名包含模式后缀) -------------------
        cache_filename = f"corr_matrix_{class_label_str}_Enhanced_RR_{suffix_mode}.npy"
        cache_path = os.path.join(cache_dir, cache_filename) 
        
        np.save(cache_path, corr_matrix)
        print(f"✅ 已将相关性矩阵缓存到文件: {cache_path}")
        # ---------------------------------------------------------------------
        
        # 绘图 (文件名也包含模式后缀，防止覆盖)
        _plot_correlation_matrix(corr_matrix, class_label_str, plot_dir, suffix=f"Enhanced_RR_{suffix_mode}")


# ----------------------------------------------------------------------------
# 主程序
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    
    try:
        cfg 
    except NameError:
        cfg = ExpConfig() 
        
    print("开始运行 network.py 主程序...")

    plot_dir = os.path.join(cfg.data_path, "network_plot")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
        print(f"已创建图片保存目录: {plot_dir}")

    # 定义缓存文件路径和 CSV 路径
    cache_file = os.path.join(cfg.data_path, "preprocessed_data_cache.npz") 
    csv_filename = 'rr_enhanced_indices_ordered.csv'
    csv_load_path = os.path.join(cfg.data_path, csv_filename)
    
    segments, labels, neuron_pos_filtered = None, None, None
    load_from_cache_successful = False
    
    # 1. 尝试加载缓存数据 (dF/F 预处理后的分段数据)
    if os.path.exists(cache_file):
        segments_cached, labels_cached, neuron_pos_filtered_cached = load_preprocessed_data_npz(cache_file)
        if segments_cached is not None:
             segments = segments_cached
             labels = labels_cached
             neuron_pos_filtered = neuron_pos_filtered_cached
             load_from_cache_successful = True

    # 2. 如果缓存加载失败，执行完整的加载和预处理流程 (dF/F 预处理)
    if not load_from_cache_successful:
        print("未找到有效缓存或缓存加载失败，执行完整的加载和预处理流程...")
        
        # 2a. 加载原始数据 (.mat, .txt, .csv)
        neuron_data_orig, neuron_pos_orig, start_edges, stimulus_data = load_data()
        
        # 2b. 执行昂贵的 dF/F 预处理和分割步骤
        # filter_and_segment_data 会返回 segments, labels, neuron_pos_filtered
        segments, labels, neuron_pos_filtered = filter_and_segment_data(
            neuron_data_orig, neuron_pos_orig, start_edges, stimulus_data, cfg
        )
        
        # 2c. 保存缓存
        save_preprocessed_data_npz(segments, labels, neuron_pos_filtered, cache_file)
        # 从此节点开始，segments/labels/neuron_pos_filtered 已被定义。
        # 同时也定义了 neuron_data_orig, neuron_pos_orig, start_edges, stimulus_data 
        # (需要用于下一步的 High-Pass Z-score 预处理)
    else:
        # 如果缓存成功加载，我们需要原始数据来计算 High-Pass Z-score，
        # 再次调用 load_data 获得原始数据和触发点。
        print("缓存加载成功，跳过原始数据加载和 dF/F 预处理步骤。但需要加载原始数据用于后续的高通滤波。")
        neuron_data_orig, neuron_pos_orig, start_edges, stimulus_data = load_data()

    
    # =========================================================================
    # 步骤 A: RR 神经元索引获取 (优先从 CSV 加载，否则重新筛选)
    # =========================================================================
    
    print("\n--- 步骤 A: RR 神经元索引获取 ---")
    
    # 1. 尝试从 CSV 文件中加载兴奋性 RR 神经元索引
    rr_enhanced_neurons = load_rr_neuron_indices_from_csv(csv_load_path)
    rr_inhibitory_neurons = [] # 默认初始化为空列表

    # 如果加载失败或文件不存在，则重新执行筛选并保存
    if rr_enhanced_neurons is None:
        print("警告: 未能从 CSV 文件加载 RR 神经元索引，将重新执行筛选和保存步骤...")
        
        # 2. 执行 RR 筛选并保存
        # segments 和 labels 是 dF/F 预处理后的分段数据 (RR 筛选的正确输入)
        rr_enhanced_neurons, rr_inhibitory_neurons = rr_selection_by_class(segments, np.array(labels)) 
        
        # 3. 保存新的筛选结果
        # 假设 save_rr_neuron_indices_to_csv 已导入
        save_rr_neuron_indices_to_csv(rr_enhanced_neurons, rr_inhibitory_neurons, csv_load_path)
    
    print(f"最终用于相关性分析的兴奋性 RR 神经元数量: {len(rr_enhanced_neurons)}")
    
    
    # =========================================================================
    # 步骤 B: 高通滤波 + Z-Score (用于最终相关性分析)
    # =========================================================================
    
    print("\n--- 步骤 B: 对全部时间序列进行 高通滤波 + Z-score 预处理 ---")
    
    # 1a. 过滤负值神经元 (保持与 four_class.py 一致的过滤逻辑，因为 high_pass_filter_and_zscore 预期 T, N 矩阵)
    mask = np.any(neuron_data_orig <= 0, axis=0) 
    keep_idx = np.where(~mask)[0]
    neuron_data_filtered = neuron_data_orig[:, keep_idx]
    
    # 1b. 执行新的高通滤波和 Z-score 预处理 (在整个时间序列上进行)
    dff_zscored = high_pass_filter_and_zscore(neuron_data_filtered)
    
    # 2. 分割神经数据 (使用新的预处理数据)
    labels_corr = reclassify(stimulus_data) # 标签保持不变
    segments_for_corr, labels_corr = segment_neuron_data(dff_zscored, start_edges, labels_corr)
    
    
    # =========================================================================
    # 步骤 C: 计算并绘制相关性矩阵
    # =========================================================================
    
    # 使用 高通滤波+Z-score 分段数据 和 筛选出的 RR 索引进行计算
    WHOLE_TRIALS = False  # <-- 在这里修改为 True 或 False
    print(f"\n--- 相关性分析模式: {'全部试次 (WHOLE=True)' if WHOLE_TRIALS else '前25个试次 (WHOLE=False)'} ---")
    # -----------------------------

    # 使用 高通滤波+Z-score 分段数据 和 筛选出的 RR 索引进行计算
    print("\n--- 步骤 C: 使用高通滤波+Z-score 分段数据计算相关性 ---")
    # 传递新增的 whole 参数
    calculate_and_plot_rr_correlation(segments_for_corr, labels_corr, rr_enhanced_neurons, cfg, plot_dir, whole=WHOLE_TRIALS)

    print("\nnetwork.py 主程序运行完成。")
import os
import numpy as np
import scipy.io
import pandas as pd
from scipy.signal import butter, filtfilt 
from scipy import stats 
import time 

# 假设 four_class_false.py 中的 ExpConfig, load_preprocessed_data_npz, 
# 以及其他必要的类和函数已通过此行导入。
from four_class_false import * # ----------------------------------------------------------------------------
def high_pass_filter_and_zscore(data_t_n, cutoff_freq=0.05, fs=4, order=3):
    """
    对 (T, N) 形状的神经元时间序列数据进行高通滤波和 Z-score 标准化。
    """
    if data_t_n.shape[0] < order:
        print("警告：时间点太少，跳过滤波。")
        return stats.zscore(data_t_n, axis=0)

    # 1. 高通滤波
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    
    # 对每列（每个神经元）进行滤波
    data_filtered = np.zeros_like(data_t_n)
    for i in range(data_t_n.shape[1]):
        data_filtered[:, i] = filtfilt(b, a, data_t_n[:, i])
    
    # 2. Z-score 标准化 (按神经元维度)
    data_zscored = stats.zscore(data_filtered, axis=0)
    
    return data_zscored
# ----------------------------------------------------------------------------


# %% 1. 计算所有神经元在刺激时间窗内的相关性矩阵
def calculate_correlation_matrix(segments, t_stimulus, l_stimulus):
    """
    计算所有神经元之间的相关性矩阵，仅使用刺激出现时间窗内的数据。
    相关性是在 Trial x Stimulus_Timepoints 的联合维度上计算的。
    """
    if segments is None or segments.size == 0:
        return None

    # segments 形状: (n_trials, n_neurons, n_timepoints_segment)
    n_trials, n_neurons, n_timepoints_segment = segments.shape

    # ----------------------------------------------------
    # 核心：筛选刺激时间窗
    # ----------------------------------------------------
    stimulus_window = np.arange(t_stimulus, t_stimulus + l_stimulus)
    
    # 检查时间窗是否越界
    if stimulus_window[-1] >= n_timepoints_segment:
        raise ValueError(
            f"刺激时间窗越界。分段总时间点: {n_timepoints_segment}，请求的结束时间点: {stimulus_window[-1]}"
        )
        
    # 截取数据: segments[:, :, stimulus_window]
    segments_stimulus = segments[:, :, stimulus_window]
    n_timepoints_stim = segments_stimulus.shape[2]
    # ----------------------------------------------------
    
    print(f"-> 神经元总数: {n_neurons} | 试验总数: {n_trials} | 刺激时间点数: {n_timepoints_stim}")

    # 1. 重塑数据: 将 (n_trials, n_neurons, n_timepoints_stim) 
    # 转换为 (n_trials * n_timepoints_stim, n_neurons) 
    data_matrix = segments_stimulus.transpose(0, 2, 1).reshape(-1, n_neurons)
    
    print(f"-> 重塑后的数据矩阵形状: {data_matrix.shape}")
    
    # 2. 计算 Pearson 相关系数矩阵
    corr_matrix = np.corrcoef(data_matrix, rowvar=False)
    
    print(f"-> 相关性矩阵形状: {corr_matrix.shape}")
    
    return corr_matrix


# %% 2. 运行主程序
if __name__ == "__main__":
    # --- 用户配置选项 ---
    CONFIG_PATH = r"IC\M79\M79.json" 
    # -------------------
    
    t_start = time.time()
    
    try:
        cfg = ExpConfig(CONFIG_PATH)
        
        # 定义缓存文件路径
        cache_file = os.path.join(cfg.data_path, "preprocessed_data_cache.npz") 
        segments_df_f = None # 用于存储从缓存加载的 dF/F 分段数据
        labels_corr = None
        load_from_cache_successful = False

        # --------------------------------------------------------------------
        # 步骤 A: 加载预处理缓存数据 (dF/F only)
        # --------------------------------------------------------------------
        print("--- 步骤 A: 尝试加载预处理缓存数据 (.npz) ---")
        if os.path.exists(cache_file):
            # 假设 load_preprocessed_data_npz 返回 segments, labels, neuron_pos_filtered
            segments_cached, labels_cached, neuron_pos_filtered_cached = load_preprocessed_data_npz(cache_file)
            
            if segments_cached is not None:
                segments_df_f = segments_cached # 加载 dF/F 分段
                labels_corr = labels_cached
                load_from_cache_successful = True
                print("-> 缓存加载成功。数据是 dF/F 形式，需执行 HPF + Z-score。")
            else:
                print("-> 缓存文件存在，但加载 segments 数据失败。")
        else:
            raise FileNotFoundError(f"未找到预处理缓存文件: {cache_file}。无法进行相关性分析。")

        if not load_from_cache_successful:
             raise RuntimeError("加载缓存文件失败，segments数据为空。无法进行相关性分析。")

        # 获取数据维度
        n_trials, n_neurons_filtered, T_seg = segments_df_f.shape
        print(f"-> 加载的分段数据形状: {segments_df_f.shape}")
        
        # --------------------------------------------------------------------
        # 步骤 B: 重新构建完整时间序列并执行 HPF + Z-score
        # --------------------------------------------------------------------
        print("\n--- 步骤 B: 重新构建完整时间序列，并进行 HPF + Z-score ---")
        
        # 1. 重建完整时间序列 (T_full, N)
        # 形状转换：(n_trials, n_neurons, T_seg) -> (n_trials, T_seg, n_neurons) -> (T_full, N)
        data_t_n_reconstructed = segments_df_f.transpose(0, 2, 1).reshape(-1, n_neurons_filtered)
        
        print(f"-> 重建的完整时间序列形状 (T_full, N): {data_t_n_reconstructed.shape}")

        # 2. 执行高通滤波和 Z-score 预处理
        dff_zscored_full = high_pass_filter_and_zscore(data_t_n_reconstructed)
        
        print(f"-> HPF + Z-score后的时间序列形状: {dff_zscored_full.shape}")

        # 3. 重新分段 (n_trials, n_neurons, T_seg)
        # 将处理后的完整时间序列重新切割回试次分段
        # 形状转换：(T_full, N) -> (n_trials, T_seg, n_neurons) -> (n_trials, n_neurons, T_seg)
        segments_for_corr = dff_zscored_full.reshape(n_trials, T_seg, n_neurons_filtered).transpose(0, 2, 1)
        
        print(f"-> 重新分段后的数据形状 (用于相关性计算): {segments_for_corr.shape}")
        
        # --------------------------------------------------------------------
        # 步骤 C: 计算相关性矩阵
        # --------------------------------------------------------------------
        
        # *** 获取刺激时间窗参数 ***
        t_stimulus = cfg.exp_info["t_stimulus"]
        l_stimulus = cfg.exp_info["l_stimulus"]
        
        print("\n--- 步骤 C: 计算相关性矩阵（仅限刺激出现时间窗） ---")
        # 注意：这里使用的 segments_for_corr 是 HPF+Z-score 后的数据
        correlation_matrix = calculate_correlation_matrix(
            segments_for_corr, t_stimulus, l_stimulus 
        )

        # --------------------------------------------------------------------
        # 步骤 D: 保存结果
        # --------------------------------------------------------------------
        if correlation_matrix is not None:
            # 使用 network_cache 目录进行保存
            CACHE_DIR = os.path.join(cfg.data_path, "network_cache")
            os.makedirs(CACHE_DIR, exist_ok=True)
            
            mouse_name = os.path.basename(os.path.dirname(CONFIG_PATH))
            filename = "AllNeurons_CorrMatrix.npy"
            save_path = os.path.join(CACHE_DIR, filename)
            
            np.save(save_path, correlation_matrix)
            
            t_end = time.time()
            print(f"\n======================================================")
            print(f"✅ 成功计算并保存相关性矩阵。")
            print(f"保存路径: {save_path}")
            print(f"矩阵维度: {correlation_matrix.shape[0]} x {correlation_matrix.shape[1]}")
            print(f"总耗时: {t_end - t_start:.2f} 秒")
            print(f"======================================================")
        else:
            print("❌ 无法计算相关性矩阵：输入数据为空或无效。")

    except Exception as e:
        print(f"\n❌ 运行过程中发生致命错误: {e}")
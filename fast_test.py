import h5py
import os
import numpy as np
import scipy.io
from scipy import stats
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# %% 定义配置
class ExpConfig:
    def __init__(self, file_path = None):
        # 加载配置文件
        if file_path is not None:
            try:
                self.load_config(file_path)
            except Exception as e:
                print(f"加载配置文件失败: {e}")
                self.set_default_config()
        else:
            self.set_default_config()
        self.preprocess_cfg = {
            'preprocess': True,
            'win_size' : 150
        }

    def load_config(self, file_path):
        # 从文件加载配置
        # 如果不是json
        if not file_path.endswith('.json'):
            raise NotImplementedError("目前仅支持JSON格式的配置文件")
        # 解析配置数据
        import json
        with open(file_path, 'r') as f:
            config_data = json.load(f)  

        # 检查必要字段
        required_keys = ['DATA_PATH']
        missing = [k for k in required_keys if k not in config_data]
        if missing:
            raise KeyError(f"配置文件缺少字段: {', '.join(missing)}")
        
        # 赋值配置
        self.data_path = config_data.get("DATA_PATH")
        self.trial_info = config_data.get("TRIAL_INFO", {})
        self.exp_info = config_data.get("EXP_INFO")


    def set_default_config(self):
        # 设置默认配置
        # 数据路径
        self.data_path = "D:\\1study\\junior1\\frontier\\analysis\\IC\\M21"
        # 试次信息
        self.trial_info = {
            "TRIAL_START_SKIP": 0,
            "TOTAL_TRIALS": 180
        }
        # 刺激参数
        self.exp_info = {
            "t_stimulus": 12,  #刺激前帧数
            "l_stimulus": 8,   #刺激持续帧数
            "l_trials": 32,    #单试次总帧数
            "IPD":2.0,
            "ISI":6.0
        }


cfg = ExpConfig("IC\M79\M79.json")

# %% 预处理相关函数定义(通用)
# 从matlab改过来的，经过检查应该无误
def process_trigger(txt_file, IPD=cfg.exp_info["IPD"], ISI=cfg.exp_info["ISI"], fre=None, min_sti_gap=4.0):
    """
    处理触发文件，修改自step1x_trigger_725right.m
    
    参数:
    txt_file: str, txt文件路径
    IPD: float, 刺激呈现时长(s)，默认2s
    ISI: float, 刺激间隔(s)，默认6s
    fre: float, 相机帧率Hz，None则从相机触发时间自动估计
    min_sti_gap: float, 相邻刺激"2"小于此间隔(s)视作同一次（用于去重合并），默认5s
    
    返回:
    dict: 包含start_edge, end_edge, stimuli_array的字典
    """
    
    # 读入文件
    data = []
    with open(txt_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    time_val = float(parts[0])
                    ch_str = parts[1]
                    abs_ts = float(parts[2]) if len(parts) >= 3 else None
                    data.append((time_val, ch_str, abs_ts))
                except ValueError:
                    continue
    
    if not data:
        raise ValueError("未能从文件中读取到有效数据")
    
    # 解析数据
    times, channels, abs_timestamps = zip(*data)
    times = np.array(times)
    
    # 转换通道为数值，非数值的设为NaN
    ch_numeric = []
    valid_indices = []
    for i, ch_str in enumerate(channels):
        try:
            ch_val = float(ch_str)
            ch_numeric.append(ch_val)
            valid_indices.append(i)
        except ValueError:
            continue
    
    if not valid_indices:
        raise ValueError("未找到有效的数值通道数据")
    
    # 只保留有效数据
    t = times[valid_indices]
    ch = np.array(ch_numeric)
    
    # 相机帧与刺激起始时间
    cam_t_raw = t[ch == 1]
    sti_t_raw = t[ch == 2]
    
    if len(cam_t_raw) == 0:
        raise ValueError("未检测到相机触发(值=1)")
    if len(sti_t_raw) == 0:
        raise ValueError("未检测到刺激触发(值=2)")
    
    # 去重/合并：将时间靠得很近的"2"视作同一次刺激
    sti_t = np.sort(sti_t_raw)
    if len(sti_t) > 0:
        keep = np.ones(len(sti_t), dtype=bool)
        for i in range(1, len(sti_t)):
            if (sti_t[i] - sti_t[i-1]) < min_sti_gap:
                keep[i] = False  # 合并到前一个
        sti_t = sti_t[keep]
    
    # 帧率估计或使用给定值
    if fre is None:
        dt = np.diff(cam_t_raw)
        fre = 1 / np.median(dt)  # 用相机帧时间戳的中位间隔

    IPD_frames = max(1, round(IPD * fre))
    isi_frames = round((IPD + ISI) * fre)
    
    # 把每个刺激时间映射到最近的相机帧索引
    cam_t = cam_t_raw.copy()
    nFrames = len(cam_t)
    start_edge = np.zeros(len(sti_t), dtype=int)        #所有刺激起始帧
    
    for k in range(len(sti_t)):
        idx = np.argmin(np.abs(cam_t - sti_t[k]))
        start_edge[k] = idx
    
    end_edge = start_edge + IPD_frames - 1
    
    # 边界裁剪，避免越界
    valid = (start_edge >= 0) & (end_edge < nFrames) & (start_edge <= end_edge)
    start_edge = start_edge[valid]
    end_edge = end_edge[valid]
    
    # 尾段完整性检查（与旧逻辑一致）
    if len(start_edge) >= 2:
        d = np.diff(start_edge)
        while len(d) > 0 and d[-1] not in [isi_frames-1, isi_frames, isi_frames+1, isi_frames+2]:
            # 丢掉最后一个可疑的刺激段
            start_edge = start_edge[:-1]
            end_edge = end_edge[:-1]
            if len(start_edge) >= 2:
                d = np.diff(start_edge)
            else:
                break
    
    # 生成0/1刺激数组（可视化/保存用）
    stimuli_array = np.zeros(nFrames)
    for i in range(len(start_edge)):
        stimuli_array[start_edge[i]:end_edge[i]+1] = 1
    
    # 保存结果到mat文件
    save_path = os.path.join(os.path.dirname(txt_file), 'visual_stimuli_with_label.mat')
    scipy.io.savemat(save_path, {
        'start_edge': start_edge,
        'end_edge': end_edge,
        'stimuli_array': stimuli_array
    })
    
    return {
        'start_edge': start_edge,
        'end_edge': end_edge,
        'stimuli_array': stimuli_array,
        'camera_frames': len(cam_t),
        'stimuli_count': len(start_edge)
    }

# ========== RR神经元筛选函数 ========== 
def rr_selection(trials, labels, t_stimulus=cfg.exp_info["t_stimulus"], l=cfg.exp_info["l_stimulus"], 
                 alpha_fdr=0.05, alpha_level=0.05, reliability_threshold=0.7, 
                 snr_threshold=0.8, effect_size_threshold=0.5, response_ratio_threshold=0.6):
    """
    RR神经元筛选函数：计算神经元的可靠性(Reliability)和响应性(Responsiveness)，
    返回兴奋性RR和混合性RR的索引列表，以及中间筛选步骤的索引列表。
    
    参数:
        trials (np.array): 维度 (试次, 神经元, 时间点) 的分割数据。
        labels (np.array): 试次标签。
        (其他为筛选阈值)
        
    返回: 
        rr_excitatory_idx (list): 兴奋性 RR 神经元的索引。
        rr_mixed_idx (list): 兴奋+抑制性 RR 神经元的索引。
        reliable_neurons_idx (list): 仅可靠性达标的神经元索引。
        excitatory_responsive_idx (list): 仅兴奋性响应达标的神经元索引。
        mixed_responsive_idx (list): 仅混合性响应达标的神经元索引。
    """
    import time
    
    # 过滤有效数据（假设所有非零类别都有效）
    valid_mask = (labels > 0) 
    valid_trials = trials[valid_mask]
    valid_labels = labels[valid_mask] 
    
    n_trials, n_neurons, n_timepoints = valid_trials.shape
    
    # 定义时间窗口
    baseline_pre = slice(0, t_stimulus)
    baseline_post = slice(t_stimulus + l, n_timepoints)
    stimulus_window = slice(t_stimulus, t_stimulus + l)
    
    # ----------------------------------------------
    # 步骤 1: 响应性检测 (Responsiveness)
    # ----------------------------------------------
    
    # 1.1 计算平均值
    baseline_pre_mean = np.mean(valid_trials[:, :, baseline_pre], axis=2)
    baseline_post_mean = np.mean(valid_trials[:, :, baseline_post], axis=2)
    # 使用刺激前后的基线平均值
    baseline_mean = (baseline_pre_mean + baseline_post_mean) / 2 
    stimulus_mean = np.mean(valid_trials[:, :, stimulus_window], axis=2)
    
    # 1.2 计算 Cohen's d 效应大小
    baseline_std = np.std(valid_trials[:, :, baseline_pre], axis=2)
    stimulus_std = np.std(valid_trials[:, :, stimulus_window], axis=2)
    
    # 计算合并标准差 (Pooled Std)
    pooled_std = np.sqrt((baseline_std**2 + stimulus_std**2) / 2)
    pooled_std_safe = pooled_std + 1e-8 # 避免除以零
    
    # 效应大小 (按试次计算): Cohen's d
    effect_size = np.abs(stimulus_mean - baseline_mean) / pooled_std_safe
    
    # 计算响应比率 (Response Ratio): 效应大小超过阈值的试次比例 (维度: 神经元)
    response_ratio = np.mean(effect_size > effect_size_threshold, axis=0)
    
    # ----------------------------------------------
    # 步骤 2: 可靠性检测 (Reliability)
    # ----------------------------------------------
    
    # 2.1 计算信噪比 (SNR)
    signal_strength = np.abs(stimulus_mean - baseline_mean)
    # 使用基线期 std 作为噪声水平
    noise_level = baseline_std 
    snr = signal_strength / (noise_level + 1e-8)
    
    # 可靠性比率 (Reliability Ratio): 信噪比超过阈值的试次比例 (维度: 神经元)
    reliability_ratio = np.mean(snr > snr_threshold, axis=0)
    
    # ----------------------------------------------
    # 步骤 3: 最终筛选与分类 (基于布尔掩码)
    # ----------------------------------------------
    
    # A. 可靠神经元掩码
    reliable_neurons_mask = (reliability_ratio > reliability_threshold)
    reliable_neurons_idx = np.where(reliable_neurons_mask)[0].tolist()
    
    # B. 兴奋性响应神经元掩码 (响应显著 AND 响应方向主要是兴奋性)
    excitatory_responsive_mask = (
        (response_ratio > response_ratio_threshold) & 
        # 兴奋性判断条件：大部分试次响应均值 > 基线均值
        (np.mean(stimulus_mean > baseline_mean, axis=0) > response_ratio_threshold) 
    )
    excitatory_responsive_idx = np.where(excitatory_responsive_mask)[0].tolist()

    # C. 混合性响应神经元掩码 (响应显著，不限制方向)
    mixed_responsive_mask = (response_ratio > response_ratio_threshold)
    mixed_responsive_idx = np.where(mixed_responsive_mask)[0].tolist()
    
    
    # D. 最终 RR 神经元分类 (取交集)
    
    # **类别 1: 兴奋性 RR 细胞 (Reliable AND Excitatory Responsive)**
    rr_excitatory_mask = reliable_neurons_mask & excitatory_responsive_mask
    rr_excitatory_idx = np.where(rr_excitatory_mask)[0].tolist()

    # **类别 2: 兴奋+抑制性 RR 细胞 (Reliable AND Mixed Responsive)**
    rr_mixed_mask = reliable_neurons_mask & mixed_responsive_mask
    rr_mixed_idx = np.where(rr_mixed_mask)[0].tolist()
    
    
    # 返回: 1/2. 最终RR索引, 3. 可靠性索引, 4/5. 响应性索引
    return rr_excitatory_idx, rr_mixed_idx, reliable_neurons_idx, excitatory_responsive_idx, mixed_responsive_idx

#  ========== 数据分割函数 ========== 
def segment_neuron_data(neuron_data, trigger_data, label, pre_frames=cfg.exp_info["t_stimulus"], post_frames=cfg.exp_info["l_trials"]-cfg.exp_info["t_stimulus"]):
    """
    改进的数据分割函数
    
    参数:
    pre_frames: 刺激前的帧数（用于基线）
    post_frames: 刺激后的帧数（用于反应）
    baseline_correct: 是否进行基线校正 (ΔF/F)
    """
    total_frames = pre_frames + post_frames
    segments = np.zeros((len(trigger_data), neuron_data.shape[1], total_frames))
    labels = []

    for i in range(len(trigger_data)): # 遍历每个触发事件
        start = trigger_data[i] - pre_frames
        end = trigger_data[i] + post_frames
        # 边界检查
        if start < 0 or end >= neuron_data.shape[0]:
            print(f"警告: 第{i}个刺激的时间窗口超出边界，跳过")
            continue
        segment = neuron_data[start:end, :]
        segments[i] = segment.T
        labels.append(label[i])
    labels = np.array(labels)
    return segments, labels

# %% 实际功能函数
# ========== 加载数据 ==============================
def load_data(data_path = cfg.data_path, start_idx=cfg.trial_info["TRIAL_START_SKIP"], end_idx=cfg.trial_info["TRIAL_START_SKIP"] + cfg.trial_info["TOTAL_TRIALS"]):
    '''
    加载神经数据、位置数据、触发数据和刺激数据
    '''
    ######### 读取神经数据 #########
    print("开始处理数据...")
    mat_file = os.path.join(data_path, 'wholebrain_output.mat')
    if not os.path.exists(mat_file):
        raise ValueError(f"未找到神经数据文件: {mat_file}")
    try:
        data = h5py.File(mat_file, 'r')
    except Exception as e:
        raise ValueError(f"无法读取mat文件: {mat_file}，错误信息: {e}")

    # 检查关键数据集是否存在
    if 'whole_trace_ori' not in data or 'whole_center' not in data:
        raise ValueError("mat文件缺少必要的数据集（'whole_trace_ori' 或 'whole_center'）")

    # ==========神经数据================
    neuron_data = data['whole_trace_ori']
    # 转化成numpy数组
    neuron_data = np.array(neuron_data)
    print(f"原始神经数据形状: {neuron_data.shape}")
    
    # 只做基本的数据清理：移除NaN和Inf
    neuron_data = np.nan_to_num(neuron_data, nan=0.0, posinf=0.0, neginf=0.0)
    neuron_pos = data['whole_center']
    # 检查和处理neuron_pos维度
    if len(neuron_pos.shape) != 2:
        raise ValueError(f"neuron_pos 应为2D数组，实际为: {neuron_pos.shape}")
    
    # 灵活处理不同维度的neuron_pos
    if neuron_pos.shape[0] > 2:
        # 标准格式 (4, n)，提取前两维
        neuron_pos = neuron_pos[0:2, :]
    elif neuron_pos.shape[0] == 2:
        # 已经是2维，直接使用
        print(f"检测到2维neuron_pos格式: {neuron_pos.shape}")
    else:
        raise ValueError(f"不支持的neuron_pos维度: {neuron_pos.shape[0]}，期望为2、3或4维")

    trigger_files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.txt')]
    trigger_data = process_trigger(trigger_files[-1])
    
    # 刺激数据
    stimulus_files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.csv')]
    stimulus_df = pd.read_csv(stimulus_files[0], header=None)
    # 假设您的 CSV 只有一列，我们将其转换为 NumPy 字符串数组
    stimulus_data = stimulus_df.iloc[:, 0].values.astype(str) # 取第一列（索引 0）并转为字符串数组
    
    # 保持指定试验数，去掉首尾 - 对触发数据和刺激数据同时处理
    start_edges = trigger_data['start_edge'][start_idx:end_idx]
    stimulus_data = stimulus_data[0:end_idx - start_idx]
    
    return neuron_data, neuron_pos, start_edges, stimulus_data 

# ========== 预处理， 去除负值神经元 + 矫正 + 分割trial ==================
def preprocess_data(neuron_data, neuron_pos, start_edge, stimulus_data, cfg=cfg):

    # =========== 第一步 提取仅有正值的神经元==================
    # 带负值的神经元索引
    mask = np.any(neuron_data <= 0, axis=0)   # 每列是否存在 <=0
    keep_idx = np.where(~mask)[0]

    # 如果 neuron_pos 与 neuron_data 的列对齐，则同步删除对应列
    if neuron_pos.shape[1] == neuron_data.shape[1]:
        # 从数据中删除这些列
        neuron_data = neuron_data[:, keep_idx]
        neuron_pos = neuron_pos[:, keep_idx]
    else:
        raise ValueError(f"警告: neuron_pos 列数({neuron_pos.shape[1]}) 与 neuron_data 列数({neuron_data.shape[1]}) 不匹配，未修改 neuron_pos")
    
    from scipy import ndimage
    # =========== 第二步 预处理 ===========================
    if cfg.preprocess_cfg["preprocess"]:
        win_size = cfg.preprocess_cfg["win_size"]
        if win_size % 2 == 0:
            win_size += 1
        T, N = neuron_data.shape
        F0_dynamic = np.zeros((T, N), dtype=float)
        for i in range(N):
            # ndimage.percentile_filter 输出每帧的窗口百分位值
            F0_dynamic[:, i] = ndimage.percentile_filter(neuron_data[:, i], percentile=8, size=win_size, mode='reflect')
        # 通常取每个神经元动态基线的中位数或逐帧使用（此处返回按神经元取中位数的 F0）
        F0 = np.median(F0_dynamic, axis=0)
    # 计算 dF/F（逐帧）
    # dff = (neuron_data - F0[np.newaxis, :]) / F0[np.newaxis, :]
    dff = (neuron_data - F0_dynamic) / F0_dynamic
    # =========== 可视化：随机挑选神经元对比原始信号和dF/F ==================
    # n_samples = min(4, N)  # 最多展示6个神经元
    # sample_indices = np.random.choice(N, size=n_samples, replace=False)
    
    # fig, axes = plt.subplots(n_samples, 2, figsize=(30, 3*n_samples))
    # if n_samples == 1:
    #     axes = axes.reshape(1, -1)
    
    # time_axis = np.arange(T)
    
    # for i, neuron_idx in enumerate(sample_indices):
    #     # 左侧：原始信号
    #     axes[i, 0].plot(time_axis, neuron_data[:, neuron_idx], 'b-', linewidth=0.8)
    #     axes[i, 0].set_ylabel('Raw Fluorescence', fontsize=10)
    #     axes[i, 0].set_title(f'Neuron {neuron_idx} - Original Signal', fontsize=11)
    #     axes[i, 0].grid(True, alpha=0.3)
        
    #     # 右侧：dF/F信号
    #     axes[i, 1].plot(time_axis, dff[:, neuron_idx], 'r-', linewidth=0.8)
    #     axes[i, 1].set_ylabel('dF/F', fontsize=10)
    #     axes[i, 1].set_title(f'Neuron {neuron_idx} - dF/F Signal', fontsize=11)
    #     axes[i, 1].grid(True, alpha=0.3)
    #     axes[i, 1].axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
        
    #     # 只在最后一行显示x轴标签
    #     if i == n_samples - 1:
    #         axes[i, 0].set_xlabel('Time (frames)', fontsize=10)
    #         axes[i, 1].set_xlabel('Time (frames)', fontsize=10)
    #     else:
    #         axes[i, 0].set_xticklabels([])
    #         axes[i, 1].set_xticklabels([])
    
    # plt.tight_layout()# %%

    # =========== 第三步 分割神经数据 =====================================
    labels = reclassify(stimulus_data)
    segments, labels = segment_neuron_data(dff, start_edge, labels)

    return segments, labels, neuron_pos
    

# %% 特殊函数（和刺激类型等相关）
def reclassify(stimulus_data):
    '''
    刺激重新分类函数：将字符串标签转换为数值类别。
    IC2->1, IC4->2, LC2->3, LC4->4
    
    参数:
    stimulus_labels_str: np.array或list，包含 'IC2', 'LC4' 等字符串标签。
    '''
    # 定义映射规则
    mapping = {
        'IC2': 1,  # 类别 1
        'IC4': 2,  # 类别 2
        'LC2': 3,  # 类别 3
        'LC4': 4,  # 类别 4
    }
    
    new_labels = []
    # 遍历输入的字符串标签
    for label in stimulus_data:
        # 使用 get() 方法，如果标签不在映射中，则默认设置为 0 (未分类)
        new_labels.append(mapping.get(label, 0))
    return np.array(new_labels)


# %% =============  主程序逻辑 =============================
# %% =============  主程序逻辑 - 参数快速测试模式 =============================
if __name__ == "__main__":
    import itertools
    import time
    
    print("开始运行主程序 - 参数快速测试模式")
    
    # 步骤 1: 运行一次耗时的 I/O 和数据分割（在循环外）
    t_start_load = time.time()
    # 确保 load_data 里的 stimulus_data 切片已修正！
    neuron_data, neuron_pos, start_edges, stimulus_data = load_data()
    segments, labels, neuron_pos = preprocess_data(neuron_data, neuron_pos, start_edges, stimulus_data)
    t_load_end = time.time()
    print(f"数据加载和预处理耗时: {t_load_end - t_start_load:.2f} 秒")
    
    # 步骤 2: 定义参数的范围和步长
    
    # [参数: 初始值, 经验下限]
    R_REL = (0.7, 0.5)  # reliability_threshold (可靠神经元比率)
    R_SNR = (0.8, 0.5)  # snr_threshold (信噪比阈值)
    R_EFF = (0.5, 0.2)  # effect_size_threshold (Cohen's d 阈值)
    R_RES = (0.6, 0.4)  # response_ratio_threshold (响应试次比率)
    STEP = -0.1
    
    # 生成参数序列
    def generate_range(start, end, step):
        # 确保包含 end
        return np.arange(start, end + step/2, step)

    # 生成所有参数组合
    params_lists = [
        generate_range(R_REL[0], R_REL[1], STEP),
        generate_range(R_SNR[0], R_SNR[1], STEP),
        generate_range(R_EFF[0], R_EFF[1], STEP),
        generate_range(R_RES[0], R_RES[1], STEP),
    ]
    
    # 使用笛卡尔积生成所有组合
    all_combinations = list(itertools.product(*params_lists))
    print(f"\n--- 共生成 {len(all_combinations)} 种参数组合进行测试 ---")
    
    # 步骤 3: 初始化结果存储
    results_excitatory = []
    results_mixed = []
    
    t_start_test = time.time()

    # 步骤 4: 进入循环，只执行 rr_selection
    for rel, snr, eff, res in all_combinations:
        # 调用 rr_selection，返回各种索引列表
        rr_exc_idx, rr_mix_idx, rel_idx, exc_res_idx, mix_res_idx = rr_selection(
            segments, 
            labels,
            reliability_threshold=rel,
            snr_threshold=snr,
            effect_size_threshold=eff,
            response_ratio_threshold=res
        )
        
        # 提取数量
        N_RR_MIXED = len(rr_mix_idx)
        N_RR_EXC = len(rr_exc_idx)
        N_REL = len(rel_idx)             # 可靠神经元数量
        N_MIX_RES = len(mix_res_idx)     # 混合性响应神经元数量
        N_EXC_RES = len(exc_res_idx)     # 兴奋性响应神经元数量


        # 存储兴奋性结果
        results_excitatory.append([
            rel, snr, eff, res, N_EXC_RES, N_REL, N_RR_EXC
        ])
        
        # 存储混合性结果
        results_mixed.append([
            rel, snr, eff, res, N_MIX_RES, N_REL, N_RR_MIXED
        ])

    t_end_test = time.time()
    print(f"\n参数测试循环耗时: {t_end_test - t_start_test:.2f} 秒")
    
    # 步骤 5: 将结果转换为 DataFrame 并保存为 CSV
    
    columns = ['rel_thresh', 'snr_thresh', 'eff_thresh', 'res_thresh', 
               'N_Responsive', 'N_Reliable', 'N_RR_Final']
    
    # 5.1 保存 Excitatory 结果
    df_exc = pd.DataFrame(results_excitatory, columns=columns)
    exc_path = os.path.join(cfg.data_path, "RR_Test_Excitatory.csv")
    df_exc.to_csv(exc_path, index=False)
    print(f"结果已保存至: {exc_path}")
    
    # 5.2 保存 Mixed 结果
    df_mix = pd.DataFrame(results_mixed, columns=columns)
    mix_path = os.path.join(cfg.data_path, "RR_Test_Mixed.csv")
    df_mix.to_csv(mix_path, index=False)
    print(f"结果已保存至: {mix_path}")
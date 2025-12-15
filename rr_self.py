import h5py
import os
import numpy as np
import scipy.io
from scipy import stats
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import ndimage # 预处理需要用到

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


cfg = ExpConfig("IC\M21\M21.json")

# %% 预处理相关函数定义(通用)
# 从matlab改过来的，经过检查应该无误
def process_trigger(txt_file, IPD=cfg.exp_info["IPD"], ISI=cfg.exp_info["ISI"], fre=None, min_sti_gap=4.0):
    """
    处理触发文件，修改自step1x_trigger_725right.m
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

# ========== RR神经元筛选函数 (保持不变) ========== 
def rr_selection(trials, labels, t_stimulus=cfg.exp_info["t_stimulus"], l=cfg.exp_info["l_stimulus"], alpha_fdr=0.05, alpha_level=0.05, reliability_threshold=0.7, snr_threshold=0.8, effect_size_threshold=0.5, response_ratio_threshold=0.6):
    """
    快速RR神经元筛选
    """
    import time
    start_time = time.time()
    
    print("使用快速RR筛选算法...")
    
    # 过滤有效数据
    valid_mask = (labels > 0)
    valid_trials = trials[valid_mask]
    valid_labels = labels[valid_mask]
    
    n_trials, n_neurons, n_timepoints = valid_trials.shape
    
    # 定义时间窗口
    baseline_pre = np.arange(0, t_stimulus)
    baseline_post = np.arange(t_stimulus + l, n_timepoints)
    stimulus_window = np.arange(t_stimulus, t_stimulus + l)
    
    print(f"处理 {n_trials} 个试次, {n_neurons} 个神经元")
    
    # 1. 响应性检测 - 向量化计算
    # 计算基线和刺激期的平均值
    baseline_pre_mean = np.mean(valid_trials[:, :, baseline_pre], axis=2)  # (trials, neurons)
    baseline_post_mean = np.mean(valid_trials[:, :, baseline_post], axis=2)  # (trials, neurons)
    # 合并前后基线的平均
    baseline_mean = (baseline_pre_mean + baseline_post_mean) / 2
    
    stimulus_mean = np.mean(valid_trials[:, :, stimulus_window], axis=2)  # (trials, neurons)
    
    # 简化的响应性检测：基于效应大小和标准误差
    baseline_pre_std = np.std(valid_trials[:, :, baseline_pre], axis=2)  # (trials, neurons)
    baseline_post_std = np.std(valid_trials[:, :, baseline_post], axis=2)  # (trials, neurons)
    # 合并前后基线的标准差
    baseline_std = (baseline_pre_std + baseline_post_std) / 2
    
    stimulus_std = np.std(valid_trials[:, :, stimulus_window], axis=2)
    
    # Cohen's d效应大小
    pooled_std = np.sqrt((baseline_std**2 + stimulus_std**2) / 2)
    effect_size = np.abs(stimulus_mean - baseline_mean) / (pooled_std + 1e-8)
    
    # 响应性标准：平均效应大小 > 阈值 且 至少指定比例试次有响应
    response_ratio = np.mean(effect_size > effect_size_threshold, axis=0)
    enhanced_neurons = np.where((response_ratio > response_ratio_threshold) & 
                                (np.mean(stimulus_mean > baseline_mean, axis=0) > response_ratio_threshold))[0].tolist()
    mixed_neurons = np.where(response_ratio > response_ratio_threshold)[0].tolist()

    # 2. 可靠性检测 - 简化版本
    # 计算每个神经元在每个试次的信噪比
    signal_strength = np.abs(stimulus_mean - baseline_mean)
    noise_level = baseline_std + 1e-8
    snr = signal_strength / noise_level
    
    # 可靠性：指定比例的试次信噪比 > 阈值
    reliability_ratio = np.mean(snr > snr_threshold, axis=0)
    reliable_neurons = np.where(reliability_ratio >= reliability_threshold)[0].tolist()
    
    # 3. 最终RR神经元
    rr_enhanced_neurons = list(set(enhanced_neurons) & set(reliable_neurons))
    rr_mixed_neurons = list(set(mixed_neurons) & set(reliable_neurons))
    
    elapsed_time = time.time() - start_time
    print(f"快速RR筛选完成，耗时: {elapsed_time:.2f}秒")
    
    return rr_enhanced_neurons, rr_mixed_neurons

#  ========== 数据分割函数 (保持不变) ========== 
def segment_neuron_data(neuron_data, trigger_data, label, pre_frames=cfg.exp_info["t_stimulus"], post_frames=cfg.exp_info["l_trials"]-cfg.exp_info["t_stimulus"]):
    """
    改进的数据分割函数
    """
    total_frames = pre_frames + post_frames
    # segment 形状: (n_triggers, n_neurons, n_timepoints)
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

# =================================================================
# %% 缓存函数 (新增)
# 针对用户的 .json 请求，使用更适合 NumPy 数组的 .npz 格式实现缓存。
# .npz 文件可以存储多个压缩的 NumPy 数组，是科学数据缓存的最佳实践。
# =================================================================
def save_preprocessed_data_npz(segments, labels, neuron_pos_filtered, file_path):
    """保存预处理中间数据 (segments, labels, filtered_neuron_pos) 到 .npz 文件。"""
    try:
        np.savez_compressed(
            file_path, 
            segments=segments, 
            labels=labels, 
            neuron_pos_filtered=neuron_pos_filtered
        )
        print(f"已将预处理中间数据保存到缓存文件: {file_path}")
    except Exception as e:
        print(f"保存预处理数据失败: {e}")

def load_preprocessed_data_npz(file_path):
    """从 .npz 文件加载预处理中间数据。"""
    try:
        # allow_pickle=True 是为了兼容旧版 numpy 数组，但这里主要用于加载多个数组
        data = np.load(file_path, allow_pickle=True)
        print(f"尝试从缓存文件加载预处理中间数据: {file_path}")
        return data['segments'], data['labels'], data['neuron_pos_filtered']
    except Exception as e:
        print(f"加载预处理数据失败: {e}")
        return None, None, None

# %% 实际功能函数
# ========== 加载数据 (保持不变) ==============================
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

    # 触发数据
    trigger_files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.txt')]
    # 确保触发文件存在
    if not trigger_files:
         raise FileNotFoundError(f"在 {data_path} 中未找到触发txt文件。")
    trigger_data = process_trigger(trigger_files[-1])
    
    # 刺激数据
    stimulus_files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.csv')]
    if not stimulus_files:
         raise FileNotFoundError(f"在 {data_path} 中未找到刺激csv文件。")
    stimulus_df = pd.read_csv(stimulus_files[0], header=None)
    # 假设您的 CSV 只有一列，我们将其转换为 NumPy 字符串数组
    stimulus_data = stimulus_df.iloc[:, 0].values.astype(str) # 取第一列（索引 0）并转为字符串数组
    
    # 保持指定试验数，去掉首尾 - 对触发数据和刺激数据同时处理
    start_edges = trigger_data['start_edge'][start_idx:end_idx]
    stimulus_data = stimulus_data[0:end_idx - start_idx]
    
    # 返回原始数据，用于后续的昂贵预处理步骤
    return neuron_data, neuron_pos, start_edges, stimulus_data 


# ========== 预处理的耗时部分：去除负值神经元 + 矫正 + 分割trial (新增函数) ==================
def filter_and_segment_data(neuron_data, neuron_pos, start_edge, stimulus_data, cfg=cfg):
    """执行耗时的神经元过滤、dF/F预处理和数据分割步骤。"""

    # =========== 第一步 提取仅有正值的神经元==================
    # 带负值的神经元索引
    mask = np.any(neuron_data <= 0, axis=0)  # 每列是否存在 <=0
    keep_idx = np.where(~mask)[0]

    # 如果 neuron_pos 与 neuron_data 的列对齐，则同步删除对应列
    if neuron_pos.shape[1] == neuron_data.shape[1]:
        neuron_data_filtered = neuron_data[:, keep_idx]
        neuron_pos_filtered = neuron_pos[:, keep_idx]
    else:
        # 如果长度不匹配，理论上应该在 load_data 阶段就报错，这里保留原始逻辑
        raise ValueError(f"警告: neuron_pos 列数({neuron_pos.shape[1]}) 与 neuron_data 列数({neuron_data.shape[1]}) 不匹配，未修改 neuron_pos")
    
    # =========== 第二步 预处理 (dF/F) ===========================
    if cfg.preprocess_cfg["preprocess"]:
        win_size = cfg.preprocess_cfg["win_size"]
        if win_size % 2 == 0:
            win_size += 1
        T, N = neuron_data_filtered.shape
        F0_dynamic = np.zeros((T, N), dtype=float)
        for i in range(N):
            # ndimage.percentile_filter 输出每帧的窗口百分位值
            F0_dynamic[:, i] = ndimage.percentile_filter(neuron_data_filtered[:, i], percentile=8, size=win_size, mode='reflect')
        # 计算 dF/F（逐帧）
        dff = (neuron_data_filtered - F0_dynamic) / F0_dynamic
    else:
        dff = neuron_data_filtered

    # =========== 第三步 分割神经数据 =====================================
    labels = reclassify(stimulus_data)
    segments, labels = segment_neuron_data(dff, start_edge, labels)

    return segments, labels, neuron_pos_filtered

# %% 特殊函数（和刺激类型等相关）
def reclassify(stimulus_data):
    # ... (Original content)
    mapping = {
        'IC2': 1,   # 类别 1
        'IC4': 2,   # 类别 2
        'LC2': 3,   # 类别 3
        'LC4': 4,   # 类别 4
    }
    
    new_labels = []
    for label in stimulus_data:
        new_labels.append(mapping.get(label, 0))
    return np.array(new_labels)

# %% 可视化相关函数定义 (保持不变)
def _rr_distribution_plot(neuron_pos, neuron_pos_rr, plot_dir, suffix, cfg=cfg):
    """RR neuron distribution plot"""
    from tifffile import imread # 确保 imread 在这里被引入

    fig, ax = plt.subplots(figsize=(8.0, 6.2))
    
    # >>> 新增的文件读取检查逻辑开始 <<<
    try:
        # 尝试读取 TIF 文件
        tif_path = os.path.join(cfg.data_path, "whole_brain_3d.tif")
        brain_img = imread(tif_path)
        
        # 成功读取后，进行处理和绘制
        mid_slice = brain_img[brain_img.shape[0] // 2, :, :].astype(float)
        mid_slice = mid_slice / np.nanmax(mid_slice)
        ax.imshow(mid_slice, cmap="Greys", alpha=0.35)
        print(f"背景脑图文件 {tif_path} 读取成功并已绘制。")
        
    except FileNotFoundError:
        # 如果文件不存在，打印警告并跳过背景绘制
        print(f"警告: 脑图文件 {cfg.data_path}/whole_brain_3d.tif 未找到，跳过背景图绘制。")
    except Exception as e:
        # 如果文件读取失败（例如损坏），打印警告并跳过背景绘制
        print(f"警告: 读取脑图文件 {cfg.data_path}/whole_brain_3d.tif 失败，跳过背景图绘制。错误信息: {e}")
    # >>> 新增的文件读取检查逻辑结束 <<<


    sns.scatterplot(
        x=neuron_pos[1, :],
        y=neuron_pos[0, :],
        s=18,
        color="#9fb3c8",
        alpha=0.35,
        edgecolor="none",
        ax=ax,
        label="All neurons",
    )
    sns.scatterplot(
        x=neuron_pos_rr[1, :],
        y=neuron_pos_rr[0, :],
        s=32,
        color="#F67280",
        edgecolor="white",
        linewidth=0.5,
        ax=ax,
        label="RR neurons",
    )

    ax.set_title('RR neuron spatial distribution', fontsize=13)
    ax.set_xlabel('X (pixels)', fontsize=11)
    ax.set_ylabel('Y (pixels)', fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc='upper right')
    ax.set_aspect('equal')
    sns.despine(ax=ax)
    fig.tight_layout()
    
    save_path = os.path.join(plot_dir, f"rr_distribution_{suffix}.png")
    fig.savefig(save_path, dpi=300)
    print(f"已保存 RR 分布图: {save_path}")
    
    plt.close(fig)

    return True

# =================可视化RR神经元响应 (保持不变) =====================
def _plot_rr_responses(segments, labels, plot_dir, suffix, n=20, cfg=cfg):
    """RR neuron response plot"""
    n_samples = min(n, segments.shape[1])
    if n_samples == 0:
        return False
    sample_indices = np.random.choice(segments.shape[1], size=n_samples, replace=False)
    time_axis = np.arange(segments.shape[2])
    class_ids = sorted(np.unique(labels))
    palette = sns.color_palette('tab10', n_colors=len(class_ids))
    color_map = {cls: palette[i] for i, cls in enumerate(class_ids)}

    n_cols = 4
    n_rows = int(np.ceil(n_samples / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.0 * n_cols, 2.6 * n_rows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, neuron_idx in zip(axes, sample_indices):
        for cls in class_ids:
            traces = segments[labels == cls, neuron_idx, :]
            if traces.size == 0:
                continue
            mean_trace = np.mean(traces, axis=0)
            sem_trace = stats.sem(traces, axis=0, nan_policy='omit')
            ax.fill_between(time_axis, mean_trace - sem_trace, mean_trace + sem_trace, color=color_map[cls], alpha=0.18)
            ax.plot(time_axis, mean_trace, color=color_map[cls], linewidth=1.6, label=f'Class {int(cls)}')
        ax.axvline(x=cfg.exp_info["t_stimulus"], color="#aa3a3a", linestyle="--", linewidth=1.0)
        ax.set_title(f'Neuron {neuron_idx}', fontsize=10)
        ax.set_ylim(-0.3, 1.3)

    for ax in axes[len(sample_indices):]:
        ax.axis('off')

    handles, labels_legend = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels_legend, frameon=False, loc='upper center', ncol=len(handles))
    for ax in axes[:len(sample_indices)]:
        sns.despine(ax=ax)
        ax.tick_params(labelsize=8)

    fig.text(0.5, 0.02, 'Time (frames)', ha='center', fontsize=11)
    fig.text(0.02, 0.5, 'dF/F', va='center', rotation='vertical', fontsize=11)
    fig.tight_layout(rect=[0.02, 0.04, 0.98, 0.95])
    
    save_path = os.path.join(plot_dir, f"rr_responses_{suffix}.png")
    fig.savefig(save_path, dpi=300)
    print(f"已保存 RR 响应图: {save_path}")
    
    plt.close(fig)

    return True

# %% =============  主程序逻辑 (修改，实现缓存检查和加载) =============================
if __name__ == "__main__":
    print("开始运行主程序")

    plot_dir = os.path.join(cfg.data_path, "plot")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
        print(f"已创建图片保存目录: {plot_dir}")

    # 定义缓存文件路径
    # 建议使用 .npz 格式存储 NumPy 数组，而不是纯 JSON。
    cache_file = os.path.join(cfg.data_path, "preprocessed_data_cache.npz") 
    print(f"预处理数据缓存文件路径: {cache_file}")

    # 1. 尝试加载缓存数据
    segments, labels, neuron_pos_filtered = None, None, None
    load_from_cache_successful = False
    
    if os.path.exists(cache_file):
        segments_cached, labels_cached, neuron_pos_filtered_cached = load_preprocessed_data_npz(cache_file)
        if segments_cached is not None:
             segments = segments_cached
             labels = labels_cached
             neuron_pos_filtered = neuron_pos_filtered_cached
             load_from_cache_successful = True

    # 2. 如果缓存加载失败，执行完整的加载和预处理流程
    if not load_from_cache_successful:
        print("未找到有效缓存或缓存加载失败，执行完整的加载和预处理流程...")
        
        # 2a. 加载原始数据 (.mat, .txt, .csv)
        neuron_data_orig, neuron_pos_orig, start_edges, stimulus_data = load_data()
        
        # 2b. 执行昂贵的预处理和分割步骤
        segments, labels, neuron_pos_filtered = filter_and_segment_data(
            neuron_data_orig, neuron_pos_orig, start_edges, stimulus_data, cfg
        )
        
        # 2c. 保存缓存
        save_preprocessed_data_npz(segments, labels, neuron_pos_filtered, cache_file)
    else:
        print("缓存加载成功，跳过原始数据加载和预处理步骤。")


    # 3. RR 神经元筛选 (每次都要执行，因为筛选参数可能变化)
    
    # 注意：此时 segments, labels, neuron_pos_filtered 已经准备好，并且是过滤后的数据
    rr_enhanced_neurons, rr_mixed_neurons = rr_selection(segments, np.array(labels))
    enhanced_segments = segments[:, rr_enhanced_neurons, :]
    enhanced_neuron_pos_rr = neuron_pos_filtered[:, rr_enhanced_neurons]

    mixed_segments = segments[:, rr_mixed_neurons, :]
    mixed_neuron_pos_rr = neuron_pos_filtered[:, rr_mixed_neurons]

    # %% 可视化RR神经元分布 (兴奋性)
    # 传入过滤后的全部神经元位置 (neuron_pos_filtered) 作为背景
    _rr_distribution_plot(neuron_pos_filtered, enhanced_neuron_pos_rr, plot_dir, "Excitatory")
    # %% 可视化RR神经元响应 (兴奋性)
    _plot_rr_responses(enhanced_segments, labels, plot_dir, "Excitatory", n=20)

    # %% 可视化RR神经元分布 (混合性)
    _rr_distribution_plot(neuron_pos_filtered, mixed_neuron_pos_rr, plot_dir, "Mixed")
    # %% 可视化RR神经元响应 (混合性)
    _plot_rr_responses(mixed_segments, labels, plot_dir, "Mixed", n=20)

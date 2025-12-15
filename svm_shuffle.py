import os
import numpy as np
import time
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# --- 新增 SVM 相关的库 ---
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.utils import shuffle
# ------------------------

# 假设 sub 模块和相关配置类已存在于您的环境中
from four_class import ExpConfig, rr_selection_by_class

# --- 全局/常量定义 ---
N_CLASSES = 4
CHANCE_LEVEL = 1 / N_CLASSES # 0.25 (偶然水平)

# %% 1. 定义特征提取函数 (沿用)
def extract_rr_features(segments, labels, rr_neuron_indices, t_stimulus):
    """
    从筛选出的 RR 神经元数据中提取特征。
    特征定义为刺激出现后两帧 (t_stimulus 到 t_stimulus + 2) 的平均 dF/F 响应。
    """
    if not rr_neuron_indices:
        return None, None

    rr_segments = segments[:, rr_neuron_indices, :]
    
    FEATURE_FRAMES = 2 
    stimulus_window = np.arange(t_stimulus, t_stimulus + FEATURE_FRAMES)
    
    if stimulus_window[-1] >= rr_segments.shape[2]:
        raise ValueError("特征提取时间窗越界。")
    
    X = np.mean(rr_segments[:, :, stimulus_window], axis=2) 
    Y = labels
    
    return X, Y

# %% 2. 数据加载和 RR 神经元筛选 (沿用)
def load_and_preprocess_data(config_file):
    """加载配置、预处理数据并筛选 RR 神经元，仅运行一次。"""
    cfg = ExpConfig(config_file)
    cache_file = os.path.join(cfg.data_path, "preprocessed_data_cache.npz")
    
    if os.path.exists(cache_file):
        print("缓存加载成功，跳过原始数据加载和预处理步骤。")
        try:
            data = np.load(cache_file, allow_pickle=True)
        except Exception: 
            data = np.load(cache_file, allow_pickle=True) 
        segments = data['segments']
        labels = data['labels']
    else:
        raise FileNotFoundError(f"未找到预处理缓存文件: {cache_file}。请先运行 four_class.py 生成缓存。")

    # RR 神经元筛选 
    rr_enhanced_neurons, rr_inhibitory_neurons = rr_selection_by_class(segments, np.array(labels))
    
    return segments, labels, rr_enhanced_neurons, rr_inhibitory_neurons, cfg


# %% 3. 结果格式化函数 (为 SVM 重新编写)
def _format_svm_results(mouse_name, feature_type, n_features, n_pca_components, pca_variance, acc_obs, chance_level, p_value, total_time, N_PERMUTATIONS, SVM_C, PCA_VAR_RATIO, TEST_SIZE):
    """将 SVM 结果格式化为字符串，包括 Shuffle 检验结果。"""
    
    results_str = f"""
======================================================
SVM 分类 Shuffle 检验报告 (Linear Kernel, C={SVM_C})
鼠 ID: {mouse_name}
特征类型: {feature_type.capitalize()} RR 神经元
神经元数量: {n_features} 个
------------------------------------------------------
--- PCA 降维信息 ---
PCA 目标保留 >= {PCA_VAR_RATIO*100:.0f}% 方差
PCA 实际保留主成分数: {n_pca_components}
PCA 实际解释方差: {pca_variance:.4f}
------------------------------------------------------
--- 分类性能 ---
数据划分: 训练集 {100*(1-TEST_SIZE):.0f}% / 测试集 {100*TEST_SIZE:.0f}% (分层抽样)
偶然水平 (Chance Level): {chance_level:.4f} ({N_CLASSES} 类)
观测到的测试集准确率 (Test Accuracy Observed): {acc_obs:.4f}

--- Shuffle 显著性检验 ({N_PERMUTATIONS} 次置换) ---
Shuffle 检验 P 值: {p_value:.4f}
结论: P < 0.05 则分类显著 (即模型性能优于随机猜测)

--- 运行时间 ---
总运行时间 (含 Shuffle): {total_time:.4f} 秒
======================================================
"""
    return results_str

# %% 4. Shuffle 准确率分布图
def plot_shuffle_distribution(acc_obs, acc_permutations, mouse_name, save_dir, SVM_C, N_PERMUTATIONS):
    """
    绘制 Shuffle 准确率的分布图，并标记观测值和偶然水平。
    """
    
    plt.figure(figsize=(10, 6))
    sns.histplot(acc_permutations, kde=True, bins=30, color='skyblue', label=f'Shuffle Accuracies ({N_PERMUTATIONS} times)')
    
    # 偶然水平
    plt.axvline(CHANCE_LEVEL, color='gray', linestyle='--', linewidth=2, label=f'Chance Level ({CHANCE_LEVEL:.2f})')
    
    # 观测到的准确率
    plt.axvline(acc_obs, color='red', linestyle='-', linewidth=3, label=f'Observed Acc. ({acc_obs:.4f})')
    
    # 计算 P 值
    p_value = np.sum(np.array(acc_permutations) >= acc_obs) / N_PERMUTATIONS
    
    # 标题和标签
    plt.title(f'{mouse_name} SVM Test Accuracy Permutation Test (P={p_value:.4f}, C={SVM_C})')
    plt.xlabel('Test Accuracy')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(axis='y', linestyle='--')
    
    # 保存图像
    filename = f'svm_shuffle_distribution_{mouse_name}_C{int(SVM_C*100)}_P{N_PERMUTATIONS}p.png'
    save_path = os.path.join(save_dir, filename) 
    plt.savefig(save_path)
    plt.close()
    print(f"   -> Shuffle 分布图已保存到: {save_path}")
    
    # 触发图示
    print("")


# %% 5. 对单一特征类型进行 SVM 分类和 Shuffle 检验
def analyze_svm_classification(segments, labels, rr_enhanced_neurons, rr_inhibitory_neurons, cfg, feature_type, PCA_VAR_RATIO, N_PERMUTATIONS, SVM_C, TEST_SIZE, RANDOM_STATE):
    """
    对指定的特征类型执行 SVM 分类和标签置换检验。
    """
    
    t_start = time.time()
    
    # --- 1. 确定神经元索引与数据加载 ---
    feature_type_lower = feature_type.lower()
    if feature_type_lower == "enhanced":
        rr_indices = rr_enhanced_neurons
    else:
        # 仅分析兴奋性 RR 神经元
        raise ValueError(f"该脚本设计为分析 'enhanced' (兴奋性) RR 神经元。")
    
    n_features = len(rr_indices)
    Y_true = np.array(labels) 
    
    if n_features == 0:
        return f"\n--- {feature_type.capitalize()} RR 神经元 (0 个) SVM 分类失败：神经元数量不足 ---\n", None
    
    print(f"\n======================================================\n")
    print(f"正在分析特征类型: {feature_type.capitalize()} RR 神经元 ({n_features} 个)")
    
    # 特征提取
    t_stimulus = cfg.exp_info["t_stimulus"]
    X_full, _ = extract_rr_features(segments, labels, rr_indices, t_stimulus)
    
    # --- 2. 标准化 + PCA 降维 ---
    print("-> 1. 标准化 (StandardScaler)...")
    sc = StandardScaler()
    X_scaled = sc.fit_transform(X_full)
    
    print(f"-> 2. PCA 降维 (目标保留 >= {PCA_VAR_RATIO*100:.0f}% 方差)...")
    pca = PCA(n_components=PCA_VAR_RATIO, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)
    n_pca_components = X_pca.shape[1]
    pca_variance = np.sum(pca.explained_variance_ratio_)
    print(f"   -> PCA 实际保留 {n_pca_components} 个主成分，共解释 {pca_variance:.4f} 的方差。")

    # --- 3. 划分训练集和测试集 (分层抽样) ---
    print(f"-> 3. 划分训练集和测试集 (Test Size={TEST_SIZE*100:.0f}%, 分层抽样)...")
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_pca, Y_true, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=Y_true
    )

    # --- 4. 观测值 SVM 训练和评估 (Observed Performance) ---
    print(f"-> 4. 训练观测值 SVM (C={SVM_C}, kernel='linear')...")
    
    # 初始化 SVM 分类器
    clf = SVC(kernel='linear', C=SVM_C, random_state=RANDOM_STATE)
    
    # 训练模型
    clf.fit(X_train, Y_train)
    
    # 测试集准确率
    Y_pred_obs = clf.predict(X_test)
    acc_obs = accuracy_score(Y_test, Y_pred_obs)
    
    print(f"   -> 观测到的测试集准确率: {acc_obs:.4f} (Chance Level: {CHANCE_LEVEL:.4f})")
    
    # 打印详细报告 (用于写入文件)
    # 使用类别名作为目标名称
    target_names = [v for k, v in sorted({1: 'IC2', 2: 'IC4', 3: 'LC2', 4: 'LC4'}.items())]
    print(f"   -> 分类报告:\n{classification_report(Y_test, Y_pred_obs, target_names=target_names)}")
    print(f"   -> 混淆矩阵:\n{confusion_matrix(Y_test, Y_pred_obs)}")


    # --- 5. 标签置换检验 (Shuffle Test) ---
    print(f"-> 5. 开始标签置换检验 ({N_PERMUTATIONS} 次置换)...")
    acc_permutations = []
    
    for i in range(N_PERMUTATIONS):
        # 随机打乱整个标签集 Y_true
        Y_shuffled_full = shuffle(Y_true, random_state=i + 100) # 使用不同的随机种子
        
        # 划分训练集和测试集（使用打乱后的标签进行分层抽样）
        # 这样确保了训练集和测试集标签都是随机的，且类别比例平衡
        X_train_s, X_test_s, Y_train_s, Y_test_s = train_test_split(
            X_pca, Y_shuffled_full, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=Y_shuffled_full
        )

        # 训练和测试模型 (使用打乱后的标签)
        clf_s = SVC(kernel='linear', C=SVM_C, random_state=RANDOM_STATE)
        clf_s.fit(X_train_s, Y_train_s)
        Y_pred_s = clf_s.predict(X_test_s)
        acc_shuffled = accuracy_score(Y_test_s, Y_pred_s)
        acc_permutations.append(acc_shuffled)
        
    # 计算 P 值：随机准确率 >= 观测准确率 的比例
    p_value = np.sum(np.array(acc_permutations) >= acc_obs) / N_PERMUTATIONS
    print(f"   -> Shuffle 检验 P 值: {p_value:.4f}")

    # --- 6. 格式化输出 ---
    total_time = time.time() - t_start
    results_str = _format_svm_results(
        cfg.mouse_name, feature_type, n_features, n_pca_components, pca_variance,
        acc_obs, CHANCE_LEVEL, p_value, total_time, N_PERMUTATIONS, SVM_C, PCA_VAR_RATIO, TEST_SIZE
    )
    
    # 返回报告字符串和 shuffle 准确率列表，用于绘图
    return results_str, acc_permutations

# %% 6. 运行主程序
if __name__ == "__main__":
    # === 用户配置选项 ===
    CONFIG_PATH = r"IC\M79\M79.json" 
    
    PCA_VAR_RATIO = 0.7 
    N_PERMUTATIONS = 1000 
    FEATURE_TYPE_TO_RUN = "enhanced" 
    
    # --- SVM 固定配置 ---
    SVM_C = 0.1
    TEST_SIZE = 0.2 
    RANDOM_STATE = 42
    # ==================
    
    try:
        # A. 提取鼠 ID
        mouse_name = os.path.basename(os.path.dirname(CONFIG_PATH))

        # B. 一次性加载数据和 RR 神经元索引
        global_start_time = time.time()
        segments, labels, rr_enhanced_neurons, rr_inhibitory_neurons, cfg = load_and_preprocess_data(CONFIG_PATH)
        cfg.mouse_name = mouse_name # 临时存储鼠名
        
        # 定义新的保存路径并创建文件夹
        SVM_DIR = os.path.join(cfg.data_path, "svm_shuffle") # 结果保存在 'svm_shuffle' 文件夹
        os.makedirs(SVM_DIR, exist_ok=True)
        
        # C. 运行 SVM 分类和 Shuffle 检验
        report_str, acc_permutations = analyze_svm_classification(
            segments, labels, rr_enhanced_neurons, rr_inhibitory_neurons, cfg, 
            FEATURE_TYPE_TO_RUN, PCA_VAR_RATIO, N_PERMUTATIONS, SVM_C, TEST_SIZE, RANDOM_STATE
        )
        
        if acc_permutations is None:
             raise RuntimeError("SVM 分析失败，未生成结果。")
             
        # 从 report_str 中提取观测准确率用于绘图
        # 通过字符串解析获取观测准确率
        acc_obs_line = [line for line in report_str.split('\n') if 'Test Accuracy Observed' in line]
        acc_obs = float(acc_obs_line[0].split(':')[-1].strip())
        
        # D. 可视化 - 绘制 Shuffle 准确率分布图
        plot_shuffle_distribution(
            acc_obs, acc_permutations, mouse_name, SVM_DIR, SVM_C, N_PERMUTATIONS
        )
        
        # E. 生成和保存报告
        final_report = report_str
        filename = f"{mouse_name}_SVM_Shuffle_C{int(SVM_C*100)}_P{N_PERMUTATIONS}.txt"
        save_path = os.path.join(SVM_DIR, filename) 
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(final_report)
        
        print(f"\n\n======================================================")
        print(f"✅ SVM Shuffle 综合报告已成功保存到文件: {save_path}")
        print(f"✅ 所有结果保存在目录: {SVM_DIR}")
        print(f"======================================================")

    except Exception as e:
        print(f"\n❌ 运行过程中发生致命错误: {e}")
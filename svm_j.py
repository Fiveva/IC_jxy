import os
import numpy as np
import time
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold # 导入分层 K 折交叉验证
from sklearn.metrics import accuracy_score
import json
from sub.four_class_false import ExpConfig, rr_selection_by_class

# %% 1. 定义特征提取函数 (不变)
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

# %% 2. 格式化结果字符串函数 (更新为 CV 结果)
def _format_results_string_cv(C_value, feature_type, n_features, mean_acc, std_acc, total_time, N_SPLITS):
    """将单个分类任务的交叉验证结果格式化为字符串。"""
    
    # 格式化输出字符串
    results_str = f"""
------------------------------------------------------
特征类型: {feature_type.capitalize()} RR 神经元
神经元数量: {n_features} 个
正则化参数 C: {C_value}
------------------------------------------------------
--- {N_SPLITS}-Fold 交叉验证性能 ---
  平均测试准确率: {mean_acc:.4f}
  准确率标准差: {std_acc:.4f}  (反映结果的波动性)

--- 运行时间 ---
  总运行时间: {total_time:.4f} 秒
"""
    return results_str

# %% 3. 数据加载和 RR 神经元筛选 (一次性执行)
def load_and_preprocess_data(config_file):
    """加载配置、预处理数据并筛选 RR 神经元，仅运行一次。"""
    cfg = ExpConfig(config_file)
    cache_file = os.path.join(cfg.data_path, "preprocessed_data_cache.npz")
    
    if os.path.exists(cache_file):
        print("缓存加载成功，跳过原始数据加载和预处理步骤。")
        try:
            data = np.load(cache_file)
        except TypeError:
            data = np.load(cache_file) 
        segments = data['segments']
        labels = data['labels']
    else:
        raise FileNotFoundError(f"未找到预处理缓存文件: {cache_file}。请先运行 four_class.py 生成缓存。")

    # RR 神经元筛选 
    rr_enhanced_neurons, rr_inhibitory_neurons = rr_selection_by_class(segments, np.array(labels))
    
    return segments, labels, rr_enhanced_neurons, rr_inhibitory_neurons, cfg

# %% 4. 对单一特征类型进行 C 值循环分析 (核心优化，现在执行 CV)
def analyze_single_feature_type(segments, labels, rr_enhanced_neurons, rr_inhibitory_neurons, cfg, feature_type, C_VALUES, N_SPLITS):
    """对指定的特征类型，循环运行所有 C 值，使用 K-Fold CV 并生成报告列表。"""
    
    # --- 1. 确定神经元索引 ---
    feature_type_lower = feature_type.lower()
    if feature_type_lower == "enhanced":
        rr_indices = rr_enhanced_neurons
    elif feature_type_lower == "inhibitory":
        rr_indices = rr_inhibitory_neurons
    elif feature_type_lower == "all":
        rr_indices = sorted(list(set(rr_enhanced_neurons) | set(rr_inhibitory_neurons)))
    else:
        raise ValueError(f"无效的 feature_type: {feature_type}。")
    
    n_features = len(rr_indices)
    Y = np.array(labels) # 转换为 numpy 数组方便索引
    
    if n_features == 0:
        return [f"\n--- {feature_type.capitalize()} RR 神经元 (0 个) 分类失败：神经元数量不足 ---\n"]
        
    print(f"\n======================================================\n")
    print(f"正在分析特征类型: {feature_type.capitalize()} RR 神经元 ({n_features} 个)")

    # --- 2. 耗时步骤只执行一次 ---
    t_preprocessing_start = time.time()
    
    # 特征提取
    t_stimulus = cfg.exp_info["t_stimulus"]
    X, _ = extract_rr_features(segments, labels, rr_indices, t_stimulus)
    
    # 初始化 K-Fold 分层交叉验证器
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    
    t_preprocessing_end = time.time()
    print(f"预处理 (特征提取) 耗时: {t_preprocessing_end - t_preprocessing_start:.4f} 秒")
    
    all_c_reports = []
    
    # --- 3. 循环训练 SVM (对每个 C 值执行完整的 K-Fold CV) ---
    for C_value in C_VALUES:
        t_cv_start = time.time()
        fold_accuracies = []
        
        print(f"  -> 开始 {N_SPLITS}-Fold CV, C={C_value}...", end=" ")
        
        # 3.1. K-Fold 循环
        for fold_idx, (train_index, test_index) in enumerate(skf.split(X, Y)):
            X_train, X_test = X[train_index], X[test_index]
            Y_train, Y_test = Y[train_index], Y[test_index]
            
            # 3.2. 标准化特征 (每次都在当前训练集上 fit)
            sc = StandardScaler()
            X_train_std = sc.fit_transform(X_train) 
            X_test_std  = sc.transform(X_test)  
            
            # 3.3. 训练和评估
            svm = SVC(kernel='linear', C=C_value, decision_function_shape='ovr', random_state=42)
            svm.fit(X_train_std, Y_train)       
            
            Y_test_pred = svm.predict(X_test_std)
            test_accuracy = accuracy_score(Y_test, Y_test_pred)
            fold_accuracies.append(test_accuracy)

        # 3.4. 聚合结果
        mean_acc = np.mean(fold_accuracies)
        std_acc = np.std(fold_accuracies)
        
        t_cv_end = time.time()
        cv_run_time = t_cv_end - t_cv_start
        print(f"平均 CV 准确率: {mean_acc:.4f} ± {std_acc:.4f} (耗时: {cv_run_time:.4f} 秒)")

        # 3.5. 格式化结果字符串
        results_str = _format_results_string_cv(
            C_value, feature_type, n_features, mean_acc, std_acc, cv_run_time, N_SPLITS
        )
        all_c_reports.append(results_str)
    
    return all_c_reports

# %% 5. 运行主程序
if __name__ == "__main__":
    # === 用户配置选项 ===
    
    CONFIG_PATH = r"IC\M21\M21.json"
    
    # 要测试的 C 值列表 (从小到大排列)
    C_VALUES = [0.01, 0.1, 1.0, 10.0] 
    
    # K 折交叉验证的折数
    N_SPLITS = 10 
    
    # 要测试的特征类型
    FEATURE_TYPES = ["enhanced", "inhibitory", "all"]
    
    # ==================
    
    try:
        # A. 提取鼠 ID 和 C 值范围字符串
        mouse_name = os.path.basename(os.path.dirname(CONFIG_PATH))
        c_min = min(C_VALUES)
        c_max = max(C_VALUES)
        c_range_str = f"{c_min}".replace('.', '_') + "_to_" + f"{c_max}".replace('.', '_')

        # B. 一次性加载数据和 RR 神经元索引
        global_start_time = time.time()
        segments, labels, rr_enhanced_neurons, rr_inhibitory_neurons, cfg = load_and_preprocess_data(CONFIG_PATH)
        
        all_results_sections = []
        
        # C. 循环运行不同特征类型的分析 (在内部执行 CV 循环)
        for feature_type in FEATURE_TYPES:
            c_reports_for_type = analyze_single_feature_type(
                segments, labels, rr_enhanced_neurons, rr_inhibitory_neurons, cfg, feature_type, C_VALUES, N_SPLITS
            )
            all_results_sections.extend(c_reports_for_type)
            
        total_analysis_time = time.time() - global_start_time

        # D. 生成最终报告
        final_report_header = f"""
======================================================
SVM 线性分类器多特征类型综合报告 (K-Fold 交叉验证)
======================================================
整体配置:
  鼠 ID: {mouse_name}
  正则化参数 C 范围: [{c_min}, {c_max}]
  交叉验证折数 (K): {N_SPLITS}
  特征时间窗: 刺激出现后 2 帧
  总分析耗时 (含加载): {total_analysis_time:.2f} 秒
"""
        final_report = final_report_header + "\n\n" + ("\n" * 2).join(all_results_sections)

        # E. 保存文件
        # 文件名格式: mouse+Cvalue_range
        filename = f"{mouse_name}_C{c_range_str}_CV{N_SPLITS}.txt"
        save_path = os.path.join(cfg.data_path, filename)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(final_report)
        
        print(f"\n\n======================================================")
        print(f"✅ 综合报告已成功保存到文件: {save_path}")
        print(f"======================================================")

    except Exception as e:
        print(f"\n❌ 运行过程中发生致命错误: {e}")
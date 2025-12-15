import os
import numpy as np
import time
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.utils import shuffle
from sklearn.manifold import TSNE 
import matplotlib.pyplot as plt 
import seaborn as sns 

# 假设 sub 模块和相关配置类已存在于您的环境中
from four_class import ExpConfig, rr_selection_by_class

# --- 新增全局/常量定义：真实标签映射 ---
# 假设您的真实标签数字 (labels) 对应关系如下，这与您的 docx 中提到的四类一致
TRUE_LABEL_MAP = {
    1: 'IC2',
    2: 'IC4',
    3: 'LC2',
    4: 'LC4'
}

# %% 1. 定义特征提取函数 (沿用您的代码)
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

# %% 2. 数据加载和 RR 神经元筛选 (沿用您的代码)
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

# %% 3. 聚类结果格式化函数 (沿用您的代码)
def _format_clustering_results(feature_type, n_features, n_clusters, var_ratio, n_pca_components, ari_obs, nmi_obs, p_value_ari, p_value_nmi, total_time, N_PERMUTATIONS):
    """将聚类结果格式化为字符串，包括 Shuffle 检验结果。"""
    
    results_str = f"""
======================================================
聚类结果 (KMeans, K={n_clusters})
特征类型: {feature_type.capitalize()} RR 神经元
神经元数量: {n_features} 个
PCA 配置: 目标保留 {var_ratio*100:.0f}% 解释方差
PCA 实际保留主成分数: {n_pca_components}
------------------------------------------------------
--- 聚类评估指标 (与真实标签的关联度) ---
  观测到的 ARI (Adjusted Rand Index): {ari_obs:.4f}
  观测到的 NMI (Normalized Mutual Information): {nmi_obs:.4f}

--- Shuffle 显著性检验 ({N_PERMUTATIONS} 次置换) ---
  ARI 的 P 值: {p_value_ari:.4f}  (P < 0.05 即为显著)
  NMI 的 P 值: {p_value_nmi:.4f}  (P < 0.05 即为显著)

--- 运行时间 ---
  总运行时间: {total_time:.4f} 秒
======================================================
"""
    return results_str

# %% 4. 对单一特征类型进行聚类分析 (核心逻辑)
def analyze_clustering(segments, labels, rr_enhanced_neurons, rr_inhibitory_neurons, cfg, feature_type, PCA_VAR_RATIO, N_CLUSTERS, N_PERMUTATIONS):
    """
    对指定的特征类型执行聚类分析（标准化 -> PCA -> KMeans），
    并使用标签置换检验评估聚类与真实标签的显著性。
    """
    
    t_start = time.time()
    
    # --- 1. 确定神经元索引与数据加载 ---
    feature_type_lower = feature_type.lower()
    if feature_type_lower == "enhanced":
        rr_indices = rr_enhanced_neurons
    else:
        raise ValueError(f"该脚本设计为分析 'enhanced' (兴奋性) RR 神经元，以符合 docx 逻辑。")
    
    n_features = len(rr_indices)
    Y_true = np.array(labels) 
    
    if n_features == 0:
        return f"\n--- {feature_type.capitalize()} RR 神经元 (0 个) 聚类失败：神经元数量不足 ---\n", None, None
    
    print(f"\n======================================================\n")
    print(f"正在分析特征类型: {feature_type.capitalize()} RR 神经元 ({n_features} 个)")
    
    # 特征提取
    t_stimulus = cfg.exp_info["t_stimulus"]
    X, _ = extract_rr_features(segments, labels, rr_indices, t_stimulus)
    
    # --- 2. docx 核心逻辑：标准化 + PCA 降维 ---
    print("-> 1. 标准化 (StandardScaler)...")
    sc = StandardScaler()
    X_scaled = sc.fit_transform(X)
    
    # PCA: 保留 70% 解释方差 (var_ratio=0.7)
    print(f"-> 2. PCA 降维 (目标保留 >= {PCA_VAR_RATIO*100:.0f}% 方差)...")
    pca = PCA(n_components=PCA_VAR_RATIO, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    n_pca_components = X_pca.shape[1]
    explained_variance = np.sum(pca.explained_variance_ratio_)
    print(f"   -> PCA 实际保留 {n_pca_components} 个主成分，共解释 {explained_variance:.4f} 的方差。")

    # --- 3. KMeans 聚类 (K=4) ---
    print(f"-> 3. KMeans 聚类 (K={N_CLUSTERS})...")
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    Y_pred = kmeans.fit_predict(X_pca) # 聚类结果 (Cluster Labels)
    
    # --- 4. 评估 (观测值) ---
    ari_obs = adjusted_rand_score(Y_true, Y_pred)
    nmi_obs = normalized_mutual_info_score(Y_true, Y_pred)

    # --- 5. 标签置换检验 (Shuffle Test) ---
    print(f"-> 4. 开始标签置换检验 ({N_PERMUTATIONS} 次置换)...")
    ari_permutations = []
    nmi_permutations = []
    
    for i in range(N_PERMUTATIONS):
        if (i + 1) % 100 == 0:
            print(f"   -> 已完成 {i+1} 次置换...")
            
        Y_shuffled = shuffle(Y_true, random_state=i) 
        ari_shuffled = adjusted_rand_score(Y_shuffled, Y_pred)
        nmi_shuffled = normalized_mutual_info_score(Y_shuffled, Y_pred)
        
        ari_permutations.append(ari_shuffled)
        nmi_permutations.append(nmi_shuffled)

    p_value_ari = np.sum(np.array(ari_permutations) >= ari_obs) / N_PERMUTATIONS
    p_value_nmi = np.sum(np.array(nmi_permutations) >= nmi_obs) / N_PERMUTATIONS
    
    # --- 6. 格式化输出 ---
    total_time = time.time() - t_start
    results_str = _format_clustering_results(
        feature_type, n_features, N_CLUSTERS, PCA_VAR_RATIO, n_pca_components,
        ari_obs, nmi_obs, p_value_ari, p_value_nmi, total_time, N_PERMUTATIONS
    )
    
    return results_str, X_pca, Y_true, Y_pred 

# %% 5. 可视化函数 (标签处理逻辑已调整)
def visualize_data_space(X_data, Y_true, Y_pred, mouse_name, save_dir):
    """
    使用 t-SNE 将 PCA 降维后的数据进一步降到 2D，
    并分别按真实标签和聚类结果着色绘图。
    """
    if X_data is None or len(X_data) == 0:
        print("无法可视化：输入数据为空。")
        return

    print("\n-> 5. 正在执行 t-SNE 降维 (基于 PCA 空间)...")
    
    tsne = TSNE(
        n_components=2, 
        perplexity=30, 
        learning_rate='auto', 
        n_iter=1000, 
        init='pca',          
        random_state=42      
    )
    
    X_tsne = tsne.fit_transform(X_data)
    
    # --- 关键修改：处理标签用于可视化 ---
    
    # 1. True Label: 将数字 (1, 2, 3, 4) 映射为字符串 ('IC2', 'IC4', 'LC2', 'LC4')
    # 使用 np.vectorize 快速应用映射
    Y_true_str = np.vectorize(TRUE_LABEL_MAP.get)(Y_true)
    
    # 2. Cluster Label: 将 K-Means 的 0-based 标签 (0, 1, 2, 3) 调整为 1-based (1, 2, 3, 4)
    # K-Means 的标签是任意的，但调整后从 1 开始显示更符合习惯。
    Y_pred_adjusted = (Y_pred + 1).astype(str) 
    
    # ------------------------------------

    # 为绘图准备数据帧
    plot_df = {}
    plot_df['tSNE-1'] = X_tsne[:, 0]
    plot_df['tSNE-2'] = X_tsne[:, 1]
    plot_df['True_Label'] = Y_true_str
    plot_df['Cluster_Label'] = Y_pred_adjusted
    
    # 确保保存目录存在 (由主程序确保)
    os.makedirs(save_dir, exist_ok=True)
    
    # ----------------------------------------------------
    # 图 1: 按真实标签着色 (True Labels: IC2/IC4/LC2/LC4)
    # ----------------------------------------------------
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x='tSNE-1', y='tSNE-2', 
        hue='True_Label', 
        data=plot_df, 
        # 使用 tab10 调色板
        palette='tab10', 
        s=50
    )
    plt.title(f'{mouse_name} Trial Activity t-SNE (Colored by TRUE Label)')
    plt.legend(title='True Label')
    save_path_true = os.path.join(save_dir, f'tsne_scatter_TrueLabel_{mouse_name}.png') 
    plt.savefig(save_path_true)
    plt.close()
    print(f"   -> 真实标签可视化图已保存到: {save_path_true}")
    
    # ----------------------------------------------------
    # 图 2: 按聚类结果着色 (Cluster Labels: 1/2/3/4)
    # ----------------------------------------------------
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x='tSNE-1', y='tSNE-2', 
        hue='Cluster_Label', 
        data=plot_df, 
        # 使用 Set1 调色板，确保颜色区别明显
        palette='Set1', 
        s=50
    )
    plt.title(f'{mouse_name} Trial Activity t-SNE (Colored by CLUSTER Label)')
    plt.legend(title='Cluster Label')
    save_path_pred = os.path.join(save_dir, f'tsne_scatter_ClusterLabel_{mouse_name}.png') 
    plt.savefig(save_path_pred)
    plt.close()
    print(f"   -> 聚类结果可视化图已保存到: {save_path_pred}")


# %% 6. 运行主程序 (保存逻辑已调整)
if __name__ == "__main__":
    # === 用户配置选项 ===
    
    CONFIG_PATH = r"IC\M21\M21.json" 
    
    PCA_VAR_RATIO = 0.7 
    N_CLUSTERS = 4 
    N_PERMUTATIONS = 1000 
    
    FEATURE_TYPE_TO_RUN = "enhanced" 
    
    # ==================
    
    try:
        # A. 提取鼠 ID
        mouse_name = os.path.basename(os.path.dirname(CONFIG_PATH))

        # B. 一次性加载数据和 RR 神经元索引
        global_start_time = time.time()
        segments, labels, rr_enhanced_neurons, rr_inhibitory_neurons, cfg = load_and_preprocess_data(CONFIG_PATH)
        
        # 定义新的保存路径并创建文件夹
        CLUSTERING_DIR = os.path.join(cfg.data_path, "clustering")
        os.makedirs(CLUSTERING_DIR, exist_ok=True)
        
        # C. 运行聚类分析
        report_str, X_pca_result, Y_true, Y_pred_result = analyze_clustering(
            segments, labels, rr_enhanced_neurons, rr_inhibitory_neurons, cfg, 
            FEATURE_TYPE_TO_RUN, PCA_VAR_RATIO, N_CLUSTERS, N_PERMUTATIONS
        )
        
        # D. **可视化** - 将 CLUSTERING_DIR 作为保存目录传入
        visualize_data_space(
            X_pca_result, Y_true, Y_pred_result, mouse_name, CLUSTERING_DIR
        )
        
        total_analysis_time = time.time() - global_start_time

        # E. 生成和保存报告
        final_report_header = f"""
======================================================
PCA-KMeans 聚类分析综合报告 (含 Shuffle 检验)
======================================================
整体配置:
  鼠 ID: {mouse_name}
  特征时间窗: 刺激出现后 2 帧
  聚类数 (K): {N_CLUSTERS}
  Shuffle 检验次数: {N_PERMUTATIONS} 次
  总分析耗时 (含加载): {total_analysis_time:.2f} 秒
"""
        final_report = final_report_header + "\n\n" + report_str

        filename = f"{mouse_name}_Clustering_K{N_CLUSTERS}_P{N_PERMUTATIONS}.txt"
        # 报告保存到 CLUSTERING_DIR
        save_path = os.path.join(CLUSTERING_DIR, filename) 
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(final_report)
        
        print(f"\n\n======================================================")
        print(f"✅ 综合报告已成功保存到文件: {save_path}")
        print(f"======================================================")

    except Exception as e:
        print(f"\n❌ 运行过程中发生致命错误: {e}")
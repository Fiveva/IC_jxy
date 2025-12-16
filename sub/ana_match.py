import numpy as np
import pandas as pd
from scipy.io import loadmat
import os

def analyze_rr_neurons(csv_path, mat_path):
    """
    读取 RR 神经元索引和脑区标签数据，并按类别进行总结。
    
    结果将显示每个脑区的神经元数量和占该类别总数的比例。

    Args:
        csv_path (str): 包含 RR 神经元索引的 CSV 文件路径 ('rr_enhanced_indices_ordered.csv').
        mat_path (str): 包含所有神经元脑区标签的 MAT 文件路径 ('brain_results.mat').
    """
    
    print("--- 开始处理神经元数据 ---")

    # 1. 读取 brain_results.mat 文件
    try:
        mat_data = loadmat(mat_path)
        print(f"成功读取 MAT 文件: {mat_path}")
    except FileNotFoundError:
        print(f"错误：未找到文件 {mat_path}。请检查路径。")
        return
    except Exception as e:
        print(f"读取 MAT 文件时发生错误: {e}")
        return

    # 提取脑区标签和 ID 数据
    # 'brain_region' 是一个 MATLAB cell 数组，需要特殊处理
    try:
        # 将 MATLAB cell 数组转换为 Python 列表
        brain_region_cells = mat_data['brain_region'][0]
        # 解包成字符串列表
        all_brain_regions = [region[0] for region in brain_region_cells]
        
        # brain_region_id 是一个 1xN 的数组
        all_brain_region_ids = mat_data['brain_region_id'][0]
        
        # 验证数据长度
        if len(all_brain_regions) != len(all_brain_region_ids):
            print("警告：脑区名称和 ID 列表长度不匹配！")
            return
            
        print(f"总神经元数量 (来自 MAT 文件): {len(all_brain_regions)}")

    except KeyError as e:
        print(f"错误：MAT 文件中缺少关键变量 {e}。请确认 'brain_results.mat' 包含 'brain_region' 和 'brain_region_id'。")
        return
    except Exception as e:
        print(f"处理 MAT 数据时发生未知错误: {e}")
        return


    # 2. 读取 rr_enhanced_indices_ordered.csv 文件
    try:
        # 假设 CSV 文件包含一列或多列神经元索引
        rr_indices_df = pd.read_csv(csv_path, header=None)
        print(f"成功读取 CSV 文件: {csv_path}")
    except FileNotFoundError:
        print(f"错误：未找到文件 {csv_path}。请检查路径。")
        return
    except Exception as e:
        print(f"读取 CSV 文件时发生错误: {e}")
        return
        
    # 3. 提取 RR 神经元数据并分类总结 (新增比例计算功能)

    results = {}
    
    for col_index in range(rr_indices_df.shape[1]):
        category_name = f"RR_Category_{col_index + 1}"
        
        # 提取非 NaN 的索引，并转换为 0-based 整数索引
        matlab_indices = rr_indices_df.iloc[:, col_index].dropna().astype(int).values
        python_indices = matlab_indices - 1 
        
        if len(python_indices) == 0:
            print(f"警告：类别 {category_name} 中没有有效神经元索引。跳过。")
            continue
            
        # 过滤掉超出范围的索引 (安全检查)
        max_idx = len(all_brain_regions)
        valid_indices = python_indices[(python_indices >= 0) & (python_indices < max_idx)]
        invalid_count = len(python_indices) - len(valid_indices)

        if invalid_count > 0:
            print(f"警告：类别 {category_name} 中有 {invalid_count} 个索引超出 MAT 数据范围，已忽略。")
            
        if len(valid_indices) == 0:
            print(f"警告：类别 {category_name} 中所有索引均无效。跳过。")
            continue

        # 获取 RR 神经元的脑区标签
        rr_brain_regions = [all_brain_regions[i] for i in valid_indices]
        
        # 统计总结
        region_counts = pd.Series(rr_brain_regions).value_counts()
        total_rr_neurons = region_counts.sum()
        
        # >>>>> 新增：计算比例 <<<<<
        region_summary_df = region_counts.rename('Count').to_frame()
        region_summary_df['Percentage'] = (region_summary_df['Count'] / total_rr_neurons) * 100
        region_summary_df['Percentage'] = region_summary_df['Percentage'].round(2) # 保留两位小数
        region_summary_df = region_summary_df.sort_values(by='Count', ascending=False)
        
        # 将结果存入字典
        results[category_name] = {
            'Total_Neurons': total_rr_neurons,
            'Region_Summary': region_summary_df
        }
        
        print(f"\n--- 类别总结: {category_name} (总数: {total_rr_neurons}) ---")
        # 打印包含数量和比例的 DataFrame
        print(region_summary_df.to_string())

    # 4. 统一输出结果 (保存到新的 CSV)
    
    if results:
        # 将所有类别的总结合并为一个 DataFrame
        summary_dfs = []
        for category, data in results.items():
            df = data['Region_Summary'].copy()
            df['Category'] = category
            # 重置索引，将 'Brain_Region' 从索引转换为列
            summary_dfs.append(df.reset_index().rename(columns={'index': 'Brain_Region'}))
        
        final_summary = pd.concat(summary_dfs)
        output_csv_path = 'rr_neuron_region_summary_21.csv'
        final_summary.to_csv(output_csv_path, index=False)
        
        print(f"\n--- 总结完成 ---")
        print(f"详细的分类脑区统计结果 (包含数量和比例) 已保存到: {output_csv_path}")
    else:
        print("\n未找到任何有效的 RR 神经元分类数据。")

# --- 脚本执行部分 ---
if __name__ == "__main__":
    
    # 替换成您的实际文件路径
    RR_CSV_FILE = 'IC/M21/rr_enhanced_indices_ordered.csv'
    BRAIN_MAT_FILE = 'IC/M21/brain_results.mat'
    
    # 检查文件是否存在
    if not os.path.exists(RR_CSV_FILE):
        print(f"文件未找到: {RR_CSV_FILE}")
    elif not os.path.exists(BRAIN_MAT_FILE):
        print(f"文件未找到: {BRAIN_MAT_FILE}")
    else:
        analyze_rr_neurons(RR_CSV_FILE, BRAIN_MAT_FILE)
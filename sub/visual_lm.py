import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt
import seaborn as sns # <-- 新增导入 Seaborn
from tifffile import imread
import h5py
from tqdm import tqdm # 保持导入，尽管在这个脚本中主要用于调试/信息输出


# --- 字体配置（解决中文乱码） ---
try:
    # 请确保您的系统或环境中安装了 SimHei 或其他中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  
    plt.rcParams['axes.unicode_minus'] = False
    print("Matplotlib 已配置 SimHei 字体支持中文。")
except Exception:
    print("警告：SimHei 字体配置失败。图表中的中文可能仍显示为乱码。")


# --- 1. 配置文件和常量 ---
# 文件路径（根据您的运行环境保持不变）
TIF_FILE = 'IC/M79/whole_brain_3d.tif'
MAT_BRAIN_RESULTS = 'IC/M79/brain_results.mat'
MAT_WHOLE_CENTER = 'IC/M79/wholebrain_output.mat' 

OUTPUT_PLOT_FILE = 'visual_lm.png' # 更改输出文件名

# 定义需要标记的新目标脑区列表
TARGET_AREAS = {
    'Primary visual area layer 1': 'red',             
    'Laterointermediate area layer 1': 'orange', 
    'Lateral visual area layer 1': 'green',           
}

# --- 2. 辅助函数：背景图生成 (不变) ---
def process_background_image(file_path):
    """
    加载 3D TIF 图像，提取中间切片并归一化，作为背景图。
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"未找到 TIF 文件: {file_path}")
    
    print(f"尝试加载背景 TIF 文件: {file_path}")
    
    try:
        brain_img = imread(file_path) 
        
        if brain_img.ndim < 3:
            mid_slice = brain_img.astype(float)
        else:
            z_dim = brain_img.shape[0]
            mid_slice_index = z_dim // 2
            mid_slice = brain_img[mid_slice_index, :, :].astype(float)
            print(f"提取中间切片 (Z={mid_slice_index}) 作为背景。")
        
        max_val = np.nanmax(mid_slice)
        if max_val > 0:
            mid_slice = mid_slice / max_val
        else:
            mid_slice = np.zeros_like(mid_slice)
            
        return mid_slice
        
    except Exception as e:
        raise IOError(f"加载或处理 TIF 文件失败: {e}")


# --- 3. 核心功能：加载并处理数据 (不变) ---
def get_neuron_positions_and_areas(mat_brain_results, mat_whole_center):
    """加载脑区结果和神经元坐标，并筛选出目标区域的所有神经元。"""
    
    # 3.1 加载脑区结果 (brain_results.mat)
    try:
        print(f"尝试读取 MAT 文件: {mat_brain_results}")
        # 兼容 v7.3 文件
        mat_data = loadmat(mat_brain_results, struct_as_record=False, squeeze_me=True)
        all_brain_regions = mat_data['brain_region'].tolist()
        total_neurons_count = len(all_brain_regions)
        print(f"总神经元数量（根据 brain_results.mat）: {total_neurons_count}")
        
    except Exception as e:
        print(f"读取 MAT 文件 {mat_brain_results} 时发生错误：{e}")
        return None, None, None

    # 3.2 加载全部神经元原始坐标 (wholebrain_output.mat) - 使用 h5py
    try:
        print(f"尝试使用 h5py 读取 MAT 文件: {mat_whole_center}")
        with h5py.File(mat_whole_center, 'r') as f:
            whole_center_data = f['whole_center'][:]
            # 检查并处理 MATLAB 数组可能需要的转置
            if whole_center_data.shape[0] < whole_center_data.shape[1]:
                 whole_center_data = whole_center_data.T
            # 提取 [X, Y] 坐标 (假设 (Y, X) => [X, Y])
            # 注意: whole_center_data[:, [1, 0]] 是 [Y, X] 轴数据，用于绘图时 X 轴是 Y 值，Y 轴是 X 值
            all_neuron_pos = whole_center_data[:, [1, 0]] 
        
        if len(all_neuron_pos) != total_neurons_count:
            print(f"警告：whole_center 数量 ({len(all_neuron_pos)}) 与 brain_region 数量 ({total_neurons_count}) 不匹配！")
            
    except Exception as e:
        print(f"读取 MAT 文件 {mat_whole_center} 时发生致命错误：{e}")
        return None, None, None
    
    # 3.3 筛选出目标脑区的 ALL 神经元
    neuron_groups = {}
    
    for area_name in TARGET_AREAS.keys():
        # 找到属于该脑区的所有神经元的 0-based 索引
        area_all_indices = np.where(np.array(all_brain_regions) == area_name)[0]
        
        if len(area_all_indices) > 0:
            # 提取这些神经元的 (X, Y) 坐标
            pos_xy = all_neuron_pos[area_all_indices, :]
            neuron_groups[area_name] = pos_xy
            print(f"  - 区域 '{area_name}' 中共有 {len(pos_xy)} 个神经元。")
        else:
            print(f"  - 区域 '{area_name}' 中没有找到神经元。")

    # 返回筛选后的神经元位置 和 所有神经元位置 (all_neuron_pos)
    return neuron_groups, all_neuron_pos, all_brain_regions


# --- 4. 绘图函数 (核心修改) ---
def plot_visual_neurons(background_img, neuron_groups, all_neuron_pos):
    """
    在背景图像上绘制所有神经元（作为灰色背景）和标记的特定视觉区域神经元。
    """
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 4.1 绘制背景图像
    # 关键修正：设置 vmax=0.999，确保归一化的白色背景显示为浅灰色，使其可见
    ax.imshow(background_img, cmap="Greys", alpha=0.35, vmin=0, vmax=0.999) 
    
    # 4.2 **新增功能：绘制全部神经元作为灰色背景**
    # all_neuron_pos 的形状是 (N, 2)，其中第二维是 [X, Y]
    # 我们绘制时需要 x 轴是 X，y 轴是 Y。
    # 根据之前对 all_neuron_pos 的处理 (whole_center_data[:, [1, 0]]，即 [Y, X])，
    # 这里的 all_neuron_pos 是 (N_neurons, [Y_coord, X_coord])
    # 因此 X 坐标是 all_neuron_pos[:, 0] (Y值)，Y 坐标是 all_neuron_pos[:, 1] (X值)。
    # **注意：** 这里的 x/y 轴的映射可能需要根据您实际图片的显示方向进行微调。

    # 为了与您提供的片段保持一致，使用您提供的颜色和样式
    sns.scatterplot(
        x=all_neuron_pos[:, 0], # Y 坐标 (在原始图像上通常是横轴或 Y 轴)
        y=all_neuron_pos[:, 1], # X 坐标 (在原始图像上通常是纵轴或 X 轴)
        s=18,
        color="#9fb3c8", # 浅灰色/蓝灰色
        alpha=0.35,
        edgecolor="none",
        ax=ax,
        label="All neurons",
    )
    
    # 4.3 绘制目标区域的神经元 (在全部神经元之上)
    total_marked = 0
    
    for area_name, pos_xy in neuron_groups.items():
        if len(pos_xy) > 0:
            color = TARGET_AREAS[area_name]
            
            # 使用 Matplotlib 绘制特定区域的神经元，以便更好地控制图层和样式
            ax.scatter(pos_xy[:, 0], pos_xy[:, 1], # X 坐标是 pos_xy[:, 0]，Y 坐标是 pos_xy[:, 1]
                       s=30, # 稍微调大 S，使其在全部神经元中突出
                       c=color, 
                       marker='o',
                       edgecolors='black', 
                       linewidths=0.5,
                       zorder=3, # 确保在背景和全部神经元之上
                       label=f'{area_name} (N={len(pos_xy)})')
            total_marked += len(pos_xy)
            
    # 设置坐标轴范围以匹配图像
    y_max, x_max = background_img.shape
    ax.set_xlim(0, x_max)
    ax.set_ylim(y_max, 0) # 反转 Y 轴，使原点在左上角
    
    # 设置标题和图例
    if total_marked == 0:
        ax.set_title("未找到目标视觉区域的神经元")
    else:
        ax.set_title(f"特定视觉区域神经元分布 (总标记: {total_marked} / 全部: {len(all_neuron_pos)})")
        
    ax.legend(loc='lower right', fontsize=8)
    ax.axis('off') 
    fig.tight_layout()
    
    # 保存图像
    plt.savefig(OUTPUT_PLOT_FILE, dpi=300)
    print(f"\n绘图已保存到: {OUTPUT_PLOT_FILE}")
    plt.show()


# --- 5. 主执行逻辑 (修改) ---
if __name__ == '__main__':
    
    # 检查所有必需的文件
    required_files = {TIF_FILE, MAT_WHOLE_CENTER, MAT_BRAIN_RESULTS}
    missing = [f for f in required_files if not os.path.exists(f)]
    
    if missing:
        print("********************************************************************************")
        print(f"*** 错误：以下必需的文件未找到，请检查文件路径：{', '.join(missing)} ***")
        print("********************************************************************************")
    else:
        try:
            # 1. 加载和处理背景图像
            background_img = process_background_image(TIF_FILE)
            
            # 2. 加载和处理神经元数据
            neuron_groups, all_neuron_pos, all_brain_regions = get_neuron_positions_and_areas(
                MAT_BRAIN_RESULTS, MAT_WHOLE_CENTER
            )
            
            # 3. 绘图 (传入 all_neuron_pos)
            if neuron_groups is not None:
                # 确保 all_neuron_pos 不是 None 且有数据
                if all_neuron_pos is not None and len(all_neuron_pos) > 0:
                    plot_visual_neurons(background_img, neuron_groups, all_neuron_pos)
                else:
                    print("\n错误：未找到任何神经元位置数据 (all_neuron_pos)。无法绘图。")
                
        except Exception as e:
            print(f"\n致命错误：脚本执行失败。{e}")
            
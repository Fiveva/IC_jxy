import os

# 💡 文件名配置
input_file_name = "mod_2025`024_M79.txt"
output_file_name = "indexed_" + input_file_name # 输出到新的文件，避免覆盖

# 🔍 要搜索的字符串
search_2 = " 2 "
search_1 = " 1 "

# 计数器：用于给包含 " 1 " 的行添加索引
index_1_count = 0
modified_lines = []

print(f"尝试读取文件: {input_file_name}")

# --- 核心逻辑 ---
try:
    # 1. 读取文件并处理每一行
    with open(input_file_name, 'r', encoding='utf-8') as infile:
        print("文件读取成功，开始处理...")
        
        for line in infile:
            # 去除行末的换行符，方便后续添加内容和新的换行符
            line = line.rstrip('\n') 
            
            new_line = line # 默认新行为原行

            # a. 优先处理包含 " 2 " 的行：添加 " a"
            # 注意：如果一行同时包含 " 2 " 和 " 1 "，它将首先被添加 " a"，然后才会被添加 " 1 " 的索引
            if search_2 in line:
                new_line += " a"
                # 如果同一行也包含 " 1 "，则继续下面的判断，追加索引

            # b. 处理包含 " 1 " 的行：添加 " [索引]"
            if search_1 in line:
                index_1_count += 1
                new_line += f" {index_1_count}"

            # 将处理后的行（带上换行符）添加到列表中
            modified_lines.append(new_line + '\n')

    # 2. 将修改后的内容写入新文件
    with open(output_file_name, 'w', encoding='utf-8') as outfile:
        outfile.writelines(modified_lines)
        
    # --- 结果输出 ---
    print("\n--- 处理完成 ---")
    print(f"原始文件: '{input_file_name}'")
    print(f"已生成新文件: '{output_file_name}'")
    print(f"共处理 {len(modified_lines)} 行。")
    print(f"共为 {index_1_count} 个包含 '{search_1}' 的行添加了索引。")
    print("----------------")

except FileNotFoundError:
    print(f"\n❌ 错误：文件 '{input_file_name}' **未找到**。")
    print("请确保该文件与你的 Python 脚本位于**同一个文件夹**内，或提供**正确的路径**。")
except Exception as e:
    print(f"\n❌ 发生其他错误: {e}")
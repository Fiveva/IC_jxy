import os

# 💡 确保这个文件名与你实际的文件名一致
file_name = "2025`024_M79.txt"
# 🔍 要搜索的字符串，注意包含两端的空格
search_string = " 2 "
count = 0

print(f"尝试打开文件: {file_name}")

# --- 核心逻辑 ---
try:
    # 使用 'with open' 确保文件在处理完毕后自动关闭
    # 'r' 表示只读模式
    with open(file_name, 'r', encoding='utf-8') as file:
        print(f"成功打开文件: {file_name}")

        # 逐行读取文件内容
        for line_number, line in enumerate(file, 1):
            # 使用字符串的 count() 方法来计算当前行中目标字符串的出现次数
            occurrences_in_line = line.count(search_string)
            count += occurrences_in_line
            
            # 如果需要显示每一行找到的次数，可以取消注释下面这行：
            # if occurrences_in_line > 0:
            #     print(f"第 {line_number} 行找到 {occurrences_in_line} 次")

    # --- 结果输出 ---
    print("\n--- 统计结果 ---")
    print(f"在文件 '{file_name}' 中，")
    print(f"包含 **\"{search_string}\"** 字符的次数为: **{count}**")
    print("----------------")

except FileNotFoundError:
    print(f"\n❌ 错误：文件 '{file_name}' **未找到**。")
    print("请确保该文件与你的 Python 脚本位于**同一个文件夹**内，或提供**正确的路径**。")
except Exception as e:
    print(f"\n❌ 发生其他错误: {e}")
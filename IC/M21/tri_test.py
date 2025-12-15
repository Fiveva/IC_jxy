import os
import sys

# 1. 确定当前脚本所在的目录
# os.path.abspath(__file__) 获取当前脚本的绝对路径，os.path.dirname 获取目录名。
# 使用 __file__ 比 os.listdir('.') 更可靠，因为 os.listdir('.') 获取的是
# 运行命令时的“当前工作目录”，而 __file__ 总是指向脚本所在的目录。
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # 针对某些环境（如交互式环境）__file__ 不存在的情况，退而求其次使用 os.getcwd()
    script_dir = os.getcwd()

# 2. 使用列表推导式获取所有 .txt 文件名
# os.listdir(script_dir) 获取目录下的所有文件名
txt_files = [f for f in os.listdir(script_dir) if f.endswith('.txt')]

# 3. 依次打印文件名
print(f"--- 正在扫描目录: {script_dir} ---")
if txt_files:
    print("找到的 .txt 文件:")
    for filename in txt_files:
        print(f"- {filename}")
else:
    print("未找到任何 .txt 文件。")
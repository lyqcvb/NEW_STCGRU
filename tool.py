import os
import glob

# 目标目录
base_dir = "EEGData"

# 查找所有包含 seed_42 的文件
files_to_delete = glob.glob(os.path.join(base_dir, "**", "*seed_42*.pth"), recursive=True)

print(f"找到 {len(files_to_delete)} 个文件：")
for f in files_to_delete:
    print(f)

# 删除文件
for f in files_to_delete:
    try:
        os.remove(f)
        print(f"已删除: {f}")
    except Exception as e:
        print(f"删除失败: {f}, 错误: {e}")

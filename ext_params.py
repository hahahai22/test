# extract_miopen.py
import re
import csv

# === 配置区 ===
input_file = 'perf.txt'          # <-- 修改为你的文件名
output_file = 'miopen_params.csv' # 输出 CSV 文件名
# ==============

# 用于匹配 -x 数字 的正则表达式（支持 -F 12 或 -F12 这两种写法）
pattern = r'(-[a-zA-Z])\s+(\d+)'

all_rows = []
headers_set = set()

# 读取文件，逐行处理
with open(input_file, 'r') as f:
    for line in f:
        line = line.strip()
        if not line.startswith('MIOpenDriver'):
            continue  # 跳过非 MIOpenDriver 命令

        matches = re.findall(pattern, line)
        row = {key: value for key, value in matches}
        all_rows.append(row)
        headers_set.update(row.keys())

# 排序列名（按字母顺序，如 -F, -H, -W, -n, -k...）
headers = sorted(headers_set)

# 写入 CSV
with open(output_file, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=headers)
    writer.writeheader()
    writer.writerows(all_rows)

print(f"✅ 提取完成！共处理 {len(all_rows)} 行。")
print(f"📊 数据已保存到: {output_file}")
print(f"📋 列名: {headers}")

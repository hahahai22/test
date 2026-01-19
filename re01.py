import re

s = '娜扎e佟丽娅热巴a代b斯'

result = re.search('佟丽娅', s)
print(result)
 
result = re.match('佟丽娅', s)
print(result)

result = re.findall('[a-z]+', s)
print(result)

"""
1. with open自动管理文件关闭
"""
with open('001.txt', 'r') as file:
    content = file.read()

result = re.findall('(?:min_time\s:\s)(\d+.\d+)', content)
print(result)

"""
2. open显式调用file.close()关闭文件
"""
try:
    file = open('001.txt', 'r')
    content = file.read()
    match = re.findall('(?:min_time\s:\s)(\d+.\d+)', content)
    print(match)
finally:
    file.close()


file = open('001.txt', 'r')
content = file.readline(-1)
pattern = "(?:min_time\s:\s)(\d+.\d+)"
match = re.findall(pattern, content)
print(match)

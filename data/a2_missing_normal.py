from decimal import Decimal, getcontext

# 打开输入文件和输出文件
with open('input.txt', 'r') as input_file, open('output.txt', 'w') as output_file:
    for line in input_file:
        # 按逗号分割每行
        parts = line.strip().split(') ')
        # print(parts)
        # exit(0)

        # 解析前面的文本部分和四个小数部分
        text = parts[0] + ') '
        # print(text)
        # exit(0)

        decimals = [float(x) for x in parts[1][0:-1].split(', ')]
        # print(decimals)
        # exit(0)

        # 删除的小数
        decimals.pop(2)
        # print(decimals)
        # exit(0)

        # 计算总和，用于归一化
        total = sum(decimals)
        # print(total)
        # exit(0)

        # 归一化小数部分并保留6位小数
        normalized_decimals = [round(Decimal(x / total), 6) for x in decimals]
        # print(normalized_decimals)

        # 计算最后一个数值并保留6位小数
        normalized_decimals[-1] = Decimal(1.0) - sum((normalized_decimals[:-1]))
        # print(normalized_decimals)

        # 将文本和归一化的小数部分写入输出文件
        output_line = "  " + text
        for p in normalized_decimals[:-1]:
            output_line += str(p) + ", "
        output_line += str(normalized_decimals[-1]) + ";" + "\n"
        # print(output_line)

        output_file.write(output_line)
        # exit(0)
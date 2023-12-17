# 打开原始文件和目标文件
with open('processed.cleveland.data', 'r') as input_file, open('heart.txt', 'w') as output_file:
    # 逐行读取原始文件
    for line in input_file:
        # 分割每行数据
        data = line.strip().split(',')
        if len(data) == 14:
            # 对特定列进行舍入操作
            data[0] = str(round(float(data[0])) // 10 * 10)  # 第1列十位取整
            data[3] = str(round(float(data[3])) // 10 * 10)  # 第4列十位取整
            data[4] = str(round(float(data[4])) // 10 * 10)  # 第5列十位取整
            data[7] = str(round(float(data[7])) // 10 * 10)  # 第8列十位取整
            data[9] = str(int(float(data[9])))  # 第10列各位取整

            # 重新构建每行数据
            new_line = ','.join(data) + '\n'

            # 将处理后的数据写入目标文件
            output_file.write(new_line)

# 打印完成消息
print("处理完成，结果已写入output.txt文件")
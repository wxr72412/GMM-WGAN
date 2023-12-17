# 打开原始文件和目标文件
with open('bone_marrow.txt', 'r') as input_file, open('processed_bone_marrow.txt', 'w') as output_file:
    # 逐行读取原始文件
    for line in input_file:
        if "?" in line:
            pass
        else:
            # 分割每行数据
            data = line.strip().split(',')
            if len(data) == 37:
                # 对特定列进行舍入操作
                # print(data)
                # exit(0)
                # 3 C Donor age             numeric 《供体》年龄
                data[2] = str(round(float(data[2])) // 5 * 5)

                # 23 C Recipient  mnumeric 《受体》年龄
                data[22] = str(round(float(data[22])) // 5 * 5)

                # 29 C CD34 kgx10d6 numeric 受体每公斤体重的CD34细胞剂量
                data[28] = str(round(float(data[28])) // 10 * 10)

                # 30 C CD3d CD34 numeric CD3细胞与CD34细胞的比例
                data[29] = str(round(float(data[29])) // 5 * 5)

                # 31 C CD3d kgx10d8 numeric 受体每公斤体重的CD3细胞剂量
                data[30] = str(round(float(data[30])) // 5 * 5)

                # 32 C Rbody mass numeric 移植时受体的体重
                data[31] = str(round(float(data[31])) // 20 * 20)

                # 33 C ANC recovery numeric 中性粒细胞恢复为定义的中性粒细胞计数的时间
                data[32] = str(round(float(data[32])) // 10 * 10)
                # 34 C PLT recovery numeric 血小板恢复定义的血小板计数的时间
                data[33] = str(round(float(data[33])) // 10 * 10)
                # 35 C time_to_aGvHD_III_IV numeric 发展为急性移植物抗宿主病III期或IV期的时间
                data[34] = str(round(float(data[34])) // 10 * 10)
                # 36 C survival_time numeric 生存时间
                data[35] = str(round(float(data[35])) // 10 * 1000)

                # new_data = data[0:32]
                new_data = data[0:2]
                print(len(new_data))
                new_data += data[4:32]
                print(len(new_data))
                new_data.append(data[36])
                print(len(new_data))
                print(new_data)
                # exit(0)

                # 重新构建每行数据
                new_line = ','.join(new_data) + '\n'
                # 将处理后的数据写入目标文件
                output_file.write(new_line)

# 打印完成消息
print("处理完成，结果已写入output.txt文件")
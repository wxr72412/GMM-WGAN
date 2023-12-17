import os
# import numpy as np




# index = 8
# for i in range(index-1, index): # 0, 1, ..., 10
for i in range(100):
    try:
        min = float('inf')
        max = float('-inf')
        list = []
        # num = 0
        fileObject = open('processed_bone_marrow.txt', 'r', encoding='utf-8')
        while (1):
            # num+=1
            # print(num)
            line_str = fileObject.readline()[0:-1]  # 消去末尾的   换行符
            if line_str == "":
                break
            line_str_split = line_str.split(",")
            # print(line_str_split)

            temp = float(line_str_split[i])
            if temp < min:
                min = temp
            if temp > max:
                max = temp
            if temp not in list:
                list.append(temp)
        print("i: " + str(i+1) + ", min=" + str(min) + ", max=" + str(max) + ", car=" + str(len(list)))
        print(list)
        fileObject.closed
    except:
        fileObject.closed
        pass
    # exit(0)


# i: 1, min=0.0, max=1.0, car=2
# [1.0, 0.0]
# i: 2, min=0.0, max=1.0, car=2
# [1.0, 0.0]

# 移除
# 3 C Donor age             numeric 《供体》年龄
# 4 Donor age 35          {0,1} 《供体》年龄

# i: 3, min=0.0, max=1.0, car=2
# [1.0, 0.0]
# i: 4, min=0.0, max=1.0, car=2
# [0.0, 1.0]
# i: 5, min=-1.0, max=2.0, car=4
# [1.0, -1.0, 2.0, 0.0]
# i: 6, min=-1.0, max=2.0, car=4
# [1.0, -1.0, 2.0, 0.0]
# i: 7, min=0.0, max=1.0, car=2
# [1.0, 0.0]
# i: 8, min=0.0, max=1.0, car=2
# [0.0, 1.0]
# i: 9, min=0.0, max=3.0, car=4
# [3.0, 0.0, 2.0, 1.0]
# i: 10, min=0.0, max=1.0, car=2
# [1.0, 0.0]
# i: 11, min=0.0, max=1.0, car=2
# [1.0, 0.0]
# i: 13, min=0.0, max=1.0, car=2
# [1.0, 0.0]
# i: 14, min=0.0, max=1.0, car=2
# [0.0, 1.0]
# i: 15, min=0.0, max=1.0, car=2
# [1.0, 0.0]
# i: 16, min=0.0, max=3.0, car=4
# [0.0, 1.0, 2.0, 3.0]
# i: 17, min=0.0, max=1.0, car=2
# [0.0, 1.0]
# i: 18, min=-1.0, max=2.0, car=4
# [-1.0, 1.0, 0.0, 2.0]
# i: 19, min=-1.0, max=3.0, car=5
# [-1.0, 0.0, 1.0, 2.0, 3.0]
# i: 20, min=0.0, max=7.0, car=7
# [0.0, 1.0, 3.0, 2.0, 4.0, 5.0, 7.0]

# i: 23, min=0.0, max=20.0, car=5
# [10.0, 0.0, 5.0, 15.0, 20.0]
# i: 24, min=0.0, max=1.0, car=2
# [0.0, 1.0]
# i: 25, min=0.0, max=2.0, car=3
# [1.0, 0.0, 2.0]
# i: 26, min=0.0, max=1.0, car=2
# [0.0, 1.0]
# i: 27, min=0.0, max=1.0, car=2
# [0.0, 1.0]
# i: 28, min=0.0, max=1.0, car=2
# [1.0, 0.0]

# i: 29, min=0.0, max=40.0, car=3
# [0.0, 40.0, 20.0]
# i: 30, min=0.0, max=100.0, car=4
# [0.0, 10.0, 100.0, 50.0]
# i: 31, min=0.0, max=20.0, car=3
# [0.0, 10.0, 20.0]
# i: 32, min=0.0, max=60.0, car=4
# [20.0, 0.0, 40.0, 60.0]

# i: 33, min=0.0, max=1.0, car=2
# [0.0, 1.0]



import time
import pulp
import numpy as np

time_start = time.time()
# system_data_1=[最大功率,最小功率,最小开机时间,最小关机时间,初始运行状况],假设周期开始前初始
system_data = np.array([[455, 150, 8, 8],
                        [455, 150, 8, 8],
                        [130, 20, 5, 5],
                        [130, 20, 5, 5],
                        [162, 25, 6, 6],
                        [80, 20, 3, 3],
                        [85, 25, 3, 3],
                        [55, 10, 1, 1]])
load_demand = np.array([1422*0.71, 1422*0.65, 1422*0.62, 1422*0.60, 1422*0.58, 1422*0.58, 1422*0.60, 1422*0.64, 1422*0.73, 1422*0.80, 1422*0.82, 1422*0.83,
                        1422*0.82, 1422*0.80, 1422*0.79,1422*0.79, 1422*0.83, 1422*0.91, 1422*0.90, 1422*0.88, 1422*0.85, 1422*0.84, 1422*0.79
                           , 1422*0.74])
productcost_ = np.array([16.19, 17.26, 16.60, 16.50, 19.70, 22.26, 27.74, 25.92])
productcost = np.tile(productcost_[:, np.newaxis], (1, 24))
start_up_cost_ = np.array([4500, 5000, 550, 560, 900, 170, 260, 30, 30, 30])
start_up_cost = np.tile(start_up_cost_[:, np.newaxis], (1, 24))
shutdown_cost_ = np.array([4500, 5000, 550, 560, 900, 170, 260, 30, 30, 30])
shutdown_cost = np.tile(shutdown_cost_[:, np.newaxis], (1, 24))

onoffstage_v = np.zeros((8, 24))
row_matrix_v = len(onoffstage_v)
col_matrix_v = len(onoffstage_v[0])
var_v = pulp.LpVariable.dicts("v", (range(row_matrix_v), range(col_matrix_v)), 0, 1, pulp.LpBinary)
# 生产功率变量 c_p 10*24
productcost_cp = np.zeros((8, 24))
row_matrix_cp = len(productcost_cp)
col_matrix_cp = len(productcost_cp[0])
var_cp = pulp.LpVariable.dicts("cp", (range(row_matrix_cp), range(col_matrix_cp)), lowBound=0)
# 设置初始开关机状态,这里决定了0时刻各个机组的开关状态
# for i in range(len(system_data[:, 4])):
#         var_v[i][0] = 1 if system_data[i, 4] == 1 else 0
# 定义要优化的问题，最小化混合整数规划问题
UC = pulp.LpProblem('UC_task', sense=pulp.LpMinimize)

# 定义目标函数：系统总成本
UC += pulp.lpSum([
    pulp.lpSum(productcost[i][t] * var_cp[i][t] for i in range(row_matrix_cp) if var_v[i][t]==1 ) +  # 发电成本
    pulp.lpSum(start_up_cost[i][t] * (var_v[i][t] - var_v[i][t - 1]) for i in range(row_matrix_v) if t > 0) +  # 启动成本
    pulp.lpSum(shutdown_cost[i][t] * (var_v[i][t - 1] - var_v[i][t]) for i in range(row_matrix_v) if t > 0)  # 停机成本
    for t in range(col_matrix_cp)
])

# 添加约束条件：系统总发电功率大于等于负载需求
for t in range(col_matrix_cp):
    UC += pulp.lpSum(var_cp[i][t] for i in range(row_matrix_cp) if var_v[i][t]==1 ) >= load_demand[t]
# 确保每个机组在每个小时内的发电功率都在其最大功率和最小功率之间
for i in range(row_matrix_cp):
    for t in range(col_matrix_cp):
        UC += var_cp[i][t] <= system_data[i][0] * var_v[i][t]  # 最大功率约束
        UC += var_cp[i][t] >= system_data[i][1] * var_v[i][t]  # 最小功率约束

# 最小开机时间约束
for i in range(row_matrix_v):
    for t in range(col_matrix_v - system_data[i][2] + 1):
        UC += pulp.lpSum(var_v[i][t + k] for k in range(system_data[i][2])) >= system_data[i][2] * var_v[i][t]

# 最小关机时间约束
for i in range(row_matrix_v):
    for t in range(col_matrix_v - system_data[i][3] + 1):
        UC += pulp.lpSum(1 - var_v[i][t + k] for k in range(system_data[i][3])) >= system_data[i][3] * (1 - var_v[i][t])

# 模型求解
UC.solve(pulp.PULP_CBC_CMD())
time_end = time.time()
print("Minimun Cost =", pulp.value(UC.objective))
print('totally cost', time_end - time_start)
print("Status:", pulp.LpStatus[UC.status])
# 打印优化变量的值
print("Optimal Variable Values:")




V=np.zeros((8,24))
P=np.zeros((8,24))
import re


def Match_V(variable_name,array,variable_number):


    # 从变量名中提取数字
    match = re.match(r'v_(\d+)_(\d+)', variable_name)
    if match:
        row = int(match.group(1))   # 减一是因为数组从零开始
        column = int(match.group(2))   # 减一是因为数组从零开始

        array[row][column] = variable_number # 在数组中填充变量名
        return array
    else:
        print("变量名格式不符合要求")
        return None


def Match_P(variable_name, array,variable_number):
    # 从变量名中提取数字
    match = re.match(r'cp_(\d+)_(\d+)', variable_name)
    if match:
        row = int(match.group(1))   # 减一是因为数组从零开始
        column = int(match.group(2))  # 减一是因为数组从零开始
        print(row, column)
        array[row][column] = variable_number # 在数组中填充变量名
        return array
    else:
        print("变量名格式不符合要求")
        return None

# for variable in UC.variables():
#     print(f"{variable.name} = {variable.varValue}")
#


for variable in UC.variables():
    if variable.name.startswith("v_"):
        V=Match_V(variable.name,V,variable.varValue)
    elif variable.name.startswith("cp_"):
        P=Match_P(variable.name,P,variable.varValue)
print(V.shape,"开停变量")
print(P.shape,'功率变量')


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

np.random.seed(42)
data = V*P

# 设置Seaborn样式
sns.set(style='darkgrid')

# 创建图表
plt.figure(figsize=(10, 6))
markers = ['o', 's', '^', 'd', 'v', '>', '<', 'p']
linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
linewidths = [1.5, 1.8, 2.0, 2.2, 2.5, 2.0, 1.8, 1.5]

for i in range(8):
    plt.plot(range(24), data[i], marker=markers[i], linestyle=linestyles[i], linewidth=linewidths[i], label=f'{i + 1} Unit')

plt.xlabel('Time')
plt.ylabel('Power')
plt.title('Result')
plt.legend()
plt.show()






load_demand_plot = np.array([1422*0.71, 1422*0.65, 1422*0.62, 1422*0.60, 1422*0.58, 1422*0.58, 1422*0.60, 1422*0.64, 1422*0.73, 1422*0.80, 1422*0.82, 1422*0.83,
                        1422*0.82, 1422*0.80, 1422*0.79,1422*0.79, 1422*0.83, 1422*0.91, 1422*0.90, 1422*0.88, 1422*0.85, 1422*0.84, 1422*0.79
                           , 1422*0.74])


load_provide=[]
for i in range(24):
    load_provide.append(np.sum(V[:,i]*P[:,i]))


# 设置Seaborn样式
sns.set(style='darkgrid')

# 创建图表
plt.figure(figsize=(10, 6))
markers = ['o', 's']
linestyles = ['-', '--']
linewidths = [1.5, 1.8, 2.0, 2.2, 2.5, 2.0, 1.8, 1.5]

plt.plot(range(24), load_demand_plot, marker=markers[0], linestyle=linestyles[0], linewidth=linewidths[0], label='Power Demand')
plt.plot(range(24), load_provide, marker=markers[1], linestyle=linestyles[1], linewidth=linewidths[1], label='Power Povided')

plt.xlabel('Time')
plt.ylabel('Power')
plt.title('Result')
plt.legend()
plt.show()


import matplotlib.pyplot as plt
import numpy as np

# 创建数据
data = V
data2 = np.random.randint(0, 2, size=(8, 24))  # 假设这里是你的 0/1 数组数据
data2[5]=[0,0,1,1,1,1,1,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0]
data2[6]=[0,0,0,0,0,0,1,1,1,1,0,0,0,1,1,1,1,1,0,0,0,1,1,1]
data=np.logical_or(data,data2)
# 绘制不同颜色的格子图
plt.figure(figsize=(10, 6))  # 设置图形大小
plt.style.use('seaborn-dark')
# 遍历每行数据并绘制对应的颜色格子
for i in range(len(data)):
    for j in range(len(data[i])):
        color = 'white' if data[i][j] == 0 else plt.cm.tab10(i / len(data))  # 根据值选择颜色
        plt.scatter(j, i, color=color, s=250,marker='s')  # 绘制格子

plt.xlabel('Time')  # x轴标签
plt.ylabel('Unit')  # y轴标签
plt.title('Grid Plot with Different Colors')  # 图标题
plt.yticks(np.arange(8), [f'Unit{i+1}' for i in range(8)])  # 设置y轴刻度
plt.grid()
plt.show()




data=data*P
# 设置Seaborn样式
sns.set(style='darkgrid')

# 创建图表
plt.figure(figsize=(10, 6))
markers = ['o', 's', '^', 'd', 'v', '>', '<', 'p']
linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
linewidths = [3.5, 3.8, 4.0, 4.2, 4.5, 3.0, 3.8, 3.5]

for i in range(8):
    plt.plot(range(24), data[i], marker=markers[i], linestyle=linestyles[i], linewidth=linewidths[i], label=f'{i + 1} Unit')

plt.xlabel('Time')
plt.ylabel('Power')
plt.title('Result')
plt.legend()
plt.show()





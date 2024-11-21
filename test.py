import cvxpy as cp
import matplotlib.pyplot as plt
import pandas as pd

# 定义任务信息
tasks = {
    "User1": [(20, 23, 1, 1), (18, 23, 1, 2), (19, 21, 1, 1), (12, 20, 1, 3), (6, 12, 1, 3),
              (18, 20, 1, 2), (4, 10, 1, 2), (12, 18, 1, 2), (7, 14, 1, 3), (8, 14, 1, 3)],
    "User2": [(11, 22, 1, 2), (5, 11, 1, 2), (5, 23, 1, 1), (6, 20, 1, 3), (19, 19, 1, 1),
              (18, 21, 1, 2), (3, 23, 1, 3), (21, 23, 1, 2), (13, 17, 1, 1), (6, 11, 1, 2)],
    "User3": [(20, 23, 1, 2), (15, 21, 1, 3), (11, 15, 1, 2), (2, 17, 1, 3), (13, 16, 1, 2),
              (10, 18, 1, 2), (21, 23, 1, 2), (20, 23, 1, 1), (7, 21, 1, 2), (0, 7, 1, 3)],
    "User4": [(1, 8, 1, 1), (11, 20, 1, 2), (12, 19, 1, 3), (11, 16, 1, 3), (16, 18, 1, 1),
              (19, 23, 1, 3), (22, 23, 1, 1), (12, 19, 1, 2), (8, 20, 1, 2), (4, 12, 1, 2)],
    "User5": [(4, 20, 1, 1), (18, 22, 1, 3), (4, 16, 1, 1), (2, 16, 1, 3), (16, 23, 1, 2),
              (6, 18, 1, 2), (2, 6, 1, 1), (13, 17, 1, 3), (15, 23, 1, 1), (17, 23, 1, 1)]
}

# 定义小时范围
hours = range(24)

# 决策变量：每小时的任务能量分配
energy_usage = {
    (user, task_idx, hour): cp.Variable(nonneg=True)
    for user, task_list in tasks.items()
    for task_idx, (start, end, max_energy, demand) in enumerate(task_list)
    for hour in hours
}

# 决策变量：每小时的总能量使用
total_energy = {hour: cp.Variable(nonneg=True) for hour in hours}

# 辅助变量：线性化总成本的辅助变量
linear_cost_vars = {hour: cp.Variable() for hour in hours}

# 目标函数：最小化线性化成本变量的总和
objective = cp.Minimize(cp.sum(list(linear_cost_vars.values())))

# 约束条件
constraints = []

# 约束每小时的总能量使用
for hour in hours:
    constraints.append(
        total_energy[hour] == cp.sum(
            [
                energy_usage[(user, task_idx, hour)]
                for user, task_list in tasks.items()
                for task_idx, (start, end, max_energy, demand) in enumerate(task_list)
                if start <= hour <= end
            ]
        )
    )

# 线性化二次定价函数
for hour in hours:
    constraints.append(linear_cost_vars[hour] >= 0.5 * total_energy[hour] ** 2)

# 约束每个任务的能量分配
for user, task_list in tasks.items():
    for task_idx, (start, end, max_energy, demand) in enumerate(task_list):
        # 总能量需求
        constraints.append(
            cp.sum(
                [energy_usage[(user, task_idx, hour)] for hour in range(start, end + 1)]
            ) == demand
        )
        # 每小时最大能量约束
        for hour in range(start, end + 1):
            constraints.append(energy_usage[(user, task_idx, hour)] <= max_energy)

# 求解问题
problem = cp.Problem(objective, constraints)
problem.solve()

if problem.status == cp.OPTIMAL:
    # 提取每小时的总能量使用
    hourly_total_energy = [total_energy[hour].value for hour in hours]
    # 计算总成本
    total_cost = sum(linear_cost_vars[hour].value for hour in hours)
    print(f"Total Cost: {total_cost:.2f} currency units")  # 输出总成本
    # 计算每小时的单价
    hourly_prices = [0.5 * energy ** 2 for energy in hourly_total_energy]
    print(f"Hourly Prices: {hourly_prices}")  # 输出每小时单价

    # 计算所有任务的能量需求总和
    total_energy_demand = sum(demand for user_tasks in tasks.values() for _, _, _, demand in user_tasks)
    print(f"Total Energy Demand: {total_energy_demand} units")  # 输出总能量需求

    # 计算每个用户每小时的能量贡献
    user_contributions = {user: [0] * len(hours) for user in tasks.keys()}
    for user, task_list in tasks.items():
        for task_idx, (start, end, max_energy, demand) in enumerate(task_list):
            for hour in range(start, end + 1):
                user_contributions[user][hour] += energy_usage[(user, task_idx, hour)].value

    # 绘制每小时的堆叠柱状图
    plt.figure(figsize=(12, 6))
    bottom = [0] * len(hours)
    for user, contributions in user_contributions.items():
        plt.bar(hours, contributions, bottom=bottom, label=user)
        bottom = [bottom[i] + contributions[i] for i in range(len(bottom))]

    plt.xlabel("Hour")
    plt.ylabel("Total Energy Usage")
    plt.title("Hourly Energy Usage by User for Minimum Cost")
    plt.xticks(hours)
    plt.legend(title="Users")
    plt.show()

    # 显示每小时的总能量使用
    results_df = pd.DataFrame({
        "Hour": hours,
        "Total Energy Usage": hourly_total_energy,
        **{user: contributions for user, contributions in user_contributions.items()}
    })
    print(results_df)
else:
    print("No optimal solution found.")
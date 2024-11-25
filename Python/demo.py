"""
SYNOPSIS

    A simple demo for Rapid trajectory generation for quadrocopters

DESCRIPTION
    
    Generates a single trajectory, and runs input and position feasibility
    tests. Then some plots are generated to visualise the results.

AUTHOR
    
    Mark W. Mueller <mwm@mwm.im>

LICENSE

    Copyright 2014 by Mark W. Mueller <mwm@mwm.im>

    This code is free software: you can redistribute
    it and/or modify it under the terms of the GNU General Public
    License as published by the Free Software Foundation, either
    version 3 of the License, or (at your option) any later version.

    This code is distributed in the hope that it will
    be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
    of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with the code.  If not, see <http://www.gnu.org/licenses/>.
    
VERSION 

    0.0

"""

from __future__ import print_function, division  # 确保代码兼容 Python 2 和 3
import quadrocoptertrajectory as quadtraj  # 导入四轴飞行器轨迹模块

# 定义初始状态
pos0 = [0, 0, 2]  # 初始位置 (x, y, z)
vel0 = [0, 0, 0]  # 初始速度 (x, y, z)
acc0 = [0, 0, 0]  # 初始加速度 (x, y, z)

# 定义目标状态
posf = [1, 0, 2]  # 目标位置 (x, y, z)
velf = [0, 0, 1]  # 目标速度 (x, y, z)
accf = [0, 9.81, 0]  # 目标加速度 (x, y, z)

# 定义轨迹持续时间
Tf = 1  # 持续时间 [秒]

# 定义输入限制
fmin = 5  # 最小推力 [m/s²]
fmax = 25  # 最大推力 [m/s²]
wmax = 20  # 最大角速度 [rad/s]
minTimeSec = 0.02  # 最小时间间隔 [s]

# 定义重力方向
gravity = [0, 0, -9.81]  # 重力加速度方向和大小 [m/s²]

# 初始化轨迹对象
traj = quadtraj.RapidTrajectory(pos0, vel0, acc0, gravity)
traj.set_goal_position(posf)  # 设置目标位置
traj.set_goal_velocity(velf)  # 设置目标速度
traj.set_goal_acceleration(accf)  # 设置目标加速度

# 如果想要在某些轴上保持状态自由，可以这样设置
# 例如，在 x 轴（轴 0）保持速度自由
# Option 1:
# traj.set_goal_velocity_in_axis(1, velf_y)
# traj.set_goal_velocity_in_axis(2, velf_z)
# Option 2:
# traj.set_goal_velocity([None, velf_y, velf_z])

# 生成轨迹
traj.generate(Tf)

# 测试输入可行性
inputsFeasible = traj.check_input_feasibility(fmin, fmax, wmax, minTimeSec)

# 测试位置可行性，避免飞行器撞地
floorPoint = [0, 0, 0]  # 地面上的点
floorNormal = [0, 0, 1]  # 法向量，表示地面方向
positionFeasible = traj.check_position_feasibility(floorPoint, floorNormal)

# 输出轨迹参数
for i in range(3):  # 遍历 x, y, z 轴
    print("Axis #", i)
    print("\talpha = ", traj.get_param_alpha(i), "\tbeta = ", traj.get_param_beta(i), "\tgamma = ", traj.get_param_gamma(i))
print("Total cost = ", traj.get_cost())  # 输出轨迹总成本
print("Input feasibility result: ", quadtraj.InputFeasibilityResult.to_string(inputsFeasible), "(", inputsFeasible, ")")  # 输出输入可行性结果
print("Position feasibility result: ", quadtraj.StateFeasibilityResult.to_string(positionFeasible), "(", positionFeasible, ")")  # 输出位置可行性结果

###########################################
# 绘制轨迹以及输入参数的图形
###########################################

import matplotlib.pyplot as plt  # 导入绘图库
import matplotlib.gridspec as gridspec  # 用于网格布局
import numpy as np  # 导入 numpy，用于数组操作

numPlotPoints = 100  # 绘图采样点数
time = np.linspace(0, Tf, numPlotPoints)  # 生成从 0 到 Tf 的时间点
position = np.zeros([numPlotPoints, 3])  # 初始化位置数组
velocity = np.zeros([numPlotPoints, 3])  # 初始化速度数组
acceleration = np.zeros([numPlotPoints, 3])  # 初始化加速度数组
thrust = np.zeros([numPlotPoints, 1])  # 初始化推力数组
ratesMagn = np.zeros([numPlotPoints, 1])  # 初始化角速度数组

# 计算不同时间点的状态参数
for i in range(numPlotPoints):
    t = time[i]
    position[i, :] = traj.get_position(t)  # 获取位置
    velocity[i, :] = traj.get_velocity(t)  # 获取速度
    acceleration[i, :] = traj.get_acceleration(t)  # 获取加速度
    thrust[i] = traj.get_thrust(t)  # 获取推力
    ratesMagn[i] = np.linalg.norm(traj.get_body_rates(t))  # 计算角速度大小

# 创建子图
figStates, axes = plt.subplots(3, 1, sharex=True)
gs = gridspec.GridSpec(6, 2)
axPos = plt.subplot(gs[0:2, 0])  # 位置子图
axVel = plt.subplot(gs[2:4, 0])  # 速度子图
axAcc = plt.subplot(gs[4:6, 0])  # 加速度子图

# 绘制状态曲线
for ax, yvals in zip([axPos, axVel, axAcc], [position, velocity, acceleration]):
    cols = ['r', 'g', 'b']  # 定义颜色
    labs = ['x', 'y', 'z']  # 定义标签
    for i in range(3):
        ax.plot(time, yvals[:, i], cols[i], label=labs[i])

axPos.set_ylabel('Pos [m]')  # 设置 y 轴标签
axVel.set_ylabel('Vel [m/s]')
axAcc.set_ylabel('Acc [m/s^2]')
axAcc.set_xlabel('Time [s]')  # 设置 x 轴标签
axPos.legend()  # 显示图例
axPos.set_title('States')  # 设置标题

# 绘制推力和角速度曲线
infeasibleAreaColour = [1, 0.5, 0.5]  # 不可行区域颜色
axThrust = plt.subplot(gs[0:3, 1])  # 推力子图
axOmega = plt.subplot(gs[3:6, 1])  # 角速度子图
axThrust.plot(time, thrust, 'k', label='command')  # 绘制推力曲线
axThrust.plot([0, Tf], [fmin, fmin], 'r--', label='fmin')  # 最小推力线
axThrust.fill_between([0, Tf], [fmin, fmin], -1000, facecolor=infeasibleAreaColour, color=infeasibleAreaColour)  # 填充不可行区域
axThrust.fill_between([0, Tf], [fmax, fmax], 1000, facecolor=infeasibleAreaColour, color=infeasibleAreaColour)
axThrust.plot([0, Tf], [fmax, fmax], 'r-.', label='fmax')  # 最大推力线

axThrust.set_ylabel('Thrust [m/s^2]')  # 设置 y 轴标签
axThrust.legend()  # 显示图例

axOmega.plot(time, ratesMagn, 'k', label='command magnitude')  # 绘制角速度曲线
axOmega.plot([0, Tf], [wmax, wmax], 'r--', label='wmax')  # 最大角速度线
axOmega.fill_between([0, Tf], [wmax, wmax], 1000, facecolor=infeasibleAreaColour, color=infeasibleAreaColour)  # 填充不可行区域
axOmega.set_xlabel('Time [s]')  # 设置 x 轴标签
axOmega.set_ylabel('Body rates [rad/s]')  # 设置 y 轴标签
axOmega.legend()  # 显示图例

axThrust.set_title('Inputs')  # 设置标题

# 设置推力和角速度图的 y 轴范围
axThrust.set_ylim([min(fmin - 1, min(thrust)), max(fmax + 1, max(thrust))])
axOmega.set_ylim([0, max(wmax + 1, max(ratesMagn))])

plt.show()  # 显示图形

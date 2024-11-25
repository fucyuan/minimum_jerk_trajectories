"""
SYNOPSIS

    快速生成四轴飞行器的轨迹

DESCRIPTION
    
    实现了论文《Rapid quadrocopter trajectory generation》中描述的算法。
    从任意初始状态生成到达目标状态的轨迹，目标状态可由位置、速度和/或加速度的任意组合描述。
    
    请参考论文以获取更多信息。

EXAMPLES

    请查看附带的 `demo.py` 脚本，了解如何使用轨迹生成器。
    `demo.py` 将根据给定的约束生成轨迹，并返回轨迹是否通过可行性测试。随后会生成一些图形来可视化结果轨迹。
    
AUTHOR
    
    Mark W. Mueller <mwm@mwm.im>

LICENSE

    版权所有 2014, Mark W. Mueller <mwm@mwm.im>

    该代码是免费软件：你可以在 GNU 通用公共许可证的条款下重新分发和/或修改此代码，
    该许可证由自由软件基金会发布，许可版本为第 3 版，或（按你的选择）任何更新版本。

    该代码的分发目的是希望它能发挥作用，但没有任何保证；
    甚至没有对特定用途适用性和商品性的暗示性保证。
    详细信息请参阅 GNU 通用公共许可证。

    你应该已随附此代码收到 GNU 通用公共许可证。如果没有，请参阅 <http://www.gnu.org/licenses/>。
    
VERSION 

    0.1

"""

import numpy as np


class SingleAxisTrajectory:
    """单轴轨迹类。
    
    用于沿一个轴构建轨迹，通过规划跃度（jerk）来实现位置、速度和/或加速度的目标条件。
    初始化时需要指定初始位置、速度和加速度。

    此轨迹在跃度平方积分方面是最优的。

    不建议单独使用该类，建议通过 "RapidTrajectory" 类使用，它将三个单轴轨迹封装在一起，并允许测试输入/状态的可行性。
    """

    def __init__(self, pos0, vel0, acc0):
        """用初始状态初始化轨迹。"""
        self._p0 = pos0  # 初始位置
        self._v0 = vel0  # 初始速度
        self._a0 = acc0  # 初始加速度
        self._pf = 0  # 目标位置
        self._vf = 0  # 目标速度
        self._af = 0  # 目标加速度
        self.reset()  # 重置轨迹

    def set_goal_position(self, posf):
        """定义轨迹的目标位置。"""
        self._posGoalDefined = True  # 标记目标位置已定义
        self._pf = posf  # 设置目标位置

    def set_goal_velocity(self, velf):
        """定义轨迹的目标速度。"""
        self._velGoalDefined = True  # 标记目标速度已定义
        self._vf = velf  # 设置目标速度

    def set_goal_acceleration(self, accf):
        """定义轨迹的目标加速度。"""
        self._accGoalDefined = True  # 标记目标加速度已定义
        self._af = accf  # 设置目标加速度

    def generate(self, Tf):
        """生成持续时间为 Tf 的轨迹。

        根据之前定义的目标终态（如位置、速度和/或加速度）生成轨迹。
        """
        # 计算初始状态和目标状态的差值
        delta_a = self._af - self._a0  # 加速度差值
        delta_v = self._vf - self._v0 - self._a0 * Tf  # 速度差值
        delta_p = self._pf - self._p0 - self._v0 * Tf - 0.5 * self._a0 * Tf * Tf  # 位置差值

        # 计算终止时间的各次方
        T2 = Tf * Tf
        T3 = T2 * Tf
        T4 = T3 * Tf
        T5 = T4 * Tf

        # 根据约束条件求解轨迹参数
        if self._posGoalDefined and self._velGoalDefined and self._accGoalDefined:
            self._a = (60 * T2 * delta_a - 360 * Tf * delta_v + 720 * delta_p) / T5
            self._b = (-24 * T3 * delta_a + 168 * T2 * delta_v - 360 * Tf * delta_p) / T5
            self._g = (3 * T4 * delta_a - 24 * T3 * delta_v + 60 * T2 * delta_p) / T5
        elif self._posGoalDefined and self._velGoalDefined:
            self._a = (-120 * Tf * delta_v + 320 * delta_p) / T5
            self._b = (72 * T2 * delta_v - 200 * Tf * delta_p) / T5
            self._g = (-12 * T3 * delta_v + 40 * T2 * delta_p) / T5
        elif self._posGoalDefined and self._accGoalDefined:
            self._a = (-15 * T2 * delta_a + 90 * delta_p) / (2 * T5)
            self._b = (15 * T3 * delta_a - 90 * Tf * delta_p) / (2 * T5)
            self._g = (-3 * T4 * delta_a + 30 * T2 * delta_p) / (2 * T5)
        elif self._velGoalDefined and self._accGoalDefined:
            self._a = 0
            self._b = (6 * Tf * delta_a - 12 * delta_v) / T3
            self._g = (-2 * T2 * delta_a + 6 * Tf * delta_v) / T3
        elif self._posGoalDefined:
            self._a = 20 * delta_p / T5
            self._b = -20 * delta_p / T4
            self._g = 10 * delta_p / T3
        elif self._velGoalDefined:
            self._a = 0
            self._b = -3 * delta_v / T3
            self._g = 3 * delta_v / T2
        elif self._accGoalDefined:
            self._a = 0
            self._b = 0
            self._g = delta_a / Tf
        else:
            # 如果没有任何约束，不需要处理
            self._a = self._b = self._g = 0

        # 计算轨迹成本
        self._cost = (self._g**2) + self._b * self._g * Tf + (self._b**2) * T2 / 3.0 + self._a * self._g * T2 / 3.0 + self._a * self._b * T3 / 4.0 + (self._a**2) * T4 / 20.0
                
    def reset(self):
        """重置轨迹参数。"""
        self._cost = float("inf")  # 轨迹成本设为无穷大
        self._accGoalDefined = self._velGoalDefined = self._posGoalDefined = False  # 重置目标状态定义
        self._accPeakTimes = [None, None]  # 重置加速度峰值时间
        pass
    
    def get_jerk(self, t):
        """返回时间 t 的跃度（jerk）。"""
        return self._g + self._b * t + (1.0 / 2.0) * self._a * t * t
    
    def get_acceleration(self, t):
        """返回时间 t 的加速度。"""
        return self._a0 + self._g * t + (1.0 / 2.0) * self._b * t * t + (1.0 / 6.0) * self._a * t * t * t

    def get_velocity(self, t):
        """返回时间 t 的速度。"""
        return self._v0 + self._a0 * t + (1.0 / 2.0) * self._g * t * t + (1.0 / 6.0) * self._b * t * t * t + (1.0 / 24.0) * self._a * t * t * t * t

    def get_position(self, t):
        """返回时间 t 的位置。"""
        return self._p0 + self._v0 * t + (1.0 / 2.0) * self._a0 * t * t + (1.0 / 6.0) * self._g * t * t * t + (1.0 / 24.0) * self._b * t * t * t * t + (1.0 / 120.0) * self._a * t * t * t * t * t
    def get_min_max_acc(self, t1, t2):
        """Return the extrema of the acceleration trajectory between t1 and t2."""
        if self._accPeakTimes[0] is None:
            #uninitialised: calculate the roots of the polynomial
            if self._a:
                #solve a quadratic
                det = self._b*self._b - 2*self._g*self._a
                if det<0:
                    #no real roots
                    self._accPeakTimes[0] = 0
                    self._accPeakTimes[1] = 0
                else:
                    self._accPeakTimes[0] = (-self._b + np.sqrt(det))/self._a
                    self._accPeakTimes[1] = (-self._b - np.sqrt(det))/self._a
            else:
                #_g + _b*t == 0:
                if self._b:
                    self._accPeakTimes[0] = -self._g/self._b
                    self._accPeakTimes[1] = 0
                else:
                    self._accPeakTimes[0] = 0
                    self._accPeakTimes[1] = 0

        #Evaluate the acceleration at the boundaries of the period:
        aMinOut = min(self.get_acceleration(t1), self.get_acceleration(t2))
        aMaxOut = max(self.get_acceleration(t1), self.get_acceleration(t2))

        #Evaluate at the maximum/minimum times:
        for i in [0,1]:
            if self._accPeakTimes[i] <= t1: continue
            if self._accPeakTimes[i] >= t2: continue
            
            aMinOut = min(aMinOut, self.get_acceleration(self._accPeakTimes[i]))
            aMaxOut = max(aMaxOut, self.get_acceleration(self._accPeakTimes[i]))
        return (aMinOut, aMaxOut)
 
    def get_max_jerk_squared(self,t1, t2):
        """Return the extrema of the jerk squared trajectory between t1 and t2."""
        jMaxSqr = max(self.get_jerk(t1)**2,self.get_jerk(t2)**2)
        
        if self._a:
            tMax = -self._b/self._a
            if(tMax>t1 and tMax<t2):
                jMaxSqr = max(pow(self.get_jerk(tMax),2),jMaxSqr)

        return jMaxSqr


    def get_param_alpha(self):
        """Return the parameter alpha which defines the trajectory."""
        return self._a

    def get_param_beta (self):
        """Return the parameter beta which defines the trajectory."""
        return self._b

    def get_param_gamma(self):
        """Return the parameter gamma which defines the trajectory."""
        return self._g

    def get_initial_acceleration(self):
        """Return the start acceleration of the trajectory."""
        return self._a0

    def get_initial_velocity(self):
        """Return the start velocity of the trajectory."""
        return self._v0

    def get_initial_position(self):
        """Return the start position of the trajectory."""
        return self._p0

    def get_cost(self):
        """Return the total cost of the trajectory."""
        return self._cost

    def get_cost(self):
        """返回轨迹总成本。"""
        return self._cost



class InputFeasibilityResult:
    """输入可行性测试的结果枚举类。

    如果测试结果不是 ``Feasible``，则返回第一个失败的段的结果。不同的结果包括：
        0: Feasible -- 轨迹在输入方面是可行的
        1: Indeterminable -- 某一段的可行性无法确定
        2: InfeasibleThrustHigh -- 某一段因超过最大推力限制而失败
        3: InfeasibleThrustLow -- 某一段因低于最小推力限制而失败
    """
    Feasible, Indeterminable, InfeasibleThrustHigh, InfeasibleThrustLow = range(4)
    
    @classmethod
    def to_string(cls, ifr):
        """返回结果的名称。"""
        if   ifr == InputFeasibilityResult.Feasible:
            return "Feasible"  # 可行
        elif ifr == InputFeasibilityResult.Indeterminable:
            return "Indeterminable"  # 无法确定
        elif ifr == InputFeasibilityResult.InfeasibleThrustHigh:
            return "InfeasibleThrustHigh"  # 超过最大推力限制
        elif ifr == InputFeasibilityResult.InfeasibleThrustLow:
            return "InfeasibleThrustLow"  # 低于最小推力限制
        return "Unknown"  # 未知结果


class StateFeasibilityResult:
    """状态可行性测试的结果枚举类。

    结果要么是可行 (0)，要么是不可行 (1)。
    """
    Feasible, Infeasible = range(2)
    
    @classmethod
    def to_string(cls, sfr):
        """返回结果的名称。"""
        if   sfr == StateFeasibilityResult.Feasible:
            return "Feasible"  # 可行
        elif sfr == StateFeasibilityResult.Infeasible:
            return "Infeasible"  # 不可行
        return "Unknown"  # 未知结果


class RapidTrajectory:
    """快速生成四轴飞行器的轨迹。

    生成四轴飞行器状态拦截的轨迹。轨迹起点由飞行器的位置、速度和加速度定义。
    加速度可以直接通过四轴飞行器的姿态和推力值计算得到。
    轨迹的持续时间固定，由用户提供。

    轨迹的目标状态可以包含飞行器的位置、速度和加速度的任意组合。
    加速度可以用来描述轨迹结束时推力的方向。

    生成的轨迹在跃度平方积分方面是最优的，但不考虑任何约束。
    轨迹可以通过递归算法测试其输入约束（推力/机体速率）是否满足条件。
    还可以高效测试轨迹中状态线性组合是否满足一定的边界条件。

    更多信息请参考论文《A computationally efficient motion primitive for quadrocopter trajectory generation》，
    可访问以下链接：http://www.mwm.im/research/publications/

    注意：论文中的轴是从1开始的索引，而这里是从0开始的索引。
    """

    def __init__(self, pos0, vel0, acc0, gravity):
        """初始化轨迹。

        初始化轨迹时需要提供四轴飞行器的初始状态以及重力方向。

        重力方向在可行性测试中是必要的。

        参数：
          pos0 (array(3)): 初始位置
          vel0 (array(3)): 初始速度
          acc0 (array(3)): 初始加速度
          gravity (array(3)): 重力加速度方向（例如，对于东-北-上坐标系为 [0,0,-9.81]）。
        """
        self._axis = [SingleAxisTrajectory(pos0[i], vel0[i], acc0[i]) for i in range(3)]  # 每个轴初始化单轴轨迹
        self._grav = gravity  # 重力方向
        self._tf = None  # 轨迹持续时间
        self.reset()

    def set_goal_position(self, pos):
        """定义目标结束位置。

        为所有三个轴定义结束位置。如果某个分量需要保持自由，
        可以在参数中将其设为 ``None``，或者使用 `set_goal_position_in_axis` 方法。

        """
        for i in range(3):
            if pos[i] is None:
                continue
            self.set_goal_position_in_axis(i, pos[i])

    def set_goal_velocity(self, vel):
        """定义目标结束速度。

        为所有三个轴定义结束速度。如果某个分量需要保持自由，
        可以在参数中将其设为 ``None``，或者使用 `set_goal_velocity_in_axis` 方法。

        """
        for i in range(3):
            if vel[i] is None:
                continue
            self.set_goal_velocity_in_axis(i, vel[i])

    def set_goal_acceleration(self, acc):
        """定义目标结束加速度。

        为所有三个轴定义结束加速度。如果某个分量需要保持自由，
        可以在参数中将其设为 ``None``，或者使用 `set_goal_acceleration_in_axis` 方法。

        """
        for i in range(3):
            if acc[i] is None:
                continue
            self.set_goal_acceleration_in_axis(i, acc[i])

    def set_goal_position_in_axis(self, axNum, pos):
        """为指定轴 `axNum` 定义目标结束位置。"""
        self._axis[axNum].set_goal_position(pos)

    def set_goal_velocity_in_axis(self, axNum, vel):
        """为指定轴 `axNum` 定义目标结束速度。"""
        self._axis[axNum].set_goal_velocity(vel)

    def set_goal_acceleration_in_axis(self, axNum, acc):
        """为指定轴 `axNum` 定义目标结束加速度。"""
        self._axis[axNum].set_goal_acceleration(acc)

    def reset(self):
        """重置轨迹生成器。

        删除所有目标状态，并重置成本。如果希望从一个初始状态生成多个轨迹，可以使用此方法。

        """
        for i in range(3):
            self._axis[i].reset()

    def generate(self, timeToGo):
        """计算持续时间为 `timeToGo` 的轨迹。

        根据当前定义的问题数据计算轨迹。如果某些目标状态（如目标位置）未定义，
        则假定其自由。

        参数：
          timeToGo (float): 轨迹持续时间。

        """
        self._tf = timeToGo
        for i in range(3):
            self._axis[i].generate(self._tf)

    def check_input_feasibility(self, fminAllowed, fmaxAllowed, wmaxAllowed, minTimeSection):
        """运行递归输入可行性测试。

        尝试证明或否定轨迹在输入约束方面的可行性。

        返回的结果为三种情况之一：
        1. 轨迹完全可行
        2. 轨迹输入不可行
        3. 无法确定输入可行性

        如果无法确定可行性，应将其视为不可行。

        参数：
          fminAllowed (float): 允许的最小推力 [m/s²]。
          fmaxAllowed (float): 允许的最大推力 [m/s²]。
          wmaxAllowed (float): 允许的最大机体角速度 [rad/s]。
          minTimeSection (float): 递归测试的最小时间间隔 [s]。

        返回：
          一个枚举值，类型为 InputFeasibilityResult。
        """
        return self._check_input_feasibility_section(fminAllowed, fmaxAllowed, wmaxAllowed, minTimeSection, 0, self._tf)

    # 以下方法为递归可行性检查、位置检查及各种轨迹相关计算，
    # 对应的中文注释请继续说明具体算法逻辑。
    def _check_input_feasibility_section(self, fminAllowed, fmaxAllowed, 
                             wmaxAllowed, minTimeSection, t1, t2):
        """递归方法，用于检查输入可行性。

        由 `check_input_feasibility` 调用。

        参数：
          fminAllowed (float): 允许的最小推力 [m/s²]。
          fmaxAllowed (float): 允许的最大推力 [m/s²]。
          wmaxAllowed (float): 允许的最大机体角速度 [rad/s]。
          minTimeSection (float): 最小时间间隔 [s]。
          t1 (float): 当前段的起始时间。
          t2 (float): 当前段的结束时间。

        返回：
          一个枚举值，类型为 InputFeasibilityResult。
        """
        # 如果时间段小于最小时间间隔，无法确定可行性
        if (t2 - t1) < minTimeSection:
            return InputFeasibilityResult.Indeterminable

        # 检查在起点和终点的推力是否超出允许范围
        if max(self.get_thrust(t1), self.get_thrust(t2)) > fmaxAllowed:
            return InputFeasibilityResult.InfeasibleThrustHigh
        if min(self.get_thrust(t1), self.get_thrust(t2)) < fminAllowed:
            return InputFeasibilityResult.InfeasibleThrustLow

        fminSqr = 0  # 最小推力平方
        fmaxSqr = 0  # 最大推力平方
        jmaxSqr = 0  # 最大跃度平方

        # 检查轨迹每个轴的加速度范围
        for i in range(3):
            amin, amax = self._axis[i].get_min_max_acc(t1, t2)

            # 与零推力点的距离
            v1 = amin - self._grav[i]  # 左侧值
            v2 = amax - self._grav[i]  # 右侧值

            # 如果超出最大推力范围，标记为不可行
            if (max(v1**2, v2**2) > fmaxAllowed**2):
                return InputFeasibilityResult.InfeasibleThrustHigh

            # 如果加速度过零点，意味着推力包含了零值
            if v1 * v2 < 0:
                fminSqr += 0
            else:
                fminSqr += min(abs(v1), abs(v2))**2

            fmaxSqr += max(abs(v1), abs(v2))**2
            jmaxSqr += self._axis[i].get_max_jerk_squared(t1, t2)

        fmin = np.sqrt(fminSqr)  # 推力最小值
        fmax = np.sqrt(fmaxSqr)  # 推力最大值

        if fminSqr > 1e-6:
            wBound = np.sqrt(jmaxSqr / fminSqr)  # 限制角速度的上界
        else:
            wBound = float("inf")  # 防止除零

        # 如果推力超出范围，标记为不可行
        if fmax < fminAllowed:
            return InputFeasibilityResult.InfeasibleThrustLow
        if fmin > fmaxAllowed:
            return InputFeasibilityResult.InfeasibleThrustHigh

        # 如果可行性无法确定，细分时间段递归检查
        if (fmin < fminAllowed) or (fmax > fmaxAllowed) or (wBound > wmaxAllowed):
            tHalf = (t1 + t2) / 2.0
            r1 = self._check_input_feasibility_section(fminAllowed, fmaxAllowed, wmaxAllowed, minTimeSection, t1, tHalf)
            
            if r1 == InputFeasibilityResult.Feasible:
                return self._check_input_feasibility_section(fminAllowed, fmaxAllowed, wmaxAllowed, minTimeSection, tHalf, t2)
            else:
                return r1

        # 如果所有条件满足，标记为可行
        return InputFeasibilityResult.Feasible

    def check_position_feasibility(self, boundaryPoint, boundaryNormal):
        """测试轨迹是否符合与平面相关的可行性。

        测试轨迹是否保持在给定平面某一侧。平面通过一个平面点和法向量定义。

        返回值为 StateFeasibilityResult 的一个实例，可能是 Feasible 或 Infeasible。

        参数：
          boundaryPoint (array(3)): 平面上的一个点。
          boundaryNormal (array(3)): 平面法向量，平面法向量方向为允许的轨迹区域。

        返回：
          一个枚举值，类型为 StateFeasibilityResult。
        """
        boundaryNormal = np.array(boundaryNormal)
        boundaryPoint  = np.array(boundaryPoint)
        
        # 确保法向量是单位向量
        boundaryNormal = boundaryNormal / np.linalg.norm(boundaryNormal)

        # 构建描述轨迹在法向量方向上的速度多项式
        coeffs = np.zeros(5)
        for i in range(3):
            coeffs[0] += boundaryNormal[i] * self._axis[i].get_param_alpha() / 24.0  # t**4
            coeffs[1] += boundaryNormal[i] * self._axis[i].get_param_beta() / 6.0   # t**3
            coeffs[2] += boundaryNormal[i] * self._axis[i].get_param_gamma() / 2.0  # t**2
            coeffs[3] += boundaryNormal[i] * self._axis[i].get_initial_acceleration() / 6.0  # t
            coeffs[4] += boundaryNormal[i] * self._axis[i].get_initial_velocity()  # 常数项

        # 计算多项式的根
        tRoots = np.roots(coeffs)
        
        # 检查这些时间点以及轨迹的起点和终点
        for t in np.append(tRoots, [0, self._tf]):
            distToPoint = np.dot(self.get_position(t) - boundaryPoint, boundaryNormal)
            if distToPoint <= 0:
                return StateFeasibilityResult.Infeasible
        
        # 如果所有点都满足条件，标记为可行
        return StateFeasibilityResult.Feasible

    def get_jerk(self, t):
        """返回时间 `t` 的3D跃度值。"""
        return np.array([self._axis[i].get_jerk(t) for i in range(3)])

    def get_acceleration(self, t):
        """返回时间 `t` 的3D加速度值。"""
        return np.array([self._axis[i].get_acceleration(t) for i in range(3)])

    def get_velocity(self, t):
        """返回时间 `t` 的3D速度值。"""
        return np.array([self._axis[i].get_velocity(t) for i in range(3)])

    def get_position(self, t):
        """返回时间 `t` 的3D位置值。"""
        return np.array([self._axis[i].get_position(t) for i in range(3)])

    def get_normal_vector(self, t):
        """返回时间 `t` 飞行器的法向量。

        法向量表示推力的方向。可以通过角速度计算出飞行器姿态需要的旋转方向。
        注意结果以规划框架表达，如果需要转换为机体框架，需要进一步旋转。

        参数：
          t (float): 时间参数。

        返回：
          np.array()，包含单位向量。
        """
        v = (self.get_acceleration(t) - self._grav)
        return v / np.linalg.norm(v)

    def get_thrust(self, t):
        """返回时间 `t` 的推力输入。

        返回轨迹上时间 `t` 所需的推力，单位为加速度。

        参数：
          t (float): 时间参数。

        返回：
          推力大小。
        """
        return np.linalg.norm(self.get_acceleration(t) - self._grav)

    def get_body_rates(self, t, dt=1e-3):
        """返回时间 `t` 的机体角速度输入（惯性框架）。

        参数：
          t (float): 时间参数。
          dt (float, optional): 离散化时间间隔，默认为1毫秒。

        返回：
          np.array()，包含机体角速度。
        """
        n0 = self.get_normal_vector(t)
        n1 = self.get_normal_vector(t + dt)
        crossProd = np.cross(n0, n1)  # 角速度方向（惯性轴）

        if np.linalg.norm(crossProd) > 1e-6:
            return np.arccos(np.dot(n0, n1)) / dt * (crossProd / np.linalg.norm(crossProd))
        else:
            return np.array([0, 0, 0])

    def get_cost(self):
        """返回轨迹的总成本。

        总成本越高，表明轨迹的输入（推力和机体速率）越激进。
        可用于比较不同轨迹的代价。
        """
        return self._axis[0].get_cost() + self._axis[1].get_cost() + self._axis[2].get_cost()

    def get_param_alpha(self, axNum):
        """返回轨迹中指定轴的 alpha 参数。"""
        return self._axis[axNum].get_param_alpha()

    def get_param_beta(self, axNum):
        """返回轨迹中指定轴的 beta 参数。"""
        return self._axis[axNum].get_param_beta()

    def get_param_gamma(self, axNum):
        """返回轨迹中指定轴的 gamma 参数。"""
        return self._axis[axNum].get_param_gamma()

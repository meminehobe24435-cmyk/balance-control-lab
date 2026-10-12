"""仿真层: RK4 定步长积分器、电机模型、整机闭环仿真器。

这一层把前面三块(dynamics / driver / control)串起来: 控制器按固定周期读传感
器、算控制量、写执行器, 积分器按更细的步长推进物理。所有指标都从这里的时间
序列里算出来。

**纯仿真**: 没有任何真实硬件、没有实时内核、没有硬件在环(HIL)。积分器是离线
定步长 RK4, 时间可以任意缩放 —— 这一点必须记住, 因为"控制器在仿真里 1 kHz
跑得动"完全不等于"在 MCU 上跑得动"。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .driver import ImuDevice, ImuConfig, PwmChannel, QuadEncoder, SimRegisterBus
from .dynamics import CartPoleParams, cartpole_deriv


# ---------------------------------------------------------------------------
# 积分器
# ---------------------------------------------------------------------------


def rk4_step(f, t, x, h):
    """经典四阶 Runge-Kutta 单步。

    ``f(t, x) -> dxdt``。四阶意味着全局误差 ``O(h^4)``: 步长减半误差降 16 倍。
    测试里直接断言这个比值(见 ``tests/test_sim.py::test_rk4_convergence_order``),
    因为它是"我写的积分器确实是 RK4"最直接的证据 —— 如果系数抄错一个 1/6,
    收敛阶会掉到 2 或 3, 比值立刻不是 16。

    为什么不用变步长 RK45: 控制回路里"物理步长 == 控制周期/N" 这个关系是刻意
    的, 变步长会让"控制器每 10 步执行一次"变得不确定。定步长的代价是在刚性
    (stiff)系统上需要很小的 h, 但本项目所有系统的时间常数都在 1e-2 s 以上。
    """
    k1 = f(t, x)
    x2 = [x[i] + 0.5 * h * k1[i] for i in range(len(x))]
    k2 = f(t + 0.5 * h, x2)
    x3 = [x[i] + 0.5 * h * k2[i] for i in range(len(x))]
    k3 = f(t + 0.5 * h, x3)
    x4 = [x[i] + h * k3[i] for i in range(len(x))]
    k4 = f(t + h, x4)
    return [x[i] + (h / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i])
            for i in range(len(x))]


def integrate_rk4(f, x0, t_end, dt, t0=0.0):
    """定步长积分到 ``t_end``, 返回 ``(t_list, x_list)``。

    步数用 ``round`` 而不是 ``int``: ``int`` 会静默丢尾, 让"积分 10 秒"变成
    9.999 秒, 这在能量漂移这种需要精确时间窗的测试里会造成假阴性。
    """
    if dt <= 0:
        raise ValueError("dt 必须为正: %r" % (dt,))
    n = int(round((t_end - t0) / dt))
    x = list(x0)
    t = t0
    ts = [t]
    xs = [list(x)]
    for _ in range(n):
        x = rk4_step(f, t, x, dt)
        t += dt
        ts.append(t)
        xs.append(list(x))
    return ts, xs


# ---------------------------------------------------------------------------
# 电机/执行器模型
# ---------------------------------------------------------------------------


@dataclass
class MotorParams:
    """执行器参数。

    ``time_constant`` 是电流环(或力矩环)的等效一阶时间常数, 典型 10~50 ms 的
    减速电机取 20 ms。``dead_zone`` 是归一化指令死区(0.03 = 3%), 来自驱动器
    最小导通时间与静摩擦。``encoder_ppr`` 是输出轴每转脉冲数。

    默认值标定方式(重要, 否则仿真的量纲会悄悄错掉): 目标是小车最大推力
    ``force_limit = 20 N``, 经 2 cm 半径的齿轮齿条, 需要电机峰值力矩
    ``0.4 N*m``。所以 ``max_torque = 0.4``、``torque_constant = 0.4`` 与
    ``Simulator.pinion_radius = 0.02`` 是**配套**的。上一版这里用了
    ``max_torque = 20``, 结果执行器实际增益是控制器的 50 倍, 系统全程饱和 ——
    这种量纲错误不会报错, 只会让"性能很差"看起来像"控制器不行"。
    """

    torque_constant: float = 0.4
    max_torque: float = 0.4
    time_constant: float = 0.02
    dead_zone: float = 0.03
    encoder_ppr: int = 4096


class MotorModel:
    """带死区、饱和、一阶惯性与编码器量化的执行器模型。

    四个非理想因素都会**实质影响**平衡控制, 所以必须建模:

    * **死区**: 小指令完全不产生力矩。平衡车静止时正好工作在小指令区, 死区会
      导致稳态极限环(左右微摆)。死区补偿(把指令重新映射到 [dead,1])能缓解,
      但代价是零指令附近增益变大, 更容易振荡。
    * **饱和**: 恢复大扰动时需要满力矩, 一旦饱和, 控制器实际变成了开环 —— 这
      是"为什么大扰动下 PID 会失败"的物理原因。
    * **一阶惯性**: 力矩不是瞬时建立的。20 ms 时间常数相对 1 kHz 控制周期是
      20 拍, 相当于给环路加了相位滞后, 会直接吃掉相位裕度。
    * **编码器量化**: 位置只能读到 2*pi/ppr 的整数倍, 差分测速的噪声被 1/dt
      放大。这是"仿真很平顺、实机很吵"最常见的来源。

    边界: 没有建模齿槽转矩、齿隙、绕组的电流环动态、温度导致的力矩常数漂移。
    真机上这些项会让低速性能进一步变差。
    """

    def __init__(self, params=None):
        self.params = params if params is not None else MotorParams()
        self.torque = 0.0
        self.command = 0.0
        self.saturated = False
        self.quantized_angle = 0.0
        self._last_quantized_angle = 0.0

    def reset(self):
        """复位内部状态。"""
        self.torque = 0.0
        self.command = 0.0
        self.saturated = False
        self.quantized_angle = 0.0
        self._last_quantized_angle = 0.0

    def apply_dead_zone(self, cmd):
        """死区 + 补偿: |cmd| <= dz 输出 0, 否则把 [dz,1] 线性映射回 [0,1]。"""
        dz = self.params.dead_zone
        if dz <= 0.0:
            return cmd
        if dz >= 1.0:
            return 0.0
        mag = abs(cmd)
        if mag <= dz:
            return 0.0
        return math.copysign((mag - dz) / (1.0 - dz), cmd)

    def step(self, cmd, dt):
        """推进一个仿真步, 返回本步实际输出的力矩 [N*m]。

        指令 ``cmd`` 归一化到 [-1, 1]。一阶惯性用**精确离散解**
        ``tau += (target - tau) * (1 - exp(-dt/tc))`` 而不是欧拉法
        ``tau += (target-tau)*dt/tc``: 后者在 ``dt > tc`` 时会发散(增益 > 1),
        而精确解永远稳定 —— 这样时间常数改大也不会悄悄把仿真搞爆。
        """
        if dt <= 0:
            raise ValueError("dt 必须为正: %r" % (dt,))
        self.command = cmd
        comp = self.apply_dead_zone(cmd)
        target = comp * self.params.torque_constant
        if target > self.params.max_torque:
            target = self.params.max_torque
            self.saturated = True
        elif target < -self.params.max_torque:
            target = -self.params.max_torque
            self.saturated = True
        else:
            self.saturated = False
        tc = self.params.time_constant
        if tc <= 0.0:
            self.torque = target
        else:
            alpha = 1.0 - math.exp(-dt / tc)
            self.torque += (target - self.torque) * alpha
        return self.torque

    def quantize_angle(self, angle):
        """把输出轴角度量化到编码器分辨率。"""
        step = 2.0 * math.pi / float(self.params.encoder_ppr)
        self._last_quantized_angle = self.quantized_angle
        self.quantized_angle = round(angle / step) * step
        return self.quantized_angle

    def quantized_speed(self, dt):
        """由**量化后**的角度差分求速度(这正是真实系统的做法)。"""
        if dt <= 0:
            raise ValueError("dt 必须为正: %r" % (dt,))
        return (self.quantized_angle - self._last_quantized_angle) / dt


# ---------------------------------------------------------------------------
# 传感器降级模型
# ---------------------------------------------------------------------------


class SensorModel:
    """把真实状态降级成"传感器真的能测到的东西"。

    做这件事的原因是: 平衡控制真正的难点不在控制律, 而在**微分量怎么来**。
    角度可以从 IMU 拿到(还行), 但角速度和线速度在低成本平台上往往是差分量,
    量化噪声被 1/dt 放大, 必须先低通。低通带来的相位滞后正是平衡控制器最难
    调的部分。如果仿真里直接给控制器喂真值, 得到的性能数字是**上界**, 不是
    预期值。

    边界: 这里是"量化 + 差分 + 一阶低通"的最小组合, 没有建模 IMU 零偏随温度
    漂移、编码器丢脉冲、丢帧(实际总线偶尔会读失败)。
    """

    def __init__(self, pos_res=1.0 / 4096.0, angle_res=6.1e-5, lpf_alpha=0.25):
        self.pos_res = pos_res
        self.angle_res = angle_res
        self.lpf_alpha = lpf_alpha
        self.prev = None
        self.v_est = 0.0
        self.w_est = 0.0

    def reset(self):
        """复位差分/滤波状态。"""
        self.prev = None
        self.v_est = 0.0
        self.w_est = 0.0

    def quantize(self, value, res):
        """量化到给定分辨率。"""
        if res <= 0:
            return value
        return round(value / res) * res

    def measure(self, state, dt):
        """返回带量化与滤波延迟的状态估计 ``[x, xd, theta, thetadot]``。"""
        if dt <= 0:
            raise ValueError("dt 必须为正: %r" % (dt,))
        x = self.quantize(state[0], self.pos_res)
        th = self.quantize(state[2], self.angle_res)
        if self.prev is None:
            self.prev = (x, th)
            return [x, 0.0, th, 0.0]
        raw_v = (x - self.prev[0]) / dt
        raw_w = (th - self.prev[1]) / dt
        self.v_est = self.lpf_alpha * raw_v + (1.0 - self.lpf_alpha) * self.v_est
        self.w_est = self.lpf_alpha * raw_w + (1.0 - self.lpf_alpha) * self.w_est
        self.prev = (x, th)
        return [x, self.v_est, th, self.w_est]


# ---------------------------------------------------------------------------
# 仿真结果
# ---------------------------------------------------------------------------


@dataclass
class SimResult:
    """仿真时间序列。

    ``t[k]`` 是第 k 个采样时刻, ``states[k]`` 是该时刻的完整状态,
    ``u[k]`` 是在 ``[t[k], t[k+1])`` 区间内保持的控制量(零阶保持)。
    """

    t: list
    states: list
    u: list
    dt: float
    label: str = "sim"
    force: list = None

    def column(self, i):
        """取第 i 个状态分量。"""
        return [s[i] for s in self.states]

    def applied_force(self):
        """执行器实际输出的力 [N]。理想档下与 ``u`` 完全相同。

        把它和 ``u`` 分开记录, 是为了能直接看出**执行器饱和/死区吃掉了多少**:
        如果 ``max|u - force|`` 很大, 那么"控制器性能差"其实是"执行器不够"。
        """
        return self.u if self.force is None else self.force

    def actuator_error(self):
        """指令力与实际力之间的最大偏差 [N]。"""
        f = self.applied_force()
        return max(abs(a - b) for a, b in zip(self.u, f)) if f else 0.0

    def position(self):
        """位置 x(t)。"""
        return self.column(0)

    def velocity(self):
        """速度 x'(t)。"""
        return self.column(1)

    def theta(self):
        """倾角 theta(t)。"""
        return self.column(2)

    def theta_dot(self):
        """角速度 theta'(t)。"""
        return self.column(3)

    def control_energy(self):
        """控制能量 ``integral u^2 dt``(梯形法)。

        用它而不是 ``integral |u| dt``: 平方项更接近电机铜损(``I^2 R``), 而且
        对大控制量惩罚更重, 能区分"平顺的小控制"和"来回猛打的开关式控制"。
        """
        total = 0.0
        for k in range(len(self.u) - 1):
            total += 0.5 * (self.u[k] ** 2 + self.u[k + 1] ** 2) * self.dt
        return total

    def control_peak(self):
        """控制量峰值 ``max|u|``。"""
        return max(abs(v) for v in self.u) if self.u else 0.0

    def max_abs_theta(self):
        """最大倾角绝对值 [rad]。"""
        return max(abs(v) for v in self.theta()) if self.states else 0.0

    def max_abs_position(self):
        """最大位移绝对值 [m]。"""
        return max(abs(v) for v in self.position()) if self.states else 0.0

    def __len__(self):
        return len(self.t)


# ---------------------------------------------------------------------------
# 整机闭环仿真器
# ---------------------------------------------------------------------------


class Simulator:
    """cart-pole 整机闭环仿真器: dynamics + (可选)执行器 + (可选)驱动栈 + control。

    三种保真度, 用两个开关组合:

    * ``use_motor=False, use_driver=False`` : 理想执行器 + 全状态真值反馈。
      用来验证**控制律本身**(LQR/PID 的增益、收敛性)。
    * ``use_motor=True``                    : 加上死区/饱和/一阶惯性/编码器量化,
      并用 ``SensorModel`` 把反馈降级。用来回答"控制律还够用吗"。
    * ``use_driver=True``                   : 再往上叠寄存器总线 + PWM 死区 +
      IMU 寄存器读写。用来验证**驱动层与控制的接口**没接错。

    为什么不用一个"全开"的默认值: 每个开关都会让指标变差。把保真度做成显式
    开关之后, README 里的性能数字可以标注清楚"这是在哪一档下测的" —— 不标注
    保真度的性能数字没有意义。
    """

    def __init__(self, params=None, dt=1e-3, control_dt=1e-2, force_limit=20.0,
                 use_motor=False, use_driver=False, motor_params=None,
                 sensor_model=None, pinion_radius=0.02, imu_config=None,
                 encoder_ppr=4096, encoder_res=0.001):
        if control_dt < dt:
            raise ValueError("control_dt 必须 >= dt")
        ratio = control_dt / dt
        if abs(ratio - round(ratio)) > 1e-9:
            raise ValueError("control_dt 必须是 dt 的整数倍: %r" % ratio)
        self.params = params if params is not None else CartPoleParams()
        self.dt = dt
        self.control_dt = control_dt
        self.ctrl_every = int(round(ratio))
        self.force_limit = force_limit
        self.use_motor = use_motor
        self.use_driver = use_driver
        self.pinion_radius = pinion_radius
        self.encoder_res = encoder_res

        self.motor = MotorModel(motor_params) if use_motor else None
        self.sensor = (sensor_model if sensor_model is not None
                       else (SensorModel() if use_motor else None))

        self.bus = None
        self.pwm = None
        self.imu = None
        self.encoder = None
        if use_driver:
            self.bus = SimRegisterBus(size=256)
            self.pwm = PwmChannel(self.bus, base=0x40, channel=0)
            self.imu = ImuDevice(self.bus, accel_reg=0x30, gyro_reg=0x36,
                                 config=imu_config or ImuConfig())
            self.encoder = QuadEncoder(self.bus, count_reg=0x20, ppr=encoder_ppr)

    # -- 执行器 ------------------------------------------------------------

    def actuate(self, u):
        """把控制量(期望力 [N])变成实际作用力 [N]。

        经过的环节按开关叠加:
          1. ``use_motor``: 归一化指令 -> 电机力矩 -> 齿轮齿条(半径 pinion) -> 力;
          2. ``use_driver``: PWM 12 位量化 + 死区压降 -> 等效占空比 -> 力。
        """
        if not self.use_motor and not self.use_driver:
            return u
        cmd = max(-1.0, min(1.0, u / self.force_limit))
        force = u
        if self.use_motor:
            torque = self.motor.step(cmd, self.control_dt)
            force = torque / self.pinion_radius
        if self.use_driver:
            self.pwm.set_duty(cmd)
            duty = self.pwm.effective_duty()
            force = duty * self.force_limit
        return force

    # -- 传感器 ------------------------------------------------------------

    def sense(self, state, t):
        """把真实状态降级成控制器能看到的反馈。

        ``use_driver`` 时走完整驱动栈: 位置 -> 编码器寄存器(含 16 位回绕) ->
        读回; 角度 -> IMU 寄存器(量化 + 噪声) -> 读回 + 加速度计反正切。
        """
        if self.use_driver:
            # 位置: 线性编码器, 1 count = encoder_res 米。用 advance 注入真实
            # 位移对应的计数差, 再由寄存器读回(含 16 位回绕语义)。
            target_counts = int(round(state[0] / self.encoder_res))
            self.encoder.advance(target_counts - self.encoder.count_total)
            x_meas = self.encoder.read_register() * self.encoder_res
            v_meas = self.encoder.update_speed(self.control_dt) * self.encoder_res
            # 角度: 由加速度计重力分量反算(量化 + 白噪声), 角速度由陀螺给
            ax = math.sin(state[2])
            az = math.cos(state[2])
            self.imu.write_sample([ax, 0.0, az], [0.0, math.degrees(state[3]), 0.0])
            sample = self.imu.read()
            th_meas = self.imu.tilt_from_accel(sample["accel_g"])
            w_meas = math.radians(sample["gyro_dps"][1])
            return [x_meas, v_meas, th_meas, w_meas]
        if self.sensor is not None:
            return self.sensor.measure(state, self.control_dt)
        return list(state)

    # -- 主循环 ------------------------------------------------------------

    def run(self, x0, controller, t_end, disturbance=None, label="sim"):
        """跑一次闭环仿真。

        ``controller(t, x_meas) -> float`` 返回期望力 [N]。
        ``disturbance(t) -> float`` 是外部扰动力 [N](可选)。
        返回 ``SimResult``。
        """
        if self.motor is not None:
            self.motor.reset()
        if self.sensor is not None:
            self.sensor.reset()
        if self.encoder is not None:
            self.encoder.reset()

        n_phys = int(round(t_end / self.dt))
        x = list(x0)
        t = 0.0
        u = 0.0
        ts = [0.0]
        states = [list(x)]
        us = []
        forces = []
        p = self.params

        for k in range(n_phys):
            if k % self.ctrl_every == 0:
                x_meas = self.sense(x, t)
                u = controller(t, x_meas)
            applied = self.actuate(u)
            total = applied
            if disturbance is not None:
                total += disturbance(t)

            def f(tt, xx, _f=total):
                return cartpole_deriv(xx, _f, p)

            x = rk4_step(f, t, x, self.dt)
            t += self.dt
            ts.append(t)
            states.append(list(x))
            # us[k] 是 ``[t[k], t[k+1])`` 区间内保持的控制量 —— 这个对齐关系是
            # 控制能量积分正确的前提。若把首元素写成占位 0, 梯形积分会在第一段上
            # 少算一半, 得到 3.998 而不是 4.0(实测踩过)。
            us.append(u)
            forces.append(applied)
        us.append(u)                                  # 补末拍, 使 len(u) == len(t)
        forces.append(forces[-1] if forces else 0.0)
        return SimResult(t=ts, states=states, u=us, dt=self.dt, label=label,
                         force=forces)


# ---------------------------------------------------------------------------
# 单关节伺服: 轨迹跟踪的评测平台
# ---------------------------------------------------------------------------


@dataclass
class ServoParams:
    """单关节伺服参数。

    ``inertia`` 是折算到关节侧的等效转动惯量 [kg*m^2], ``damping`` 是粘性阻尼
    [N*m*s/rad], ``torque_limit`` 是电机峰值力矩 [N*m]。默认值约等于一个小型
    协作机械臂的单关节(惯量 0.02、峰值 2 N*m)。
    """

    inertia: float = 0.02
    damping: float = 0.05
    torque_limit: float = 2.0
    encoder_ppr: int = 16384


def encoder_rms_floor(ppr):
    """编码器量化给出的跟踪误差 RMSE 地板 ``(2*pi/ppr)/sqrt(12)``。

    均匀量化的误差标准差是 ``step/sqrt(12)``。它是一条**物理下限**: 无论控制算法
    多好, 只要误差是用同一个编码器测的, RMSE 就不可能显著低于它。把这条数字显式
    算出来, 才能判断"跟踪误差 1e-4 rad"到底是算法好还是已经到底了。
    """
    return (2.0 * math.pi / float(ppr)) / math.sqrt(12.0)


@dataclass
class ServoResult:
    """单关节伺服的时间序列。

    ``q`` / ``qd`` 是**实测**(含编码器量化与差分测速), ``q_ref`` / ``v_ref`` 是
    轨迹给定的参考值。跟踪误差用实测减参考计算 —— 用真值算会高估性能, 那正是
    仿真里最容易自我欺骗的地方。
    """

    params: ServoParams = None
    t: list = None
    q: list = None
    qd: list = None
    u: list = None
    q_ref: list = None
    v_ref: list = None
    dt: float = 1e-4
    label: str = "servo"
    q_true: list = None

    def error(self):
        """实测跟踪误差序列 ``q - q_ref``。"""
        return [a - b for a, b in zip(self.q, self.q_ref)]

    def error_true(self):
        """真实跟踪误差序列 ``q_true - q_ref``。"""
        if self.q_true is None:
            return self.error()
        return [a - b for a, b in zip(self.q_true, self.q_ref)]

    @staticmethod
    def _rms(seq):
        return math.sqrt(sum(v * v for v in seq) / len(seq)) if seq else 0.0

    def tracking_rms(self):
        """**实测**跟踪误差 RMS [rad](驱动器自己能看到的口径)。

        注意它有一个**量化地板**: 编码器分辨率 ``2*pi/ppr`` 均匀量化误差的 RMS
        是 ``step/sqrt(12)``。ppr=16384 时约 1.1e-4 rad —— 也就是说, 无论控制
        算法多好, 这个口径下的 RMS 都不可能显著低于它。
        """
        return self._rms(self.error())

    def tracking_peak(self):
        """**实测**跟踪误差峰值 [rad]。"""
        e = self.error()
        return max(abs(v) for v in e) if e else 0.0

    def tracking_rms_true(self):
        """**真实**跟踪误差 RMS [rad](用外部测量仪器才能拿到的口径)。"""
        return self._rms(self.error_true())

    def tracking_peak_true(self):
        """**真实**跟踪误差峰值 [rad]。"""
        e = self.error_true()
        return max(abs(v) for v in e) if e else 0.0

    def encoding_floor(self):
        """编码器量化给出的 RMSE 地板 ``(2*pi/ppr)/sqrt(12)``。"""
        ppr = self.params.encoder_ppr if self.params else 16384
        return encoder_rms_floor(ppr)

    def control_energy(self):
        """控制能量 ``integral tau^2 dt`` [(N*m)^2*s]。"""
        total = 0.0
        for k in range(len(self.u) - 1):
            total += 0.5 * (self.u[k] ** 2 + self.u[k + 1] ** 2) * self.dt
        return total

    def max_torque(self):
        """峰值力矩 [N*m]。"""
        return max(abs(v) for v in self.u) if self.u else 0.0


class ServoSim:
    """一自由度关节伺服的闭环仿真, 用来评测关节轨迹规划。

    ``J q'' = tau - b q'``, 力矩指令经 ``MotorModel``(死区 + 饱和 + 一阶惯性),
    反馈经编码器量化 + 差分测速 + 一阶低通。

    为什么轨迹跟踪不在 cart-pole 上评: cart-pole 的位置模态带宽极低(必须先倾斜
    才能移动), 用它评轨迹跟踪得到的 RMS 主要反映的是"平衡约束"而不是"轨迹规划
    质量"。单关节伺服才是关节轨迹规划的真实落地场景, 而且模型足够简单, 出现跟踪
    误差时可以逐项归因(前馈不够? 速度环带宽不够? 还是编码器量化到底了?)。

    边界: 没有建模减速器柔性、齿隙、摩擦(库仑/Stribeck)、关节力矩传感器噪声。
    真实关节的跟踪误差里齿隙和摩擦占大头 —— 所以本项目的 RMS 数字是**乐观**的,
    它衡量的是"算法在理想机械上的表现"。
    """

    def __init__(self, params=None, dt=1e-4, control_dt=1e-3, motor_params=None,
                 use_motor=True, feedforward=True, lpf_alpha=0.3):
        if control_dt < dt:
            raise ValueError("control_dt 必须 >= dt")
        ratio = control_dt / dt
        if abs(ratio - round(ratio)) > 1e-9:
            raise ValueError("control_dt 必须是 dt 的整数倍")
        self.params = params if params is not None else ServoParams()
        self.dt = dt
        self.control_dt = control_dt
        self.ctrl_every = int(round(ratio))
        self.use_motor = use_motor
        self.feedforward = feedforward
        self.lpf_alpha = lpf_alpha
        self.motor = MotorModel(motor_params or MotorParams(
            torque_constant=self.params.torque_limit,
            max_torque=self.params.torque_limit,
            dead_zone=0.02,
            encoder_ppr=self.params.encoder_ppr)) if use_motor else None
        self._prev_q = None
        self._v_est = 0.0

    def _sense(self, state, dt):
        """编码器量化 + 差分测速 + 一阶低通。"""
        step = 2.0 * math.pi / float(self.params.encoder_ppr)
        q_meas = round(state[0] / step) * step
        if self._prev_q is None:
            self._prev_q = q_meas
            return q_meas, 0.0
        raw = (q_meas - self._prev_q) / dt
        self._prev_q = q_meas
        self._v_est = self.lpf_alpha * raw + (1.0 - self.lpf_alpha) * self._v_est
        return q_meas, self._v_est

    def run(self, profile, controller, t_end=None):
        """跟踪给定位移轮廓。

        ``controller`` 需要带 ``step(pos_ref, pos_meas, vel_meas, dt)`` 接口的串级
        控制器(``control.CascadePID``)。前馈打开时, 按**计算力矩**形式补上
        ``tau_ff = J*a_ref + b*v_ref`` —— 这是轨迹跟踪里最有效的一招: 纯反馈对付
        斜坡输入必然有正比于速度的稳态滞后误差(``e = v/kp``), 前馈直接把它消掉。
        代价是前馈依赖轨迹的**准确微分**; 本项目轮廓给的是解析加速度, 所以前馈是
        理想的, 实机上必须先把参考加速度滤一遍再用。

        注意**反馈回路里没有前馈**: 控制器只看到量化后的编码器读数, 前馈项是在
        控制器输出之后叠加的。这样"前馈能补多少"和"反馈能做到多少"是两个可以
        分开看的量。

        传感器**只在控制周期采样一次**(``ctrl_every`` 拍), 这与真实驱动一致:
        1 kHz 控制环不会去读 10 kHz 的编码器。
        """
        if t_end is None:
            t_end = profile.duration
        n_phys = int(round(t_end / self.dt))
        state = [profile.p0, 0.0]
        t = 0.0
        u = 0.0
        self._prev_q = None
        self._v_est = 0.0
        if self.motor is not None:
            self.motor.reset()
        controller.reset()

        ts, qs, qts, qds, us, qrefs, vrefs = [], [], [], [], [], [], []
        q_meas, v_meas = self._sense(state, self.control_dt)
        for k in range(n_phys + 1):
            q_ref = profile.position(t)
            v_ref = profile.velocity(t)
            if k % self.ctrl_every == 0:
                q_meas, v_meas = self._sense(state, self.control_dt)
                u = controller.step(q_ref, q_meas, v_meas, self.control_dt,
                                    vel_ref=v_ref if self.feedforward else 0.0)
                if self.feedforward:
                    u += (self.params.inertia * profile.acceleration(t)
                          + self.params.damping * v_ref)
                    u = max(-self.params.torque_limit,
                            min(self.params.torque_limit, u))
            ts.append(t)
            qs.append(q_meas)
            qts.append(state[0])
            qds.append(v_meas)
            us.append(u)
            qrefs.append(q_ref)
            vrefs.append(v_ref)
            if k < n_phys:
                torque = self._actuate(u)
                state = self._integrate(state, torque, self.dt)
                t += self.dt
        return ServoResult(params=self.params, t=ts, q=qs, qd=qds, u=us,
                           q_ref=qrefs, v_ref=vrefs, dt=self.dt,
                           label="servo", q_true=qts)

    def _actuate(self, u):
        if self.motor is None:
            return u
        cmd = max(-1.0, min(1.0, u / self.params.torque_limit))
        return self.motor.step(cmd, self.control_dt)

    def _integrate(self, state, torque, h):
        """RK4 单步(伺服动力学很小, 直接内联避免闭包开销)。"""
        p = self.params

        def f(q, qd):
            return (qd, (torque - p.damping * qd) / p.inertia)

        q, qd = state
        k1 = f(q, qd)
        k2 = f(q + 0.5 * h * k1[0], qd + 0.5 * h * k1[1])
        k3 = f(q + 0.5 * h * k2[0], qd + 0.5 * h * k2[1])
        k4 = f(q + h * k3[0], qd + h * k3[1])
        return [q + h / 6.0 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]),
                qd + h / 6.0 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])]

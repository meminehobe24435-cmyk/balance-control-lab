"""状态估计: 互补滤波与卡尔曼滤波估计姿态(含陀螺零偏)。

为什么需要这一层: 陀螺积分短期准、长期漂(零偏积分成角度漂移, 0.05 deg/s 的
零偏一小时能漂 3 度); 加速度计长期准(重力方向是绝对的)、短期烂(车体一加速就
把加速度当成倾角)。两者的**频域特性刚好互补**, 所以估计器的本质就是一个
"低频信加速度计、高频信陀螺"的融合问题。

两个实现, 用途不同:

* ``ComplementaryFilter`` —— 一阶高通(陀螺) + 一阶低通(加速度计)。只有一个参数
  ``tau``, 一行代码, 在 MCU 上几十微秒。**代价**: 它无法估计陀螺零偏, 零偏会
  造成 ``bias * tau`` 量级的稳态角度误差(实测 tau=0.5 s、bias=0.05 rad/s
  时约 0.025 rad = 1.4 度), 而且 tau 越小误差越小、但加速度计的噪声进来越多
  —— 存在一个无法同时满足的最优点。
* ``KalmanFilter2D`` —— 状态 ``[角度, 陀螺零偏]``。多出来的零偏状态让它能**在线
  辨识并扣掉零偏**, 于是稳态误差可以压到远小于互补滤波的水平。代价是需要
  整定 ``Q/R`` 三个参数, 且对模型失配(比如零偏在缓慢漂移)不如互补滤波鲁棒。

``KalmanFilter1D`` 是只有角度状态的版本, 用来对比"多了零偏状态到底值多少"。

**纯仿真**: 数据由 ``simulate_imu`` 合成(白噪声 + 常值零偏), 不含真实 IMU, 也
没有建模零偏不稳定性(1/f 噪声)、温度漂移、轴间不对准。真实 IMU 的主要误差
恰恰是零偏不稳定性, 所以本模块的卡尔曼误差数字**偏乐观**。
"""

from __future__ import annotations

import math
import random


# ---------------------------------------------------------------------------
# 互补滤波
# ---------------------------------------------------------------------------


class ComplementaryFilter(object):
    """一阶互补滤波: ``theta = a*(theta + w*dt) + (1-a)*theta_accel``。

    ``a = tau / (tau + dt)``, ``tau`` 是"信任陀螺的时间尺度"。物理含义: 加速度计
    的权重相当于一个时间常数 ``tau`` 的一阶低通, 陀螺的权重是它的补(高通)。

    为什么用这个形式而不是"固定权重": 固定权重会让滤波特性随采样率变化 ——
    同一份代码从 100 Hz 换到 1 kHz, 等效带宽会变 10 倍。按 ``tau`` 参数化之后
    采样率变了也不用重调。

    边界: **不估计零偏**。零偏 ``b`` 造成的稳态角度误差约 ``b * tau``。想减小它
    只能减小 tau, 但那会把加速度计的噪声按 ``sqrt(1/tau)`` 放大 —— 这就是互补
    滤波的天花板, 也是卡尔曼存在的理由。
    """

    def __init__(self, tau=0.5, angle0=0.0):
        if tau <= 0.0:
            raise ValueError("tau 必须为正: %r" % (tau,))
        self.tau = float(tau)
        self.angle = float(angle0)

    def reset(self, angle0=0.0):
        """重置角度估计。"""
        self.angle = float(angle0)

    def alpha(self, dt):
        """由 tau 和 dt 算互补系数。"""
        return self.tau / (self.tau + dt)

    def step(self, gyro_rate, accel_angle, dt):
        """推进一拍并返回融合后的角度。"""
        if dt <= 0.0:
            raise ValueError("dt 必须为正: %r" % (dt,))
        a = self.alpha(dt)
        self.angle = a * (self.angle + gyro_rate * dt) + (1.0 - a) * accel_angle
        return self.angle


# ---------------------------------------------------------------------------
# 卡尔曼滤波
# ---------------------------------------------------------------------------


class KalmanFilter1D(object):
    """只有角度状态的卡尔曼滤波(用来量化"零偏状态值多少钱")。

    预测: ``theta <- theta + w*dt``, 过程噪声 ``q_angle``。
    更新: 观测 ``z = theta_accel``, 观测噪声 ``r_measure``。

    标量形式下它就是互补滤波的一个特例(增益自适应), 但因为没有零偏状态, 零偏
    造成的误差**不会被消除**, 只会被观测慢慢拉回 —— 表现为一个恒定偏置。
    """

    def __init__(self, q_angle=1e-4, r_measure=1e-2, angle=0.0, p=1.0):
        self.q_angle = float(q_angle)
        self.r_measure = float(r_measure)
        self.angle = float(angle)
        self.p = float(p)
        self.gain = 0.0

    def reset(self, angle=0.0, p=1.0):
        """重置状态与协方差。"""
        self.angle = float(angle)
        self.p = float(p)
        self.gain = 0.0

    def predict(self, gyro_rate, dt):
        """预测步。"""
        if dt <= 0.0:
            raise ValueError("dt 必须为正: %r" % (dt,))
        self.angle += gyro_rate * dt
        self.p += self.q_angle

    def update(self, measured_angle):
        """更新步, 返回本拍的卡尔曼增益。"""
        self.gain = self.p / (self.p + self.r_measure)
        self.angle += self.gain * (measured_angle - self.angle)
        self.p *= (1.0 - self.gain)
        return self.gain

    def step(self, gyro_rate, measured_angle, dt):
        """预测 + 更新, 返回角度估计。"""
        self.predict(gyro_rate, dt)
        self.update(measured_angle)
        return self.angle


class KalmanFilter2D(object):
    """状态 ``[角度, 陀螺零偏]`` 的卡尔曼滤波(姿态估计的标准做法)。

    模型::

        theta_{k+1} = theta_k + (w_k - b_k) * dt        (陀螺测到的是 真值 + 零偏)
        b_{k+1}     = b_k                                (零偏视为随机游走)
        z_k         = theta_k + v_k                     (加速度计给的角度)

    ``F = [[1, -dt], [0, 1]]``, ``B = [dt, 0]'``, ``H = [1, 0]``,
    ``Q = diag(q_angle, q_bias)``, ``R = [r_measure]``。

    关键的整定直觉(这是本类最实用的部分):
      * ``q_bias`` 相对 ``q_angle`` **越小**, 零偏估计越稳但收敛越慢;
        越大则收敛快但对噪声敏感, 甚至把真实角度变化误当成零偏变化。
      * ``r_measure`` 越大表示"越不信加速度计", 稳态对加速度扰动的抑制越好,
        但收敛越慢。

    数值实现说明: 这里手写 2x2 协方差递推(不用通用矩阵库), 因为 2x2 的展开式
    比通用矩阵乘法快一个数量级, 而且是最容易和教科书逐行对照的形式。
    代价: 没有做平方根滤波/UD 分解, 长期运行时协方差可能失去正定性。
    在本项目的量级(几十秒、每秒 200 拍)不会出现, 但不能据此认为它能跑几个月。
    """

    def __init__(self, q_angle=1e-4, q_bias=1e-6, r_measure=1e-2,
                 angle=0.0, bias=0.0, p00=1.0, p11=1e-2):
        self.q_angle = float(q_angle)
        self.q_bias = float(q_bias)
        self.r_measure = float(r_measure)
        self.angle = float(angle)
        self.bias = float(bias)
        self.p00 = float(p00)
        self.p01 = 0.0
        self.p10 = 0.0
        self.p11 = float(p11)
        self.gain_angle = 0.0
        self.gain_bias = 0.0

    def reset(self, angle=0.0, bias=0.0, p00=1.0, p11=1e-2):
        """重置状态与协方差矩阵。"""
        self.angle = float(angle)
        self.bias = float(bias)
        self.p00 = float(p00)
        self.p01 = 0.0
        self.p10 = 0.0
        self.p11 = float(p11)
        self.gain_angle = 0.0
        self.gain_bias = 0.0

    def predict(self, gyro_rate, dt):
        """预测步(含协方差传播 ``P = F P F' + Q``)。"""
        if dt <= 0.0:
            raise ValueError("dt 必须为正: %r" % (dt,))
        self.angle += (gyro_rate - self.bias) * dt
        p00, p01, p10, p11 = self.p00, self.p01, self.p10, self.p11
        self.p00 = p00 - dt * (p10 + p01) + dt * dt * p11 + self.q_angle
        self.p01 = p01 - dt * p11
        self.p10 = p10 - dt * p11
        self.p11 = p11 + self.q_bias

    def update(self, measured_angle):
        """更新步, 返回 ``(增益_角度, 增益_零偏)``。"""
        s = self.p00 + self.r_measure
        k0 = self.p00 / s
        k1 = self.p10 / s
        innovation = measured_angle - self.angle
        self.angle += k0 * innovation
        self.bias += k1 * innovation
        p00, p01, p10, p11 = self.p00, self.p01, self.p10, self.p11
        self.p00 = (1.0 - k0) * p00
        self.p01 = (1.0 - k0) * p01
        self.p10 = p10 - k1 * p00
        self.p11 = p11 - k1 * p01
        # 强制对称: 浮点误差会让 P 轻微不对称, 长期累积会破坏正定性
        avg = 0.5 * (self.p01 + self.p10)
        self.p01 = avg
        self.p10 = avg
        self.gain_angle = k0
        self.gain_bias = k1
        return k0, k1

    def step(self, gyro_rate, measured_angle, dt):
        """预测 + 更新, 返回角度估计。"""
        self.predict(gyro_rate, dt)
        self.update(measured_angle)
        return self.angle


# ---------------------------------------------------------------------------
# 数据合成与评估
# ---------------------------------------------------------------------------


def simulate_imu(angle_fn, duration, dt, gyro_bias=0.05, gyro_noise=0.01,
                 accel_noise=0.05, seed=20240517):
    """合成一段带白噪声与常值零偏的 IMU 数据。

    参数:
      * ``angle_fn(t) -> 真实倾角[rad]``;
      * ``gyro_bias`` 陀螺常值零偏 [rad/s](0.05 rad/s 约合 2.9 deg/s, 是低端
        MEMS 的典型量级);
      * ``gyro_noise`` / ``accel_noise`` 白噪声标准差。

    返回 dict: ``t``, ``true_angle``, ``gyro``(含零偏与噪声的陀螺读数),
    ``accel_angle``(加速度计反算的角度, 含噪声), ``gyro_bias``。

    边界: 噪声是**高斯白噪声**。真实 IMU 是零偏不稳定性(1/f) + 白噪声 + 量化
    噪声的叠加, 而且零偏会随温度漂。所以这里的估计器性能是上界。
    """
    if dt <= 0.0:
        raise ValueError("dt 必须为正: %r" % (dt,))
    rng = random.Random(seed)
    n = int(round(duration / dt)) + 1
    out = {"t": [], "true_angle": [], "gyro": [], "accel_angle": [],
           "gyro_bias": gyro_bias}
    for i in range(n):
        t = i * dt
        true_angle = angle_fn(t)
        true_rate = (angle_fn(t + 1e-6) - angle_fn(t - 1e-6)) / 2e-6
        out["t"].append(t)
        out["true_angle"].append(true_angle)
        out["gyro"].append(true_rate + gyro_bias + rng.gauss(0.0, gyro_noise))
        out["accel_angle"].append(true_angle + rng.gauss(0.0, accel_noise))
    return out


def gyro_integrate(gyro, dt, angle0=0.0):
    """纯陀螺积分(基线)。零偏会在这里被**积分成角度漂移**, 所以它必然越跑越差。"""
    out = []
    angle = angle0
    for i, rate in enumerate(gyro):
        if i > 0:
            angle += rate * dt
        out.append(angle)
    return out


def rmse(estimate, truth):
    """均方根误差。"""
    if len(estimate) != len(truth):
        raise ValueError("长度不一致: %d vs %d" % (len(estimate), len(truth)))
    if not estimate:
        raise ValueError("空序列没有 RMSE")
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(estimate, truth))
                     / len(estimate))


def max_abs_error(estimate, truth):
    """最大绝对误差。"""
    if len(estimate) != len(truth):
        raise ValueError("长度不一致")
    return max(abs(a - b) for a, b in zip(estimate, truth))


def run_estimators(data, tau=0.5, kf1_params=None, kf2_params=None):
    """在同一份 IMU 数据上跑四种估计, 返回误差对比表。

    返回 dict::

        {"dt":..., "pure_gyro": {"rmse","max_err","final_err"},
         "complementary": {...}, "kalman_1d": {...}, "kalman_2d": {...},
         "kalman_2d_bias_estimate": float}

    把"纯积分 / 互补 / 卡尔曼1维 / 卡尔曼2维"放在同一张表里, 是为了让每个组成部分
    的**边际贡献**都可见: 纯积分 -> 互补 证明了"需要加速度计"; 互补 -> 卡尔曼1维
    量化了"自适应增益"的价值; 卡尔曼1维 -> 卡尔曼2维 量化了"零偏状态"的价值。
    """
    t = data["t"]
    dt = t[1] - t[0]
    truth = data["true_angle"]
    gyro = data["gyro"]
    accel_angle = data["accel_angle"]

    kf1 = KalmanFilter1D(**(kf1_params or {}))
    kf2 = KalmanFilter2D(**(kf2_params or {}))
    comp = ComplementaryFilter(tau=tau)

    est_gyro = gyro_integrate(gyro, dt, angle0=truth[0])
    est_comp = []
    est_kf1 = []
    est_kf2 = []
    for i in range(len(t)):
        est_comp.append(comp.step(gyro[i], accel_angle[i], dt) if i > 0
                        else comp.angle)
        est_kf1.append(kf1.step(gyro[i], accel_angle[i], dt) if i > 0
                       else kf1.angle)
        est_kf2.append(kf2.step(gyro[i], accel_angle[i], dt) if i > 0
                       else kf2.angle)

    def summarize(est):
        return {"rmse": rmse(est, truth),
                "max_err": max_abs_error(est, truth),
                "final_err": abs(est[-1] - truth[-1])}

    return {
        "dt": dt,
        "samples": len(t),
        "true_bias": data["gyro_bias"],
        "pure_gyro": summarize(est_gyro),
        "complementary": summarize(est_comp),
        "kalman_1d": summarize(est_kf1),
        "kalman_2d": summarize(est_kf2),
        "kalman_2d_bias_estimate": kf2.bias,
    }

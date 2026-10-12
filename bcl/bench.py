"""评测指标与一键评测套件 ``run_suite()``。

这一层的作用是**把"我觉得调好了"变成数字**。README 里所有性能数字都来自这里,
CLI 的 ``bench`` 子命令打印的也是这里的结果。

七个指标(每个都有明确、可复现的定义 —— 定义不明确, 数字就可以随便解释):

==========================  ========================================================
稳定时间                    |y - target| 最后一次超出给定带之后的时刻
超调量                      越过终值最大的那一截, 按 |终值 - 初值| 归一化
稳态误差                    末段 10% 的均值与目标之差(不是最后一个采样点 —— 单点会被噪声骗)
抗扰恢复时间                扰动后 |y| 回到"扰动峰值 2%"带内并保持的时刻 - 扰动起始时刻
轨迹跟踪 RMS / 峰值误差     实测关节角与轨迹参考值之差的 RMS / 最大绝对值
控制能量                    integral u^2 dt(平方项更接近电机铜损 I^2 R)
最大倾角                    max|theta|, 反映过程中"歪得最狠"的程度
==========================  ========================================================

两条贯穿全篇的原则:

1. **所有数字都来自本机纯仿真**, 没有真实硬件。不同保真度(理想执行器 / 带死区
   与饱和的电机 / 完整驱动栈)下数字差别很大, 所以每个结果都带 ``fidelity``
   标签 —— 不标注保真度的性能数字没有意义。
2. **指标定义写在函数里**, 不写在 README 里。这样口径和数字一起演进, 不会出现
   "README 里写着 1 秒、代码里换了个定义变成 0.3 秒"的情况。
"""

from __future__ import annotations

import math

from .control import (AnglePIDController, CascadePID, LQRController, PID,
                      cartpole_lqr_gain)
from .dynamics import (CartPoleParams, PendulumParams, cartpole_energy,
                       cartpole_linearize, cartpole_omega_sq, energy_drift,
                       estimate_period, pendulum_ode,
                       pendulum_small_angle_period)
from .estimator import run_estimators, simulate_imu
from .sim import ServoParams, ServoSim, Simulator, integrate_rk4
from .trajectory import (QuinticProfile, SCurveProfile, TrapezoidalProfile,
                         sync_trajectories)


# ---------------------------------------------------------------------------
# 指标函数
# ---------------------------------------------------------------------------


def settle_index(t, y, band, target=0.0):
    """返回"最后一次超出带的采样下标", 从未超出则返回 None。

    抽成独立函数是因为它有两个容易写错的边界: (1) 必须取**最后一次**超出而不是
    第一次进入 —— 否则振荡收敛的系统会得到虚低的稳定时间; (2) 从未超出时应返回
    None(表示"一开始就在带内"), 而不是 0 —— 两者含义完全不同。
    """
    last = None
    for k in range(len(y)):
        if abs(y[k] - target) > band:
            last = k
    return last


def settling_time(t, y, band, target=0.0):
    """稳定时间 [s]: 最后一次超出带之后的时刻。从未超出返回 0.0。"""
    last = settle_index(t, y, band, target)
    if last is None:
        return 0.0
    if last + 1 >= len(t):
        return t[-1] - t[0]
    return t[last + 1] - t[0]


def overshoot_percent(y, initial, final):
    """超调量 [%]: 越过终值最多的那一截, 按 |final - initial| 归一化。

    用方向投影 ``direction*(y - final)`` 而不是简单的 ``max(y)``: 这样上升阶跃和
    下降阶跃(比如 cart-pole 从 theta=0.05 回到 0)都能用同一个公式, 不用为符号
    分支写两套代码 —— 两套代码就是两倍出错机会。
    """
    span = abs(final - initial)
    if span <= 0.0:
        return 0.0
    direction = 1.0 if final >= initial else -1.0
    worst = max(direction * (v - final) for v in y)
    return max(0.0, worst) / span * 100.0


def steady_state_error(t, y, target=0.0, tail_fraction=0.1):
    """稳态误差: 末段 ``tail_fraction`` 内 y 的均值与 target 之差。"""
    if not y:
        raise ValueError("空序列")
    n = len(y)
    k = max(1, int(round(n * tail_fraction)))
    tail = y[-k:]
    return sum(tail) / len(tail) - target


def recovery_time(t, y, t_start, band_fraction=0.02):
    """抗扰恢复时间 [s]。

    定义: 以扰动发生后的 ``max|y|`` 为峰值, 带 = ``band_fraction * 峰值``;
    恢复时间 = (最后一次 |y| 超出带的时刻) - (扰动起始时刻)。

    为什么带按"扰动峰值"而不是按绝对值取: 绝对带(比如 0.001 rad)在不同幅值的
    扰动下没有可比性 —— 小扰动会"瞬间恢复", 大扰动永远恢复不了。按峰值归一化后,
    这个指标衡量的是"抑制扰动花了多少个自己的时间尺度", 可以跨扰动幅值比较。
    """
    k0 = 0
    for k in range(len(t)):
        if t[k] >= t_start:
            k0 = k
            break
    window = y[k0:]
    if not window:
        return 0.0
    peak = max(abs(v) for v in window)
    if peak <= 0.0:
        return 0.0
    band = band_fraction * peak
    last = None
    for k in range(k0, len(y)):
        if abs(y[k]) > band:
            last = k
    if last is None:
        return 0.0
    if last + 1 >= len(t):
        return t[-1] - t_start
    return t[last + 1] - t_start


def tracking_errors(ref, actual):
    """返回 ``(rms, peak)`` 跟踪误差。"""
    if len(ref) != len(actual):
        raise ValueError("长度不一致: %d vs %d" % (len(ref), len(actual)))
    diff = [a - b for a, b in zip(actual, ref)]
    rms = math.sqrt(sum(v * v for v in diff) / len(diff)) if diff else 0.0
    return rms, (max(abs(v) for v in diff) if diff else 0.0)


def control_energy(t, u):
    """控制能量 ``integral u^2 dt``(梯形法)。"""
    total = 0.0
    for k in range(len(u) - 1):
        total += 0.5 * (u[k] ** 2 + u[k + 1] ** 2) * (t[k + 1] - t[k])
    return total


# ---------------------------------------------------------------------------
# 场景控制器
# ---------------------------------------------------------------------------


def default_servo_controller(v_max=6.0, torque_limit=2.0):
    """单关节轨迹跟踪用的串级 PID(输出侧限幅到 ``torque_limit``)。

    增益 ``(200, 2.0, 40)`` 是实测选出来的: 再小则内环跟不上参考加速度, 再大则
    力矩指令顶到限幅反而变差(实测 kpt=3.0 时峰值力矩 2.0 = 满饱和, RMS 从
    6.6e-4 恶化到 4.5e-3)。这一类"增益加大反而变差"的非单调行为, 只有跑仿真
    才能看出来。
    """
    outer = PID(kp=200.0, ki=0.0, kd=0.0, out_min=-v_max, out_max=v_max)
    inner = PID(kp=2.0, ki=40.0, kd=0.0,
                out_min=-torque_limit, out_max=torque_limit)
    return CascadePID(outer, inner, v_max=v_max)


def pulse_disturbance(amplitude, t_on, duration):
    """生成一个方波扰动力 ``f(t) -> N``。

    为什么用方波而不是冲激: 冲激在数值仿真里需要一个采样周期才能实现, 幅值依赖
    dt, 不同步长下不可比。方波的能量是确定的(``A^2 * duration``), 换步长结果
    不变 —— 评测指标必须与仿真步长无关, 否则"调小 dt 提高精度"会顺带改变指标。
    """
    def f(t):
        return amplitude if t_on <= t < t_on + duration else 0.0
    return f


# ---------------------------------------------------------------------------
# 各场景
# ---------------------------------------------------------------------------


def _summarize_balance(sim_result, y, target=0.0, band=1e-3):
    """把一个平衡仿真结果压成指标字典。"""
    return {
        "settling_time_s": settling_time(sim_result.t, y, band, target),
        "overshoot_pct": overshoot_percent(y, y[0], target),
        "steady_state_error_rad": steady_state_error(sim_result.t, y, target),
        "control_energy": sim_result.control_energy(),
        "peak_control_N": sim_result.control_peak(),
        "actuator_error_N": sim_result.actuator_error(),
        "max_abs_tilt_rad": sim_result.max_abs_theta(),
        "max_abs_position_m": sim_result.max_abs_position(),
        "final_position_m": sim_result.position()[-1],
        "final_tilt_rad": y[-1],
    }


def scenario_lqr(params=None, use_motor=False, use_driver=False, theta0=0.05,
                 t_end=8.0, control_dt=0.01, force_limit=20.0):
    """LQR 从 ``theta0`` 起稳 cart-pole。"""
    p = params if params is not None else CartPoleParams()
    k, info = cartpole_lqr_gain(p, control_dt=control_dt)
    sim = Simulator(p, dt=1e-3, control_dt=control_dt, force_limit=force_limit,
                    use_motor=use_motor, use_driver=use_driver)
    ctrl = LQRController(k, u_limit=force_limit)
    res = sim.run([0.0, 0.0, theta0, 0.0], ctrl, t_end, label="lqr")
    out = _summarize_balance(res, res.theta())
    out["fidelity"] = ("driver" if use_driver else
                       ("motor" if use_motor else "ideal"))
    out["gain"] = list(k[0])
    out["riccati_residual"] = info["residual"]
    out["closed_loop_spectral_radius"] = info["spectral_radius"]
    out["riccati_iterations"] = info["iterations"]
    return out, res


def scenario_pid(params=None, theta0=0.05, t_end=8.0, control_dt=0.01,
                 force_limit=20.0):
    """基线: 只反馈摆角的单环 PID(与 LQR 同一个初始扰动)。"""
    p = params if params is not None else CartPoleParams()
    sim = Simulator(p, dt=1e-3, control_dt=control_dt, force_limit=force_limit)
    ctrl = AnglePIDController(kp=40.0, kd=8.0, u_limit=force_limit)
    res = sim.run([0.0, 0.0, theta0, 0.0], ctrl, t_end, label="pid")
    out = _summarize_balance(res, res.theta())
    out["fidelity"] = "ideal"
    return out, res


def scenario_disturbance(params=None, amplitude=6.0, t_on=2.0, duration=0.1,
                         t_end=8.0, controller="lqr", control_dt=0.01):
    """脉冲扰动下的抗扰恢复。"""
    p = params if params is not None else CartPoleParams()
    sim = Simulator(p, dt=1e-3, control_dt=control_dt, force_limit=20.0)
    if controller == "lqr":
        k, _info = cartpole_lqr_gain(p, control_dt=control_dt)
        ctrl = LQRController(k, u_limit=20.0)
    else:
        ctrl = AnglePIDController(kp=40.0, kd=8.0, u_limit=20.0)
    dist = pulse_disturbance(amplitude, t_on, duration)
    res = sim.run([0.0, 0.0, 0.0, 0.0], ctrl, t_end, disturbance=dist,
                  label="dist-" + controller)
    theta = res.theta()
    return {
        "controller": controller,
        "disturbance_N": amplitude,
        "disturbance_duration_s": duration,
        "peak_tilt_rad": max(abs(v) for v in theta),
        "recovery_time_s": recovery_time(res.t, theta, t_on, 0.02),
        "cart_drift_m": abs(res.position()[-1]),
        "control_energy": res.control_energy(),
        "peak_control_N": res.control_peak(),
    }, res


def scenario_trajectory(kind="scurve", distance=1.5, vmax=1.5, amax=6.0,
                        jmax=40.0, feedforward=True):
    """单关节伺服跟踪一条关节轨迹。"""
    if kind == "scurve":
        profile = SCurveProfile(0.0, distance, vmax, amax, jmax)
    elif kind == "trapezoid":
        profile = TrapezoidalProfile(0.0, distance, vmax, amax)
    elif kind == "quintic":
        profile = QuinticProfile(0.0, distance, vmax, amax)
    else:
        raise ValueError("未知轨迹类型: %r" % (kind,))

    sim = ServoSim(ServoParams(), dt=1e-4, control_dt=1e-3,
                   feedforward=feedforward)
    res = sim.run(profile, default_servo_controller())
    peak_v = profile.max_abs("velocity")
    peak_a = profile.max_abs("acceleration")
    peak_j = profile.max_abs("jerk")
    return {
        "kind": kind,
        "feedforward": feedforward,
        "duration_s": profile.duration,
        "tracking_rms_rad": res.tracking_rms(),
        "tracking_peak_rad": res.tracking_peak(),
        "tracking_rms_true_rad": res.tracking_rms_true(),
        "tracking_peak_true_rad": res.tracking_peak_true(),
        "encoding_floor_rad": res.encoding_floor(),
        "peak_velocity": peak_v,
        "peak_acceleration": peak_a,
        "peak_jerk": peak_j,
        "vmax_limit": vmax,
        "amax_limit": amax,
        "jmax_limit": jmax,
        # 五次多项式**不接受** jerk 限幅, 只接受时长; 所以它可能超限。
        # 把"是否超限"显式记录成布尔量, 而不是让读者自己去比两个数。
        "violates_vmax": peak_v > vmax * (1.0 + 1e-9),
        "violates_amax": peak_a > amax * (1.0 + 1e-9),
        "violates_jmax": peak_j > jmax * (1.0 + 1e-9),
        "peak_torque_Nm": res.max_torque(),
        "control_energy": res.control_energy(),
    }, res


def scenario_sync(distance=0.7, vmax=1.2, amax=5.0, jmax=30.0):
    """三条不同长度的轨迹做时间同步, 检查是否同时到达。"""
    profiles = [SCurveProfile(0.0, distance, vmax, amax, jmax),
                TrapezoidalProfile(0.0, distance * 0.3, vmax, amax),
                QuinticProfile(-distance * 0.5, distance * 0.2, vmax, amax)]
    synced = sync_trajectories(profiles)
    tmax = max(s.duration for s in synced)
    ends = [s.position(tmax) for s in synced]
    targets = [p.p1 for p in profiles]
    return {
        "joints": len(profiles),
        "raw_durations_s": [p.duration for p in profiles],
        "synced_duration_s": tmax,
        "end_positions_rad": ends,
        "targets_rad": targets,
        "max_end_error_rad": max(abs(a - b) for a, b in zip(ends, targets)),
        "end_time_spread_s": max(abs(s.duration - tmax) for s in synced),
    }


def scenario_dynamics(params=None):
    """动力学对账指标(能量漂移、线性化误差、开环不稳定特征值)。"""
    p = params if params is not None else CartPoleParams()
    from .dynamics import linearize, cartpole_deriv, cartpole_unstable_eig

    worst_energy, _ts, _es = energy_drift(
        lambda t, x: cartpole_deriv(x, 0.0, p), [0.0, 0.0, 0.2, 0.0],
        lambda s: cartpole_energy(s, p), 10.0, 1e-3)

    a_an, b_an = cartpole_linearize(p)
    a_num, b_num = linearize(lambda x, u: cartpole_deriv(x, u, p),
                             [0.0, 0.0, 0.0, 0.0], 0.0)
    worst_a = 0.0
    for i in range(4):
        for j in range(4):
            ref = a_an[i][j]
            if ref != 0.0:
                worst_a = max(worst_a, abs(a_an[i][j] - a_num[i][j]) / abs(ref))

    pend = PendulumParams()
    ts, xs = integrate_rk4(pendulum_ode(0.0, pend), [0.02, 0.0], 20.0, 5e-4)
    period = estimate_period(ts, [s[0] for s in xs])
    analytic_period = pendulum_small_angle_period(pend)

    return {
        "cartpole_energy_drift_10s": worst_energy,
        "linearization_max_rel_err": worst_a,
        "open_loop_unstable_eig": cartpole_unstable_eig(p),
        "analytic_omega": math.sqrt(cartpole_omega_sq(p)),
        "pendulum_period_sim_s": period,
        "pendulum_period_analytic_s": analytic_period,
        "pendulum_period_rel_err": abs(period - analytic_period) / analytic_period,
    }


def scenario_estimator(duration=20.0, dt=0.005, gyro_bias=0.05,
                       gyro_noise=0.01, accel_noise=0.05):
    """IMU 姿态估计对比(纯积分 / 互补 / 卡尔曼 1 维 / 卡尔曼 2 维)。"""
    def angle_fn(t):
        return 0.30 * math.sin(2.0 * math.pi * 0.20 * t) + 0.05

    data = simulate_imu(angle_fn, duration, dt, gyro_bias=gyro_bias,
                        gyro_noise=gyro_noise, accel_noise=accel_noise)
    return run_estimators(data)


# ---------------------------------------------------------------------------
# 一键评测
# ---------------------------------------------------------------------------


def run_suite(params=None, include_driver=True):
    """跑完整评测套件, 返回嵌套 dict(CLI 打印它, README 引用它)。"""
    p = params if params is not None else CartPoleParams()
    out = {"meta": {"description": "balance-control-lab 纯仿真评测套件"}}

    out["dynamics"] = scenario_dynamics(p)

    lqr_ideal, _ = scenario_lqr(p, use_motor=False, use_driver=False)
    out["balance"] = {"lqr_ideal": lqr_ideal}

    lqr_motor, _ = scenario_lqr(p, use_motor=True, use_driver=False)
    out["balance"]["lqr_motor"] = lqr_motor

    if include_driver:
        lqr_driver, _ = scenario_lqr(p, use_motor=True, use_driver=True)
        out["balance"]["lqr_driver"] = lqr_driver

    pid_base, _ = scenario_pid(p)
    out["balance"]["pid_angle_only"] = pid_base

    dist_lqr, _ = scenario_disturbance(p, controller="lqr")
    out["disturbance"] = {"lqr": dist_lqr}

    out["trajectory"] = {
        "scurve": scenario_trajectory("scurve", feedforward=True)[0],
        "scurve_no_ff": scenario_trajectory("scurve", feedforward=False)[0],
        "trapezoid": scenario_trajectory("trapezoid", feedforward=True)[0],
        "quintic": scenario_trajectory("quintic", feedforward=True)[0],
        "sync": scenario_sync(),
    }

    out["estimator"] = scenario_estimator()
    return out


# ---------------------------------------------------------------------------
# 打印
# ---------------------------------------------------------------------------


def _fmt(value, digits=6):
    """把值格式化成紧凑字符串。"""
    if isinstance(value, float):
        if value == 0.0:
            return "0"
        if abs(value) < 1e-3 or abs(value) >= 1e5:
            return "%.3e" % value
        return ("%%.%df" % digits) % value
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_fmt(v, 3) for v in value) + "]"
    return str(value)


def format_report(result, indent=0):
    """把 ``run_suite`` 的结果排成可读文本。"""
    lines = []
    pad = "  " * indent

    def walk(node, prefix, level):
        p = "  " * level
        if isinstance(node, dict):
            for k, v in node.items():
                if isinstance(v, dict):
                    lines.append("%s%s%s:" % (p, prefix, k))
                    walk(v, prefix, level + 1)
                else:
                    lines.append("%s%s%-34s %s" % (p, prefix, k, _fmt(v)))
        else:
            lines.append("%s%s%s" % (p, prefix, _fmt(node)))

    walk(result, pad, 0)
    return "\n".join(lines)


def main(argv=None):
    """``python -m bcl bench`` 的入口实现。"""
    del argv
    result = run_suite()
    print(format_report(result))
    return 0

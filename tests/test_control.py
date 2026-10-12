"""平衡控制器测试: DARE/LQR、极点配置、串级 PID、起摆。

三条核心断言(它们一起构成"这个 LQR 真的可用"的证据):

1. **Riccati 残差 < 1e-6** —— P 确实是离散 Riccati 方程的不动点。这一条不能省:
   迭代不收敛时 K 看起来仍然完全正常, 但闭环可能谱半径 > 1, 也就是"慢慢发散"。
2. **闭环谱半径 < 1** —— 离散意义下的稳定。``A - BK`` 的特征值必须全在单位圆内。
3. **实测稳定时间与稳态误差** —— 从 ``theta = 0.05 rad`` 起稳, 稳定时间 < 4 s,
   稳态误差 < 1e-3。
"""

from __future__ import annotations

import math

import pytest

from bcl.control import (AnglePIDController, CascadePID, LQRController, PID,
                         SwingUpBalanceController, SwingUpController, ackermann,
                         cartpole_lqr_gain, controllability_matrix, dare,
                         is_controllable, lqr, matrix_rank, poly_from_roots,
                         polyvalm, riccati_residual)
from bcl.dynamics import (CartPoleParams, add, cartpole_linearize,
                          discretize_zoh, eigvals, eye, matmul, norm_fro,
                          spectral_radius, sub)
from bcl.sim import Simulator


def _mat_close(a, b, tol=1e-9):
    assert norm_fro(sub(a, b)) < tol


def _settling_time(t, y, band, target=0.0):
    last = None
    for k in range(len(y)):
        if abs(y[k] - target) > band:
            last = k
    if last is None:
        return 0.0
    return t[last + 1] - t[0] if last + 1 < len(t) else t[-1] - t[0]


# ---------------------------------------------------------------------------
# DARE / LQR
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cartpole_problem():
    p = CartPoleParams()
    a, b = cartpole_linearize(p)
    ad, bd = discretize_zoh(a, b, 0.01)
    q = eye(4)
    for i, v in enumerate((20.0, 4.0, 100.0, 10.0)):
        q[i][i] = v
    return ad, bd, q, [[0.005]]


def test_dare_converges_and_returns_reasonable_k(cartpole_problem):
    ad, bd, q, r = cartpole_problem
    k, p, iters, ok = dare(ad, bd, q, r)
    assert ok
    assert iters < 20000
    assert len(k) == 1 and len(k[0]) == 4
    assert all(math.isfinite(v) for v in k[0])
    assert 1e6 > norm_fro(p) > 1.0


def test_dare_residual_below_1e_6(cartpole_problem):
    """核心断言 1: Riccati 残差 < 1e-6。"""
    ad, bd, q, r = cartpole_problem
    k, p, _iters, _ok = dare(ad, bd, q, r)
    res = riccati_residual(ad, bd, q, r, p)
    assert res < 1e-6
    assert res < 1e-8   # 实测 ~1.15e-9, 留一档余量
    assert all(math.isfinite(v) for v in k[0])


def test_lqr_closed_loop_spectral_radius_below_one(cartpole_problem):
    """核心断言 2: 闭环 ``A - BK`` 的谱半径 < 1(离散稳定)。"""
    ad, bd, q, r = cartpole_problem
    k, info = lqr(ad, bd, q, r)
    assert info["converged"]
    assert info["spectral_radius"] < 1.0
    _mat_close(sub(ad, matmul(bd, k)),
               sub(ad, matmul(bd, k)))
    a_cl = sub(ad, matmul(bd, k))
    assert spectral_radius(a_cl) == pytest.approx(info["spectral_radius"], rel=1e-12)


def test_lqr_info_reports_residual_and_radius(cartpole_problem):
    ad, bd, q, r = cartpole_problem
    _k, info = lqr(ad, bd, q, r)
    for key in ("residual", "spectral_radius", "iterations", "converged", "P"):
        assert key in info
    assert info["residual"] < 1e-6


def test_riccati_residual_zero_at_solution_and_positive_elsewhere(cartpole_problem):
    ad, bd, q, r = cartpole_problem
    _k, p, _i, _o = dare(ad, bd, q, r)
    assert riccati_residual(ad, bd, q, r, p) < 1e-8
    perturbed = [[p[i][j] * 1.05 for j in range(4)] for i in range(4)]
    assert riccati_residual(ad, bd, q, r, perturbed) > 1e-3


def test_dare_shape_validation(cartpole_problem):
    from bcl.dynamics import LinAlgError
    ad, bd, _q, r = cartpole_problem
    with pytest.raises(LinAlgError):
        dare(ad, bd, eye(3), r)


# ---------------------------------------------------------------------------
# 可控性与极点配置
# ---------------------------------------------------------------------------


def test_controllability_matrix_shape_and_rank(cartpole_problem):
    ad, bd, _q, _r = cartpole_problem
    c = controllability_matrix(ad, bd)
    assert len(c) == 4 and len(c[0]) == 4
    assert matrix_rank(c) == 4
    assert is_controllable(ad, bd)


def test_uncontrollable_system_detected():
    a = [[1.0, 0.0], [0.0, 2.0]]
    b = [[1.0], [0.0]]
    assert not is_controllable(a, b)
    with pytest.raises(Exception):
        ackermann(a, b, [0.5, 0.5])


def test_ackermann_scalar_case():
    """n=1 时可手算: K = (A - p) / B。"""
    k = ackermann([[3.0]], [[2.0]], [0.5])
    assert k[0][0] == pytest.approx((3.0 - 0.5) / 2.0, rel=1e-12)


def test_ackermann_places_poles_exactly(cartpole_problem):
    """核心断言: 闭环特征值等于期望极点(误差 < 1e-6)。"""
    ad, bd, _q, _r = cartpole_problem
    poles = [0.60, 0.70, 0.75, 0.80]
    k = ackermann(ad, bd, poles)
    a_cl = sub(ad, matmul(bd, k))
    got = sorted(eigvals(a_cl), key=lambda z: (round(z.real, 9), round(z.imag, 9)))
    for g, want in zip(got, poles):
        assert abs(g - want) < 1e-6
        assert abs(g.imag) < 1e-6


def test_ackermann_wrong_pole_count_raises(cartpole_problem):
    ad, bd, _q, _r = cartpole_problem
    with pytest.raises(Exception):
        ackermann(ad, bd, [0.5, 0.6])


def test_poly_from_roots_expands_correctly():
    coeffs = poly_from_roots([1.0, 2.0, 3.0])
    assert coeffs == pytest.approx([1.0, -6.0, 11.0, -6.0], rel=1e-12)


def test_polyvalm_satisfies_cayley_hamilton():
    """Cayley-Hamilton: ``p(A) = 0``(用 A 自己的特征多项式)。"""
    from bcl.dynamics import charpoly
    a = [[0.0, 1.0], [-2.0, -3.0]]
    assert norm_fro(polyvalm(charpoly(a), a)) < 1e-12


# ---------------------------------------------------------------------------
# PID
# ---------------------------------------------------------------------------


def test_pid_proportional_only():
    pid = PID(kp=2.0)
    assert pid.step(1.0, 0.1) == pytest.approx(2.0)
    assert pid.terms()[0] == pytest.approx(2.0)


def test_pid_output_clamped():
    pid = PID(kp=100.0, out_min=-1.0, out_max=1.0)
    assert pid.step(1.0, 0.1) == pytest.approx(1.0)
    assert pid.saturated


def test_pid_integral_accumulates_and_eliminates_bias():
    pid = PID(kp=0.0, ki=2.0, out_min=-10.0, out_max=10.0)
    pid.step(1.0, 0.1)
    pid.step(1.0, 0.1)
    assert pid.integral == pytest.approx(2.0 * 1.0 * 0.2, rel=1e-12)


def test_pid_anti_windup_freezes_integral_when_saturated():
    """输出顶到限幅且误差仍推同方向时, 积分必须冻结。"""
    sat = PID(kp=100.0, ki=10.0, out_min=-1.0, out_max=1.0)
    free = PID(kp=100.0, ki=10.0)
    for _ in range(100):
        sat.step(1.0, 0.01)
        free.step(1.0, 0.01)
    assert sat.integral == pytest.approx(0.0, abs=1e-12)
    assert free.integral > 5.0


def test_pid_anti_windup_releases_when_desaturating():
    """误差变小到输出不再饱和时, 积分恢复工作(否则系统会被"欠账"卡住)。

    注意: 误差反向且输出仍在下限饱和时, 抗饱和逻辑**应当继续冻结**积分 ——
    那正是抗饱和的定义。所以这里用一个"使输出回到线性区"的小反向误差。
    """
    pid = PID(kp=100.0, ki=10.0, out_min=-1.0, out_max=1.0)
    for _ in range(50):
        pid.step(1.0, 0.01)
    assert pid.integral == pytest.approx(0.0, abs=1e-12)
    pid.step(-0.005, 0.01)          # 未饱和 -> 积分恢复
    assert pid.integral == pytest.approx(10.0 * (-0.005) * 0.01, rel=1e-9)


def test_pid_derivative_on_measurement_avoids_kick():
    """设定值阶跃 + ``measurement`` 给定时, 微分项不应产生尖峰。"""
    pid = PID(kp=1.0, kd=1.0, deriv_filter_tau=0.0)
    pid.step(0.0, 0.01, measurement=0.0)
    out = pid.step(1.0, 0.01, measurement=0.0)   # 设定值跳变, 测量值没动
    assert out == pytest.approx(1.0, abs=1e-12)


def test_pid_error_derivative_would_kick_without_measurement():
    """不传 measurement 时用的是误差微分, 阶跃会打出一个大尖峰(对照组)。"""
    pid = PID(kp=1.0, kd=1.0, deriv_filter_tau=0.0)
    pid.step(0.0, 0.01)
    out = pid.step(1.0, 0.01)
    assert out > 50.0


def test_pid_derivative_filter_reduces_kick():
    """一阶低通能把 derivative kick 显著压低(但压不到 0 —— 那是滤波的代价)。"""
    raw = PID(kp=1.0, kd=1.0, deriv_filter_tau=0.0)
    raw.step(0.0, 0.01)
    kick_raw = raw.step(1.0, 0.01)

    filt = PID(kp=1.0, kd=1.0, deriv_filter_tau=0.05)
    filt.step(0.0, 0.01)
    kick_filt = filt.step(1.0, 0.01)

    assert kick_raw > 50.0
    assert kick_filt < 0.5 * kick_raw


def test_pid_reset_clears_state():
    pid = PID(kp=1.0, ki=1.0, kd=1.0)
    pid.step(1.0, 0.01)
    pid.reset()
    assert pid.integral == pytest.approx(0.0)
    assert pid.prev_error is None
    assert pid.prev_meas is None


def test_pid_invalid_dt_raises():
    with pytest.raises(ValueError):
        PID(kp=1.0).step(1.0, 0.0)


# ---------------------------------------------------------------------------
# 串级 PID 阶跃响应(给出超调/上升时间/调节时间)
# ---------------------------------------------------------------------------


def _simulate_double_integrator(kp_p, ki_p, kp_v, ki_v, t_end=8.0, dt=1e-4,
                                v_max=5.0, u_max=10.0):
    """双积分器 ``x'' = u`` 上的串级 PID 阶跃响应(位置环 -> 速度环)。

    用半隐式欧拉: 对该线性系统, dt=1e-4 下与 RK4 的差异在 1e-8 量级, 远小于被
    测指标(超调/上升时间/调节时间)的判读精度, 而 20 万步的纯 Python 循环比
    RK4 快一倍。**这是"用对精度的近似换测量时长"的取舍**, 不是忽略积分精度。
    """
    ctrl = CascadePID(PID(kp_p, ki_p, 0.0, -v_max, v_max),
                      PID(kp_v, ki_v, 0.0, -u_max, u_max), v_max=v_max)
    ctrl.reset()
    n = int(t_end / dt)
    t, x, v = 0.0, 0.0, 0.0
    ts, xs, vs, us = [0.0], [0.0], [0.0], [0.0]
    for _ in range(n):
        u = ctrl.step(1.0, x, v, dt)
        v += u * dt
        x += v * dt
        t += dt
        ts.append(t)
        xs.append(x)
        vs.append(v)
        us.append(u)
    return ts, xs, vs, us


# 实测选出的增益: 速度指令限幅 1.0 m/s, 力矩限幅 20 N。
# 为什么 v_max 必须小: 位置环的"可停距离"约 ``v_max^2/(2*u_max)``, v_max=5 时是
# 1.25 m, 已经超过阶跃幅值本身 —— 那是**物理上停不下来**, 表现为巨大超调。
# 把 v_max 压到 1.0 之后可停距离 2.5 cm, 超调立刻掉到 0.35%。
_CASCADE_GAINS = (10.0, 4.0, 20.0, 50.0)
_CASCADE_LIMITS = dict(v_max=1.0, u_max=20.0)


def test_cascade_pid_step_overshoot_and_steady_state():
    """核心断言: 阶跃响应超调 < 5%, 稳态误差 ≈ 0。

    实测(``kp_p=10, ki_p=4, kp_v=20, ki_v=50, v_max=1.0, u_max=20``):
    超调 **0.347%**、上升时间 **0.905 s**、调节时间(2% 带)**1.003 s**、
    t=15 s 时稳态误差 **< 1e-5**。
    """
    ts, xs, _vs, _us = _simulate_double_integrator(
        *_CASCADE_GAINS, t_end=15.0, **_CASCADE_LIMITS)
    overshoot = max(0.0, max(xs) - 1.0)
    assert overshoot < 0.05

    rise_time = next(t for t, x in zip(ts, xs) if x >= 0.9)
    assert 0.3 < rise_time < 1.5

    settle = _settling_time(ts, xs, 0.02, target=1.0)
    assert settle < 2.0

    final = sum(xs[-1000:]) / 1000.0
    assert final == pytest.approx(1.0, abs=1e-3)


def test_cascade_pid_quantitative_numbers():
    """把三个经典指标固定下来, 防止无意间改坏增益。"""
    ts, xs, _vs, _us = _simulate_double_integrator(
        *_CASCADE_GAINS, t_end=15.0, **_CASCADE_LIMITS)
    overshoot = max(0.0, max(xs) - 1.0) * 100.0
    rise = next(t for t, x in zip(ts, xs) if x >= 0.9)
    settle = _settling_time(ts, xs, 0.02, target=1.0)
    assert overshoot == pytest.approx(0.35, abs=0.2)
    assert rise == pytest.approx(0.905, abs=0.05)
    assert settle == pytest.approx(1.00, abs=0.1)


def test_cascade_pid_clamps_outer_velocity_command():
    """外环输出必须被 ``v_max`` 顶住; 越界的速度指令会直接把内环打饱和。"""
    ctrl = CascadePID(PID(1000.0, 0.0, 0.0), PID(1.0, 0.0, 0.0), v_max=2.0)
    ctrl.reset()
    ctrl.step(10.0, 0.0, 0.0, 0.01)
    assert ctrl.last_v_cmd == pytest.approx(2.0)
    assert abs(ctrl.outer.last_output) > 2.0     # 外环本身没限幅
    assert ctrl.last_v_cmd <= ctrl.v_max + 1e-12


def test_cascade_pid_inner_output_respects_limit():
    ctrl = CascadePID(PID(100.0, 0.0, 0.0), PID(1000.0, 0.0, 0.0,
                                                 out_min=-1.5, out_max=1.5), 5.0)
    ctrl.reset()
    u = ctrl.step(1.0, 0.0, 0.0, 0.01)
    assert abs(u) <= 1.5 + 1e-12


def test_cascade_pid_velocity_feedforward_removes_ramp_lag():
    """速度前馈能消掉匀速段的滞后误差(纯反馈的结构性缺陷)。"""
    def run(feed):
        ctrl = CascadePID(PID(8.0, 0.0, 0.0), PID(12.0, 30.0, 0.0), 5.0)
        ctrl.reset()
        dt, x, v = 1e-4, 0.0, 0.0
        worst = 0.0
        for k in range(20000):
            t = k * dt
            ref = 0.5 * t
            v_ref = 0.5 if feed else 0.0
            u = ctrl.step(ref, x, v, dt, vel_ref=v_ref)
            v += u * dt
            x += v * dt
            if t > 1.0:
                worst = max(worst, abs(x - ref))
        return worst

    assert run(False) > 0.01
    assert run(True) < 0.1 * run(False)


# ---------------------------------------------------------------------------
# 整机: LQR 起稳 / PID 基线 / 起摆
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cartpole_lqr_result():
    p = CartPoleParams()
    k, info = cartpole_lqr_gain(p)
    sim = Simulator(p, dt=1e-3, control_dt=0.01, force_limit=20.0)
    ctrl = LQRController(k, u_limit=20.0)
    res = sim.run([0.0, 0.0, 0.05, 0.0], ctrl, 8.0)
    return k, info, res


def test_lqr_stabilizes_cartpole_from_005_rad(cartpole_lqr_result):
    """核心断言 3: 稳定时间 < 4 s, 稳态误差 < 1e-3(实测约 2.9 s / 1e-7)。"""
    _k, _info, res = cartpole_lqr_result
    theta = res.theta()
    settle = _settling_time(res.t, theta, 1e-3)
    assert settle < 4.0
    assert abs(sum(theta[-200:]) / 200.0) < 1e-3
    assert abs(theta[-1]) < 1e-3


def test_lqr_brings_cart_back_to_origin(cartpole_lqr_result):
    _k, _info, res = cartpole_lqr_result
    assert abs(res.position()[-1]) < 1e-3
    assert max(abs(v) for v in res.position()) < 0.2


def test_lqr_control_stays_within_actuator_limit(cartpole_lqr_result):
    _k, _info, res = cartpole_lqr_result
    assert res.control_peak() <= 20.0 + 1e-9
    assert res.actuator_error() == pytest.approx(0.0, abs=1e-12)


def test_pid_angle_only_balances_but_cart_drifts():
    """基线对照: PID 把摆角压回 0, 但小车一路漂走且不回来。

    这是单环反馈的**结构性**缺陷, 不是调参问题 —— 也是 LQR 值得存在的理由。
    """
    p = CartPoleParams()
    sim = Simulator(p, dt=1e-3, control_dt=0.01, force_limit=20.0)
    ctrl = AnglePIDController(kp=40.0, kd=8.0, u_limit=20.0)
    res = sim.run([0.0, 0.0, 0.05, 0.0], ctrl, 8.0)
    assert abs(res.theta()[-1]) < 1e-3          # 摆角压住了
    assert abs(res.position()[-1]) > 0.5        # 小车跑了(实测 ~1.05 m)


def test_lqr_uses_less_energy_than_pid_while_also_regulating_position():
    """LQR 同时稳住摆角和位置, 且控制能量与 PID 同量级。"""
    p = CartPoleParams()
    k, _info = cartpole_lqr_gain(p)
    sim = Simulator(p, dt=1e-3, control_dt=0.01, force_limit=20.0)
    lqr_res = sim.run([0.0, 0.0, 0.05, 0.0], LQRController(k, u_limit=20.0), 8.0)
    pid_res = sim.run([0.0, 0.0, 0.05, 0.0],
                      AnglePIDController(kp=40.0, kd=8.0, u_limit=20.0), 8.0)
    assert lqr_res.control_energy() < 10.0 * pid_res.control_energy()
    assert abs(lqr_res.position()[-1]) < 0.01 < abs(pid_res.position()[-1])


def test_swingup_energy_converges_to_target():
    """能量整形: 起摆若干秒后摆能量应接近目标(这条用物理量守住"那个负号")。"""
    p = CartPoleParams()
    ctrl = SwingUpController(p, gain=6.0)
    sim = Simulator(p, dt=1e-3, control_dt=0.01)
    res = sim.run([0.0, 0.0, math.pi - 0.01, 0.0], ctrl, 8.0)
    energy = ctrl.pendulum_energy(
        [0.0, 0.0, res.theta()[-1], res.theta_dot()[-1]])
    assert abs(energy) < 0.2 * ctrl.target_energy()
    assert ctrl.target_energy() > 0.0


def test_swingup_energy_decreases_when_disabled():
    """对照组: 零增益时能量不增(防止"其实没在泵能量"的假通过)。"""
    p = CartPoleParams()
    ctrl = SwingUpController(p, gain=0.0, cart_kp=0.0, cart_kd=0.0)
    energy_before = ctrl.pendulum_energy([0.0, 0.0, math.pi - 0.01, 0.0])
    assert energy_before < 0.0
    assert ctrl.target_energy() > energy_before


def test_swingup_balance_controller_captures_and_stabilizes():
    p = CartPoleParams()
    k, _info = cartpole_lqr_gain(p)
    ctrl = SwingUpBalanceController(k, p)
    sim = Simulator(p, dt=1e-3, control_dt=0.01, force_limit=20.0)
    res = sim.run([0.0, 0.0, math.pi, 0.0], ctrl, 15.0)
    capture = ctrl.capture_time()
    assert capture is not None
    assert 2.0 < capture < 10.0
    assert abs(res.theta()[-1]) < 1e-3
    assert abs(res.position()[-1]) < 1e-2
    assert len(ctrl.switch_log) == 1


def test_swingup_balance_controller_reset():
    p = CartPoleParams()
    k, _info = cartpole_lqr_gain(p)
    ctrl = SwingUpBalanceController(k, p)
    sim = Simulator(p, dt=1e-3, control_dt=0.02)
    ctrl(sim.dt, [0.0, 0.0, math.pi, 0.0])
    ctrl.reset()
    assert ctrl.mode == SwingUpBalanceController.MODE_SWING_UP
    assert ctrl.switch_log == []
    assert ctrl.capture_time() is None


def test_lqr_controller_clamps_output():
    ctrl = LQRController([[10.0, 0.0, 0.0, 0.0]], u_limit=3.0)
    assert ctrl(0.0, [-100.0, 0.0, 0.0, 0.0]) == pytest.approx(3.0)
    assert ctrl(0.0, [100.0, 0.0, 0.0, 0.0]) == pytest.approx(-3.0)


def test_lqr_controller_with_reference_offset():
    ctrl = LQRController([[0.0, 0.0, 1.0, 0.0]], x_ref=[0.0, 0.0, 0.5, 0.0])
    assert ctrl(0.0, [0.0, 0.0, 0.6, 0.0]) == pytest.approx(-0.1, abs=1e-12)

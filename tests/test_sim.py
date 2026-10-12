"""仿真层与评测指标测试: RK4 收敛阶、电机模型、闭环仿真、伺服跟踪、bench 指标。

最关键的一条是 **RK4 的收敛阶**: 步长减半, 误差降约 16 倍。它是"我写的积分器
确实是四阶"最直接的证据 —— 抄错任何一个 1/6 系数, 收敛阶都会掉到 2 或 3,
比值立刻不是 16。用解析解可得的非线性方程 ``x' = x^2`` 做基准, 避免线性系统
偶然的超收敛。

bench 的指标函数也在这里测: 指标定义错了, README 里所有数字就都错了。
"""

from __future__ import annotations

import math

import pytest

from bcl.bench import (control_energy, format_report, overshoot_percent,
                       recovery_time, run_suite, scenario_dynamics,
                       scenario_estimator, scenario_trajectory,
                       settling_time, steady_state_error, tracking_errors)
from bcl.control import LQRController, cartpole_lqr_gain
from bcl.dynamics import CartPoleParams
from bcl.sim import (MotorModel, MotorParams, SensorModel, ServoParams, ServoSim,
                     Simulator, encoder_rms_floor, integrate_rk4, rk4_step)


# ---------------------------------------------------------------------------
# RK4
# ---------------------------------------------------------------------------


def test_rk4_step_linear_exponential():
    h = 1e-2
    got = rk4_step(lambda t, x: [x[0]], 0.0, [1.0], h)[0]
    assert got == pytest.approx(math.exp(h), rel=1e-9)


def test_rk4_step_harmonic_oscillator():
    h = 1e-3
    x = rk4_step(lambda t, s: [s[1], -s[0]], 0.0, [1.0, 0.0], h)
    assert x[0] == pytest.approx(math.cos(h), rel=1e-10)
    assert x[1] == pytest.approx(-math.sin(h), rel=1e-9)


def test_rk4_convergence_order_is_four():
    """核心断言: 步长减半误差降约 16 倍(4 阶)。

    基准方程 ``x' = x^2``, ``x(0) = 1``, 解析解 ``1/(1-t)``, 积分到 t=0.5。
    它是非线性的, 所以不会出现"线性系统上 RK4 恰好精确"这种假通过。
    """
    def exact(t):
        return 1.0 / (1.0 - t)

    def f(t, x):
        return [x[0] * x[0]]

    def err(dt):
        _ts, xs = integrate_rk4(f, [1.0], 0.5, dt)
        return abs(xs[-1][0] - exact(0.5))

    e1 = err(0.02)
    e2 = err(0.01)
    e3 = err(0.005)
    assert e1 > 0.0
    assert 12.0 < e1 / e2 < 22.0
    assert 12.0 < e2 / e3 < 22.0


def test_integrate_rk4_length_and_endpoint():
    ts, xs = integrate_rk4(lambda t, x: [0.0], [3.0], 1.0, 0.1)
    assert len(ts) == 11
    assert len(xs) == 11
    assert ts[-1] == pytest.approx(1.0, rel=1e-12)
    assert xs[-1][0] == pytest.approx(3.0, abs=1e-12)


def test_integrate_rk4_rejects_non_positive_dt():
    with pytest.raises(ValueError):
        integrate_rk4(lambda t, x: [0.0], [0.0], 1.0, 0.0)


# ---------------------------------------------------------------------------
# 电机模型
# ---------------------------------------------------------------------------


def test_motor_first_order_reaches_63_percent_in_one_tau():
    mp = MotorParams(torque_constant=1.0, max_torque=10.0, time_constant=0.1,
                     dead_zone=0.0)
    motor = MotorModel(mp)
    torque = 0.0
    for _ in range(100):
        torque = motor.step(1.0, 0.001)
    assert torque == pytest.approx(1.0 - math.exp(-1.0), rel=1e-3)


def test_motor_reaches_steady_state():
    mp = MotorParams(torque_constant=1.0, max_torque=10.0, time_constant=0.02,
                     dead_zone=0.0)
    motor = MotorModel(mp)
    for _ in range(2000):
        torque = motor.step(1.0, 0.001)
    assert torque == pytest.approx(1.0, rel=1e-6)


def test_motor_saturates_and_reports_it():
    mp = MotorParams(torque_constant=5.0, max_torque=0.4, time_constant=0.02,
                     dead_zone=0.0)
    motor = MotorModel(mp)
    for _ in range(500):
        torque = motor.step(1.0, 0.001)
    assert torque == pytest.approx(0.4, rel=1e-6)
    assert motor.saturated


def test_motor_dead_zone_blocks_small_command():
    mp = MotorParams(torque_constant=1.0, max_torque=10.0, time_constant=0.01,
                     dead_zone=0.05)
    motor = MotorModel(mp)
    for _ in range(200):
        torque = motor.step(0.04, 0.001)
    assert torque == pytest.approx(0.0, abs=1e-15)


def test_motor_dead_zone_compensation_keeps_full_scale():
    mp = MotorParams(torque_constant=1.0, max_torque=10.0, time_constant=0.01,
                     dead_zone=0.05)
    motor = MotorModel(mp)
    for _ in range(2000):
        torque = motor.step(1.0, 0.001)
    assert torque == pytest.approx(1.0, rel=1e-6)


def test_motor_is_stable_for_dt_larger_than_time_constant():
    """一阶惯性用精确离散解, 所以 dt > tau 也不会发散(欧拉法会)。"""
    mp = MotorParams(torque_constant=1.0, max_torque=10.0, time_constant=0.01,
                     dead_zone=0.0)
    motor = MotorModel(mp)
    torque = 0.0
    for _ in range(100):
        torque = motor.step(1.0, 0.05)
    assert math.isfinite(torque)
    assert torque == pytest.approx(1.0, rel=1e-6)


def test_motor_encoder_quantization():
    mp = MotorParams(encoder_ppr=1024)
    motor = MotorModel(mp)
    step = 2.0 * math.pi / 1024.0
    q = motor.quantize_angle(10.0 * step + 0.3 * step)
    assert q == pytest.approx(10.0 * step, rel=1e-12)


def test_motor_quantized_speed_uses_two_samples():
    mp = MotorParams(encoder_ppr=4096)
    motor = MotorModel(mp)
    motor.quantize_angle(0.0)
    step = 2.0 * math.pi / 4096.0
    motor.quantize_angle(step)
    assert motor.quantized_speed(0.1) == pytest.approx(step / 0.1, rel=1e-12)


def test_motor_reset_clears_state():
    motor = MotorModel(MotorParams(dead_zone=0.0))
    for _ in range(100):
        motor.step(1.0, 0.001)
    motor.reset()
    assert motor.torque == pytest.approx(0.0)
    assert not motor.saturated


def test_motor_rejects_non_positive_dt():
    with pytest.raises(ValueError):
        MotorModel().step(1.0, 0.0)


# ---------------------------------------------------------------------------
# 传感器降级模型
# ---------------------------------------------------------------------------


def test_sensor_model_quantizes_position_and_angle():
    sm = SensorModel(pos_res=0.01, angle_res=0.01)
    out = sm.measure([0.014, 0.0, 0.014, 0.0], 0.01)
    assert out[0] == pytest.approx(0.01)
    assert out[2] == pytest.approx(0.01)


def test_sensor_model_first_sample_has_zero_rate():
    sm = SensorModel()
    out = sm.measure([1.0, 5.0, 0.3, 2.0], 0.01)
    assert out[1] == pytest.approx(0.0)
    assert out[3] == pytest.approx(0.0)


def test_sensor_model_rate_estimate_tracks_constant_velocity():
    sm = SensorModel(pos_res=1e-6, angle_res=1e-9, lpf_alpha=0.5)
    x = 0.0
    dt = 0.01
    for _ in range(2000):
        x += 0.5 * dt
        out = sm.measure([x, 0.5, 0.0, 0.0], dt)
    assert out[1] == pytest.approx(0.5, abs=1e-3)


def test_sensor_model_low_pass_reduces_quantization_noise():
    """位置量化 + 差分会让速度估计极吵, 一阶低通必须把它压下去。"""
    dt = 0.01
    sm = SensorModel(pos_res=1e-3, angle_res=1e-6, lpf_alpha=0.1)
    vals = []
    for k in range(500):
        x = 0.25 * k * dt
        out = sm.measure([x, 0.25, 0.0, 0.0], dt)
        vals.append(out[1])
    tail = vals[100:]
    mean = sum(tail) / len(tail)
    std = math.sqrt(sum((v - mean) ** 2 for v in tail) / len(tail))
    assert std < 0.25


def test_sensor_model_reset():
    sm = SensorModel()
    sm.measure([1.0, 0.0, 0.0, 0.0], 0.01)
    sm.reset()
    assert sm.prev is None


# ---------------------------------------------------------------------------
# 整机仿真
# ---------------------------------------------------------------------------


def test_simulator_rejects_control_dt_not_multiple_of_dt():
    with pytest.raises(ValueError):
        Simulator(dt=1e-3, control_dt=1.5e-3)
    with pytest.raises(ValueError):
        Simulator(dt=1e-3, control_dt=5e-4)


def test_simulator_result_shapes_and_columns():
    p = CartPoleParams()
    k, _info = cartpole_lqr_gain(p)
    sim = Simulator(p, dt=1e-3, control_dt=0.01)
    res = sim.run([0.0, 0.0, 0.05, 0.0], LQRController(k), 1.0)
    assert len(res.t) == len(res.states) == len(res.u) == len(res.force)
    assert len(res) == 1001
    assert res.dt == pytest.approx(1e-3)
    assert len(res.theta()) == 1001
    assert res.position()[0] == pytest.approx(0.0)
    assert res.theta()[0] == pytest.approx(0.05)


def test_simulator_control_energy_matches_trapezoid_integral():
    """``u`` 与时间轴的对齐必须正确, 否则第一段会少算一半。"""
    p = CartPoleParams()
    sim = Simulator(p, dt=1e-3, control_dt=0.01)
    res = sim.run([0.0, 0.0, 0.0, 0.0], lambda t, x: 2.0, 1.0)
    assert res.u[0] == pytest.approx(2.0)
    assert res.control_energy() == pytest.approx(4.0, rel=1e-12)


def test_simulator_ideal_actuator_passes_command_through():
    p = CartPoleParams()
    sim = Simulator(p, dt=1e-3, control_dt=0.01)
    res = sim.run([0.0, 0.0, 0.0, 0.0], lambda t, x: 3.0, 0.1)
    assert res.actuator_error() == pytest.approx(0.0, abs=1e-12)
    assert res.control_peak() == pytest.approx(3.0)


def test_simulator_with_motor_reports_actuator_error():
    """带上死区/饱和/一阶惯性后, 指令与实际力之间必须有可见的差异。"""
    p = CartPoleParams()
    k, _info = cartpole_lqr_gain(p)
    sim = Simulator(p, dt=1e-3, control_dt=0.01, use_motor=True)
    res = sim.run([0.0, 0.0, 0.05, 0.0], LQRController(k), 4.0)
    assert res.actuator_error() > 0.1
    assert abs(res.theta()[-1]) < 0.05


def test_simulator_disturbance_shifts_then_recovers():
    p = CartPoleParams()
    k, _info = cartpole_lqr_gain(p)
    sim = Simulator(p, dt=1e-3, control_dt=0.01)
    kick = lambda t: 6.0 if 2.0 <= t < 2.1 else 0.0   # noqa: E731
    res = sim.run([0.0, 0.0, 0.0, 0.0], LQRController(k), 8.0, disturbance=kick)
    theta = res.theta()
    assert max(abs(v) for v in theta) > 0.005
    assert abs(theta[-1]) < 1e-4
    assert abs(res.position()[-1]) < 1e-2


def test_simulator_driver_stack_runs_end_to_end():
    """完整驱动栈(寄存器 + PWM 死区 + 编码器 + IMU)必须能闭环跑通。"""
    p = CartPoleParams()
    k, _info = cartpole_lqr_gain(p)
    sim = Simulator(p, dt=1e-3, control_dt=0.01, use_motor=True, use_driver=True)
    res = sim.run([0.0, 0.0, 0.05, 0.0], LQRController(k), 4.0)
    assert sim.bus is not None
    assert sim.bus.stats.reads > 0
    assert sim.bus.stats.writes > 0
    assert sim.bus.stats.error_count == 0
    assert abs(res.theta()[-1]) < 0.05
    assert math.isfinite(res.position()[-1])


# ---------------------------------------------------------------------------
# 单关节伺服与轨迹跟踪
# ---------------------------------------------------------------------------


def test_servo_tracks_scurve_with_feedforward():
    from bcl.trajectory import SCurveProfile
    prof = SCurveProfile(0.0, 1.5, 1.5, 6.0, 40.0)
    from bcl.bench import default_servo_controller
    sim = ServoSim(ServoParams(), dt=1e-4, control_dt=1e-3, feedforward=True)
    res = sim.run(prof, default_servo_controller())
    assert res.tracking_rms() < 5e-3
    assert res.tracking_rms_true() < 5e-3
    assert res.max_torque() < ServoParams().torque_limit


def test_servo_feedforward_dramatically_reduces_lag_error():
    """纯反馈对斜坡输入必然有 ``e = v/Kp`` 的滞后误差, 前馈把它消掉。"""
    from bcl.trajectory import SCurveProfile
    from bcl.bench import default_servo_controller
    prof = SCurveProfile(0.0, 1.5, 1.5, 6.0, 40.0)
    with_ff = ServoSim(ServoParams(), dt=1e-4, control_dt=1e-3, feedforward=True)
    no_ff = ServoSim(ServoParams(), dt=1e-4, control_dt=1e-3, feedforward=False)
    r_ff = with_ff.run(prof, default_servo_controller())
    r_no = no_ff.run(prof, default_servo_controller())
    assert r_ff.tracking_rms_true() < 0.2 * r_no.tracking_rms_true()


def test_servo_true_error_reaches_encoder_floor():
    """前馈到位后, 真实跟踪误差会落到编码器量化噪声的地板上。"""
    from bcl.trajectory import SCurveProfile
    from bcl.bench import default_servo_controller
    prof = SCurveProfile(0.0, 1.5, 1.5, 6.0, 40.0)
    sim = ServoSim(ServoParams(), dt=1e-4, control_dt=1e-3, feedforward=True)
    res = sim.run(prof, default_servo_controller())
    assert res.tracking_rms_true() < 3.0 * res.encoding_floor()


def test_servo_encoding_floor_formula():
    assert encoder_rms_floor(4096) == pytest.approx(
        (2.0 * math.pi / 4096.0) / math.sqrt(12.0), rel=1e-12)
    sim = ServoSim(ServoParams(encoder_ppr=4096))
    from bcl.trajectory import SCurveProfile
    from bcl.bench import default_servo_controller
    res = sim.run(SCurveProfile(0.0, 0.2, 1.0, 4.0, 20.0),
                  default_servo_controller())
    assert res.encoding_floor() == pytest.approx(encoder_rms_floor(4096), rel=1e-12)


# ---------------------------------------------------------------------------
# bench 指标函数
# ---------------------------------------------------------------------------


def test_settling_time_basic():
    t = [i * 0.1 for i in range(11)]
    y = [1.0, 0.5, 0.1, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # |0.1| > 0.05 的最后下标是 2, 所以稳定时刻是 t[3] = 0.3
    assert settling_time(t, y, 0.05) == pytest.approx(0.3)


def test_settling_time_uses_last_excursion_not_first():
    """必须取**最后一次**超出 —— 否则振荡收敛的系统会得到虚低的稳定时间。"""
    t = [i * 0.1 for i in range(11)]
    y = [1.0, 0.1, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert settling_time(t, y, 0.05) == pytest.approx(0.6)


def test_settling_time_never_exceeded_is_zero():
    assert settling_time([0.0, 1.0], [0.0, 0.0], 0.1) == pytest.approx(0.0)


def test_overshoot_percent_rising_and_falling():
    assert overshoot_percent([0.0, 1.0, 1.2, 1.0], 0.0, 1.0) == pytest.approx(20.0)
    assert overshoot_percent([0.05, -0.01, 0.0, 0.0], 0.05, 0.0) == pytest.approx(20.0)
    assert overshoot_percent([0.0, 1.0, 1.0], 0.0, 1.0) == pytest.approx(0.0)


def test_overshoot_percent_zero_span():
    assert overshoot_percent([1.0, 2.0], 1.0, 1.0) == pytest.approx(0.0)


def test_steady_state_error_uses_tail_mean():
    """末段取整: ``round(10 * 0.2) = 2`` 个点, 即 [1.0, 2.0], 均值 1.5。"""
    y = [1.0] * 9 + [2.0]
    err = steady_state_error(None, y, 0.0, tail_fraction=0.2)
    assert err == pytest.approx(1.5)


def test_steady_state_error_exact():
    y = [5.0] * 10 + [1.0] * 10
    assert steady_state_error(None, y, 1.0, tail_fraction=0.5) == pytest.approx(0.0)


def test_recovery_time_measures_return_to_band():
    """带 = 2% 峰值 = 0.01; 扰动在 t=2.0~2.4, 最后超出带的时刻是 t=2.4,
    所以恢复时间 = t[25] - 2.0 = 0.5 s。"""
    ts = [i * 0.1 for i in range(40)]
    y = [0.0] * 40
    for k in range(20, 25):
        y[k] = 0.5
    for k in range(25, 28):
        y[k] = 0.004
    assert recovery_time(ts, y, 2.0, 0.02) == pytest.approx(0.5)


def test_recovery_time_no_disturbance():
    ts = [i * 0.1 for i in range(10)]
    assert recovery_time(ts, [0.0] * 10, 0.0, 0.02) == pytest.approx(0.0)


def test_tracking_errors_known_values():
    rms, peak = tracking_errors([0.0, 1.0, 2.0], [0.0, 1.5, 2.0])
    assert rms == pytest.approx(math.sqrt(0.25 / 3.0))
    assert peak == pytest.approx(0.5)


def test_tracking_errors_length_mismatch_raises():
    with pytest.raises(ValueError):
        tracking_errors([1.0], [1.0, 2.0])


def test_control_energy_constant_signal():
    ts = [0.0, 1.0, 2.0, 3.0]
    us = [2.0, 2.0, 2.0, 2.0]
    assert control_energy(ts, us) == pytest.approx(12.0)


# ---------------------------------------------------------------------------
# bench 场景与套件
# ---------------------------------------------------------------------------


def test_scenario_dynamics_reports_reconciled_numbers():
    d = scenario_dynamics()
    assert d["cartpole_energy_drift_10s"] < 1e-6
    assert d["linearization_max_rel_err"] < 1e-6
    assert d["pendulum_period_rel_err"] < 0.01
    assert d["open_loop_unstable_eig"] == pytest.approx(d["analytic_omega"], rel=1e-6)


def test_scenario_estimator_reports_error_ordering():
    e = scenario_estimator(duration=10.0)
    assert e["kalman_2d"]["rmse"] < e["pure_gyro"]["rmse"]
    assert e["kalman_2d"]["rmse"] < e["complementary"]["rmse"]


def test_scenario_trajectory_reports_peaks_and_limits():
    r, _res = scenario_trajectory("scurve", distance=1.5, vmax=1.5, amax=6.0,
                                  jmax=40.0)
    assert r["peak_velocity"] == pytest.approx(1.5, rel=1e-12)
    assert r["peak_acceleration"] == pytest.approx(6.0, rel=1e-12)
    assert r["peak_jerk"] == pytest.approx(40.0, rel=1e-12)
    assert r["tracking_rms_true_rad"] < 1e-3


def test_run_suite_returns_all_sections():
    result = run_suite()
    for section in ("meta", "dynamics", "balance", "disturbance", "trajectory",
                    "estimator"):
        assert section in result
    for key in ("lqr_ideal", "lqr_motor", "lqr_driver", "pid_angle_only"):
        assert key in result["balance"]
    assert result["balance"]["lqr_ideal"]["settling_time_s"] < 4.0
    assert result["balance"]["lqr_ideal"]["steady_state_error_rad"] < 1e-3
    assert result["balance"]["lqr_ideal"]["riccati_residual"] < 1e-6
    assert result["balance"]["lqr_ideal"]["closed_loop_spectral_radius"] < 1.0
    assert abs(result["balance"]["pid_angle_only"]["final_position_m"]) > 0.5
    assert result["disturbance"]["lqr"]["recovery_time_s"] > 0.0


def test_format_report_produces_text():
    text = format_report({"a": {"b": 1.0, "c": [1.0, 2.0]}})
    assert "a:" in text
    assert "b" in text and "c" in text

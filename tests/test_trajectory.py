"""关节轨迹规划的测试。

核心是**峰值对账**: 梯形/S 曲线的峰值速度必须"恰好等于" vmax(除非位移太短),
S 曲线的峰值加速度恰好等于 amax、峰值 jerk 恰好等于 jmax。这类断言比"曲线看起来
平滑"有用得多 —— 一个符号写反的规划器仍然会生成平滑曲线, 但峰值会差一个符号或
一个系数。

**端点约定**: 轨迹在 ``t <= 0`` 与 ``t >= T`` 返回静止值(见 ``trajectory.py``
的模块 docstring)。所以端点断言在三条轨迹上都严格成立; 同时用 ``t = 1e-9``
的采样暴露梯形曲线端点加速度阶跃这一事实(它没有被约定掩盖)。
"""

from __future__ import annotations

import math

import pytest

from bcl.trajectory import (QuinticProfile, SCurveProfile, TimeScaledProfile,
                            TrapezoidalProfile, numeric_derivative,
                            sync_trajectories)


def _rel(a, b):
    return abs(a - b) / max(abs(b), 1e-300)


# ---------------------------------------------------------------------------
# 梯形速度
# ---------------------------------------------------------------------------


def test_trapezoid_reaches_vmax_exactly():
    tr = TrapezoidalProfile(0.0, 1.0, 1.0, 2.0)
    assert tr.max_abs("velocity") == pytest.approx(1.0, rel=1e-12)


def test_trapezoid_duration_matches_closed_form():
    tr = TrapezoidalProfile(0.0, 1.0, 1.0, 2.0)
    # ta = v/a = 0.5, tv = (d - v^2/a)/v = 0.5, T = 2*ta + tv = 1.5
    assert tr.duration == pytest.approx(1.5, rel=1e-15)


def test_trapezoid_short_move_reduces_peak_velocity():
    """位移太短走不到 vmax 时必须降速, 且给出最短时间解。"""
    d, vmax, amax = 0.02, 1.0, 2.0
    tr = TrapezoidalProfile(0.0, d, vmax, amax)
    v_peak = tr.max_abs("velocity")
    assert v_peak < vmax
    assert v_peak == pytest.approx(math.sqrt(d * amax), rel=1e-9)
    assert tr.duration == pytest.approx(2.0 * math.sqrt(d / amax), rel=1e-12)


def test_trapezoid_boundary_between_short_and_long():
    """恰好等于 ``vmax^2/amax`` 的位移: 三角形与梯形给出同样的时长。"""
    vmax, amax = 1.0, 2.0
    d = vmax * vmax / amax
    tr = TrapezoidalProfile(0.0, d, vmax, amax)
    assert tr.max_abs("velocity") == pytest.approx(vmax, rel=1e-9)
    assert tr.duration == pytest.approx(2.0 * vmax / amax, rel=1e-12)


def test_trapezoid_acceleration_plateau_equals_amax():
    tr = TrapezoidalProfile(0.0, 1.0, 1.0, 2.0)
    assert tr.acceleration(0.25) == pytest.approx(2.0, rel=1e-15)
    assert tr.acceleration(1.25) == pytest.approx(-2.0, rel=1e-15)
    assert tr.acceleration(0.75) == pytest.approx(0.0, abs=1e-15)


def test_trapezoid_endpoints_are_at_rest():
    tr = TrapezoidalProfile(0.0, 1.0, 1.0, 2.0)
    for t in (0.0, tr.duration):
        assert tr.velocity(t) == pytest.approx(0.0, abs=1e-15)
        assert tr.acceleration(t) == pytest.approx(0.0, abs=1e-15)
        assert tr.jerk(t) == pytest.approx(0.0, abs=1e-15)
    assert tr.position(0.0) == pytest.approx(0.0, abs=1e-15)
    assert tr.position(tr.duration) == pytest.approx(1.0, rel=1e-12)


def test_trapezoid_acceleration_is_discontinuous_just_inside():
    """端点约定掩盖不了事实: 进入运动后一刹那加速度就是 amax。"""
    tr = TrapezoidalProfile(0.0, 1.0, 1.0, 2.0)
    assert tr.acceleration(1e-9) == pytest.approx(2.0, rel=1e-9)


def test_trapezoid_position_monotonic():
    tr = TrapezoidalProfile(0.0, 1.0, 1.0, 2.0)
    prev = -math.inf
    for i in range(501):
        p = tr.position(tr.duration * i / 500.0)
        assert p >= prev - 1e-15
        prev = p


def test_trapezoid_negative_displacement_is_symmetric():
    pos = TrapezoidalProfile(0.0, 1.0, 1.0, 2.0)
    neg = TrapezoidalProfile(0.0, -1.0, 1.0, 2.0)
    assert neg.duration == pytest.approx(pos.duration, rel=1e-15)
    for i in range(21):
        t = pos.duration * i / 20.0
        assert neg.position(t) == pytest.approx(-pos.position(t), abs=1e-14)
        assert neg.velocity(t) == pytest.approx(-pos.velocity(t), abs=1e-14)


# ---------------------------------------------------------------------------
# S 曲线(七段式, jerk 受限)
# ---------------------------------------------------------------------------


def test_scurve_reaches_vmax_and_amax_exactly():
    tr = SCurveProfile(0.0, 1.0, 1.0, 2.0, 10.0)
    assert tr.reaches_vmax
    assert tr.reaches_amax
    assert tr.max_abs("velocity") == pytest.approx(1.0, rel=1e-12)
    assert tr.max_abs("acceleration") == pytest.approx(2.0, rel=1e-12)


def test_scurve_peak_jerk_is_exactly_jmax():
    tr = SCurveProfile(0.0, 1.0, 1.0, 2.0, 10.0)
    assert tr.max_abs("jerk") == pytest.approx(10.0, rel=1e-12)


def test_scurve_has_seven_segments_when_all_limits_reached():
    tr = SCurveProfile(0.0, 1.0, 1.0, 2.0, 10.0)
    assert len(tr.segments) == 7
    assert len(tr.segment_boundaries()) == 7


def test_scurve_short_move_reduces_both_peaks():
    """位移很短时 vmax 与 amax 都到不了, 但位移必须精确到达。"""
    tr = SCurveProfile(0.0, 0.02, 1.0, 2.0, 10.0)
    assert not tr.reaches_vmax
    assert not tr.reaches_amax
    assert tr.max_abs("velocity") < 1.0
    assert tr.max_abs("acceleration") < 2.0
    assert tr.max_abs("jerk") == pytest.approx(10.0, rel=1e-12)
    assert tr.position(tr.duration) == pytest.approx(0.02, rel=1e-9)
    assert len(tr.segments) == 4


def test_scurve_mid_move_reaches_amax_only():
    tr = SCurveProfile(0.0, 0.3, 1.0, 2.0, 10.0)
    assert tr.reaches_amax
    assert not tr.reaches_vmax
    assert tr.max_abs("acceleration") == pytest.approx(2.0, rel=1e-12)
    assert tr.position(tr.duration) == pytest.approx(0.3, rel=1e-9)
    assert len(tr.segments) == 6


def test_scurve_minimum_peak_velocity_formula_for_pure_jerk_limited():
    """纯 jerk 受限(加速度到不了)时有闭式 ``v_peak = (d^2 j / 4)^(1/3)``。"""
    d, j = 0.01, 10.0
    tr = SCurveProfile(0.0, d, 100.0, 100.0, j)
    v_peak = tr.max_abs("velocity")
    assert v_peak == pytest.approx((d * d * j / 4.0) ** (1.0 / 3.0), rel=1e-6)


def test_scurve_duration_longer_than_trapezoid():
    """同参数下 S 曲线必然更慢 —— 这是 jerk 受限要付的代价。"""
    trap = TrapezoidalProfile(0.0, 1.5, 1.5, 6.0)
    scurve = SCurveProfile(0.0, 1.5, 1.5, 6.0, 40.0)
    assert scurve.duration > trap.duration


def test_scurve_endpoints_strictly_at_rest():
    tr = SCurveProfile(0.0, 1.0, 1.0, 2.0, 10.0)
    assert tr.position(0.0) == pytest.approx(0.0, abs=1e-15)
    assert tr.velocity(0.0) == pytest.approx(0.0, abs=1e-12)
    assert tr.acceleration(0.0) == pytest.approx(0.0, abs=1e-12)
    assert tr.jerk(0.0) == pytest.approx(0.0, abs=1e-12)
    assert tr.position(tr.duration) == pytest.approx(1.0, rel=1e-12)
    assert tr.velocity(tr.duration) == pytest.approx(0.0, abs=1e-12)
    assert tr.acceleration(tr.duration) == pytest.approx(0.0, abs=1e-12)
    assert tr.jerk(tr.duration) == pytest.approx(0.0, abs=1e-12)


def test_scurve_respects_limits_over_dense_sampling():
    tr = SCurveProfile(0.0, 0.8, 1.2, 5.0, 30.0)
    v_lim = tr.max_abs("velocity")
    a_lim = tr.max_abs("acceleration")
    n = 2001
    for i in range(n):
        t = tr.duration * i / (n - 1)
        assert abs(tr.velocity(t)) <= v_lim * (1.0 + 1e-9)
        assert abs(tr.acceleration(t)) <= a_lim * (1.0 + 1e-9)
        assert abs(tr.jerk(t)) <= 30.0 * (1.0 + 1e-9)


def test_scurve_position_monotonic():
    tr = SCurveProfile(0.0, 0.5, 1.0, 3.0, 20.0)
    prev = -math.inf
    for i in range(1001):
        p = tr.position(tr.duration * i / 1000.0)
        assert p >= prev - 1e-12
        prev = p


def test_scurve_zero_displacement_is_degenerate():
    tr = SCurveProfile(0.2, 0.2, 1.0, 2.0, 10.0)
    assert tr.duration == 0.0
    s = tr.sample(0.0)
    assert s["position"] == pytest.approx(0.2)
    assert s["velocity"] == pytest.approx(0.0)


def test_scurve_invalid_limits_raise():
    with pytest.raises(ValueError):
        SCurveProfile(0.0, 1.0, 0.0, 2.0, 10.0)
    with pytest.raises(ValueError):
        SCurveProfile(0.0, 1.0, 1.0, 2.0, 0.0)


# ---------------------------------------------------------------------------
# 五次多项式
# ---------------------------------------------------------------------------


def test_quintic_endpoints_strictly_at_rest():
    tr = QuinticProfile(0.0, 1.0, 1.0, 2.0)
    for t, pos in ((0.0, 0.0), (tr.duration, 1.0)):
        assert tr.position(t) == pytest.approx(pos, abs=1e-12)
        assert tr.velocity(t) == pytest.approx(0.0, abs=1e-12)
        assert tr.acceleration(t) == pytest.approx(0.0, abs=1e-12)
        assert tr.jerk(t) == pytest.approx(0.0, abs=1e-12)


def test_quintic_analytic_peaks():
    tr = QuinticProfile(0.0, 1.0, 1.0, 2.0)
    assert tr.max_abs("velocity") == pytest.approx(1.875 / tr.duration, rel=1e-12)
    assert tr.max_abs("acceleration") == pytest.approx(
        10.0 * math.sqrt(3.0) / 3.0 / tr.duration ** 2, rel=1e-12)


def test_quintic_sampled_peaks_match_analytic():
    """扫点法算出的峰值与闭式解析峰值应相差 < 1%(交叉校验解析公式没写错)。"""
    tr = QuinticProfile(0.0, 1.5, 1.5, 6.0)
    sampled_v = max(abs(tr.sample(tr.duration * i / 20000.0)["velocity"])
                    for i in range(20001))
    assert _rel(sampled_v, tr.max_abs("velocity")) < 0.01


def test_quintic_endpoint_jerk_is_60d_over_T3():
    tr = QuinticProfile(0.0, 1.0, 1.0, 2.0)
    assert tr.endpoint_jerk() == pytest.approx(60.0 / tr.duration ** 3, rel=1e-12)
    assert tr.jerk(1e-9) == pytest.approx(tr.endpoint_jerk(), rel=1e-6)


def test_quintic_jerk_peak_much_larger_than_scurve():
    """同位移、同限幅下, 五次多项式完全不理会 jerk 限幅, 峰值 jerk 远大于 S 曲线。

    这是"为什么高速高精场合用 S 曲线而不是五次多项式"的量化版本。
    """
    scurve = SCurveProfile(0.0, 1.0, 1.0, 2.0, 5.0)
    quintic = QuinticProfile(0.0, 1.0, 1.0, 2.0)
    assert scurve.max_abs("jerk") == pytest.approx(5.0, rel=1e-12)
    assert quintic.max_abs("jerk") > 3.0 * scurve.max_abs("jerk")


def test_quintic_violates_vmax_when_duration_taken_from_trapezoid():
    """五次多项式不接受 vmax 参数, 给定时长下不保证满足限幅。"""
    vmax = 1.5
    quintic = QuinticProfile(0.0, 1.5, vmax, 6.0)
    assert quintic.max_abs("velocity") > vmax


def test_quintic_explicit_duration_is_respected():
    tr = QuinticProfile(0.0, 1.0, 1.0, 2.0, duration=3.0)
    assert tr.duration == pytest.approx(3.0)
    assert tr.position(3.0) == pytest.approx(1.0, rel=1e-12)


# ---------------------------------------------------------------------------
# 数值微分对账与多关节同步
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["trapezoid", "scurve", "quintic"])
def test_numeric_derivative_matches_trajectory_velocity(kind):
    """中心差分速度与轨迹自带速度的相对误差 < 1e-4。

    两个必须做的排除(否则测的就不是实现而是采样):
      * **段边界附近**: jerk 在那里跳变, 中心差分跨过跳变时误差掉到 O(dt);
      * **速度接近 0 处**: 相对误差的分母趋零, 数学上没有意义。
        这里取 |v| >= 1% 峰值速度作为有效区间。
    """
    if kind == "trapezoid":
        tr = TrapezoidalProfile(0.0, 1.0, 1.0, 2.0)
    elif kind == "scurve":
        tr = SCurveProfile(0.0, 1.0, 1.0, 2.0, 10.0)
    else:
        tr = QuinticProfile(0.0, 1.0, 1.0, 2.0)
    dt = 1e-4
    data = tr.sample_dt(dt)
    numeric = numeric_derivative(data["t"], data["position"])
    boundaries = tr.segment_boundaries()
    v_ref_scale = 0.01 * tr.max_abs("velocity")
    worst = 0.0
    checked = 0
    for i, t in enumerate(data["t"]):
        if any(abs(t - b) < 3.0 * dt for b in boundaries):
            continue
        if abs(data["velocity"][i]) < v_ref_scale:
            continue
        worst = max(worst, _rel(numeric[i], data["velocity"][i]))
        checked += 1
    assert checked > 100
    assert worst < 1e-4


def test_sample_series_shapes_and_endpoint_values():
    tr = SCurveProfile(0.0, 1.0, 1.0, 2.0, 10.0)
    data = tr.sample_series(101)
    assert len(data["t"]) == 101
    assert len(data["position"]) == 101
    assert data["t"][0] == pytest.approx(0.0)
    assert data["t"][-1] == pytest.approx(tr.duration)
    assert data["position"][-1] == pytest.approx(1.0, rel=1e-12)


def test_sample_dt_includes_end_time():
    tr = TrapezoidalProfile(0.0, 1.0, 1.0, 2.0)
    data = tr.sample_dt(0.1)
    assert data["t"][-1] == pytest.approx(tr.duration)


def test_sync_makes_all_joints_arrive_together():
    profiles = [SCurveProfile(0.0, 0.7, 1.2, 5.0, 30.0),
                TrapezoidalProfile(0.0, 0.21, 1.2, 5.0),
                QuinticProfile(-0.35, 0.14, 1.2, 5.0)]
    synced = sync_trajectories(profiles)
    tmax = max(s.duration for s in synced)
    assert max(abs(s.duration - tmax) for s in synced) < 1e-9
    for s, p in zip(synced, profiles):
        assert s.position(tmax) == pytest.approx(p.p1, rel=1e-9)
        assert s.velocity(tmax) == pytest.approx(0.0, abs=1e-12)


def test_sync_scales_down_peaks_never_up():
    """时间拉伸只会降低速度峰值, 绝不会违反原限幅。"""
    p = SCurveProfile(0.0, 1.0, 1.0, 2.0, 10.0)
    slow = TrapezoidalProfile(0.0, 0.2, 1.0, 2.0)
    synced = sync_trajectories([p, slow])
    assert synced[0].duration == pytest.approx(synced[1].duration, rel=1e-12)
    assert synced[0].duration >= p.duration
    assert synced[0].peak_velocity() <= p.max_abs("velocity") + 1e-12


def test_sync_empty_list():
    assert sync_trajectories([]) == []


def test_time_scaled_profile_rejects_bad_scale():
    p = TrapezoidalProfile(0.0, 1.0, 1.0, 2.0)
    with pytest.raises(ValueError):
        TimeScaledProfile(p, 0.0)

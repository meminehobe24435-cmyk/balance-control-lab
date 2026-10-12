"""状态估计测试: 互补滤波 vs 卡尔曼滤波(1 维/2 维)在含噪声与零偏的 IMU 数据上的表现。

核心断言: **卡尔曼的姿态 RMSE 小于纯陀螺积分的 1/2**(实测约 1/50), 并给出量化数字。

另外两条"解析对账"性质的断言, 它们比"KF 比别的都好"更有说服力, 因为给出了一个
可以手算的期望值:

* 互补滤波的稳态角度误差约等于 ``gyro_bias * tau``(零偏被一阶低通"记住了");
* 纯陀螺积分的漂移在时间上是线性的, 末值约等于 ``bias * T``。
"""

from __future__ import annotations

import math

import pytest

from bcl.estimator import (ComplementaryFilter, KalmanFilter1D,
                           KalmanFilter2D, gyro_integrate, max_abs_error, rmse,
                           run_estimators, simulate_imu)


def _oscillating_angle(t):
    """测试用的真实姿态: 0.3 rad 幅值、0.2 Hz 的慢摆 + 0.05 rad 偏置。"""
    return 0.30 * math.sin(2.0 * math.pi * 0.20 * t) + 0.05


@pytest.fixture(scope="module")
def imu_data():
    return simulate_imu(_oscillating_angle, 20.0, 0.005, gyro_bias=0.05,
                        gyro_noise=0.01, accel_noise=0.05, seed=4242)


# ---------------------------------------------------------------------------
# 互补滤波
# ---------------------------------------------------------------------------


def test_complementary_alpha_formula():
    f = ComplementaryFilter(tau=0.5)
    assert f.alpha(0.5) == pytest.approx(0.5, rel=1e-12)
    assert f.alpha(0.005) == pytest.approx(0.5 / 0.505, rel=1e-12)


def test_complementary_alpha_increases_with_sample_rate():
    """tau 固定时, 采样越快越信任陀螺(alpha 越接近 1)。"""
    f = ComplementaryFilter(tau=0.5)
    assert f.alpha(1e-3) > f.alpha(1e-2)


def test_complementary_tracks_constant_angle():
    f = ComplementaryFilter(tau=0.2)
    for _ in range(2000):
        angle = f.step(0.0, 1.0, 0.005)
    assert angle == pytest.approx(1.0, abs=1e-6)


def test_complementary_rejects_accel_noise_better_than_raw_accel():
    """互补滤波对方差固定的加速度计白噪声有明显抑制(理论增益 ~ 1/sqrt(1+tau/dt))。"""
    rng_bias = 0.0
    f = ComplementaryFilter(tau=0.2)
    est = []
    import random
    rng = random.Random(7)
    for _ in range(4000):
        z = 0.0 + rng.gauss(0.0, 0.05)
        est.append(f.step(0.0, z, 0.005))
    assert rmse(est, [0.0] * len(est)) < 0.5 * 0.05
    assert rng_bias == 0.0


def test_complementary_steady_state_error_is_bias_times_tau():
    """解析对账: 常值零偏 b 造成的稳态角度误差约等于 ``b * tau``。"""
    bias, tau = 0.05, 0.5
    f = ComplementaryFilter(tau=tau)
    angle = 0.0
    for _ in range(20000):
        angle = f.step(bias, 0.0, 0.005)
    assert angle == pytest.approx(bias * tau, rel=0.05)


def test_complementary_rejects_non_positive_tau():
    with pytest.raises(ValueError):
        ComplementaryFilter(tau=0.0)


def test_complementary_reset():
    f = ComplementaryFilter(tau=0.2, angle0=1.0)
    f.step(0.1, 0.2, 0.01)
    f.reset(0.0)
    assert f.angle == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 卡尔曼滤波
# ---------------------------------------------------------------------------


def test_kalman1d_gain_decreases_towards_steady_state():
    """初始协方差很大时增益接近 1, 收敛后降到稳态值 ``~sqrt(q/r)``。

    注意 ``step()`` 返回的是**角度**, 不是增益 —— 增益要通过 ``kf.gain`` 读。
    稳态增益的解析值: 由 ``p = (p+q) r/(p+q+r)`` 得 ``p ~ sqrt(q r)``,
    于是 ``K = (p+q)/(p+q+r) ~ sqrt(q/r)``。q=1e-5、r=1e-2 时约 0.032。
    """
    q_angle, r_measure = 1e-5, 1e-2
    kf = KalmanFilter1D(q_angle=q_angle, r_measure=r_measure)
    gains = []
    for _ in range(200):
        kf.step(0.0, 1.0, 0.005)
        gains.append(kf.gain)
    assert gains[0] > 0.9
    assert gains[-1] < 0.2
    assert gains[-1] == pytest.approx(math.sqrt(q_angle / r_measure), rel=0.5)


def test_kalman1d_predict_integrates_gyro():
    kf = KalmanFilter1D(angle=0.0, p=1.0)
    kf.predict(1.0, 0.1)
    assert kf.angle == pytest.approx(0.1, rel=1e-12)


def test_kalman1d_covariance_decreases_on_update():
    kf = KalmanFilter1D(angle=0.0, p=1.0)
    p_before = kf.p
    kf.update(0.0)
    assert kf.p < p_before


def test_kalman2d_predict_subtracts_bias():
    kf = KalmanFilter2D(angle=0.0, bias=0.1)
    kf.predict(0.5, 0.1)
    assert kf.angle == pytest.approx((0.5 - 0.1) * 0.1, rel=1e-12)


def test_kalman2d_covariance_stays_symmetric_and_positive():
    kf = KalmanFilter2D()
    for i in range(500):
        kf.step(0.1 * math.sin(i * 0.01), 0.1 * math.sin(i * 0.01), 0.005)
    assert kf.p00 > 0.0
    assert kf.p11 > 0.0
    assert kf.p01 == pytest.approx(kf.p10, abs=1e-18)
    assert kf.p00 * kf.p11 - kf.p01 * kf.p10 > 0.0


def test_kalman2d_estimates_gyro_bias(imu_data):
    """零偏状态必须真的收敛到真值附近(这是 2 维相对 1 维的全部价值)。"""
    res = run_estimators(imu_data)
    assert res["kalman_2d_bias_estimate"] == pytest.approx(res["true_bias"],
                                                           rel=0.15)


def test_kalman_invalid_dt_raises():
    with pytest.raises(ValueError):
        KalmanFilter1D().predict(0.1, 0.0)
    with pytest.raises(ValueError):
        KalmanFilter2D().predict(0.1, -1.0)


def test_kalman_reset_restores_state():
    kf = KalmanFilter2D(angle=1.0, bias=0.2)
    kf.reset(0.0, 0.0)
    assert kf.angle == pytest.approx(0.0)
    assert kf.bias == pytest.approx(0.0)
    assert kf.p00 == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 端到端对比(核心断言)
# ---------------------------------------------------------------------------


def test_kalman_rmse_much_better_than_pure_gyro_integration(imu_data):
    """核心断言: 卡尔曼的 RMSE < 纯陀螺积分的 1/2(实测约 1/50)。"""
    res = run_estimators(imu_data)
    assert res["kalman_2d"]["rmse"] < 0.5 * res["pure_gyro"]["rmse"]
    assert res["kalman_2d"]["rmse"] < 0.05 * res["pure_gyro"]["rmse"]


def test_kalman_beats_complementary_filter(imu_data):
    res = run_estimators(imu_data)
    assert res["kalman_2d"]["rmse"] < res["complementary"]["rmse"]


def test_pure_gyro_integration_drift_is_linear_in_time(imu_data):
    """解析对账: 纯积分的末值误差约等于 ``bias * T``。"""
    dt = imu_data["t"][1] - imu_data["t"][0]
    est = gyro_integrate(imu_data["gyro"], dt, angle0=imu_data["true_angle"][0])
    final_err = abs(est[-1] - imu_data["true_angle"][-1])
    expected = imu_data["gyro_bias"] * imu_data["t"][-1]
    assert final_err == pytest.approx(expected, rel=0.05)


def test_all_estimators_reported_and_ordered(imu_data):
    res = run_estimators(imu_data)
    for key in ("pure_gyro", "complementary", "kalman_1d", "kalman_2d"):
        assert key in res
        for field in ("rmse", "max_err", "final_err"):
            assert field in res[key]
            assert res[key][field] >= 0.0
    assert res["pure_gyro"]["rmse"] > res["complementary"]["rmse"]
    assert res["complementary"]["rmse"] > res["kalman_2d"]["rmse"]


def test_kalman_2d_max_error_bounded_by_accel_noise(imu_data):
    """卡尔曼的最大误差应被加速度计噪声量级压住(不被陀螺零偏拖走)。"""
    res = run_estimators(imu_data)
    assert res["kalman_2d"]["max_err"] < 0.2
    assert res["pure_gyro"]["max_err"] > 0.5


# ---------------------------------------------------------------------------
# 数据合成与误差函数
# ---------------------------------------------------------------------------


def test_simulate_imu_is_reproducible_with_seed():
    a = simulate_imu(_oscillating_angle, 1.0, 0.01, seed=1)
    b = simulate_imu(_oscillating_angle, 1.0, 0.01, seed=1)
    c = simulate_imu(_oscillating_angle, 1.0, 0.01, seed=2)
    assert a["gyro"] == pytest.approx(b["gyro"])
    assert a["gyro"] != pytest.approx(c["gyro"])


def test_simulate_imu_shapes_and_units():
    data = simulate_imu(_oscillating_angle, 1.0, 0.01, gyro_bias=0.05,
                        gyro_noise=0.0, accel_noise=0.0, seed=3)
    assert len(data["t"]) == 101
    assert len(data["true_angle"]) == 101
    assert data["t"][0] == pytest.approx(0.0)
    assert data["t"][-1] == pytest.approx(1.0)
    # 无噪声时: 陀螺读数 = 真实角速率 + 零偏
    mid = 50
    true_rate = (_oscillating_angle(data["t"][mid] + 1e-6)
                 - _oscillating_angle(data["t"][mid] - 1e-6)) / 2e-6
    assert data["gyro"][mid] == pytest.approx(true_rate + 0.05, rel=1e-6)
    assert data["accel_angle"][mid] == pytest.approx(data["true_angle"][mid])


def test_simulate_imu_invalid_dt_raises():
    with pytest.raises(ValueError):
        simulate_imu(_oscillating_angle, 1.0, 0.0)


def test_rmse_and_max_abs_error_known_values():
    est = [1.0, 1.0, 1.0]
    truth = [0.0, 0.0, 0.0]
    assert rmse(est, truth) == pytest.approx(1.0)
    assert max_abs_error([0.0, 3.0], [1.0, 1.0]) == pytest.approx(2.0)


def test_rmse_length_mismatch_raises():
    with pytest.raises(ValueError):
        rmse([1.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        max_abs_error([1.0], [1.0, 2.0])


def test_gyro_integrate_constant_rate():
    out = gyro_integrate([2.0] * 11, 0.1, angle0=0.5)
    assert out[0] == pytest.approx(0.5)
    assert out[-1] == pytest.approx(0.5 + 2.0 * 1.0)

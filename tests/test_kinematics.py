"""运动学测试: 正解、逆解(双解)、雅可比对账、奇异检测与速度放大。

最有工程价值的两条:

* **雅可比解析式与数值中心差分逐元素差 < 1e-6** —— 这等价于对正运动学做了一次
  独立复核, 因为数值 Jacobian 只依赖 ``forward()``。
* **接近奇异时关节速度被放大** —— 给出定量数字(``|qdot|`` 与 ``1/sin(q2)`` 成正比),
  这是实机上"机器人突然抽一下"的根因。
"""

from __future__ import annotations

import math

import pytest

from bcl.dynamics import norm_fro, sub
from bcl.kinematics import (Planar2R, Planar3R, SingularityError,
                            UnreachableError, cartesian_linear_path,
                            forward_kinematics_chain, jacobian_chain,
                            jacobian_condition_estimate)


def _jac_rel_err(analytic, numeric):
    """逐元素相对误差最大值(对 0 元素退化为绝对误差)。"""
    worst = 0.0
    for i in range(len(analytic)):
        for j in range(len(analytic[0])):
            ref = analytic[i][j]
            if abs(ref) < 1e-12:
                worst = max(worst, abs(analytic[i][j] - numeric[i][j]))
            else:
                worst = max(worst, abs(analytic[i][j] - numeric[i][j]) / abs(ref))
    return worst


# ---------------------------------------------------------------------------
# 平面 2R
# ---------------------------------------------------------------------------


def test_2r_forward_at_zero_angles():
    arm = Planar2R(1.0, 0.8)
    assert arm.forward([0.0, 0.0]) == pytest.approx((1.8, 0.0), abs=1e-15)


def test_2r_forward_quarter_turn():
    arm = Planar2R(1.0, 0.8)
    x, y = arm.forward([math.pi / 2, 0.0])
    assert x == pytest.approx(0.0, abs=1e-15)
    assert y == pytest.approx(1.8, abs=1e-15)


def test_2r_forward_folded_is_at_inner_boundary():
    arm = Planar2R(1.0, 0.4)
    assert arm.forward([0.0, math.pi]) == pytest.approx((0.6, 0.0), abs=1e-15)
    assert arm.reach_min == pytest.approx(0.6)
    assert arm.reach_max == pytest.approx(1.4)


def test_2r_joint_positions_length():
    arm = Planar2R(1.0, 0.8)
    pts = arm.joint_positions([0.3, 0.7])
    assert len(pts) == 3
    assert pts[0] == (0.0, 0.0)
    assert pts[2] == pytest.approx(arm.forward([0.3, 0.7]))


@pytest.mark.parametrize("elbow", ["up", "down"])
def test_2r_inverse_roundtrip(elbow):
    arm = Planar2R(1.0, 0.8)
    q = [0.4, 1.0] if elbow == "up" else [0.4, -1.0]
    target = arm.forward(q)
    q_sol = arm.inverse(target[0], target[1], elbow)
    assert q_sol[0] == pytest.approx(q[0], abs=1e-12)
    assert q_sol[1] == pytest.approx(q[1], abs=1e-12)


def test_2r_inverse_both_solutions_hit_same_point():
    arm = Planar2R(1.0, 0.8)
    x, y = arm.forward([0.6, 0.9])
    up, down = arm.inverse_both(x, y)
    assert up[1] > 0.0
    assert down[1] < 0.0
    for q in (up, down):
        fx, fy = arm.forward(q)
        assert fx == pytest.approx(x, abs=1e-12)
        assert fy == pytest.approx(y, abs=1e-12)


def test_2r_inverse_matches_analytic_elbow_down():
    """肘下解的闭式值(用余弦定理手算), 防止只测"往返一致"这种自证。"""
    l1, l2 = 1.0, 1.0
    arm = Planar2R(l1, l2)
    x, y = 1.0, 1.0
    q1, q2 = arm.inverse(x, y, "down")
    r2 = x * x + y * y
    c2 = (r2 - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)
    assert math.cos(q2) == pytest.approx(c2, abs=1e-12)
    assert q2 < 0.0


def test_2r_unreachable_raises():
    arm = Planar2R(1.0, 0.5)
    with pytest.raises(UnreachableError):
        arm.inverse(5.0, 0.0)
    with pytest.raises(UnreachableError):
        arm.inverse(0.1, 0.0)


def test_2r_invalid_elbow_raises():
    arm = Planar2R(1.0, 0.8)
    with pytest.raises(ValueError):
        arm.inverse(0.5, 0.5, "sideways")


def test_2r_is_reachable_bounds():
    arm = Planar2R(1.0, 0.5)
    assert arm.is_reachable(1.5, 0.0)
    assert arm.is_reachable(0.5, 0.0)
    assert not arm.is_reachable(1.6, 0.0)
    assert not arm.is_reachable(0.4, 0.0)


def test_2r_jacobian_matches_numeric():
    arm = Planar2R(1.0, 0.8)
    q = [0.4, 1.0]
    assert _jac_rel_err(arm.jacobian(q), arm.jacobian_numeric(q)) < 1e-6


def test_2r_jacobian_determinant_closed_form():
    arm = Planar2R(1.0, 0.8)
    q = [0.4, 1.0]
    j = arm.jacobian(q)
    det_numeric = j[0][0] * j[1][1] - j[0][1] * j[1][0]
    assert arm.det_jacobian(q) == pytest.approx(det_numeric, rel=1e-12)
    assert arm.det_jacobian(q) == pytest.approx(
        arm.l1 * arm.l2 * math.sin(q[1]), rel=1e-12)


def test_2r_singularity_detection():
    arm = Planar2R(1.0, 0.8)
    assert arm.is_singular([0.3, 0.0])
    assert arm.is_singular([0.3, math.pi])
    assert not arm.is_singular([0.3, math.pi / 2])
    assert arm.singularity_margin([0.3, math.pi / 2]) == pytest.approx(1.0)


def test_2r_inverse_velocity_roundtrip():
    arm = Planar2R(1.0, 0.8)
    q = [0.3, 1.1]
    j = arm.jacobian(q)
    qdot = [0.7, -0.4]
    ee = [j[0][0] * qdot[0] + j[0][1] * qdot[1],
          j[1][0] * qdot[0] + j[1][1] * qdot[1]]
    back = arm.inverse_velocity(q, ee)
    assert back[0] == pytest.approx(qdot[0], abs=1e-10)
    assert back[1] == pytest.approx(qdot[1], abs=1e-10)


def test_2r_inverse_velocity_at_singularity_raises():
    arm = Planar2R(1.0, 0.8)
    with pytest.raises(SingularityError):
        arm.inverse_velocity([0.2, 1e-9], [1.0, 0.0])


def test_2r_velocity_amplification_scales_as_one_over_sin_q2():
    """定量例子: 接近奇异时关节速度被放大, 且放大倍数有闭式。

    取 ``q1 = 0``、末端沿 +x 的单位速度, 则 ``J = [[-l2 s2, -l2 s2],
    [l1 + l2 c2, l2 c2]]``, ``det = l1 l2 sin(q2)``, 于是

        |qdot| = sqrt(l2^2 cos^2 q2 + (l1 + l2 cos q2)^2) / (l1 l2 |sin q2|)

    分母里的 ``sin(q2)`` 就是全部放大来源。这里同时校验"绝对量级"和"除以 sin"
    这两件事, 而不是只验证一个大概的倍数。
    """
    arm = Planar2R(1.0, 0.8)
    l1, l2 = arm.l1, arm.l2

    def closed_form(q2):
        return math.sqrt(l2 ** 2 * math.cos(q2) ** 2
                         + (l1 + l2 * math.cos(q2)) ** 2) \
            / (l1 * l2 * abs(math.sin(q2)))

    v_good = math.hypot(*arm.velocity_amplification((1.0, 0.0), math.pi / 2))
    v_bad = math.hypot(*arm.velocity_amplification((1.0, 0.0), 0.01))
    assert v_good == pytest.approx(closed_form(math.pi / 2), rel=1e-12)
    assert v_good == pytest.approx(1.25, rel=1e-9)
    assert v_bad == pytest.approx(closed_form(0.01), rel=1e-9)
    assert v_bad / v_good > 90.0


def test_2r_velocity_amplification_monotonic_in_margin():
    """裕度越小放大越严重(单调), 这是"接近奇异"这个判据的意义。"""
    arm = Planar2R(1.0, 0.8)
    mags = [math.hypot(*arm.velocity_amplification((1.0, 0.0), q2))
            for q2 in (1.0, 0.1, 0.01, 0.001)]
    assert all(mags[i] < mags[i + 1] for i in range(len(mags) - 1))


def test_2r_invalid_link_length_raises():
    with pytest.raises(ValueError):
        Planar2R(0.0, 0.5)


# ---------------------------------------------------------------------------
# 平面 3R
# ---------------------------------------------------------------------------


def test_3r_forward_and_orientation():
    arm = Planar3R(1.0, 0.8, 0.5)
    x, y, phi = arm.forward([0.3, 0.4, 0.5])
    assert phi == pytest.approx(1.2)
    x2, y2 = Planar2R(1.0, 0.8).forward([0.3, 0.4])
    assert x == pytest.approx(x2 + 0.5 * math.cos(1.2))
    assert y == pytest.approx(y2 + 0.5 * math.sin(1.2))


@pytest.mark.parametrize("elbow", ["up", "down"])
def test_3r_inverse_roundtrip(elbow):
    arm = Planar3R(1.0, 0.8, 0.5)
    q = [0.3, 1.1, 0.4] if elbow == "up" else [0.3, -1.1, 0.9]
    pose = arm.forward(q)
    sol = arm.inverse(pose[0], pose[1], pose[2], elbow)
    for i in range(3):
        assert sol[i] == pytest.approx(q[i], abs=1e-10)


def test_3r_jacobian_matches_numeric():
    arm = Planar3R(1.0, 0.8, 0.5)
    q = [0.3, 1.1, 0.4]
    assert _jac_rel_err(arm.jacobian(q), arm.jacobian_numeric(q)) < 1e-6


def test_3r_jacobian_third_row_is_ones():
    arm = Planar3R(1.0, 0.8, 0.5)
    assert arm.jacobian([0.3, 1.1, 0.4])[2] == pytest.approx([1.0, 1.0, 1.0])


def test_3r_jacobian_determinant_closed_form():
    arm = Planar3R(1.0, 0.8, 0.5)
    q = [0.3, 1.1, 0.4]
    j = arm.jacobian(q)
    det = (j[0][0] * (j[1][1] * j[2][2] - j[1][2] * j[2][1])
           - j[0][1] * (j[1][0] * j[2][2] - j[1][2] * j[2][0])
           + j[0][2] * (j[1][0] * j[2][1] - j[1][1] * j[2][0]))
    assert arm.det_jacobian(q) == pytest.approx(det, rel=1e-11)
    assert arm.det_jacobian(q) == pytest.approx(
        arm.l1 * arm.l2 * math.sin(q[1]), rel=1e-12)


def test_3r_determinant_independent_of_third_joint():
    """闭式结论: 3R 的行列式只取决于 q2。"""
    arm = Planar3R(1.0, 0.8, 0.5)
    d1 = arm.det_jacobian([0.3, 0.9, 0.0])
    d2 = arm.det_jacobian([0.3, 0.9, 2.5])
    assert d1 == pytest.approx(d2, rel=1e-12)


def test_3r_singularity_detection():
    arm = Planar3R(1.0, 0.8, 0.5)
    assert arm.is_singular([0.3, 0.0, 0.4])
    assert not arm.is_singular([0.3, math.pi / 2, 0.4])


def test_3r_unreachable_raises():
    arm = Planar3R(1.0, 0.8, 0.5)
    with pytest.raises(UnreachableError):
        arm.inverse(10.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# 通用链与辅助函数
# ---------------------------------------------------------------------------


def test_chain_fk_matches_2r_implementation():
    arm = Planar2R(0.9, 0.6)
    q = [0.25, -0.8]
    assert forward_kinematics_chain([0.9, 0.6], q) == pytest.approx(
        arm.forward(q), abs=1e-14)


def test_chain_jacobian_matches_2r_numeric():
    arm = Planar2R(0.9, 0.6)
    q = [0.25, -0.8]
    assert norm_fro(sub(jacobian_chain([0.9, 0.6], q), arm.jacobian_numeric(q))) < 1e-6


def test_chain_length_mismatch_raises():
    with pytest.raises(ValueError):
        forward_kinematics_chain([1.0, 1.0], [0.1])


def test_cartesian_linear_path_endpoints_and_count():
    pts = cartesian_linear_path((0.0, 0.0), (1.0, 2.0), 11)
    assert len(pts) == 11
    assert pts[0] == pytest.approx((0.0, 0.0))
    assert pts[-1] == pytest.approx((1.0, 2.0))
    assert pts[5] == pytest.approx((0.5, 1.0))


def test_condition_estimate_is_finite_away_from_singularity():
    arm = Planar2R(1.0, 0.8)
    c = jacobian_condition_estimate(arm.jacobian([0.3, math.pi / 2]))
    assert math.isfinite(c)
    assert c >= 1.0


def test_condition_estimate_is_infinite_at_singularity():
    arm = Planar2R(1.0, 0.8)
    assert jacobian_condition_estimate(arm.jacobian([0.3, 0.0])) == float("inf")

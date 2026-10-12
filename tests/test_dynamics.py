"""动力学与线性代数核心的测试。

这里最重要的三条断言(它们共同构成"动力学方程写对了"的证据):

1. **能量守恒**: 无外力无阻尼时, 10 秒 RK4 仿真的相对能量漂移 < 1e-6。
2. **解析对账**: 单摆小角度周期与 ``2*pi*sqrt(L/g)`` 相差 < 1%; cart-pole 倒立点
   线性化频率与 ``g(M+m)/(Ml)`` 相差 < 1e-3。
3. **数值 Jacobian 对账**: 中心差分 Jacobian 与解析 Jacobian 逐元素相差 < 1e-6。
"""

from __future__ import annotations

import math

import pytest

from bcl.dynamics import (CartPoleParams, LinAlgError, PendulumParams,
                          WheeledCartParams, add, cartpole_deriv, cartpole_energy,
                          cartpole_linearize, cartpole_ode, cartpole_omega_sq,
                          cartpole_unstable_eig, charpoly, discretize_zoh,
                          eigvals, energy_drift, estimate_period, expm, eye,
                          inv, linearize, matmul, matvec, norm_fro,
                          pendulum_deriv, pendulum_energy, pendulum_ode,
                          pendulum_small_angle_period, power_iteration, scale,
                          solve, spectral_radius, sub, trace, transpose,
                          wheeled_cart_deriv,
                          wheeled_cart_energy, wheeled_cart_linearize,
                          wheeled_cart_ode, zeros)
from bcl.sim import integrate_rk4, rk4_step


# ---------------------------------------------------------------------------
# 线性代数核心
# ---------------------------------------------------------------------------


def _mat_close(a, b, tol=1e-12):
    """矩阵逐元素比较。

    为什么不用 ``pytest.approx`` 直接比嵌套列表: 它只比较"容器长度 + 逐元素
    approx", 对不等长嵌套结构会抛 TypeError 而不是给出清晰断言失败。统一走
    Frobenius 范数差, 失败信息也是数值, 便于判断差了多少。
    """
    assert norm_fro(sub(a, b)) < tol


def test_matmul_identity():
    a = [[1.0, 2.0], [3.0, 4.0]]
    _mat_close(matmul(a, eye(2)), a, 1e-15)


def test_matmul_dimension_mismatch_raises():
    with pytest.raises(LinAlgError):
        matmul([[1.0, 2.0]], [[1.0, 2.0]])


def test_matvec_matches_matmul():
    a = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    x = [0.5, -1.0, 2.0]
    expected = [row[0] for row in matmul(a, [[v] for v in x])]
    assert matvec(a, x) == pytest.approx(expected, rel=1e-15)


def test_transpose_trace_and_add():
    a = [[1.0, 2.0], [3.0, 4.0]]
    _mat_close(transpose(a), [[1.0, 3.0], [2.0, 4.0]])
    assert trace(a) == pytest.approx(5.0)
    _mat_close(add(a, a), [[2.0, 4.0], [6.0, 8.0]])


def test_inverse_times_matrix_is_identity():
    a = [[4.0, 7.0, 2.0], [3.0, 6.0, 1.0], [2.0, 5.0, 3.0]]
    ai = inv(a)
    assert norm_fro(sub(matmul(a, ai), eye(3))) < 1e-12


def test_inverse_singular_raises():
    with pytest.raises(LinAlgError):
        inv([[1.0, 2.0], [2.0, 4.0]])


def test_norm_fro_known_value():
    assert norm_fro([[3.0, 0.0], [0.0, 4.0]]) == pytest.approx(5.0)


def test_solve_linear_system():
    a = [[2.0, 1.0], [1.0, 3.0]]
    x = solve(a, [3.0, 5.0])
    assert matvec(a, x) == pytest.approx([3.0, 5.0], rel=1e-12)


def test_zeros_and_eye_shapes():
    z = zeros(2, 3)
    assert len(z) == 2 and len(z[0]) == 3
    assert all(v == 0.0 for row in z for v in row)
    assert eye(3) == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def test_scale_multiplies_every_element():
    assert scale([[1.0, -2.0]], 3.0) == [[3.0, -6.0]]


def test_matvec_dimension_mismatch_raises():
    with pytest.raises(LinAlgError):
        matvec([[1.0, 2.0]], [1.0])


def test_expm_zero_is_identity():
    _mat_close(expm(zeros(3, 3)), eye(3), 1e-15)


def test_expm_diagonal_is_elementwise():
    a = [[1.0, 0.0], [0.0, -2.0]]
    got = expm(a)
    assert got[0][0] == pytest.approx(math.e, rel=1e-12)
    assert got[1][1] == pytest.approx(math.exp(-2.0), rel=1e-12)
    assert got[0][1] == pytest.approx(0.0, abs=1e-15)


def test_expm_semigroup_for_commuting_matrices():
    a = [[0.0, -1.0], [1.0, 0.0]]
    b = scale(eye(2), -0.3)
    _mat_close(matmul(expm(a), expm(b)), expm(add(a, b)), 1e-12)


def test_discretize_zoh_matches_augmented_expm():
    a = [[0.0, 1.0], [0.0, 0.0]]
    b = [[0.0], [1.0]]
    dt = 0.01
    ad, bd = discretize_zoh(a, b, dt)
    assert ad[0][0] == pytest.approx(1.0)
    assert ad[0][1] == pytest.approx(dt)
    assert ad[1][1] == pytest.approx(1.0)
    assert bd[0][0] == pytest.approx(0.5 * dt * dt, rel=1e-10)
    assert bd[1][0] == pytest.approx(dt)


def test_charpoly_of_diagonal_matrix():
    coeffs = charpoly([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
    # (l-1)(l-2)(l-3) = l^3 - 6l^2 + 11l - 6
    assert coeffs == pytest.approx([1.0, -6.0, 11.0, -6.0], rel=1e-12)


def test_eigvals_diagonal():
    got = sorted(z.real for z in eigvals([[3.0, 0.0], [0.0, -1.0]]))
    assert got == pytest.approx([-1.0, 3.0], abs=1e-10)


def test_eigvals_complex_pair():
    got = eigvals([[0.0, -1.0], [1.0, 0.0]])
    imags = sorted(z.imag for z in got)
    assert imags == pytest.approx([-1.0, 1.0], abs=1e-10)
    assert all(abs(z.real) < 1e-10 for z in got)


def test_spectral_radius_known():
    assert spectral_radius([[0.5, 0.0], [0.0, -0.9]]) == pytest.approx(0.9)


def test_power_iteration_needs_shift_for_symmetric_spectrum():
    """谱里出现 ``+w`` / ``-w`` 同模时, 不加移位幂迭代不收敛; 加移位才对。"""
    a = [[0.0, 1.0], [4.0, 0.0]]  # 特征值 +-2
    assert abs(abs(power_iteration(a)) - 2.0) > 1e-6
    assert power_iteration(a, shift=1.0) == pytest.approx(2.0, rel=1e-9)


# ---------------------------------------------------------------------------
# cart-pole
# ---------------------------------------------------------------------------


def test_cartpole_deriv_shape_and_zero_state():
    d = cartpole_deriv([0.0, 0.0, 0.0, 0.0], 0.0)
    assert len(d) == 4
    assert d == pytest.approx([0.0, 0.0, 0.0, 0.0], abs=1e-15)


def test_cartpole_deriv_upright_unstable_sign():
    """theta > 0 且无外力时, 角加速度必须为正(往倾倒方向加速)。

    取 ``theta = 1e-4`` 而不是 0.01: 非线性项 ``sin(theta)/theta = 1 - theta^2/6``
    在 0.01 rad 时还有 1.7e-5 的偏差, 会把"与解析式对账"的容差顶穿 —— 那不是
    实现错, 是**测试点选得太靠外**。这条本身也是一次线性化有效范围的校验。
    """
    p = CartPoleParams()
    theta = 1e-4
    d = cartpole_deriv([0.0, 0.0, theta, 0.0], 0.0)
    assert d[3] > 0.0
    assert d[3] == pytest.approx(cartpole_omega_sq(p) * theta, rel=1e-8)


def test_cartpole_hanging_equilibrium():
    """theta = pi(下垂)是稳定平衡点: 该处角加速度为 0。"""
    d = cartpole_deriv([0.0, 0.0, math.pi, 0.0], 0.0)
    assert d[3] == pytest.approx(0.0, abs=1e-14)


def test_cartpole_force_accelerates_cart():
    d = cartpole_deriv([0.0, 0.0, 0.0, 0.0], 10.0)
    p = CartPoleParams()
    assert d[1] == pytest.approx(10.0 / p.m_cart)
    assert d[3] == pytest.approx(-10.0 / (p.m_cart * p.l_pole))


def test_cartpole_deriv_never_divides_by_cos():
    """theta = 90 度时 cos = 0, 但方程仍必须良定义(这是显式消元的意义)。"""
    d = cartpole_deriv([0.0, 0.0, math.pi / 2, 0.0], 1.0)
    for v in d:
        assert math.isfinite(v)


def test_cartpole_energy_at_upright_rest():
    p = CartPoleParams()
    e = cartpole_energy([0.0, 0.0, 0.0, 0.0], p)
    assert e == pytest.approx(p.m_pole * p.g * p.l_pole)


def test_cartpole_energy_is_positive_definite_kinetic():
    p = CartPoleParams()
    e_rest = cartpole_energy([0.0, 0.0, 0.5, 0.0], p)
    e_move = cartpole_energy([0.0, 1.0, 0.5, 1.0], p)
    assert e_move > e_rest


def test_cartpole_energy_conservation_10s():
    """核心断言: 无外力无阻尼时 10 秒内相对能量漂移 < 1e-6。"""
    p = CartPoleParams()
    worst, _t, _e = energy_drift(
        cartpole_ode(0.0, p), [0.0, 0.0, 0.2, 0.0],
        lambda s: cartpole_energy(s, p), 10.0, 1e-3)
    assert worst < 1e-6


def test_cartpole_energy_drift_fourth_order_in_dt():
    """步长减半, 能量漂移按 4 阶下降(比值应远大于 4)。"""
    p = CartPoleParams()
    d1, _a, _b = energy_drift(cartpole_ode(0.0, p), [0.0, 0.0, 0.3, 0.0],
                              lambda s: cartpole_energy(s, p), 4.0, 2e-3)
    d2, _a, _b = energy_drift(cartpole_ode(0.0, p), [0.0, 0.0, 0.3, 0.0],
                              lambda s: cartpole_energy(s, p), 4.0, 1e-3)
    assert d1 / max(d2, 1e-18) > 8.0


def test_cartpole_linearize_matches_numeric_jacobian():
    p = CartPoleParams()
    a_an, b_an = cartpole_linearize(p)
    a_num, b_num = linearize(lambda x, u: cartpole_deriv(x, u, p),
                             [0.0, 0.0, 0.0, 0.0], 0.0)
    for i in range(4):
        for j in range(4):
            ref = a_an[i][j]
            if ref != 0.0:
                assert abs(a_an[i][j] - a_num[i][j]) / abs(ref) < 1e-6
        ref = b_an[i][0]
        if ref != 0.0:
            assert abs(b_an[i][0] - b_num[i][0]) / abs(ref) < 1e-6


def test_cartpole_numeric_jacobian_offdiag_are_zero():
    p = CartPoleParams()
    a_num, _b = linearize(lambda x, u: cartpole_deriv(x, u, p),
                          [0.0, 0.0, 0.0, 0.0], 0.0)
    assert a_num[0][1] == pytest.approx(1.0, abs=1e-9)
    assert a_num[2][3] == pytest.approx(1.0, abs=1e-9)
    assert a_num[1][1] == pytest.approx(0.0, abs=1e-9)
    assert a_num[3][3] == pytest.approx(0.0, abs=1e-9)


def test_cartpole_omega_sq_matches_closed_form():
    p = CartPoleParams(m_cart=1.0, m_pole=0.1, l_pole=0.5, g=9.81)
    assert cartpole_omega_sq(p) == pytest.approx(9.81 * 1.1 / 0.5, rel=1e-12)
    a, _b = cartpole_linearize(p)
    assert a[3][2] == pytest.approx(9.81 * 1.1 / 0.5, rel=1e-12)


def test_cartpole_linearized_frequency_from_time_simulation():
    """用仿真测出的**发散速率**与解析 ``sqrt(g(M+m)/(Ml))`` 对账(相对误差 < 1e-3)。

    倒立模态是不稳定的, 解是 ``theta(t) = theta0 * cosh(omega t)`` 而不是振荡 ——
    所以不能用"过零周期"去测(那是下垂摆的做法)。这里用增长到 2 倍所需的时刻
    ``t2 = arccosh(2)/omega`` 反推 omega。起值取 1e-5 rad 保证全程在线性区。
    """
    p = CartPoleParams()
    theta0 = 1e-5
    f = cartpole_ode(0.0, p)
    dt = 2e-4
    ts, xs = integrate_rk4(f, [0.0, 0.0, theta0, 0.0], 0.5, dt)
    t2 = None
    for t, s in zip(ts, xs):
        if s[2] >= 2.0 * theta0:
            t2 = t
            break
    assert t2 is not None and t2 > 0.0
    omega_sim = math.acosh(2.0) / t2
    omega_an = math.sqrt(cartpole_omega_sq(p))
    assert abs(omega_sim - omega_an) / omega_an < 1e-3


def test_cartpole_unstable_eigenvalue_positive():
    p = CartPoleParams()
    lam = cartpole_unstable_eig(p)
    assert lam > 0.0
    assert lam == pytest.approx(math.sqrt(cartpole_omega_sq(p)), rel=1e-9)


def test_cartpole_open_loop_has_positive_real_part():
    p = CartPoleParams()
    a, _b = cartpole_linearize(p)
    assert max(z.real for z in eigvals(a)) > 1.0


def test_cartpole_eigvals_pair_is_symmetric():
    p = CartPoleParams()
    a, _b = cartpole_linearize(p)
    lam = sorted(z.real for z in eigvals(a))
    assert lam[-1] == pytest.approx(-lam[0], abs=1e-6)
    assert lam[1] == pytest.approx(0.0, abs=1e-6)


def test_linearize_two_input_system():
    """多输入系统: B 的列数等于输入维数, 且与解析值一致。"""
    a_mat = [[0.0, 1.0], [-2.0, -3.0]]
    b_mat = [[0.0, 0.0], [1.0, 2.0]]
    f = lambda x, u: [a_mat[0][0] * x[0] + a_mat[0][1] * x[1] + b_mat[0][0] * u[0] + b_mat[0][1] * u[1],
                      a_mat[1][0] * x[0] + a_mat[1][1] * x[1] + b_mat[1][0] * u[0] + b_mat[1][1] * u[1]]
    a_num, b_num = linearize(f, [0.3, -0.2], [0.5, 0.7])
    assert norm_fro(sub(a_num, a_mat)) < 1e-9
    assert norm_fro(sub(b_num, b_mat)) < 1e-9


# ---------------------------------------------------------------------------
# 单摆
# ---------------------------------------------------------------------------


def test_pendulum_deriv_rest_at_bottom():
    assert pendulum_deriv([0.0, 0.0]) == pytest.approx([0.0, 0.0], abs=1e-15)


def test_pendulum_deriv_restoring_torque():
    """phi > 0(偏向上方一侧)时角加速度必须为负, 即回到最低点。"""
    d = pendulum_deriv([0.1, 0.0])
    assert d[1] < 0.0


def test_pendulum_torque_sign():
    p = PendulumParams()
    d = pendulum_deriv([0.0, 0.0], torque=1.0, p=p)
    assert d[1] == pytest.approx(1.0 / (p.m * p.l * p.l))


def test_pendulum_energy_at_bottom_is_zero():
    assert pendulum_energy([0.0, 0.0]) == pytest.approx(0.0, abs=1e-15)


def test_pendulum_energy_conservation():
    p = PendulumParams()
    worst, _t, _e = energy_drift(pendulum_ode(0.0, p), [0.3, 0.0],
                                 lambda s: pendulum_energy(s, p), 10.0, 1e-3)
    assert worst < 1e-6


def test_pendulum_small_angle_period_matches_simulation():
    """核心解析对账: 小角度仿真周期与 ``2*pi*sqrt(L/g)`` 相对误差 < 1%。"""
    p = PendulumParams(m=1.0, l=0.5, g=9.81)
    ts, xs = integrate_rk4(pendulum_ode(0.0, p), [0.02, 0.0], 20.0, 5e-4)
    period = estimate_period(ts, [s[0] for s in xs])
    assert period == pytest.approx(pendulum_small_angle_period(p), rel=1e-2)
    assert abs(period - pendulum_small_angle_period(p)) / pendulum_small_angle_period(p) < 0.01


def test_pendulum_period_scales_with_sqrt_length():
    """周期与 ``sqrt(L)`` 成正比(换长度再对账一次, 防止"凑巧对上")。"""
    p1 = PendulumParams(l=0.4)
    p2 = PendulumParams(l=1.6)
    ts, xs = integrate_rk4(pendulum_ode(0.0, p2), [0.01, 0.0], 20.0, 5e-4)
    period2 = estimate_period(ts, [s[0] for s in xs])
    assert period2 / pendulum_small_angle_period(p1) == pytest.approx(2.0, rel=1e-2)


def test_pendulum_damping_dissipates_energy():
    p = PendulumParams(damping=0.5)
    ts, xs = integrate_rk4(pendulum_ode(0.0, p), [0.5, 0.0], 10.0, 1e-3)
    e0 = pendulum_energy(xs[0], p)
    e1 = pendulum_energy(xs[-1], p)
    assert e1 < 0.1 * e0
    assert len(ts) == len(xs)


def test_estimate_period_requires_two_crossings():
    with pytest.raises(ValueError):
        estimate_period([0.0, 0.1, 0.2], [1.0, 0.5, 0.8])


def test_estimate_period_on_synthetic_sine():
    n = 4001
    dt = 1.0 / 1000.0
    ts = [i * dt for i in range(n)]
    xs = [math.sin(2.0 * math.pi * 3.0 * t) for t in ts]
    assert estimate_period(ts, xs) == pytest.approx(1.0 / 3.0, rel=1e-3)


# ---------------------------------------------------------------------------
# 双轮自平衡车
# ---------------------------------------------------------------------------


def test_wheeled_cart_deriv_rest_upright():
    d = wheeled_cart_deriv([0.0, 0.0, 0.0, 0.0], 0.0)
    assert d == pytest.approx([0.0, 0.0, 0.0, 0.0], abs=1e-15)


def test_wheeled_cart_upright_unstable_sign():
    d = wheeled_cart_deriv([0.0, 0.0, 0.01, 0.0], 0.0)
    assert d[3] > 0.0


def test_wheeled_cart_torque_drives_both_modes():
    d = wheeled_cart_deriv([0.0, 0.0, 0.0, 0.0], 1.0)
    assert d[1] > 0.0   # 正力矩推动车前进
    assert d[3] < 0.0   # 反作用力矩让车身向后倾


def test_wheeled_cart_energy_conservation():
    p = WheeledCartParams()
    worst, _t, _e = energy_drift(
        wheeled_cart_ode(0.0, p), [0.0, 0.0, 0.2, 0.0],
        lambda s: wheeled_cart_energy(s, p), 10.0, 5e-4)
    assert worst < 1e-6


def test_wheeled_cart_linearize_matches_numeric_jacobian():
    p = WheeledCartParams()
    a_an, b_an = wheeled_cart_linearize(p)
    a_num, b_num = linearize(lambda x, u: wheeled_cart_deriv(x, u, p),
                             [0.0, 0.0, 0.0, 0.0], 0.0)
    for i in range(4):
        for j in range(4):
            ref = a_an[i][j]
            if ref != 0.0:
                assert abs(a_an[i][j] - a_num[i][j]) / abs(ref) < 1e-6
        ref = b_an[i][0]
        if ref != 0.0:
            assert abs(b_an[i][0] - b_num[i][0]) / abs(ref) < 1e-6


def test_wheeled_cart_linearize_B_keeps_reaction_term():
    """B 的第 1 行必须含车身反作用项, 不能只写 ``1/(1.5 Mw r)``。

    只写第一项在轻车身时误差不明显, 车身一重就差 25% —— 这条断言就是为它设的。
    """
    p = WheeledCartParams()
    _a, b = wheeled_cart_linearize(p)
    naive = 1.0 / (1.5 * p.m_wheel * p.r_wheel)
    assert abs(b[1][0] - naive) > 0.1 * abs(naive)


def test_wheeled_cart_open_loop_unstable():
    p = WheeledCartParams()
    a, _b = wheeled_cart_linearize(p)
    assert max(z.real for z in eigvals(a)) > 0.1


def test_wheeled_cart_positive_definite_kinetic():
    p = WheeledCartParams()
    e0 = wheeled_cart_energy([0.0, 0.0, 0.0, 0.0], p)
    e1 = wheeled_cart_energy([0.0, 1.0, 0.0, 1.0], p)
    assert e1 > e0


def test_rk4_step_matches_exponential_for_linear_ode():
    """rk4_step 对 ``x' = x`` 的一步解应与 ``exp(h)`` 相差 O(h^5)。"""
    h = 1e-3
    x1 = rk4_step(lambda t, x: [x[0]], 0.0, [1.0], h)[0]
    assert abs(x1 - math.exp(h)) < 1e-15

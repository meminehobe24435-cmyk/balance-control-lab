#!/usr/bin/env python
"""独立于 pytest 的自检脚本。

**为什么要独立于 pytest**: 这个脚本可以在没有装 pytest 的环境里跑(部署机、CI 的
最小镜像、别人的机器)。它只检查**关键不变量** —— 那些一旦被破坏, 整个项目的结论
就全都不成立的东西。任何一项失败即以退出码 1 结束。

检查项(每一项都对应 README 里的一个数字或一段论证):

1.  cart-pole 能量守恒(10 s RK4, 相对漂移 < 1e-6)
2.  单摆小角度周期与 ``2*pi*sqrt(L/g)`` 相对误差 < 1%
3.  cart-pole 数值 Jacobian 与解析线性化逐元素相对误差 < 1e-6
4.  DARE 的 Riccati 残差 < 1e-6 且闭环谱半径 < 1
5.  S 曲线峰值速度/加速度/jerk 恰好等于限幅值
6.  轨迹端点位置/速度/加速度严格为 0
7.  PWM 扫遍全周期无上下桥臂直通
8.  编码器 16 位回绕后不回绕累计值连续
9.  关节雅可比解析式与数值中心差分一致, 且接近奇异时关节速度被放大
10. RK4 的收敛阶为 4(步长减半误差降约 16 倍)
11. `bcl/` 只 import 标准库(AST 扫描, 支持 pyproject 里"零运行时依赖"的承诺)

用法::

    python tools/selfcheck.py               # 跑全部
    python tools/selfcheck.py --deps-only   # 只跑依赖扫描(CI 的独立一步)
"""

from __future__ import annotations

import ast
import math
import os
import sys

# 允许 ``python tools/selfcheck.py`` 直接运行: 此时 sys.path[0] 是 tools/ 而不是
# 项目根目录, 所以要把根目录插进去才能 import bcl。
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from bcl.control import cartpole_lqr_gain  # noqa: E402
from bcl.dynamics import (CartPoleParams, PendulumParams, WheeledCartParams,  # noqa: E402
                          cartpole_deriv, cartpole_energy, cartpole_linearize,
                          cartpole_omega_sq, cartpole_ode, energy_drift,
                          estimate_period, linearize, pendulum_energy,
                          pendulum_ode, pendulum_small_angle_period,
                          wheeled_cart_energy, wheeled_cart_ode)
from bcl.driver import PwmChannel, QuadEncoder, SimRegisterBus  # noqa: E402
from bcl.kinematics import SingularityError, Planar2R  # noqa: E402
from bcl.sim import integrate_rk4, rk4_step  # noqa: E402
from bcl.trajectory import (QuinticProfile, SCurveProfile,  # noqa: E402
                            TrapezoidalProfile)
from bcl.dynamics import eigvals, norm_fro, sub  # noqa: E402


VERBOSE = "-v" in sys.argv or "--verbose" in sys.argv
_RESULTS = []


def check(name, condition, detail=""):
    """记录一项检查结果。``condition`` 必须是一个已经算好的布尔值。"""
    _RESULTS.append((name, bool(condition), detail))
    status = "PASS" if condition else "FAIL"
    line = "[%s] %-46s %s" % (status, name, detail)
    print(line, flush=True)
    return bool(condition)


def _rel_err_matrix(a, b):
    """逐元素相对误差的最大值(参考值为 0 时退化为绝对误差)。"""
    worst = 0.0
    for i in range(len(a)):
        for j in range(len(a[0])):
            ref = a[i][j]
            if abs(ref) < 1e-12:
                worst = max(worst, abs(a[i][j] - b[i][j]))
            else:
                worst = max(worst, abs(a[i][j] - b[i][j]) / abs(ref))
    return worst


def check_energy_conservation():
    """检查 1: 三个系统的能量守恒。"""
    cp = CartPoleParams()
    drift_cp, _t, _e = energy_drift(
        cartpole_ode(0.0, cp), [0.0, 0.0, 0.2, 0.0],
        lambda s: cartpole_energy(s, cp), 10.0, 1e-3)
    check("cart-pole 能量守恒 10s", drift_cp < 1e-6, "drift=%.3e" % drift_cp)

    pd = PendulumParams()
    drift_pd, _t, _e = energy_drift(
        pendulum_ode(0.0, pd), [0.3, 0.0],
        lambda s: pendulum_energy(s, pd), 10.0, 1e-3)
    check("单摆能量守恒 10s", drift_pd < 1e-6, "drift=%.3e" % drift_pd)

    wc = WheeledCartParams()
    drift_wc, _t, _e = energy_drift(
        wheeled_cart_ode(0.0, wc), [0.0, 0.0, 0.2, 0.0],
        lambda s: wheeled_cart_energy(s, wc), 10.0, 5e-4)
    check("双轮车能量守恒 10s", drift_wc < 1e-6, "drift=%.3e" % drift_wc)
    return drift_cp < 1e-6 and drift_pd < 1e-6 and drift_wc < 1e-6


def check_pendulum_period():
    """检查 2: 单摆小角度周期解析对账。"""
    pd = PendulumParams()
    ts, xs = integrate_rk4(pendulum_ode(0.0, pd), [0.02, 0.0], 20.0, 5e-4)
    period = estimate_period(ts, [s[0] for s in xs])
    analytic = pendulum_small_angle_period(pd)
    rel = abs(period - analytic) / analytic
    check("单摆周期解析对账 (<1%)", rel < 0.01,
          "sim=%.6f analytic=%.6f rel=%.2e" % (period, analytic, rel))
    return rel < 0.01


def check_linearization():
    """检查 3: 数值 Jacobian 与解析线性化对账, 并核对闭式频率。"""
    cp = CartPoleParams()
    a_an, b_an = cartpole_linearize(cp)
    a_num, b_num = linearize(lambda x, u: cartpole_deriv(x, u, cp),
                             [0.0, 0.0, 0.0, 0.0], 0.0)
    err_a = _rel_err_matrix(a_an, a_num)
    err_b = _rel_err_matrix(b_an, b_num)
    check("cart-pole 线性化 A 对账 (<1e-6)", err_a < 1e-6, "rel=%.3e" % err_a)
    check("cart-pole 线性化 B 对账 (<1e-6)", err_b < 1e-6, "rel=%.3e" % err_b)

    omega_closed = math.sqrt(cartpole_omega_sq(cp))
    omega_lin = math.sqrt(a_an[3][2])
    rel = abs(omega_closed - omega_lin) / omega_closed
    check("倒立点频率 = sqrt(g(M+m)/(Ml))", rel < 1e-3,
          "analytic=%.6f from A=%.6f rel=%.2e" % (omega_closed, omega_lin, rel))
    return err_a < 1e-6 and err_b < 1e-6 and rel < 1e-3


def check_lqr():
    """检查 4: Riccati 残差与闭环谱半径。"""
    cp = CartPoleParams()
    k, info = cartpole_lqr_gain(cp)
    residual = info["residual"]
    rho = info["spectral_radius"]
    check("Riccati 残差 < 1e-6", residual < 1e-6, "residual=%.3e" % residual)
    check("闭环谱半径 < 1", rho < 1.0, "rho=%.9f" % rho)
    check("DARE 迭代收敛标志", info["converged"],
          "iterations=%d" % info["iterations"])
    return residual < 1e-6 and rho < 1.0 and info["converged"]


def check_trajectory():
    """检查 5/6: S 曲线峰值对账与端点严格静止。"""
    tr = SCurveProfile(0.0, 1.5, 1.5, 6.0, 40.0)
    v_err = abs(tr.max_abs("velocity") - 1.5)
    a_err = abs(tr.max_abs("acceleration") - 6.0)
    j_err = abs(tr.max_abs("jerk") - 40.0)
    check("S 曲线峰值速度 = vmax", v_err < 1e-9, "err=%.3e" % v_err)
    check("S 曲线峰值加速度 = amax", a_err < 1e-9, "err=%.3e" % a_err)
    check("S 曲线峰值 jerk = jmax", j_err < 1e-9, "err=%.3e" % j_err)

    endpoints_ok = True
    detail = []
    for name, prof in (("梯形", TrapezoidalProfile(0.0, 1.0, 1.0, 2.0)),
                       ("S曲线", tr),
                       ("五次", QuinticProfile(0.0, 1.0, 1.0, 2.0))):
        worst = 0.0
        for t in (0.0, prof.duration):
            s = prof.sample(t)
            worst = max(worst, abs(s["velocity"]), abs(s["acceleration"]),
                        abs(s["jerk"]))
        detail.append("%s=%.1e" % (name, worst))
        if worst > 1e-12:
            endpoints_ok = False
    check("三条轨迹端点速度/加速度/jerk = 0", endpoints_ok, " ".join(detail))

    # 五次多项式不理会 jerk 限幅, 而 S 曲线严格满足
    quintic = QuinticProfile(0.0, 1.0, 1.0, 2.0)
    check("五次多项式 jerk 峰值 > 3x S 曲线 jmax",
          quintic.max_abs("jerk") > 3.0 * 5.0,
          "quintic=%.3f scurve_jmax=5.0" % quintic.max_abs("jerk"))
    return v_err < 1e-9 and a_err < 1e-9 and j_err < 1e-9 and endpoints_ok


def check_pwm_dead_time():
    """检查 7: PWM 全周期无直通, 且死区确实存在。"""
    bus = SimRegisterBus(size=256)
    pwm = PwmChannel(bus, base=0x40, channel=0)
    worst = []
    for duty in (-0.95, -0.5, -0.1, 0.0, 0.1, 0.5, 0.95, 1.0):
        pwm.set_duty(duty)
        hits = pwm.sweep_shoot_through()
        if hits:
            worst.append((duty, len(hits)))
    check("PWM 无上下桥臂直通", not worst, "shoot-through=%r" % (worst,))

    pwm.set_duty(0.5)
    ticks = pwm.duty_ticks()
    gap_ok = ticks > 0 and pwm.max_duty_ticks == pwm.period_ticks - pwm.dead_time_ticks
    check("PWM 死区确实存在且限住占空比", gap_ok,
          "duty_ticks=%d max=%d dead=%d" % (ticks, pwm.max_duty_ticks,
                                            pwm.dead_time_ticks))
    return not worst and gap_ok


def check_encoder_wrap():
    """检查 8: 编码器 16 位回绕后不回绕累计值连续。"""
    bus = SimRegisterBus(size=256)
    enc = QuadEncoder(bus, count_reg=0x20, ppr=1024)
    enc.seed_count(32766)
    for a, b in [(0, 1), (1, 1), (1, 0), (0, 0)]:
        enc.update(a, b)
    ok = (enc.count_raw == -32766 and enc.count_total == 32770
          and enc.wrap_count == 1)
    check("编码器回绕后累计值连续", ok,
          "raw=%d total=%d wraps=%d" % (enc.count_raw, enc.count_total,
                                        enc.wrap_count))

    enc2 = QuadEncoder(bus, count_reg=0x20, ppr=1024)
    forward = [(0, 1), (1, 1), (1, 0), (0, 0)]
    reverse = [(1, 0), (1, 1), (0, 1), (0, 0)]
    for a, b in forward:
        enc2.update(a, b)
    fwd_count = enc2.count_total
    for a, b in reverse:
        enc2.update(a, b)
    check("编码器正反一圈回到原点", enc2.count_total == 0,
          "forward=%d net=%d state=%d" % (fwd_count, enc2.count_total, enc2._state))
    return ok and enc2.count_total == 0


def check_kinematics():
    """检查 9: 雅可比对账与奇异放大。"""
    arm = Planar2R(1.0, 0.8)
    q = [0.4, 1.0]
    err = _rel_err_matrix(arm.jacobian(q), arm.jacobian_numeric(q))
    check("2R 雅可比解析 vs 数值 (<1e-6)", err < 1e-6, "rel=%.3e" % err)

    v_good = math.hypot(*arm.velocity_amplification((1.0, 0.0), math.pi / 2))
    v_bad = math.hypot(*arm.velocity_amplification((1.0, 0.0), 0.001))
    check("接近奇异时关节速度被放大", v_bad / v_good > 100.0,
          "q2=pi/2:%.3f q2=0.001:%.1f ratio=%.0f" % (v_good, v_bad, v_bad / v_good))

    raised = False
    try:
        arm.inverse_velocity([0.2, 0.0], [1.0, 0.0])
    except SingularityError:
        raised = True
    check("奇异位形求逆显式报错", raised, "SingularityError raised=%s" % raised)
    return err < 1e-6 and v_bad / v_good > 100.0 and raised


def check_rk4_order():
    """检查 10: RK4 收敛阶为 4。"""
    def exact(t):
        return 1.0 / (1.0 - t)

    def f(t, x):
        return [x[0] * x[0]]

    def err(dt):
        _ts, xs = integrate_rk4(f, [1.0], 0.5, dt)
        return abs(xs[-1][0] - exact(0.5))

    e1, e2, e3 = err(0.02), err(0.01), err(0.005)
    r1, r2 = e1 / e2, e2 / e3
    ok = 12.0 < r1 < 22.0 and 12.0 < r2 < 22.0
    check("RK4 收敛阶 = 4 (比值约 16)", ok,
          "r1=%.2f r2=%.2f" % (r1, r2))
    return ok


# 允许出现在 bcl/ 里的顶层模块名。手工列白名单而不是用
# ``sys.stdlib_module_names``: 后者是 Python 3.10 才有的, 而本项目要求 >= 3.8,
# 在一个 3.8 的解释器上跑这个自检时必须同样有效。
_ALLOWED_IMPORTS = frozenset([
    "__future__", "abc", "argparse", "array", "bisect", "cmath", "collections",
    "contextlib", "dataclasses", "enum", "functools", "heapq", "io", "itertools",
    "json", "math", "operator", "os", "random", "re", "statistics", "struct",
    "subprocess", "sys", "time", "typing", "unicodedata", "warnings",
])

# 明确点名禁止的第三方库(即使它们"看起来像标准库"或曾经是)。
_FORBIDDEN_IMPORTS = frozenset([
    "numpy", "scipy", "control", "sympy", "pandas", "matplotlib", "torch",
    "jax", "casadi", "cvxpy", "slycot", "nlopt", "sklearn", "numba", "cython",
])


def scan_imports(package_dir):
    """用 AST 扫出 ``package_dir`` 下所有模块 import 的**顶层包名**。

    为什么用 AST 而不是 ``grep '^import'``: 正则会把注释里的 "import numpy" 和
    字符串里的示例代码也算进去(本项目的 docstring 里就写了 numpy), 产生假阳性;
    AST 只看真正的 import 语句。

    为什么忽略相对导入(``from .dynamics import ...``): 包内互相引用本来就应该
    用相对导入, 它不是外部依赖。
    """
    found = {}
    for name in sorted(os.listdir(package_dir)):
        if not name.endswith(".py"):
            continue
        path = os.path.join(package_dir, name)
        with open(path, "r", encoding="utf-8") as fh:
            tree = ast.parse(fh.read(), filename=path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    found.setdefault(top, set()).add(name)
            elif isinstance(node, ast.ImportFrom):
                if node.level and node.level > 0:
                    continue
                if node.module:
                    top = node.module.split(".")[0]
                    found.setdefault(top, set()).add(name)
    return found


def check_no_third_party_imports():
    """检查 11: ``bcl/`` 只 import 标准库。"""
    pkg = os.path.join(_ROOT, "bcl")
    if not os.path.isdir(pkg):
        check("bcl/ 目录存在", False, pkg)
        return False
    imports = scan_imports(pkg)
    forbidden = sorted(set(imports) & _FORBIDDEN_IMPORTS)
    unknown = sorted(set(imports) - _ALLOWED_IMPORTS - _FORBIDDEN_IMPORTS)
    ok = not forbidden and not unknown
    check("bcl/ 无第三方依赖 (AST 扫描)", ok,
          "imports=%d forbidden=%s unknown=%s"
          % (len(imports), forbidden or "-", unknown or "-"))
    return ok


def check_numeric_derivative():
    """附加检查: 轨迹自带速度与中心差分一致(平滑段相对误差 < 1e-4)。"""
    from bcl.trajectory import numeric_derivative
    tr = SCurveProfile(0.0, 1.0, 1.0, 2.0, 10.0)
    dt = 1e-4
    data = tr.sample_dt(dt)
    numeric = numeric_derivative(data["t"], data["position"])
    bounds = tr.segment_boundaries()
    scale = 0.01 * tr.max_abs("velocity")
    worst = 0.0
    for i, t in enumerate(data["t"]):
        if any(abs(t - b) < 3.0 * dt for b in bounds):
            continue
        if abs(data["velocity"][i]) < scale:
            continue
        worst = max(worst, abs(numeric[i] - data["velocity"][i])
                    / abs(data["velocity"][i]))
    check("轨迹速度 vs 数值微分 (<1e-4)", worst < 1e-4, "worst=%.3e" % worst)
    return worst < 1e-4


def main():
    """跑全部检查, 任一失败返回 1。"""
    print("=" * 74)
    print("balance-control-lab 自检 (纯仿真, 无真实硬件)")
    print("=" * 74)
    if "--deps-only" in sys.argv:
        print("(仅依赖扫描模式)")
        check_no_third_party_imports()
        failed = [name for name, ok, _d in _RESULTS if not ok]
        if failed:
            print("FAILED: %s" % "; ".join(failed))
            return 1
        print("DEPENDENCY CHECK PASSED")
        return 0
    check_energy_conservation()
    check_pendulum_period()
    check_linearization()
    check_lqr()
    check_trajectory()
    check_pwm_dead_time()
    check_encoder_wrap()
    check_kinematics()
    check_rk4_order()
    check_numeric_derivative()
    check_no_third_party_imports()
    print("-" * 74)
    failed = [name for name, ok, _d in _RESULTS if not ok]
    total = len(_RESULTS)
    if failed:
        print("FAILED %d/%d: %s" % (len(failed), total, "; ".join(failed)))
        return 1
    print("ALL %d CHECKS PASSED" % total)
    return 0


if __name__ == "__main__":
    sys.exit(main())

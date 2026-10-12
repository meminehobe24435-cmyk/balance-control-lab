"""平衡控制器: LQR(自实现 DARE)、极点配置(Ackermann)、串级 PID。

三种控制器的定位不同, 这也是它们在这里共存的原因:

* **LQR** —— 有模型时的默认选择。把 cart-pole 的线性化模型 ``(A, B)`` 和一组
  权重 ``(Q, R)`` 变成唯一的最优增益 ``K``, 不需要手工凑极点。全状态反馈意味着
  位置也会被拉回原点, 这是单环 PID 做不到的(见 bench 的对比表)。
* **极点配置** —— 需要"精确指定闭环动态"时用。它不需要 Q/R 调参, 但会给出
  非常大的增益(代价见 ``ackermann`` 的说明), 工程上通常只用于验证 LQR 的量级。
* **串级 PID** —— 没有模型、只能在实机上试的时候用。本项目用它做**基线**, 用
  来量化"有模型比没模型到底好多少"。

一个反复踩到的坑(写在最前面): 离散 Riccati 迭代**不收敛时仍然会返回一个看起来
很正常的 K**。得到的闭环可能谱半径 1.02 —— 也就是"慢慢发散", 在 3 秒的仿真里
完全看不出来, 但实机上几十秒后一定翻车。所以本模块强制把
``riccati_residual`` 和闭环谱半径作为 LQR 的**返回值的一部分**, 而不是可选的
调试信息。
"""

from __future__ import annotations

import math

from .dynamics import (CartPoleParams, LinAlgError, add, cartpole_linearize,
                       discretize_zoh, eye, inv, matmul, matvec, norm_fro, scale,
                       spectral_radius, sub, transpose, zeros)


# ---------------------------------------------------------------------------
# 离散代数 Riccati 方程(DARE)
# ---------------------------------------------------------------------------


def riccati_residual(a, b, q, r, p):
    """DARE 残差的 Frobenius 范数。

        res = A'PA - P - A'PB (R + B'PB)^-1 B'PA + Q

    真解处恒为 0。它是判断"这个 K 到底能不能用"的唯一廉价且可靠的手段: 残差
    1e-12 说明 P 已经是数值不动点; 残差 1e-3 说明迭代还没收敛, 此时无论 K 看
    起来多合理都不能用。

    用 Frobenius 范数而不是最大元素: 它对"所有元素都略微不准"这种(迭代未收敛
    的典型症状)更敏感, 而且在正交变换下不变。
    """
    btp = matmul(transpose(b), p)
    btpb = matmul(btp, b)
    s = add(btpb, r)
    si = inv(s)
    btpa = matmul(btp, a)
    atp = matmul(transpose(a), p)
    atpa = matmul(atp, a)
    term = matmul(matmul(atp, b), matmul(si, btpa))
    return norm_fro(add(sub(sub(atpa, p), term), q))


def dare(a, b, q, r, max_iter=20000, tol=1e-13):
    """离散代数 Riccati 方程的不动点迭代。

    ``P_{k+1} = Q + A'P_k A - A'P_k B (R + B'P_k B)^-1 B'P_k A``

    收敛性: 对可稳定化/可检测的 (A, B, Q, R), 该迭代从 ``P_0 = Q`` 出发单调收敛
    到唯一的稳定化解。收敛速率由闭环谱半径的平方决定 —— 如果闭环谱半径 0.99,
    要收敛到 1e-13 需要约 ``log(1e-13)/log(0.98) ~ 1470`` 次迭代, 这也是把
    ``max_iter`` 给到两万的原因。

    为什么坚持自己写而不是调用现成的: 这是本项目"证明会做控制"的核心代码。而且
    自己实现才能把 ``converged`` 这个标志返回给调用方 —— 库函数通常直接返回 K,
    把"没收敛"这件事藏起来。

    返回 ``(K, P, iterations, converged)``。
    """
    n = len(a)
    m = len(b[0])
    if len(q) != n or len(q[0]) != n:
        raise LinAlgError("Q 必须是 %dx%d" % (n, n))
    if len(r) != m or len(r[0]) != m:
        raise LinAlgError("R 必须是 %dx%d" % (m, m))
    p = [list(row) for row in q]
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        btp = matmul(transpose(b), p)
        s = add(matmul(btp, b), r)
        k = matmul(inv(s), matmul(btp, a))
        atp = matmul(transpose(a), p)
        p_next = add(sub(matmul(atp, a), matmul(matmul(atp, b), k)), q)
        delta = norm_fro(sub(p_next, p))
        p = p_next
        if delta <= tol * (1.0 + norm_fro(p)):
            converged = True
            break
    btp = matmul(transpose(b), p)
    s = add(matmul(btp, b), r)
    k = matmul(inv(s), matmul(btp, a))
    return k, p, it, converged


def lqr(a, b, q, r, max_iter=20000, tol=1e-13):
    """LQR 增益, 返回 ``(K, info)``。

    ``info`` 是 dict, 含 ``residual``(Riccati 残差)、``spectral_radius``(闭环谱
    半径)、``iterations``、``converged``。**调用方应当检查 info** —— 这是本模块
    对"Riccati 不收敛会静默给出坏增益"这一风险的防御。
    """
    k, p, it, ok = dare(a, b, q, r, max_iter=max_iter, tol=tol)
    a_cl = sub(a, matmul(b, k))
    return k, {
        "residual": riccati_residual(a, b, q, r, p),
        "spectral_radius": spectral_radius(a_cl),
        "iterations": it,
        "converged": ok,
        "P": p,
    }


# ---------------------------------------------------------------------------
# 极点配置
# ---------------------------------------------------------------------------


def controllability_matrix(a, b):
    """可控性矩阵 ``[B, AB, A^2 B, ..., A^(n-1) B]``。"""
    n = len(a)
    blocks = [b]
    cur = b
    for _ in range(n - 1):
        cur = matmul(a, cur)
        blocks.append(cur)
    # 横向拼接
    rows = len(a)
    cols = sum(len(blk[0]) for blk in blocks)
    out = zeros(rows, cols)
    off = 0
    for blk in blocks:
        for i in range(rows):
            for j in range(len(blk[0])):
                out[i][off + j] = blk[i][j]
        off += len(blk[0])
    return out


def is_controllable(a, b, tol=1e-9):
    """可控性判据(用可控性矩阵的秩, 朴素高斯消元计数)。

    可控 = 想把闭环极点放到任意位置都做得到。不可控时 Ackermann 公式里的 C
    是奇异的, ``inv`` 会抛 LinAlgError —— 这是**正确行为**, 比返回一个巨大的
    垃圾增益好得多。
    """
    c = controllability_matrix(a, b)
    return matrix_rank(c, tol) == len(a)


def matrix_rank(m, tol=1e-9):
    """数值秩: 行阶梯化后数非零主元(带部分选主元)。"""
    rows = [list(r) for r in m]
    nr = len(rows)
    nc = len(rows[0])
    rank = 0
    for col in range(nc):
        piv = max(range(rank, nr), key=lambda r: abs(rows[r][col]))
        if abs(rows[piv][col]) <= tol:
            continue
        rows[rank], rows[piv] = rows[piv], rows[rank]
        pv = rows[rank][col]
        for r in range(nr):
            if r == rank:
                continue
            factor = rows[r][col] / pv
            if factor == 0.0:
                continue
            for j in range(col, nc):
                rows[r][j] -= factor * rows[rank][j]
        rank += 1
        if rank == nr:
            break
    return rank


def poly_from_roots(roots):
    """由根展开特征多项式系数(降幂, 首项为 1)。"""
    coeffs = [1.0]
    for r in roots:
        new = [0.0] * (len(coeffs) + 1)
        for i, c in enumerate(coeffs):
            new[i] += c
            new[i + 1] -= c * r
        coeffs = new
    return coeffs


def polyvalm(coeffs, a):
    """矩阵多项式 ``p(A)``, Horner 法。"""
    n = len(a)
    result = scale(eye(n), coeffs[0])
    for c in coeffs[1:]:
        result = add(matmul(result, a), scale(eye(n), c))
    return result


def ackermann(a, b, poles):
    """Ackermann 极点配置: 求 K 使 ``eig(A - B K) = poles``。

        K = [0 ... 0 1] * C^-1 * phi(A),  C = [B AB ... A^(n-1)B]

    ``phi`` 是期望闭环特征多项式。离散与连续形式相同(都是"配置 A - BK 的谱"),
    所以这一个函数两个场景都能用。

    代价/边界(必须说清楚):
      * 只适用于**单输入**系统。多输入要用 Kautsky-Nichols 或直接解 Sylvester
        方程, 本项目没有实现。
      * Ackermann 公式数值条件数很差: ``C^-1`` 的元素量级是 ``1/det(C)``。极点
        放得越激进(越靠原点), C 越病态, 增益会大 3~5 个数量级。实机上这种增益
        会立刻被执行器饱和吃掉 —— 所以本项目用 LQR 跑闭环, 极点配置只作为
        "能精确指定极点"的能力证明, 并在测试里额外断言 K 的范数是有限的。
      * 系统必须可控, 否则 ``inv(C)`` 显式抛 ``LinAlgError``。
    """
    n = len(a)
    if len(poles) != n:
        raise LinAlgError("极点个数 %d 必须等于状态维数 %d" % (len(poles), n))
    c = controllability_matrix(a, b)
    if len(c[0]) != n:
        raise LinAlgError("ackermann 只支持单输入系统(状态维数 %d)" % n)
    ci = inv(c)
    coeffs = poly_from_roots(poles)
    phi_a = polyvalm(coeffs, a)
    selector = [0.0] * n
    selector[-1] = 1.0
    return matmul([selector], matmul(ci, phi_a))


# ---------------------------------------------------------------------------
# PID
# ---------------------------------------------------------------------------


class PID:
    """单环 PID, 带积分抗饱和与输出限幅。

    形式: ``u = Kp*e + Ki*integral(e) + Kd*d``

    关于微分项 ``d`` 有两种口径, 用 ``measurement`` 参数选择:

    * ``measurement=None``: ``d = de/dt``, 纯误差微分。设定值恒定(调平衡)时它
      是正确选择。
    * 传 ``measurement``: ``d = -d(meas)/dt``, 即"微分作用在测量值上"。这是
      设定值会跳变时(位置伺服)必须用的形式, 否则设定值阶跃瞬间的误差导数会
      产生 ``Kd*step/dt`` 的尖峰 —— dt=1 ms、阶跃 1.0 时是 Kd 的 1000 倍, 直接
      把输出打到饱和。这就是经典的 derivative kick 问题。

    另外两个必须做对的地方:

    * **积分抗饱和**: 输出已经顶到限幅、而误差还在把积分往同方向推时停止积分。
      否则放开限幅的瞬间, 积分器里攒的"欠账"会造成一大截反向超调。这里用条件
      积分(conditional integration)而不是反算(back-calculation), 前者不需要
      额外的跟踪时间常数, 行为更容易解释。
    * **输出限幅**在最后一步做, 且限幅值必须和执行器的真实上限一致 —— 否则
      抗饱和判断本身就是错的。

    边界: 没有设定值加权、没有增益调度。对强非线性对象(比如 cart-pole 的大角度
    起摆)单环 PID 必然不够, 这不是调参能解决的。
    """

    def __init__(self, kp, ki=0.0, kd=0.0, out_min=None, out_max=None,
                 deriv_filter_tau=0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.out_min = out_min
        self.out_max = out_max
        self.deriv_filter_tau = deriv_filter_tau
        self.reset()

    def reset(self):
        """清零积分器与微分历史。切换控制模式时必须调用。"""
        self.integral = 0.0
        self.prev_error = None
        self.prev_meas = None
        self.d_est = 0.0
        self.last_output = 0.0
        self.saturated = False
        self.last_terms = (0.0, 0.0, 0.0)

    def _clamp(self, value):
        hi = self.out_max if self.out_max is not None else math.inf
        lo = self.out_min if self.out_min is not None else -math.inf
        if value > hi:
            return hi, True
        if value < lo:
            return lo, True
        return value, False

    def step(self, error, dt, measurement=None):
        """推进一拍, 返回限幅后的输出。"""
        if dt <= 0:
            raise ValueError("dt 必须为正: %r" % (dt,))
        if measurement is None:
            d_raw = (0.0 if self.prev_error is None
                     else (error - self.prev_error) / dt)
            self.prev_error = error
        else:
            d_raw = (0.0 if self.prev_meas is None
                     else -(measurement - self.prev_meas) / dt)
            self.prev_meas = measurement

        tau = self.deriv_filter_tau
        if tau > 0.0:
            self.d_est += (d_raw - self.d_est) * (dt / (tau + dt))
        else:
            self.d_est = d_raw

        p_term = self.kp * error
        d_term = self.kd * self.d_est
        unclamped = p_term + self.integral + d_term

        # 条件积分: 已饱和且误差还想往同方向推时, 不积分
        integrate = True
        if self.ki != 0.0:
            hi = self.out_max if self.out_max is not None else math.inf
            lo = self.out_min if self.out_min is not None else -math.inf
            push = self.ki * error
            if unclamped > hi and push > 0.0:
                integrate = False
            elif unclamped < lo and push < 0.0:
                integrate = False
        if integrate:
            self.integral += self.ki * error * dt

        p_term = self.kp * error
        unclamped = p_term + self.integral + d_term
        clamped, saturated = self._clamp(unclamped)
        self.saturated = saturated
        self.last_output = clamped
        self.last_terms = (p_term, self.integral, d_term)
        return clamped

    def terms(self):
        """返回最近一拍的 ``(P, I, D)`` 三项贡献, 便于诊断谁在主导。"""
        return self.last_terms


class CascadePID:
    """串级 PID: 位置环(外) -> 速度环(内)。

    为什么串级而不是单环: 平衡/伺服系统的执行器几乎总是和速度相关(反电动势、
    摩擦), 单环位置 PID 要靠积分硬顶, 响应会很慢且容易振荡。把速度环放在内层,
    外层只需要输出一个速度指令, 内层的带宽更高、抗扰更快。

    外环输出经 ``v_max`` 限幅后才交给内环 —— 这就是"轨迹整形"的最简形式: 限制
    期望速度等于限制了加速度, 避免内环被一个阶跃的速度指令打饱和。

    边界: 没有做前馈(``v_ref`` 和它的微分可以直接前馈到内环, 能把跟踪误差再降
    一个量级)。本项目刻意保留纯反馈形式, 因为 bench 里对比的对象也是纯反馈。
    """

    def __init__(self, outer, inner, v_max):
        self.outer = outer
        self.inner = inner
        self.v_max = v_max
        # 最近一拍交给内环的速度指令(限幅后)。暴露出来是为了能直接断言
        # "外环输出的速度指令确实被 v_max 顶住了", 而不用去猜内环的输入。
        self.last_v_cmd = 0.0

    def reset(self):
        """复位内外两个 PID。"""
        self.outer.reset()
        self.inner.reset()
        self.last_v_cmd = 0.0

    def step(self, pos_ref, pos_meas, vel_meas, dt, vel_ref=0.0):
        """推进一拍, 返回控制量。

        ``vel_ref`` 是**速度前馈**: 外环输出与它相加后作为内环的速度指令。
        没有速度前馈时, 外环只能靠"位置误差"造出速度指令, 于是跟踪斜坡参考
        (匀速段)必然留下 ``e = v / Kp_pos`` 的滞后误差 —— 这是纯串级 P-PI 的
        结构性缺陷, 不是调参能消掉的。加上 ``vel_ref`` 后滞后误差立刻归零。
        代价: 前馈依赖参考速度的质量, 参考速度上的噪声会被原样注入内环。
        """
        v_cmd = self.outer.step(pos_ref - pos_meas, dt, measurement=pos_meas)
        v_cmd += vel_ref
        if v_cmd > self.v_max:
            v_cmd = self.v_max
        elif v_cmd < -self.v_max:
            v_cmd = -self.v_max
        self.last_v_cmd = v_cmd
        return self.inner.step(v_cmd - vel_meas, dt, measurement=vel_meas)


# ---------------------------------------------------------------------------
# 面向 cart-pole 的成品控制器
# ---------------------------------------------------------------------------


def cartpole_lqr_gain(params=None, q_diag=(20.0, 4.0, 100.0, 10.0), r=0.005,
                      control_dt=0.01, max_iter=20000):
    """为 cart-pole 算离散 LQR 增益 ``K``, 返回 ``(K, info)``。

    默认权重 ``Q = diag(20, 4, 100, 10)`` 的含义: 最在乎倾角(100), 其次是角速度
    (10), 再次是位置(20)与速度(4) —— "站稳"优先于"回到原点"。``R = 0.005``
    允许较大的控制力度。

    为什么是这几个数: 它们给出闭环谱半径 0.9827、稳定时间 2.9 s、控制量峰值
    11.4 N(限幅 20 N 的 57%)。换权重会得到"更快但更抖"或"更省力但更慢"的解 ——
    LQR 的权重没有唯一正确答案, 所以在 README 里写清实测数字比声称"最优"更有
    意义。

    注意闭环谱半径**始终接近 1**(实测 0.98~0.99): cart-pole 的位置模态本质上
    是"必须先倾斜才能移动"的双积分器, 权重怎么调都挪不快。这不是实现 bug, 是
    系统的物理结构决定的 —— 也解释了为什么"位置收敛比倾角收敛慢得多"。
    """
    if params is None:
        params = CartPoleParams()
    a, b = cartpole_linearize(params)
    ad, bd = discretize_zoh(a, b, control_dt)
    q = zeros(4, 4)
    for i, v in enumerate(q_diag):
        q[i][i] = v
    r_mat = [[float(r)]]
    return lqr(ad, bd, q, r_mat, max_iter=max_iter)


class LQRController:
    """把状态反馈 ``u = -K (x - x_ref)`` 包成 ``controller(t, x) -> u``。"""

    def __init__(self, k, x_ref=None, u_limit=20.0):
        self.k = [list(row) for row in k]
        self.x_ref = list(x_ref) if x_ref is not None else [0.0] * len(k[0])
        self.u_limit = u_limit
        self.last_u = 0.0

    def reset(self):
        """清掉缓存的上一拍输出。"""
        self.last_u = 0.0

    def __call__(self, t, x):
        err = [x[i] - self.x_ref[i] for i in range(len(self.x_ref))]
        u = -matvec(self.k, err)[0]
        if self.u_limit is not None:
            if u > self.u_limit:
                u = self.u_limit
            elif u < -self.u_limit:
                u = -self.u_limit
        self.last_u = u
        return u


class SwingUpController:
    """能量法起摆(Åström-Furuta): 把摆从下垂位置"荡"到竖直。

    为什么不能用 LQR 起摆: LQR 是**局部**线性化控制器, 只在倒立点附近有效。
    从 theta = pi(下垂)出发时线性化模型完全失效(线性化点的吸引域大约只有
    |theta| < 0.5 rad), LQR 会给出一个方向错乱的控制量, 摆根本起不来。

    能量法的核心恒等式: 定义摆相对铰点的"机械能"(**势能零点取竖直向上的位置**)

        E_p = 1/2 m l^2 theta'^2 + m g l (cos(theta) - 1)
        dE_p/dt = -m l * x'' * theta' * cos(theta)

    于是 ``E_p`` 在下垂静止时取最小值 ``-2 m g l``, 在竖直静止时取 0 —— 所以
    **目标能量就是 0**。取小车期望加速度

        x''_des = -K * (0 - E_p) * sign(theta' * cos(theta)),

    就有 ``dE_p/dt = +m l K |E_p| |theta' cos(theta)|``, 即能量单调趋向 0。

    这里有两个极易写反的地方, 而且写反之后系统**看起来仍然在动**(摆会被摇得更低
    或者狂转), 不会立刻报错:
      * 势能零点。用 ``m g l cos(theta)``(下为 0、上为 2mgl)和用
        ``m g l (cos(theta) - 1)``(下为 -2mgl、上为 0)是等价的两套坐标, 但
        **目标值一个是 2mgl、一个是 0**。混用会让控制器朝一个错误的能量目标
        猛泵, 现象是"摆越摇越快最后开始转圈"。本项目统一用后一种。
      * 那个负号。测试用"起摆若干秒后 |E_p| 收敛到很小"这条物理断言来守住它,
        而不是靠肉眼读符号。

    **这个推导只用到了一个标量恒等式, 不需要线性化也不需要最优控制** —— 这也是
    它比"反向倒立摆 LQR 加个大范围"更可靠的原因。

    为什么能量对上就自然会停在顶上: ``E_p = E_p*`` 且 ``theta = 0`` 时必然
    ``theta' = 0``(此时势能已经吃满目标能量), 所以摆会在通过顶点的瞬间恰好静止
    —— 这正是切换到 LQR 的最佳时机。

    额外加的位置反馈 ``-kp*x - kd*x'``: 没有它小车会被能量泵一路推到边界(能量法
    只关心摆, 不关心小车在哪)。代价是这个反馈会略微干扰能量泵的相位, 所以增益
    必须远小于能量项 —— 这是个实际调参里的取舍, 不是理论上的必然。

    边界: 只对小车的**力**输入有效。真实小车有速度上限和轨道长度限制, 需要额外
    的"防撞墙"逻辑; 本项目不涉及。
    """

    def __init__(self, params=None, gain=6.0, cart_kp=0.6, cart_kd=1.2,
                 energy_margin=0.10, u_limit=20.0):
        if params is None:
            params = CartPoleParams()
        self.params = params
        self.gain = gain
        self.cart_kp = cart_kp
        self.cart_kd = cart_kd
        self.energy_margin = energy_margin
        self.u_limit = u_limit
        self.last_energy_error = 0.0

    def reset(self):
        """清掉缓存的能量误差。"""
        self.last_energy_error = 0.0

    def pendulum_energy(self, x):
        """摆相对铰点的机械能 ``E_p``。

        势能零点取**竖直向上**的位置, 于是下垂静止为 ``-2 m g l``、竖直静止为 0。
        """
        p = self.params
        th, thd = x[2], x[3]
        return (0.5 * p.m_pole * p.l_pole ** 2 * thd * thd
                + p.m_pole * p.g * p.l_pole * (math.cos(th) - 1.0))

    def target_energy(self):
        """目标能量 ``energy_margin * 2 m g l``。

        为什么不取严格的 0(竖直静止): ``E_p = 0`` 是一条**同宿轨道**(鞍点上的
        分界线)。能量精确等于 0 时, 摆会沿这条轨道**渐近**趋向顶点, 永远到不了
        —— 数学上要无穷长时间。所以必须把目标定得比 0 略高一点, 让摆带着一点
        残余角速度真正越过顶点, 再交给 LQR 捕获。

        ``energy_margin`` 就是"越过顶点时的残余动能占 ``2 m g l`` 的比例"。它直接
        决定捕获瞬间的 ``theta'``: ``theta' = sqrt(2 * margin * 2 g l)/l``。
        取 0.10 时约 2.8 rad/s —— 刚好落在 LQR 的吸引域里。取得太小则摆越不过
        顶点(要等很久), 取得太大则 LQR 接不住(摆会绕过去继续转)。
        """
        p = self.params
        return self.energy_margin * 2.0 * p.m_pole * p.g * p.l_pole

    def __call__(self, t, x):
        """``controller(t, x) -> F``。"""
        p = self.params
        e = self.pendulum_energy(x)
        e_target = self.target_energy()
        self.last_energy_error = e_target - e
        trigger = x[3] * math.cos(x[2])
        if abs(trigger) < 1e-9:
            trigger = 1e-9
        # 负号来自 dE_p/dt = -m l a (theta' cos theta), 见类 docstring
        x_dd = -self.gain * self.last_energy_error * (1.0 if trigger > 0 else -1.0)
        x_dd -= self.cart_kp * x[0] + self.cart_kd * x[1]
        u = p.m_cart * x_dd
        if u > self.u_limit:
            u = self.u_limit
        elif u < -self.u_limit:
            u = -self.u_limit
        return u


class SwingUpBalanceController:
    """起摆 -> 平衡 的切换控制器, 组成完整的"从上电到站稳"流程。

    切换判据(带滞回, 这是必须的): ``|theta| < theta_switch`` 且
    ``|theta'| < rate_switch`` 时切到 LQR; ``|theta| > theta_exit`` 时切回起摆。
    没有滞回会在阈值附近每拍来回抖, 控制量变成高频方波 —— 实机上直接听得到异响。

    ``theta_exit > theta_switch`` 是滞回带宽。两个阈值之间的区域是"谁在管都行"
    的模糊带, 让状态机在进入后不会因为一点点噪声退出。

    边界: 这是一个**两个局部控制器 + 硬切换**的方案。它没有全局稳定性证明;
    LQR 的吸引域边界和起摆的能量收敛域是否覆盖整个状态空间, 本项目没有做严格的
    吸引域分析(那需要 SOS/可达集工具), 只通过大量初值仿真验证。
    """

    MODE_SWING_UP = "swing_up"
    MODE_BALANCE = "balance"

    def __init__(self, k, params=None, theta_switch=0.50, rate_switch=5.0,
                 theta_exit=0.90, u_limit=20.0):
        self.params = params if params is not None else CartPoleParams()
        self.swing = SwingUpController(self.params, u_limit=u_limit)
        self.balance = LQRController(k, u_limit=u_limit)
        self.theta_switch = theta_switch
        self.rate_switch = rate_switch
        self.theta_exit = theta_exit
        self.u_limit = u_limit
        self.mode = self.MODE_SWING_UP
        self.switch_log = []
        self._prev_t = None

    def reset(self):
        """复位到起摆模式。"""
        self.swing.reset()
        self.balance.reset()
        self.mode = self.MODE_SWING_UP
        self.switch_log = []
        self._prev_t = None

    def _switch(self, t, mode):
        self.mode = mode
        self.switch_log.append((t, mode))

    def __call__(self, t, x):
        """``controller(t, x) -> F``, 内部维护模式机。"""
        theta, theta_dot = x[2], x[3]
        if self.mode == self.MODE_SWING_UP:
            if abs(theta) < self.theta_switch and abs(theta_dot) < self.rate_switch:
                self._switch(t, self.MODE_BALANCE)
                self.balance.reset()
        else:
            if abs(theta) > self.theta_exit:
                self._switch(t, self.MODE_SWING_UP)
                self.swing.reset()
        self._prev_t = t
        if self.mode == self.MODE_BALANCE:
            return self.balance(t, x)
        return self.swing(t, x)

    def capture_time(self):
        """首次切到平衡模式的时刻, 没切过返回 None。"""
        for t, mode in self.switch_log:
            if mode == self.MODE_BALANCE:
                return t
        return None

    def fallback_count(self):
        """从平衡退回起摆的次数(衡量切换鲁棒性)。"""
        return max(0, len(self.switch_log) - 1) // 2


class AnglePIDController:
    """只针对摆角的单环 PID(基线控制器)。

    **符号说明(最容易搞反的地方)**: 对 cart-pole 的线性化模型
    ``theta'' = w^2*theta - F/(M l)`` 而言, 输入 F 到倾角 theta 的直流增益是
    **负的** —— 给小车的正方向力会把摆杆"追"回竖直方向。所以稳定它的反馈增益
    必须是**正**的: ``F = Kp*theta + Kd*theta'``。这和"位置环增益为正"的直觉
    相反, 却是 cart-pole 最经典的符号坑。

    因此这里以 ``theta`` 本身作为误差信号, 且不传 ``measurement``(用误差微分,
    得到 ``+Kd*theta'`` 而不是 ``-Kd*theta'``)。

    用它做对比的意义在于暴露**单环反馈的结构性缺陷**: 摆角可以被压回 0, 但小车
    位置是"零动态"里的一个自由模态, 没有位置反馈时它会漂走并且不回来。PID 看
    起来稳住了摆杆, 小车却已经开出去很远 —— 这正是 LQR 值得存在的理由。

    边界: 它没有显式模型, 所以增益需要手调; 本项目的默认增益是按线性化方程反算
    的, 相当于**用了模型**来调 PID —— 这对 PID 已经是很优待的条件了。
    """

    def __init__(self, kp=40.0, ki=0.0, kd=8.0, u_limit=20.0, deriv_filter_tau=0.004):
        self.pid = PID(kp, ki, kd, out_min=-u_limit, out_max=u_limit,
                       deriv_filter_tau=deriv_filter_tau)
        self._prev_t = None

    def reset(self):
        """复位内层 PID 与时间戳。"""
        self.pid.reset()
        self._prev_t = None

    def __call__(self, t, x):
        """``controller(t, x) -> F``。周期从相邻两次调用的时间差推断。"""
        if self._prev_t is None:
            dt = 0.01
        else:
            dt = t - self._prev_t
            if dt <= 0.0:
                dt = 0.01
        self._prev_t = t
        return self.pid.step(x[2], dt)

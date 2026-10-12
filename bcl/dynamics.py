"""拉格朗日动力学建模 + 零依赖线性代数核心。

本模块是 balance-control-lab 的物理与数学地基。它回答两个问题:

1. **系统怎么动** —— 用拉格朗日方程推导 cart-pole(小车+摆杆)、单摆、双轮
   自平衡车(简化)三个系统的运动方程, 并把它们写成 ``dxdt`` 形式供积分器调用。
2. **在平衡点附近怎么近似** —— 数值 Jacobian(中心差分)与解析线性化的对账,
   以及"这个平衡点稳不稳"的特征值判据。

为什么用拉格朗日而不是牛顿-欧拉
--------------------------------
这是本项目最重要的一个设计决定。牛顿-欧拉需要对每个刚体列受力/力矩平衡、
手工消去约束力(小车与地面的法向力、摆杆的铰链反力)。小系统里这些约束力
没有物理意义, 却占了推导的一大半, 而且一旦符号错了, 仿真"看起来还能跑",
只是能量慢慢漂移 —— 极难察觉。

拉格朗日法的产物是标量函数 ``L = T - V``, 于是:

* **能量项可以直接对账**。``T`` 与 ``V`` 都是我们亲手写的表达式, 总能量守恒
  就成了一条可执行的测试(见 ``cartpole_energy`` / ``energy_drift``)。
  方程写错时能量必然不守恒, 测试立刻红。牛顿-欧拉没有这么便宜的检查。
* **线性化有闭式解可以硬对账**。cart-pole 在倒立点线性化后
  ``omega^2 = g(M+m)/(Ml)`` 是教科书闭式结果, 数值 Jacobian 与解析 Jacobian
  可以逐元素比到 1e-6, 这等价于对非线性方程本身做了一次独立复核。
* **加自由度是机械劳动**。多一个关节只是多一个广义坐标, 不用重新做受力分析。

代价/边界(必须说清楚):

* 拉格朗日法对**非完整约束**(轮子打滑、接触切换)不友好, 那类问题要回到
  Newton-Euler 或 Gauss 原理。本项目所有系统都简化为完整约束(轮子纯滚动),
  所以避开了这个坑。
* 解析求导容易出错, 所以本模块**不手工展开二阶导** —— 见下面的推导注记。
* 纯 Python 矩阵运算: 4x4 的 DARE 迭代、5x5 的矩阵指数都还好, 但自由度上到
  6 以上(比如带 float base 的机械臂)就会有明显可感的变慢, 这是"零依赖"的代价。

三个系统的建模口径(为什么这样简化)
----------------------------------
* **cart-pole**: 摆杆 = 无质量轻杆 + 末端集中质量 ``m``, 集中质量距铰点 ``l``。
  之所以不用均匀细杆(那要带转动惯量 ``I = m*l^2/3``), 是因为只有集中质量模型
  才有 ``omega^2 = g(M+m)/(Ml)`` 这个闭式频率, 能和解析式对账。代价: 丢掉了杆
  自身的转动惯量, 与真实实物的频率会有几个百分点的差异。
* **单摆**: 角度 ``phi`` 从**竖直向下**量起(稳定平衡点), 因为待对账的
  ``T = 2*pi*sqrt(L/g)`` 是小角度下绕**稳定**平衡点的周期; 若从向上量起,
  小角度下是指数发散, 没有周期可言。
* **双轮自平衡车**: 车身 = 质量 ``m``、质心在轮轴上方 ``l`` 的倒立摆; 轮子 =
  质量 ``Mw``、半径 ``r`` 的均匀圆盘, 纯滚动(无打滑)。输入取轮轴电机力矩
  ``tau``。代价: 没有转向自由度、没有轮胎柔性/滑移、没有摩擦耗散, 所以它是
  "能不能平衡"的定性模型, 不是可标定的数字孪生。

单位统一为 SI(m, kg, s, rad, N, N*m)。
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# 零依赖线性代数核心
# ---------------------------------------------------------------------------
# 为什么把矩阵工具放在 dynamics.py 而不是单独开一个 linalg 模块?
# 因为本项目刻意规定了一个极简的目录结构(见 README 的模块地图), 而矩阵规模
# 全程 <= 6x6。多一个模块带来的 import 跳转成本大于收益, 所以这里只保留
# 一个"够用就好"的实现, 并被 control.py / estimator.py 复用同一份数值代码
# —— 两份独立的求逆实现是数值 bug 的常见来源。
#
# 约定: 矩阵 = 行列表 ``[[...], [...]]``, 向量 = 一维列表。
# 边界: 全部是稠密 O(n^3) 实现, n > 6 之后纯 Python 开销明显(求逆 6x6 已经
# 是几千次浮点乘加), 不适合再往上扩。


class LinAlgError(ValueError):
    """矩阵维数不匹配或矩阵奇异。显式抛出而不是返回 nan, 便于定位。"""


def zeros(n, m):
    """零矩阵 n x m。"""
    return [[0.0] * m for _ in range(n)]


def eye(n):
    """单位矩阵 n x n。"""
    out = zeros(n, n)
    for i in range(n):
        out[i][i] = 1.0
    return out


def matmul(a, b):
    """稠密矩阵乘法 A @ B(外两层循环 + 内层累加, 对小矩阵最快)。"""
    n, k, m = len(a), len(b), len(b[0])
    if len(a[0]) != k:
        raise LinAlgError("matmul: 内维不匹配 %d != %d" % (len(a[0]), k))
    out = zeros(n, m)
    for i in range(n):
        oi = out[i]
        ai = a[i]
        for p in range(k):
            aip = ai[p]
            if aip == 0.0:
                continue
            bp = b[p]
            for j in range(m):
                oi[j] += aip * bp[j]
    return out


def matvec(a, x):
    """矩阵-向量乘法 A @ x。"""
    if len(a[0]) != len(x):
        raise LinAlgError("matvec: 维数不匹配")
    return [sum(a[i][j] * x[j] for j in range(len(x))) for i in range(len(a))]


def transpose(a):
    """转置。"""
    return [list(col) for col in zip(*a)]


def add(a, b):
    """逐元素相加。"""
    return [[a[i][j] + b[i][j] for j in range(len(a[0]))] for i in range(len(a))]


def sub(a, b):
    """逐元素相减。"""
    return [[a[i][j] - b[i][j] for j in range(len(a[0]))] for i in range(len(a))]


def scale(a, s):
    """数乘。"""
    return [[v * s for v in row] for row in a]


def trace(a):
    """迹。"""
    return sum(a[i][i] for i in range(len(a)))


def norm_fro(a):
    """Frobenius 范数。Riccati 残差就是用它度量的。"""
    return math.sqrt(sum(v * v for row in a for v in row))


def inv(a):
    """高斯-约当求逆, 带部分选主元。

    为什么必须选主元: 极点配置里的可控性矩阵 C 在接近奇异时会给出量级 1e12
    的增益, 不选主元会把这种病态直接变成 nan。选主元后至少能把"奇异"变成
    一个显式的 LinAlgError, 而不是静默污染后面的 K。
    """
    n = len(a)
    m = [list(row) + [1.0 if i == j else 0.0 for j in range(n)]
         for i, row in enumerate(a)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-300:
            raise LinAlgError("inv: 矩阵奇异(第 %d 列主元为 0)" % col)
        if piv != col:
            m[col], m[piv] = m[piv], m[col]
        pv = m[col][col]
        invp = 1.0 / pv
        rowc = m[col]
        for j in range(2 * n):
            rowc[j] *= invp
        for r in range(n):
            if r == col:
                continue
            factor = m[r][col]
            if factor == 0.0:
                continue
            rowr = m[r]
            for j in range(2 * n):
                rowr[j] -= factor * rowc[j]
    return [row[n:] for row in m]


def solve(a, b):
    """解线性方程组 A x = b(通过求逆, 规模小所以不另写 LU)。"""
    return matvec(inv(a), b)


def expm(a):
    """矩阵指数 exp(A), 缩放-平方 + Taylor 级数。

    为什么需要它: 连续系统离散化要用零阶保持(ZOH), 而 ZOH 的精确解就是
    矩阵指数。用欧拉法 ``Ad = I + A*dt`` 也能跑, 但它只有一阶精度, 会让
    "离散闭环谱半径 < 1" 的结论依赖于 dt 的大小 —— 那是一种假装严格的测试。

    算法: 先取 s 使得 ``||A/2^s||_inf <= 0.5``, 此时 Taylor 12 项的截断误差
    已经远低于 double 精度; 再平方 s 次还原。代价: 只对 ||A|| 不太大的矩阵
    数值安全(本项目 A*dt 的范数 ~0.05, 属于最理想的情形)。
    """
    n = len(a)
    nrm = max(sum(abs(v) for v in row) for row in a)
    s = 0
    while nrm > 0.5:
        nrm *= 0.5
        s += 1
    asc = scale(a, 1.0 / (2 ** s))
    result = eye(n)
    term = eye(n)
    for k in range(1, 13):
        term = scale(matmul(term, asc), 1.0 / k)
        result = add(result, term)
    for _ in range(s):
        result = matmul(result, result)
    return result


def charpoly(a):
    """特征多项式系数, Faddeev-LeVerrier 递推。

    返回降幂系数 ``[1, -p1, -p2, ..., -pn]``, 即
    ``det(lambda*I - A) = lambda^n - p1*lambda^(n-1) - ... - pn``。

    为什么不用 QR 迭代: 幂迭代/QR 求的是特征值本身, 而极点配置的 Ackermann
    公式需要的是**期望**闭环特征多项式的系数。有了 charpoly, 求实际特征值
    只需要再加一个多项式求根(见 eigvals), 一举两得。
    """
    n = len(a)
    m = eye(n)
    p = []
    for k in range(1, n + 1):
        am = matmul(a, m)
        pk = trace(am) / k
        p.append(pk)
        m = sub(am, scale(eye(n), pk))
    return [1.0] + [-v for v in p]


def _polyval(coeffs, z):
    """Horner 法求复系数多项式值。"""
    r = 0j
    for c in coeffs:
        r = r * z + c
    return r


def eigvals(a, tol=1e-14, max_iter=3000):
    """全部特征值, Durand-Kerner(Weierstrass)同时迭代。

    选它的理由: 纯 Python 里它最短、无分支、天然处理复共轭对, 而且对
    "闭环极点是否等于期望极点"这种**已知答案**的验证场景足够准。

    边界(重要): 重根时 Durand-Kerner 只有线性收敛, 可能停在第 1e-6 量级。
    所以:
      * 求谱半径(闭环矩阵, 特征值互异)用它, 精度约 1e-12;
      * 判断开环倒立点不稳定性改用 ``power_iteration``(见下), 那里恰好有
        一对重零特征值, 正是 Durand-Kerner 的弱项。
    """
    coeffs = charpoly(a)
    n = len(coeffs) - 1
    if n == 0:
        return []
    roots = [(0.4 + 0.9j) ** k for k in range(n)]
    for _ in range(max_iter):
        worst = 0.0
        updated = []
        for i in range(n):
            num = _polyval(coeffs, roots[i])
            den = 1 + 0j
            for j in range(n):
                if j != i:
                    den *= (roots[i] - roots[j])
            if den == 0:
                den = 1e-300
            step = num / den
            updated.append(roots[i] - step)
            mag = abs(step)
            if mag > worst:
                worst = mag
        roots = updated
        if worst < tol:
            break
    return roots


def spectral_radius(a):
    """谱半径 max|lambda|。离散系统稳定的充要条件是它 < 1。"""
    return max(abs(z) for z in eigvals(a))


def power_iteration(a, shift=0.0, tol=1e-14, max_iter=20000):
    """幂迭代求**实的主特征值**(Rayleigh 商收敛判据)。

    ``shift`` 是对原矩阵做的正移位: 迭代 ``A + shift*I`` 再把结果减回 shift。

    为什么必须有这个参数(这是踩过的坑): 倒立点的谱是 ``{0, 0, +w, -w}``,
    ``+w`` 与 ``-w`` **模长完全相等**。此时幂迭代的向量在 ``v+`` 与 ``v-``
    两个方向之间来回跳, Rayleigh 商在 ``+w`` / ``-w`` 之间振荡, 永远不收敛
    —— 实测它会一直跑到 max_iter, 返回一个毫无意义的中间值(本项目默认参数
    下曾得到 11.29, 而真值是 4.6456)。

    加一个正移位 ``sigma`` 后就唯一了: ``A + sigma*I`` 的谱变成
    ``{sigma, sigma, w+sigma, sigma-w}``, 只要 ``w + sigma > |sigma - w|``
    即 ``sigma > 0``, 主特征值就是唯一且严格占优的。

    边界:
      * 主导特征值是一对复共轭时(欠阻尼振荡)幂迭代不收敛, 返回值没有意义,
        请改用 ``eigvals``;
      * 主特征值有重根时同样只有线性收敛, 精度会明显下降;
      * 本函数**不报错也不自检**, 因为"判断它有没有收敛"本身就需要另一个
        特征值算法。所以调用方必须知道谱的形状 —— 这是刻意保留的粗糙接口,
        它换来的代码量只有一个循环。
    """
    n = len(a)
    shifted = a if shift == 0.0 else add(a, scale(eye(n), shift))
    x = [1.0 / math.sqrt(n)] * n
    lam = 0.0
    for _ in range(max_iter):
        ax = matvec(shifted, x)
        nrm = math.sqrt(sum(v * v for v in ax))
        if nrm < 1e-300:
            return -shift
        x = [v / nrm for v in ax]
        ax = matvec(shifted, x)
        lam_new = sum(x[i] * ax[i] for i in range(n))
        if abs(lam_new - lam) <= tol * max(1.0, abs(lam_new)):
            return lam_new - shift
        lam = lam_new
    return lam - shift


def discretize_zoh(a, b, dt):
    """连续 (A, B) 的零阶保持离散化, 用增广矩阵 + 矩阵指数精确求解。

    构造 ``M = [[A, B], [0, 0]]``, 则 ``expm(M*dt)`` 的上半部分就是
    ``[[Ad, Bd], [0, I]]``。这比"分别算 Ad 和 Bd 的级数"更不容易写错,
    因为它只用到一个已经测试过的 expm。

    边界: dt 必须远小于系统最快时间常数, 否则 ZOH 本身没问题, 但"用离散
    LQR 控制连续被控对象"的近似会失真(控制器认为一步内输入恒定)。
    """
    n, m = len(a), len(b[0])
    big = zeros(n + m, n + m)
    for i in range(n):
        for j in range(n):
            big[i][j] = a[i][j] * dt
        for j in range(m):
            big[i][n + j] = b[i][j] * dt
    e = expm(big)
    ad = [[e[i][j] for j in range(n)] for i in range(n)]
    bd = [[e[i][n + j] for j in range(m)] for i in range(n)]
    return ad, bd


def linearize(f, x0, u0, eps=1e-6):
    """数值 Jacobian(中心差分), 返回连续时间 (A, B)。

    ``f(x, u) -> dxdt``, ``u0`` 可以是标量(单输入)或列表(多输入)。

    为什么用中心差分而不是前向差分: 中心差分误差 O(eps^2), 前向差分 O(eps)。
    要拿到 1e-6 的相对精度, 前向差分必须把 eps 调到 1e-8 附近, 那里已经被
    浮点抵消(~1e-8)污染; 中心差分在 eps=1e-6 时截断误差 ~1e-12、舍入误差
    ~1e-10, 两头都安全。这就是"能对到 1e-6"的原因。

    边界: 只对**平滑**的 f 有效。含库仑摩擦/死区的 f 在 eps 跨度上不连续,
    差分结果会随 eps 乱跳 —— 那种系统必须改用解析或分段线性化。
    """
    nx = len(x0)
    single = not isinstance(u0, (list, tuple))
    us = [float(u0)] if single else [float(v) for v in u0]
    nu = len(us)

    def eval_f(x, u):
        return f(x, u[0] if single else list(u))

    a = zeros(nx, nx)
    for j in range(nx):
        h = eps * max(1.0, abs(x0[j]))
        xp = list(x0)
        xm = list(x0)
        xp[j] += h
        xm[j] -= h
        fp = eval_f(xp, us)
        fm = eval_f(xm, us)
        for i in range(nx):
            a[i][j] = (fp[i] - fm[i]) / (2.0 * h)

    b = zeros(nx, nu)
    for j in range(nu):
        h = eps * max(1.0, abs(us[j]))
        up = list(us)
        um = list(us)
        up[j] += h
        um[j] -= h
        fp = eval_f(x0, up)
        fm = eval_f(x0, um)
        for i in range(nx):
            b[i][j] = (fp[i] - fm[i]) / (2.0 * h)
    return a, b


# ---------------------------------------------------------------------------
# 系统 1: cart-pole(小车 + 摆杆)
# ---------------------------------------------------------------------------


@dataclass
class CartPoleParams:
    """cart-pole 参数。默认值是一台"桌面级"装置(1 kg 小车 / 100 g 摆锤)。"""

    m_cart: float = 1.0     # M: 小车质量 [kg]
    m_pole: float = 0.1     # m: 摆杆末端集中质量 [kg]
    l_pole: float = 0.5     # l: 铰点到集中质量的距离 [m]
    g: float = 9.81         # 重力加速度 [m/s^2]


def cartpole_deriv(state, force=0.0, p=None):
    """cart-pole 非线性动力学 dxdt, 状态 ``[x, xdot, theta, thetadot]``。

    符号约定: ``theta`` 从**竖直向上**量起, 逆时针为正(即正 theta 让摆锤偏向
    +x 侧)。``force`` 是作用在小车上的水平外力 [N], 正方向为 +x。

    推导(拉格朗日, 摆杆 = 无质量轻杆 + 末端集中质量 m):

        T = 1/2 (M+m) x'^2 + m l cos(theta) x' theta' + 1/2 m l^2 theta'^2
        V = m g l cos(theta)

    ``x`` 是循环坐标, 两条 Euler-Lagrange 方程为:

        (M+m) x'' + m l cos(theta) theta'' = F + m l sin(theta) theta'^2     (1)
        cos(theta) x'' + l theta''          = g sin(theta)                   (2)

    这里**显式解 2x2 线性系统**, 而不是先解出 theta'' 再回代 x''。原因: (2) 式
    解 theta'' 会除以 cos(theta), 在 theta -> +-90 度时爆炸。显式消元得到的

        theta'' = [(M+m) g sin(th) - m l sin(th) cos(th) th'^2 - F cos(th)]
                  / [ l (M + m sin^2(th)) ]

    分母恒 >= l*M > 0, 在全状态空间都良定义。这一步是代码层面最容易踩的坑。
    """
    if p is None:
        p = CartPoleParams()
    m_cart, m_pole = p.m_cart, p.m_pole
    l, g = p.l_pole, p.g

    x, xd, th, thd = state
    sin_t = math.sin(th)
    cos_t = math.cos(th)

    den = l * (m_cart + m_pole * sin_t * sin_t)
    th_dd = ((m_cart + m_pole) * g * sin_t
             - m_pole * l * sin_t * cos_t * thd * thd
             - force * cos_t) / den
    x_dd = (force + m_pole * l * sin_t * thd * thd
            - m_pole * l * cos_t * th_dd) / (m_cart + m_pole)
    return [xd, x_dd, thd, th_dd]


def cartpole_energy(state, p=None):
    """cart-pole 总能量(动能 + 势能), 势能零点取铰点高度 ``theta = pi/2``。

    这是**唯一能廉价证伪动力学方程**的量: 无外力、无阻尼时它必须守恒。
    只要某处符号写反, 能量会在 10 秒内漂到 1e-2 量级, 而位移曲线看起来
    仍然"很像那么回事"。
    """
    if p is None:
        p = CartPoleParams()
    xd, th, thd = state[1], state[2], state[3]
    t_kin = (0.5 * (p.m_cart + p.m_pole) * xd * xd
             + p.m_pole * p.l_pole * math.cos(th) * xd * thd
             + 0.5 * p.m_pole * p.l_pole ** 2 * thd * thd)
    v_pot = p.m_pole * p.g * p.l_pole * math.cos(th)
    return t_kin + v_pot


def cartpole_linearize(p=None):
    """倒立点 ``theta=0, F=0`` 的**解析**连续线性化 (A, B)。

    由 (1)(2) 在 theta -> 0、二阶小量忽略后解得:

        x''     = (F - m g theta) / M
        theta'' = g (M+m)/(M l) theta - F / (M l)

    所以 ``omega^2 = g(M+m)/(M l)`` —— 这就是测试里对账的闭式频率。
    注意: 它成立的前提正是"点质量摆杆", 换成均匀细杆常数会变。
    """
    if p is None:
        p = CartPoleParams()
    m_cart, m_pole, l, g = p.m_cart, p.m_pole, p.l_pole, p.g
    a = [[0.0, 1.0, 0.0, 0.0],
         [0.0, 0.0, -m_pole * g / m_cart, 0.0],
         [0.0, 0.0, 0.0, 1.0],
         [0.0, 0.0, g * (m_cart + m_pole) / (m_cart * l), 0.0]]
    b = [[0.0],
         [1.0 / m_cart],
         [0.0],
         [-1.0 / (m_cart * l)]]
    return a, b


def cartpole_omega_sq(p=None):
    """倒立点线性化频率平方 ``g(M+m)/(M l)``(解析式)。"""
    if p is None:
        p = CartPoleParams()
    return p.g * (p.m_cart + p.m_pole) / (p.m_cart * p.l_pole)


def cartpole_unstable_eig(p=None):
    """倒立点 A 的不稳定特征值(应为正实数, 即系统开环不稳定)。

    这里 ``shift=1.0`` 是必需的: A 的谱是 ``{0, 0, +w, -w}``, ``+-w`` 同模,
    不加移位幂迭代不收敛(详见 ``power_iteration`` 的 docstring)。
    """
    a, _ = cartpole_linearize(p)
    return power_iteration(a, shift=1.0)


def cartpole_ode(force_fn, p=None):
    """把 cart-pole 包成 ``f(t, x)``, 供 RK4 使用。

    ``force_fn`` 可以是常数, 也可以是 ``f(t, x) -> float`` 的控制器回调。
    """
    if p is None:
        p = CartPoleParams()
    if callable(force_fn):
        def f(t, x):
            return cartpole_deriv(x, force_fn(t, x), p)
        return f

    def f_const(t, x):
        return cartpole_deriv(x, force_fn, p)
    return f_const


# ---------------------------------------------------------------------------
# 系统 2: 单摆(解析对账基准)
# ---------------------------------------------------------------------------


@dataclass
class PendulumParams:
    """单摆参数。``damping`` 单位 1/s(速度阻尼系数), 0 表示无阻尼。"""

    m: float = 1.0
    l: float = 0.5
    g: float = 9.81
    damping: float = 0.0


def pendulum_deriv(state, torque=0.0, p=None):
    """单摆动力学, 状态 ``[phi, phidot]``。

    ``phi`` 从**竖直向下**量起(稳定平衡点)。方程

        phi'' = -(g/L) sin(phi) - damping*phi' + torque/(m L^2)

    为什么向下量: 待对账的小角度周期 ``2*pi*sqrt(L/g)`` 描述的是绕稳定平衡点
    的振荡。如果从向上量起, 小角度下方程是 ``phi'' = +(g/L) phi``, 解是
    指数发散, 根本没有"周期"这个概念 —— 那样这条对账就没有意义了。
    """
    if p is None:
        p = PendulumParams()
    phi, phid = state
    phi_dd = (-(p.g / p.l) * math.sin(phi) - p.damping * phid
              + torque / (p.m * p.l * p.l))
    return [phid, phi_dd]


def pendulum_energy(state, p=None):
    """单摆总能量, 势能零点取最低点。无阻尼无外力时必须守恒。"""
    if p is None:
        p = PendulumParams()
    phi, phid = state
    t_kin = 0.5 * p.m * p.l * p.l * phid * phid
    v_pot = p.m * p.g * p.l * (1.0 - math.cos(phi))
    return t_kin + v_pot


def pendulum_small_angle_period(p=None):
    """小角度解析周期 ``2*pi*sqrt(L/g)``。"""
    if p is None:
        p = PendulumParams()
    return 2.0 * math.pi * math.sqrt(p.l / p.g)


def pendulum_ode(torque_fn=0.0, p=None):
    """把单摆包成 ``f(t, x)``。"""
    if p is None:
        p = PendulumParams()
    if callable(torque_fn):
        def f(t, x):
            return pendulum_deriv(x, torque_fn(t, x), p)
        return f

    def f_const(t, x):
        return pendulum_deriv(x, torque_fn, p)
    return f_const


def estimate_period(t, x):
    """从时间序列里量振荡周期: 上穿零点之间的平均间隔(线性插值)。

    为什么插值: 采样点是离散的, 直接数"第几个采样点穿过零"会把分辨率限制
    在 dt 上。线性插值把过零时刻的精度提到 O(dt^2), 这样测出的周期才能和
    ``2*pi*sqrt(L/g)`` 对到 1% 以内 —— 否则测出来的误差里混着采样误差,
    根本无法判断动力学写得对不对。

    至少需要 2 次上穿(即 > 1 个完整周期), 否则抛 ValueError。
    """
    crossings = []
    for i in range(1, len(t)):
        if x[i - 1] < 0.0 <= x[i]:
            span = x[i] - x[i - 1]
            frac = 0.0 if span == 0.0 else (-x[i - 1]) / span
            crossings.append(t[i - 1] + frac * (t[i] - t[i - 1]))
    if len(crossings) < 2:
        raise ValueError("estimate_period: 需要至少 2 次上穿零点, 实际 %d 次"
                         % len(crossings))
    total = crossings[-1] - crossings[0]
    return total / (len(crossings) - 1)


# ---------------------------------------------------------------------------
# 系统 3: 双轮自平衡车(简化)
# ---------------------------------------------------------------------------


@dataclass
class WheeledCartParams:
    """双轮自平衡车参数(两轮并联, 左右对称, 无转向自由度)。

    ``m_body`` 车身质量, ``l_body`` 车身质心到轮轴的距离, ``m_wheel`` 单轮质量
    (两轮等效后按总轮质量理解), ``r_wheel`` 轮半径。
    """

    m_wheel: float = 0.5
    m_body: float = 5.0
    l_body: float = 0.4
    r_wheel: float = 0.1
    g: float = 9.81


def wheeled_cart_deriv(state, torque=0.0, p=None):
    """双轮自平衡车动力学, 状态 ``[x, xdot, theta, thetadot]``。

    ``theta`` 从竖直向上量起; ``torque`` 是轮轴电机力矩 [N*m]。

    拉格朗日推导(轮子纯滚动, 等效轮惯量 ``3/4 Mw x'^2`` 来自平动 + 转动):

        T = 3/4 Mw x'^2 + 1/2 m (x'^2 + 2 l cos(th) x' th' + l^2 th'^2)
        V = m g l cos(th)

    广义力由**虚功**给出: 轮上力矩 tau 通过地面反力等效为水平力 ``tau/r``,
    同时车身受到等大反向的反作用力矩 ``-tau``。于是

        [ A          , m l cos(th) ] [x''  ]   [ tau/r + m l sin(th) th'^2 ]
        [ m l cos(th), m l^2       ] [th'' ] = [ m g l sin(th) - tau         ]

    其中 ``A = 3/2 Mw + m``。用 2x2 Cramer 法则显式求解, 行列式
    ``D = m l^2 (3/2 Mw + m sin^2 th) >= m l^2 * 1.5 Mw > 0``, 同样在全状态
    空间良定义(避免了像 cart-pole 那样除以 cos)。

    简化代价: 车身当点质量(忽略车体自身转动惯量)、无轮胎滑移/柔性、无摩擦、
    两轮合成为一个等效轮。因此它适合验证"控制器结构对不对", 不能当作实车
    标定模型。
    """
    if p is None:
        p = WheeledCartParams()
    mw, m, l, r, g = (p.m_wheel, p.m_body, p.l_body, p.r_wheel, p.g)

    x, xd, th, thd = state
    sin_t = math.sin(th)
    cos_t = math.cos(th)

    a_coef = 1.5 * mw + m
    ml = m * l
    det = a_coef * (ml * l) - (ml * cos_t) ** 2
    rhs1 = torque / r + ml * sin_t * thd * thd
    rhs2 = m * g * l * sin_t - torque
    x_dd = (rhs1 * ml * l - ml * cos_t * rhs2) / det
    th_dd = (a_coef * rhs2 - ml * cos_t * rhs1) / det
    return [xd, x_dd, thd, th_dd]


def wheeled_cart_energy(state, p=None):
    """双轮自平衡车总能量(势能零点取轮轴高度 ``theta = pi/2``)。"""
    if p is None:
        p = WheeledCartParams()
    mw, m, l, g = p.m_wheel, p.m_body, p.l_body, p.g
    xd, th, thd = state[1], state[2], state[3]
    t_kin = (0.75 * mw * xd * xd
             + 0.5 * m * (xd * xd + 2.0 * l * math.cos(th) * xd * thd
                          + l * l * thd * thd))
    v_pot = m * g * l * math.cos(th)
    return t_kin + v_pot


def wheeled_cart_linearize(p=None):
    """倒立点 ``theta=0, tau=0`` 的解析线性化 (A, B), 状态 ``[x, xdot, th, thd]``。

    由上面的 2x2 方程在 theta -> 0 处展开(记 ``A = 3/2 Mw + m``):

        x''     = -2 m g/(3 Mw) * theta + [(1/r + A/(m l))/(1.5 Mw) - 1/(m l)] * tau
        theta'' = A g/(1.5 Mw l) * theta - (1/r + A/(m l))/(1.5 Mw l) * tau

    注意 ``d x''/d tau`` 的两项: 第一项是电机力矩通过地面反力推动小车, 第二项
    是车身反作用力矩。只写第一项(即 ``1/(1.5 Mw r)``)在轻车身时看着还行, 车身
    一重就差 25% —— 这正是"解析式必须和数值 Jacobian 对账"的价值所在。

    与 cart-pole 一样, ``d(theta'')/d(theta) > 0`` —— 这也是开环不稳定的
    结构性原因: 重力矩与倾角同号。
    """
    if p is None:
        p = WheeledCartParams()
    mw, m, l, r, g = (p.m_wheel, p.m_body, p.l_body, p.r_wheel, p.g)
    a_coef = 1.5 * mw + m
    a = [[0.0, 1.0, 0.0, 0.0],
         [0.0, 0.0, -2.0 * m * g / (3.0 * mw), 0.0],
         [0.0, 0.0, 0.0, 1.0],
         [0.0, 0.0, a_coef * g / (1.5 * mw * l), 0.0]]
    b = [[0.0],
         [(1.0 / r + a_coef / (m * l)) / (1.5 * mw) - 1.0 / (m * l)],
         [0.0],
         [-(1.0 / r + a_coef / (m * l)) / (1.5 * mw * l)]]
    return a, b


def wheeled_cart_ode(torque_fn=0.0, p=None):
    """把双轮车包成 ``f(t, x)``。"""
    if p is None:
        p = WheeledCartParams()
    if callable(torque_fn):
        def f(t, x):
            return wheeled_cart_deriv(x, torque_fn(t, x), p)
        return f

    def f_const(t, x):
        return wheeled_cart_deriv(x, torque_fn, p)
    return f_const


# ---------------------------------------------------------------------------
# 能量漂移工具(动力学正确性的主判据)
# ---------------------------------------------------------------------------


def energy_drift(f, x0, energy_fn, t_end, dt, substep=1):
    """定步长 RK4 积分并返回(相对能量漂移, 时间序列, 能量序列)。

    ``substep`` 用来让"物理积分步"比"采样步"更细: 采样只是记录, 精度由
    实际积分步长决定。返回的相对漂移定义为

        max_t |E(t) - E(0)| / max(|E(0)|, 1e-12)

    为什么除以 E(0) 而不是除以 E(t) 的最大值: E(0) 是**不可变**的参考量,
    用它做分母时这个指标不会因为分母自己变化而变得"看起来更好"。
    """
    from .sim import rk4_step  # 局部导入: sim.py 依赖本模块, 顶层导入会成环

    h = dt / substep
    x = list(x0)
    t = 0.0
    e0 = energy_fn(x)
    denom = max(abs(e0), 1e-12)
    ts = [0.0]
    es = [e0]
    worst = 0.0
    steps = int(round(t_end / h))
    for _ in range(steps):
        x = rk4_step(f, t, x, h)
        t += h
        if _ % substep == substep - 1:
            e = energy_fn(x)
            ts.append(t)
            es.append(e)
            d = abs(e - e0) / denom
            if d > worst:
                worst = d
    return worst, ts, es

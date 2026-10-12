"""平面机械臂运动学: 正解、逆解(双解)、雅可比(解析 + 数值对账)、奇异检测。

为什么运动学也要放进"平衡控制"项目里: JD 把"运动学与动力学"并列, 而一个只会
做平衡不会做机械臂的仓库证明不了前者。更实际的原因是 —— **雅可比与奇异**是
运动学里唯一有工程后果的部分: 奇异点附近关节速度会被放大到失控, 这是实机上
"机器人突然抽一下"最常见的原因之一。

本模块只做**平面 2R / 3R**(所有关节都在同一平面内)。理由: 平面机构有闭式逆解
和闭式雅可比行列式, 可以和数值 Jacobian 逐元素对到 1e-6, 也能显式地写出奇异
条件 ``det(J) = l1*l2*sin(q2) = 0``。空间 6R 机械臂的逆解要多解 16 组并处理
腕部奇异(欧拉角万向锁), 那是另一个量级的工作量 —— 本项目不声称做了它。

**纯仿真/纯数学**: 这里没有任何真实机械臂、没有关节标定、没有减速比、没有
连杆柔性。角度单位是弧度, 长度单位与 ``link`` 参数一致。
"""

from __future__ import annotations

import math

from .dynamics import LinAlgError, inv, matvec


class UnreachableError(ValueError):
    """目标点超出工作空间。显式抛出, 而不是返回一个"最接近"的假解。

    为什么抛错而不是夹取到边界: 夹取会让调用方以为"到了", 实际停在几十厘米外
    —— 这在标定/抓取流程里是灾难。上位机应该自己决定怎么处理(缩放目标、报警、
    或运动到边界)。
    """


class SingularityError(ValueError):
    """雅可比接近奇异(或完全奇异), 无法求逆。"""


class Planar2R(object):
    """平面 2R 机械臂: 两个转动关节, 连杆长 ``l1`` / ``l2``。

    约定: ``q = (q1, q2)`` 弧度, 绝对角(不是相对角), 逆时针为正。

        x = l1 cos(q1) + l2 cos(q1+q2)
        y = l1 sin(q1) + l2 sin(q1+q2)
    """

    def __init__(self, l1=1.0, l2=0.8):
        if l1 <= 0.0 or l2 <= 0.0:
            raise ValueError("连杆长度必须为正")
        self.l1 = float(l1)
        self.l2 = float(l2)

    # -- 正运动学 ----------------------------------------------------------

    def forward(self, q):
        """正运动学, 返回末端 ``(x, y)``。"""
        q1, q2 = q[0], q[1]
        c1, s1 = math.cos(q1), math.sin(q1)
        c12, s12 = math.cos(q1 + q2), math.sin(q1 + q2)
        return (self.l1 * c1 + self.l2 * c12, self.l1 * s1 + self.l2 * s12)

    def joint_positions(self, q):
        """返回 ``[(0,0), 关节2位置, 末端位置]``, 便于画图/检查。"""
        q1, q2 = q[0], q[1]
        p1 = (self.l1 * math.cos(q1), self.l1 * math.sin(q1))
        return [(0.0, 0.0), p1, self.forward(q)]

    @property
    def reach_max(self):
        """最大可达半径 ``l1 + l2``。"""
        return self.l1 + self.l2

    @property
    def reach_min(self):
        """最小可达半径 ``|l1 - l2|``(内边界, 对应折叠位形)。"""
        return abs(self.l1 - self.l2)

    def is_reachable(self, x, y, tol=1e-9):
        """目标点是否在环形工作空间内(含边界, 带容差)。"""
        r = math.hypot(x, y)
        return (r <= self.reach_max + tol) and (r >= self.reach_min - tol)

    # -- 逆运动学 ----------------------------------------------------------

    def inverse(self, x, y, elbow="up"):
        """解析逆解, 返回 ``(q1, q2)``。

        ``elbow="up"`` 取 ``q2 > 0`` 的解, ``"down"`` 取 ``q2 < 0`` 的解。

        推导: 由余弦定理得 ``cos(q2) = (r^2 - l1^2 - l2^2) / (2 l1 l2)``,
        两个符号对应两组解。然后
        ``q1 = atan2(y, x) - atan2(l2 sin q2, l1 + l2 cos q2)``。

        为什么用 ``atan2`` 而不是 ``acos``: ``acos`` 的值域只有 ``[0, pi]``, 二连杆
        的 q1 需要覆盖整个圆周, 用 acos 会漏掉一半工作空间。这是最常见的实现 bug。
        """
        if elbow not in ("up", "down"):
            raise ValueError("elbow 只能是 'up' 或 'down', 实际 %r" % (elbow,))
        r2 = x * x + y * y
        c2 = (r2 - self.l1 ** 2 - self.l2 ** 2) / (2.0 * self.l1 * self.l2)
        if c2 > 1.0 + 1e-9 or c2 < -1.0 - 1e-9:
            raise UnreachableError(
                "目标 (%.6f, %.6f) 不可达: cos(q2)=%.9f 超出 [-1,1]" % (x, y, c2))
        c2 = max(-1.0, min(1.0, c2))
        s2 = math.sqrt(max(0.0, 1.0 - c2 * c2))
        if elbow == "down":
            s2 = -s2
        q2 = math.atan2(s2, c2)
        q1 = math.atan2(y, x) - math.atan2(self.l2 * s2, self.l1 + self.l2 * c2)
        return (q1, q2)

    def inverse_both(self, x, y):
        """返回两组解 ``(elbow_up, elbow_down)``。完全伸直/折叠时两组重合。"""
        return (self.inverse(x, y, "up"), self.inverse(x, y, "down"))

    # -- 雅可比 ------------------------------------------------------------

    def jacobian(self, q):
        """解析雅可比 ``J``: ``d(x,y)/d(q1,q2)``。"""
        q1, q2 = q[0], q[1]
        s1, c1 = math.sin(q1), math.cos(q1)
        s12, c12 = math.sin(q1 + q2), math.cos(q1 + q2)
        return [[-self.l1 * s1 - self.l2 * s12, -self.l2 * s12],
                [self.l1 * c1 + self.l2 * c12, self.l2 * c12]]

    def jacobian_numeric(self, q, eps=1e-6):
        """数值 Jacobian(中心差分), 用于和解析式对账。"""
        out = [[0.0, 0.0], [0.0, 0.0]]
        for j in range(2):
            qp = list(q)
            qm = list(q)
            qp[j] += eps
            qm[j] -= eps
            fp = self.forward(qp)
            fm = self.forward(qm)
            out[0][j] = (fp[0] - fm[0]) / (2.0 * eps)
            out[1][j] = (fp[1] - fm[1]) / (2.0 * eps)
        return out

    def det_jacobian(self, q):
        """雅可比行列式, 有闭式: ``l1*l2*sin(q2)``。

        这就是奇异的全部条件 —— 与 q1 无关。物理含义: q1 只决定机械臂朝向,
        不影响"能不能在该方向上产生末端速度"; 起决定作用的是**肘部弯曲角 q2**。
        """
        return self.l1 * self.l2 * math.sin(q[1])

    def singularity_margin(self, q):
        """到奇异的归一化裕度 ``|sin(q2)|``, 0 表示奇异。

        用 ``|sin(q2)|`` 而不是 ``|det|`` 做裕度: 后者带 ``l1*l2`` 的量纲, 换一根
        连杆长度阈值就得重调。归一化之后阈值(比如 0.05)在不同机械臂间可复用。
        """
        return abs(math.sin(q[1]))

    def is_singular(self, q, tol=1e-6):
        """是否奇异(裕度 < tol)。"""
        return self.singularity_margin(q) < tol

    def inverse_velocity(self, q, ee_velocity, tol=1e-6):
        """由末端速度求关节速度 ``qdot = J^-1 * xdot``。

        奇异附近会**放大关节速度**: ``|qdot| ~ |xdot| / (l1 l2 |sin q2|)``。
        这是本模块最想展示的工程后果 —— 见 ``velocity_amplification``。
        奇异(裕度 < tol)时显式抛 ``SingularityError``, 而不是返回巨大的数值。
        """
        margin = self.singularity_margin(q)
        if margin < tol:
            raise SingularityError(
                "位形 q=%r 接近奇异 (|sin q2|=%.3e < %.1e), 无法求逆"
                % (tuple(q), margin, tol))
        return matvec(inv(self.jacobian(q)), list(ee_velocity))

    def velocity_amplification(self, direction, q2, q1=0.0):
        """定量例子: 在给定 ``q2`` 下, 单位末端速度需要多大的关节速度。

        返回 ``|qdot|``。``direction`` 是末端速度方向(单位向量, 2 元)。

        为什么固定 ``q1 = 0``: 奇异裕度只取决于 ``q2``, 而 ``|J^-1|`` 还与 q1 有关
        (q1 是旋转对称的 —— 转 q1 只是把整个机构转一圈)。取 q1 = 0 得到的数字
        是这一族里的代表值, 且与直觉一致(末端要沿 x 正向动)。

        边界: 这不是范数上界。真正的最坏放大倍数由 ``sigma_min(J)`` 给出, 而
        本项目没有写 SVD —— 对 2x2 矩阵可以用闭式奇异值公式, 但这里只做"特定
        方向"的例子, 已经足够说明问题。
        """
        return matvec(inv(self.jacobian([q1, q2])), list(direction))


class Planar3R(object):
    """平面 3R 机械臂: 三个转动关节, 末端位姿是 ``(x, y, phi)``。

    冗余被"指定末端姿态"消掉了: 给定 ``(x, y, phi)`` 时解唯一(除了肘部两解),
    因为第三个关节角由 ``q3 = phi - q1 - q2`` 后验确定。这比用伪逆解冗余要
    简单和确定得多, 代价是**无法利用冗余**(比如避障、避奇异)。
    """

    def __init__(self, l1=1.0, l2=0.8, l3=0.5):
        if l1 <= 0.0 or l2 <= 0.0 or l3 <= 0.0:
            raise ValueError("连杆长度必须为正")
        self.l1 = float(l1)
        self.l2 = float(l2)
        self.l3 = float(l3)
        self.two_r = Planar2R(l1, l2)

    def forward(self, q):
        """正运动学, 返回末端 ``(x, y, phi)``, ``phi = q1+q2+q3``。"""
        q1, q2, q3 = q[0], q[1], q[2]
        c1, s1 = math.cos(q1), math.sin(q1)
        c12, s12 = math.cos(q1 + q2), math.sin(q1 + q2)
        c123, s123 = math.cos(q1 + q2 + q3), math.sin(q1 + q2 + q3)
        return (self.l1 * c1 + self.l2 * c12 + self.l3 * c123,
                self.l1 * s1 + self.l2 * s12 + self.l3 * s123,
                q1 + q2 + q3)

    def inverse(self, x, y, phi, elbow="up"):
        """解析逆解: 先用姿态把腕点反解出来, 再对 2R 求逆。"""
        if elbow not in ("up", "down"):
            raise ValueError("elbow 只能是 'up' 或 'down', 实际 %r" % (elbow,))
        wx = x - self.l3 * math.cos(phi)
        wy = y - self.l3 * math.sin(phi)
        q1, q2 = self.two_r.inverse(wx, wy, elbow)
        q3 = phi - q1 - q2
        return (q1, q2, q3)

    def jacobian(self, q):
        """解析 3x3 雅可比。

        第三行恒为 ``[1, 1, 1]`` —— 因为末端姿态就是三个角的和, 每个关节对
        姿态的贡献都是 1, 与位形无关。
        """
        q1, q2, q3 = q[0], q[1], q[2]
        s1, c1 = math.sin(q1), math.cos(q1)
        s12, c12 = math.sin(q1 + q2), math.cos(q1 + q2)
        s123, c123 = math.sin(q1 + q2 + q3), math.cos(q1 + q2 + q3)
        return [[-self.l1 * s1 - self.l2 * s12 - self.l3 * s123,
                 -self.l2 * s12 - self.l3 * s123,
                 -self.l3 * s123],
                [self.l1 * c1 + self.l2 * c12 + self.l3 * c123,
                 self.l2 * c12 + self.l3 * c123,
                 self.l3 * c123],
                [1.0, 1.0, 1.0]]

    def jacobian_numeric(self, q, eps=1e-6):
        """数值 Jacobian(中心差分), 3x3。"""
        out = [[0.0] * 3 for _ in range(3)]
        for j in range(3):
            qp = list(q)
            qm = list(q)
            qp[j] += eps
            qm[j] -= eps
            fp = self.forward(qp)
            fm = self.forward(qm)
            for i in range(3):
                out[i][j] = (fp[i] - fm[i]) / (2.0 * eps)
        return out

    def det_jacobian(self, q):
        """雅可比行列式, 闭式: ``l1*l2*sin(q2)`` —— 与 q3 无关。"""
        return self.l1 * self.l2 * math.sin(q[1])

    def singularity_margin(self, q):
        """到奇异的归一化裕度 ``|sin(q2)|``。"""
        return abs(math.sin(q[1]))

    def is_singular(self, q, tol=1e-6):
        """是否奇异。"""
        return self.singularity_margin(q) < tol


def cartesian_linear_path(start, end, n):
    """笛卡尔直线插补(给轨迹跟踪测试当参考路径用)。

    返回 ``[(x, y), ...]``, 共 n 个点, 含起点与终点。**只做几何插补, 不做时间
    参数化** —— 时间信息由 ``trajectory.py`` 的轮廓提供, 两者正交。
    """
    if n < 2:
        raise ValueError("n 至少为 2")
    out = []
    for i in range(n):
        s = i / float(n - 1)
        out.append((start[0] + (end[0] - start[0]) * s,
                    start[1] + (end[1] - start[1]) * s))
    return out


def jacobian_condition_estimate(jac):
    """2x2/3x3 小矩阵的条件数估计 ``||J||_F * ||J^-1||_F``。

    为什么用 Frobenius 范数而不是 2-范数(需要 SVD): 这里只是给一个"离奇异多远"
    的量级指标, 用于诊断打印。**它不是真正的条件数**, 会高估(因为 Frobenius
    范数 >= 谱范数), 所以不要用它做阈值判断 —— 阈值判断请用
    ``singularity_margin``。
    """
    n = len(jac)
    nrm = math.sqrt(sum(v * v for row in jac for v in row))
    try:
        ji = inv([list(row) for row in jac])
    except LinAlgError:
        return float("inf")
    nrm_i = math.sqrt(sum(v * v for row in ji for v in row))
    return nrm * nrm_i


def forward_kinematics_chain(lengths, q):
    """任意长度平面转动链的正运动学(通用版), 返回末端 ``(x, y)``。

    与 ``Planar2R.forward`` 的区别: 这个函数不固定关节数, 用来交叉验证
    "2R 的实现不是特例硬编码"。复杂度 O(n), 但没用它替换上面的闭式实现 ——
    闭式版更快且更容易读, 保留通用版只是为了交叉验证。
    """
    if len(lengths) != len(q):
        raise ValueError("连杆数 %d 与关节数 %d 不一致" % (len(lengths), len(q)))
    x = 0.0
    y = 0.0
    ang = 0.0
    for li, qi in zip(lengths, q):
        ang += qi
        x += li * math.cos(ang)
        y += li * math.sin(ang)
    return (x, y)


def jacobian_chain(lengths, q):
    """任意长度平面转动链的雅可比(通用数值版, 中心差分)。"""
    n = len(q)
    out = [[0.0] * n for _ in range(2)]
    for j in range(n):
        qp = list(q)
        qm = list(q)
        eps = 1e-6
        qp[j] += eps
        qm[j] -= eps
        fp = forward_kinematics_chain(lengths, qp)
        fm = forward_kinematics_chain(lengths, qm)
        out[0][j] = (fp[0] - fm[0]) / (2 * eps)
        out[1][j] = (fp[1] - fm[1]) / (2 * eps)
    return out

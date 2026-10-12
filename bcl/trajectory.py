"""关节轨迹规划: 梯形速度、七段式 S 曲线(jerk 受限)、五次多项式。

三种轨迹解决的是同一件事的三个层次, 所以放在一起对比 —— 而对比本身就是本项目
想证明的能力: 知道每种方法的**代价**在哪里, 而不只是"会调用"。

* **梯形速度**: 只有加速度约束。时间最短, 但加速度在起止点**阶跃**(jerk 无穷),
  机械上表现为冲击与振动。电机电流也会阶跃 —— 对减速器是实打实的损伤。
* **七段式 S 曲线**: 再加 jerk 约束, 加速度连续。代价是时间更长(通常 +30%~80%),
  而且峰值加速度经常到不了 ``amax``(因为 jerk 限制了爬升速度)。
* **五次多项式**: 位置/速度/加速度都连续, 数学上最漂亮。但它的 jerk 在**起止点
  有跳变**(端点 jerk = 60*d/T^3), 换言之它把冲击移到了端点。所以在高速高精度
  场合反而不如 S 曲线 —— 这一点在测试里用数字直接对比。

统一接口::

    traj = SCurveProfile(p0, p1, vmax, amax, jmax)
    traj.sample(t)      -> {"t","position","velocity","acceleration","jerk"}
    traj.sample_series(1001) -> {"t":[...], "position":[...], ...}

**端点约定**: 轨迹在 ``t <= 0`` 与 ``t >= T`` 上都返回静止值(position = p0/p1,
velocity = acceleration = jerk = 0)。也就是说整条曲线被理解为"运动前后各接一段
静止保持"。这样端点断言(position/velocity/acceleration 全为 0)可以严格成立, 同时
**不掩盖**梯形曲线在端点加速度阶跃这一事实 —— 用 ``t = 1e-9`` 采样立刻能暴露它
(见 ``tests/test_trajectory.py``)。

**量纲与精度**: 全部 SI。峰值对账的目标是"恰好等于限幅值", 实测误差在 1e-12 量级
(不是浮点噪声, 而是 S 曲线峰值速度由二分法求出、自然带一点残差)。
"""

from __future__ import annotations

import math


class _Plan(object):
    """分段多项式计划的公共实现。

    每一段是"常 jerk 段", 由 ``(duration, s0, v0, a0, jerk)`` 描述, 段内解析解为::

        s(t) = s0 + v0 t + 0.5 a0 t^2 + jerk t^3 / 6
        v(t) = v0 + a0 t + 0.5 jerk t^2
        a(t) = a0 + jerk t

    为什么用分段解析解而不是查表/插值: 解析解的位置/速度/加速度/jerk 在**任意时刻**
    都能精确求值, 采样的疏密不影响精度。查表插值会让"峰值速度恰好等于 vmax"这种
    对账变成"取决于采样点落在哪", 那就没法用它验证规划算法了。
    """

    def __init__(self, p0, p1, vmax, amax, jmax=0.0):
        for name, val in (("vmax", vmax), ("amax", amax)):
            if not val > 0.0:
                raise ValueError("%s 必须为正: %r" % (name, val))
        self.p0 = float(p0)
        self.p1 = float(p1)
        self.vmax = abs(float(vmax))
        self.amax = abs(float(amax))
        self.jmax = abs(float(jmax))
        self.distance = abs(self.p1 - self.p0)
        self.sign = 1.0 if self.p1 >= self.p0 else -1.0
        self.segments = []
        self.duration = 0.0
        self.peak_velocity = 0.0
        self.peak_acceleration = 0.0
        self.peak_jerk = 0.0

    # -- 段构建 ------------------------------------------------------------

    def _make_segments(self, specs):
        """由 ``[(duration, jerk), ...]`` 或 ``[(duration, jerk, a0), ...]`` 生成段列表。

        每个 spec 可选地给出该段的**初始加速度** ``a0``。为什么需要它: 梯形速度
        轮廓的中间段是"恒定加速度"段, 而恒定加速度对应 jerk = 0 —— 用有限 jerk
        是表达不出来的(梯形轮廓的 jerk 在段边界是狄拉克函数)。如果偷懒写成
        ``(ta, amax)``(把加速度当 jerk 传), 得到的会是一条加速度只升不降的曲线,
        而且**位置居然还是能对上**, 因为错得很"对称"。这类 bug 只有靠"峰值速度
        恰好等于 vmax"这种硬对账才能抓出来。

        负时长/零时长段直接丢弃: 它等价于"这一段不存在", 保留会在求值时产生
        退化分支。丢弃后段数减少, 但求值逻辑不变 —— S 曲线的 7 段常常退化成
        4 段甚至 3 段。
        """
        segs = []
        s = v = a = 0.0
        for spec in specs:
            if len(spec) == 2:
                dur, jerk = spec
                a0 = a
            else:
                dur, jerk, a0 = spec
            if dur <= 0.0:
                a = a0
                continue
            segs.append((dur, s, v, a0, jerk))
            s += v * dur + 0.5 * a0 * dur * dur + jerk * dur ** 3 / 6.0
            v += a0 * dur + 0.5 * jerk * dur * dur
            a = a0 + jerk * dur
        self.segments = segs
        self.duration = sum(seg[0] for seg in segs)
        return s, v, a

    def segment_boundaries(self):
        """各段结束的时刻(升序), 末尾含总时长。

        数值微分对账必须排除这些点附近: jerk 在段边界跳变, 中心差分跨过跳变时
        误差是 O(dt) 而不是 O(dt^2), 会把"微分对得上"这条结论污染掉。
        """
        out = []
        acc = 0.0
        for seg in self.segments:
            acc += seg[0]
            out.append(acc)
        return out

    # -- 求值 --------------------------------------------------------------

    def _eval(self, t):
        """在正向位移坐标系里求 ``(s, v, a, jerk)``。"""
        for dur, s0, v0, a0, jerk in self.segments:
            if t < dur:
                return (s0 + v0 * t + 0.5 * a0 * t * t + jerk * t ** 3 / 6.0,
                        v0 + a0 * t + 0.5 * jerk * t * t,
                        a0 + jerk * t,
                        jerk)
            t -= dur
        if not self.segments:
            return 0.0, 0.0, 0.0, 0.0
        dur, s0, v0, a0, jerk = self.segments[-1]
        return (s0 + v0 * dur + 0.5 * a0 * dur * dur + jerk * dur ** 3 / 6.0,
                v0 + a0 * dur + 0.5 * jerk * dur * dur,
                a0 + jerk * dur,
                jerk)

    def sample(self, t):
        """在时刻 t 采样, 返回含 position/velocity/acceleration/jerk 的 dict。"""
        if self.duration <= 0.0:
            return {"t": t, "position": self.p1, "velocity": 0.0,
                    "acceleration": 0.0, "jerk": 0.0}
        if t <= 0.0:
            return {"t": t, "position": self.p0, "velocity": 0.0,
                    "acceleration": 0.0, "jerk": 0.0}
        if t >= self.duration:
            return {"t": t, "position": self.p1, "velocity": 0.0,
                    "acceleration": 0.0, "jerk": 0.0}
        s, v, a, j = self._eval(t)
        return {"t": t,
                "position": self.p0 + self.sign * s,
                "velocity": self.sign * v,
                "acceleration": self.sign * a,
                "jerk": self.sign * j}

    def position(self, t):
        """位置 p(t)。"""
        return self.sample(t)["position"]

    def velocity(self, t):
        """速度 v(t)。"""
        return self.sample(t)["velocity"]

    def acceleration(self, t):
        """加速度 a(t)。"""
        return self.sample(t)["acceleration"]

    def jerk(self, t):
        """jerk j(t)。梯形/五次多项式在段内为 0, S 曲线在 jerk 段内为 +-jmax。"""
        return self.sample(t)["jerk"]

    def sample_series(self, n):
        """在 ``[0, T]`` 上等间隔采 n 个点, 返回列字典。"""
        if n < 2:
            raise ValueError("n 至少为 2, 实际 %r" % (n,))
        keys = ("t", "position", "velocity", "acceleration", "jerk")
        out = {k: [] for k in keys}
        for i in range(n):
            t = self.duration * i / float(n - 1) if self.duration > 0 else 0.0
            s = self.sample(t)
            for k in keys:
                out[k].append(s[k])
        return out

    def sample_dt(self, dt):
        """以固定步长 dt 采样 ``[0, T]``(末尾强制含 T)。"""
        if dt <= 0.0:
            raise ValueError("dt 必须为正: %r" % (dt,))
        n = int(math.floor(self.duration / dt)) + 1
        keys = ("t", "position", "velocity", "acceleration", "jerk")
        out = {k: [] for k in keys}
        for i in range(n):
            s = self.sample(i * dt)
            for k in keys:
                out[k].append(s[k])
        if out["t"][-1] < self.duration - 1e-15:
            s = self.sample(self.duration)
            for k in keys:
                out[k].append(s[k])
        return out

    def max_abs(self, key, n=2001):
        """在 ``[0, T]`` 上求该分量的最大绝对值。

        采样点 = 均匀网格 + **每一段的中点/四分点 + 各段紧邻边界的内侧点**。
        为什么不能只靠均匀网格: S 曲线的 jerk 只在 ``t1`` 这么长的段里等于
        ``jmax``, 当位移很小导致 ``t1 ~ 1e-6 s`` 时, 2001 点的均匀网格会整段
        跳过它, 于是"峰值 jerk 恰好 = jmax"这条断言会因为**采样点没落在段里**
        而失败 —— 那是测试的缺陷, 不是实现的缺陷。补上段内点后, 只要段存在就
        一定被采到。

        代价: 段数多时(七段)会多算十几次求值, 可忽略。真正没有覆盖的是**连续
        变化**的极值(五次多项式的 jerk 是抛物线), 但它的极值在端点, 而端点已经
        被均匀网格采到了。
        """
        if self.duration <= 0.0:
            return 0.0
        best = 0.0
        samples = [self.duration * i / float(n - 1) for i in range(n)]
        acc = 0.0
        for dur, _s0, _v0, _a0, _j in self.segments:
            for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
                samples.append(acc + frac * dur)
            acc += dur
        for t in samples:
            val = abs(self.sample(t)[key])
            if val > best:
                best = val
        return best


class TrapezoidalProfile(_Plan):
    """梯形速度轮廓: 加速 -> 匀速 -> 减速。

    最短时间的解析解:

    * 若 ``d >= vmax^2/amax``: ``ta = vmax/amax`` 能加到 vmax, 匀速段
      ``tv = (d - vmax^2/amax)/vmax``, 总时长 ``T = 2 ta + tv``。
    * 否则**走不到 vmax**, 必须降速: ``ta = sqrt(d/amax)``, 峰值速度
      ``v_peak = amax*ta = sqrt(d*amax)``, 总时长 ``T = 2 ta``(三角形轮廓)。

    第二种情况就是"位移太短要降速"的正确最短时间解。很多实现会直接把 vmax 当成
    峰值速度硬跑, 结果是算出来的总时长比真实最短时间还短, 位置对不上 —— 这类
    bug 在长距离测试里完全看不出来。

    代价: 加速度在 ``t=0, ta, ta+tv, T`` 四处阶跃, jerk 是狄拉克函数。电机电流
    会跟着阶跃, 对减速器寿命和末端振动都不友好。这正是 S 曲线存在的理由。
    """

    def __init__(self, p0, p1, vmax, amax):
        _Plan.__init__(self, p0, p1, vmax, amax, jmax=0.0)
        d = self.distance
        if d <= 0.0:
            self._make_segments([])
            return
        v2a = self.vmax * self.vmax / self.amax
        if d >= v2a:
            ta = self.vmax / self.amax
            v_peak = self.vmax
            tv = (d - v2a) / self.vmax
        else:
            ta = math.sqrt(d / self.amax)
            v_peak = self.amax * ta
            tv = 0.0
        self.peak_velocity = v_peak
        self.peak_acceleration = self.amax
        self.peak_jerk = 0.0
        # 恒定加速度段: jerk = 0, 初始加速度显式给定(见 _make_segments 的说明)
        self._make_segments([(ta, 0.0, self.amax),
                             (tv, 0.0, 0.0),
                             (ta, 0.0, -self.amax)])
        self.peak_jerk = float("inf")  # 段边界处加速度阶跃


class SCurveProfile(_Plan):
    """七段式 S 曲线: jerk 受限, 加速度连续。

    段结构(对称, ``v0 = v1 = 0``, ``a0 = a1 = 0``)::

        1. jerk = +jmax, 时长 t1    加速度 0 -> A
        2. jerk =  0   , 时长 t2    恒加速(A 平台)
        3. jerk = -jmax, 时长 t1    加速度 A -> 0
        4. jerk =  0   , 时长 t4    匀速(vmax 平台)
        5. jerk = -jmax, 时长 t1    加速度 0 -> -A
        6. jerk =  0   , 时长 t2    恒减速
        7. jerk = +jmax, 时长 t1    加速度 -A -> 0

    关键约束: 从 0 把加速度爬到 ``A`` 需要 ``t1 = A/jmax``, 期间速度已经涨了
    ``A^2/(2 jmax)``。所以**要真正达到 A, 峰值速度至少得有 ``A^2/jmax``**。
    如果分的位移不够, 就进入"加速度到不了 amax"的情形, 此时 ``A = sqrt(v*jmax)``、
    ``t2 = 0``(退化成 4 段)。

    求解方式: 把半程距离 ``D_half`` 写成峰值速度 ``v`` 的单调增函数, 用**二分法**
    反解 ``2*D_half(v) = d``。为什么用二分而不是分类解析求根: 七段式一共有
    7 段, 随 (vmax, amax, jmax, d) 的组合会退化成 3/4/5/6 段, 分类讨论有十几种
    情况, 每一种都要单独验算 —— 那是 bug 的重灾区。二分法只有一种代码路径, 单
    调性由物理保证(位移随峰值速度单调增), 200 次迭代把区间压到 1e-15, 代价是
    每次规划约 200 次标量运算(微秒级)。

    代价/边界:
      * 假设对称轮廓(起止都静止)。起点或终点速度非零的"非对称 S 曲线"本项目
        没有实现 —— 那需要把 t1/t2 分成两半独立求解, 代码量翻倍。
      * 二分法是**数值**解。峰值速度的相对误差约 1e-15, 位置误差约 1e-16 —— 但
        它不是闭式解, 所以"峰值速度恰好等于 vmax"这条断言在"走不到 vmax"的
        情形下要注意: 此时峰值速度是算出来的, 不是限幅值。
    """

    def __init__(self, p0, p1, vmax, amax, jmax):
        if not jmax > 0.0:
            raise ValueError("jmax 必须为正: %r" % (jmax,))
        _Plan.__init__(self, p0, p1, vmax, amax, jmax=jmax)
        d = self.distance
        if d <= 0.0:
            self._make_segments([])
            return

        v_switch = self.amax * self.amax / self.jmax
        if 2.0 * self._half_distance(self.vmax)[0] >= d:
            # 走不到 vmax: 二分求峰值速度
            lo, hi = 0.0, self.vmax
            for _ in range(200):
                mid = 0.5 * (lo + hi)
                if 2.0 * self._half_distance(mid)[0] < d:
                    lo = mid
                else:
                    hi = mid
                if hi - lo <= 1e-16 * max(1.0, hi):
                    break
            v_peak = 0.5 * (lo + hi)
            t4 = 0.0
        else:
            v_peak = self.vmax
            t4 = (d - 2.0 * self._half_distance(self.vmax)[0]) / self.vmax

        dist_half, a_peak, t1, t2 = self._half_distance(v_peak)
        self.peak_velocity = v_peak
        self.peak_acceleration = a_peak
        self.peak_jerk = self.jmax
        self.v_switch = v_switch
        self.reaches_amax = bool(a_peak >= self.amax - 1e-12 * self.amax)
        self.reaches_vmax = bool(v_peak >= self.vmax - 1e-12 * self.vmax)
        self._make_segments([(t1, self.jmax), (t2, 0.0), (t1, -self.jmax),
                             (t4, 0.0),
                             (t1, -self.jmax), (t2, 0.0), (t1, self.jmax)])

    def _half_distance(self, v):
        """峰值速度为 ``v`` 时, 加速半程走过的距离与推导出的 (A, t1, t2)。"""
        j = self.jmax
        if v <= 0.0:
            return 0.0, 0.0, 0.0, 0.0
        v_switch = self.amax * self.amax / j
        if v <= v_switch:
            a_peak = math.sqrt(v * j)
            t1 = a_peak / j
            t2 = 0.0
        else:
            a_peak = self.amax
            t1 = self.amax / j
            t2 = (v - v_switch) / self.amax
        x1 = j * t1 ** 3 / 6.0
        v1 = 0.5 * j * t1 * t1
        x2 = v1 * t2 + 0.5 * a_peak * t2 * t2
        v2 = v1 + a_peak * t2
        x3 = v2 * t1 + 0.5 * a_peak * t1 * t1 - j * t1 ** 3 / 6.0
        return x1 + x2 + x3, a_peak, t1, t2


class QuinticProfile(_Plan):
    """五次多项式轮廓 ``p(s) = p0 + d*(10 s^3 - 15 s^4 + 6 s^5)``, ``s = t/T``。

    边界条件全 0: ``p(0)=p0``, ``v(0)=a(0)=0``, ``p(T)=p1``, ``v(T)=a(T)=0``。

    峰值(闭式): ``v_peak = 1.875*d/T``, ``a_peak = 5.7735*d/T^2``,
    端点 ``jerk = 60*d/T^3``。最后一个式子就是它的软肋: **端点 jerk 非零**,
    而且是通过阶跃从 0 跳上去的 —— 冲击被移到了运动的开始和结束瞬间。

    ``duration`` 不给定时, 用同参数的梯形轮廓时长, 这样"五次多项式 vs S 曲线"
    的对比是在**同样的运动时间**下进行的, 否则比出来的 jerk 差异没有意义。
    """

    def __init__(self, p0, p1, vmax, amax, duration=None):
        _Plan.__init__(self, p0, p1, vmax, amax, jmax=0.0)
        if self.distance <= 0.0:
            self._make_segments([])
            return
        if duration is None:
            duration = TrapezoidalProfile(p0, p1, vmax, amax).duration
        if duration <= 0.0:
            raise ValueError("duration 必须为正: %r" % (duration,))
        self.duration = float(duration)
        self.peak_velocity = 15.0 * self.distance / (8.0 * self.duration)
        self.peak_acceleration = 10.0 * math.sqrt(3.0) / 3.0 \
            * self.distance / (self.duration ** 2)
        self.peak_jerk = float("inf")  # 端点处 jerk 从 0 阶跃到 60d/T^3

    def _eval(self, t):
        """五次多项式的解析解(jerk 是 t 的二次函数, 不是分段常数)。"""
        d = self.distance
        T = self.duration
        s = t / T
        pos = d * (10.0 * s ** 3 - 15.0 * s ** 4 + 6.0 * s ** 5)
        vel = d / T * (30.0 * s ** 2 - 60.0 * s ** 3 + 30.0 * s ** 4)
        acc = d / (T * T) * (60.0 * s - 180.0 * s ** 2 + 120.0 * s ** 3)
        jrk = d / (T ** 3) * (60.0 - 360.0 * s + 360.0 * s ** 2)
        return pos, vel, acc, jrk

    def endpoint_jerk(self):
        """端点 jerk 的解析值 ``60*d/T^3``(用来和 S 曲线的 jmax 做对比)。"""
        if self.duration <= 0.0:
            return 0.0
        return 60.0 * self.distance / (self.duration ** 3)

    def max_abs(self, key, n=2001):
        """用**解析峰值**覆盖扫点结果。

        三次以上的分量(velocity/acceleration/jerk)在本轮廓上都有闭式峰值:
        ``1.875 d/T``、``5.7735 d/T^2``、``60 d/T^3``。扫点在这些点附近只能逼近,
        而"quintic 的 jerk 峰值到底比 S 曲线的 jmax 大多少倍"正是 bench 要用的
        数字, 用逼近似会引入 0.3% 的伪误差。所以直接给解析值, 扫点结果由测试
        单独校验一次(两者相差应 < 1%)。位置没有闭式峰值, 回退到扫点。
        """
        if key == "velocity":
            return self.peak_velocity
        if key == "acceleration":
            return self.peak_acceleration
        if key == "jerk":
            return self.endpoint_jerk()
        return _Plan.max_abs(self, key, n=n)

    def segment_boundaries(self):
        """五次多项式整体连续(C^2), 没有分段边界, 返回空列表。"""
        return []


class TimeScaledProfile(object):
    """把一条轨迹在时间轴上拉伸 ``scale`` 倍(用于多关节同步)。

    为什么时间拉伸是安全的多关节同步方式: 把 ``t -> t/scale`` 代入后, 速度峰值
    变成原来的 ``1/scale``、加速度 ``1/scale^2``、jerk ``1/scale^3`` —— 只要
    ``scale >= 1``, 所有限幅只会**更宽松**, 绝不会被违反。代价是整体运动变慢,
    慢的比例等于"最长的那条轨迹比最短的长多少"。

    另一种做法是给每个关节提高 vmax, 但那需要重新规划; 时间拉伸是零成本的。
    边界: 它同步的是**总时长**, 不保证中间过程同步(比如两关节的峰值时刻仍然
    不同)。对点到点定位这足够了; 对连续路径跟踪需要时间参数化(如 TOPP),
    本项目没有做。
    """

    def __init__(self, inner, scale):
        if scale <= 0.0:
            raise ValueError("scale 必须为正: %r" % (scale,))
        self.inner = inner
        self.scale = float(scale)
        self.duration = inner.duration * self.scale
        self.p0 = inner.p0
        self.p1 = inner.p1

    def sample(self, t):
        """按缩放后的时间采样(内部换算回原轨迹时间)。"""
        out = dict(self.inner.sample(t / self.scale))
        out["t"] = t
        return out

    def position(self, t):
        """位置 p(t)。"""
        return self.sample(t)["position"]

    def velocity(self, t):
        """速度 v(t)(峰值被压到原来的 1/scale)。"""
        return self.sample(t)["velocity"]

    def peak_velocity(self):
        """缩放后的速度峰值。"""
        return self.inner.max_abs("velocity") / self.scale


def sync_trajectories(profiles):
    """多关节同步: 按最长时长统一时间缩放, 使所有关节**同时到达**。

    返回与输入等长的 ``TimeScaledProfile`` 列表, 但**只包含 duration > 0 的
    轨迹对应的缩放对象**; 零位移的轨迹本身没有运动, 也在返回列表里(scale=1),
    以便调用方按下标对应回原关节。
    """
    if not profiles:
        return []
    tmax = max(p.duration for p in profiles)
    out = []
    for p in profiles:
        if p.duration <= 0.0 or tmax <= 0.0:
            out.append(TimeScaledProfile(p, 1.0))
        else:
            out.append(TimeScaledProfile(p, tmax / p.duration))
    return out


def numeric_derivative(ts, values):
    """对等间隔序列做中心差分求导(端点用单侧差分)。

    用来做"轨迹自带速度 vs 数值微分"的对账。段边界附近误差是 O(dt) 而非
    O(dt^2), 调用方应配合 ``segment_boundaries`` 排除这些点。
    """
    n = len(values)
    if n < 2:
        raise ValueError("至少需要 2 个采样点")
    dt = ts[1] - ts[0]
    out = [0.0] * n
    for i in range(n):
        if i == 0:
            out[i] = (values[1] - values[0]) / dt
        elif i == n - 1:
            out[i] = (values[-1] - values[-2]) / dt
        else:
            out[i] = (values[i + 1] - values[i - 1]) / (2.0 * dt)
    return out

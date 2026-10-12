"""底层驱动抽象: 寄存器总线、PWM/死区、正交编码器、IMU、周期调度器。

这一层回答的是"控制量怎么落到硬件、状态怎么从硬件读回来"。在真实机器人上,
平衡控制能不能跑起来, 一半以上的功夫花在这里: 总线会超时、CRC 会错、PWM 上
下桥臂直通会炸管子、编码器 16 位计数器会回绕、1 kHz 控制周期会被抖动吃掉。

**本模块全部是内存/软件模拟, 不接触任何真实硬件**(没有串口、没有 SPI/I2C、
没有 /dev、没有 MCU)。这么做的理由是: 我们想验证**寄存器语义、时序语义和
数值量化**这三件事, 而它们与"用的是哪颗 MCU"无关。代价见下面各处的边界说明。

设计原则: **抽象接口 (RegisterBus) 与 后端实现 (SimRegisterBus) 分离**。换到真
硬件时只需要再写一个 ``SpiRegisterBus`` 子类实现 ``_raw_read`` / ``_raw_write``
两个方法, 上面所有外设模型(PWM/编码器/IMU)一行都不用改 —— 这也是这一层存在
的唯一理由。注意我们没有真的去接硬件, 所以这个"可替换性"是设计上的, 未经实
机验证。
"""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# 异常与 CRC
# ---------------------------------------------------------------------------


class BusError(Exception):
    """总线层错误基类。所有错误都显式抛出, 绝不静默返回 0。"""


class RegisterError(BusError):
    """非法寄存器地址/非法位域写入。"""


class ChecksumError(BusError):
    """CRC 校验失败。真实总线上这通常由电磁干扰或线缆过长引起。"""


class BusTimeout(BusError):
    """从设备未在超时窗口内响应。"""


def crc8(data, poly=0x07, init=0x00):
    """CRC-8/ATM(多项式 x^8+x^2+x+1), 查表法的直算版本。

    为什么用 CRC-8 而不是累加和: 累加和对字节交换不敏感, 也抓不到传输中
    某一位翻转后的偶数个错误。CRC-8 能检出全部单比特/双比特错误和所有
    长度 <= 8 的突发错误 —— 对一个寄存器读写通道来说足够了。
    """
    crc = init
    for byte in data:
        crc ^= (byte & 0xFF)
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ poly) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
    return crc


@dataclass
class BusStats:
    """总线访问统计。调实时系统时, 这是判断"是不是总线在拖后腿"的第一手数据。"""

    reads: int = 0
    writes: int = 0
    retries: int = 0
    crc_errors: int = 0
    timeouts: int = 0

    @property
    def transfers(self):
        """成功的读写次数(不含重试)。"""
        return self.reads + self.writes

    @property
    def error_count(self):
        """CRC 错误 + 超时的总次数。"""
        return self.crc_errors + self.timeouts

    def as_dict(self):
        """展开成 dict, 便于 CLI/bench 打印。"""
        return {
            "reads": self.reads,
            "writes": self.writes,
            "retries": self.retries,
            "crc_errors": self.crc_errors,
            "timeouts": self.timeouts,
        }


# ---------------------------------------------------------------------------
# 寄存器总线抽象
# ---------------------------------------------------------------------------

_WIDTH_BYTES = {8: 1, 16: 2, 32: 4}


class RegisterBus(ABC):
    """寄存器总线抽象基类。

    子类**只需要**实现两个字节级方法 ``_raw_read(addr)`` / ``_raw_write(addr, value)``,
    其余(u8/u16/u32、位域读写、重试、统计)都在这里复用。这是刻意的: 底层的
    可变部分尽量窄, 免得每个后端都要重新实现一遍位域语义。

    重试策略: 只在 ``BusTimeout`` / ``ChecksumError`` 上重试。``RegisterError``
    (地址越界)是编程错误, 重试一万次也不会变好 —— 立刻抛出才是正确行为。
    """

    def __init__(self, size, retries=2):
        self.size = size
        self.retries = retries
        self.stats = BusStats()

    # -- 子类实现 ----------------------------------------------------------

    @abstractmethod
    def _raw_read(self, addr):
        """读一个字节; 可能抛 BusTimeout / ChecksumError / RegisterError。"""

    @abstractmethod
    def _raw_write(self, addr, value):
        """写一个字节; 可能抛 BusTimeout / ChecksumError / RegisterError。"""

    # -- 重试包装 ----------------------------------------------------------

    def _guarded(self, op):
        """执行 op, 按 self.retries 次重试瞬时错误; 记录统计。"""
        attempt = 0
        while True:
            try:
                return op()
            except BusTimeout:
                self.stats.timeouts += 1
                if attempt >= self.retries:
                    raise
            except ChecksumError:
                self.stats.crc_errors += 1
                if attempt >= self.retries:
                    raise
            attempt += 1
            self.stats.retries += 1

    # -- 宽度读写 ----------------------------------------------------------

    def _check_width_addr(self, addr, width):
        nbytes = _WIDTH_BYTES[width]
        if addr < 0 or addr + nbytes > self.size:
            raise RegisterError(
                "地址越界: addr=%d width=%d size=%d" % (addr, width, self.size))
        if addr % nbytes != 0:
            raise RegisterError(
                "%d 位寄存器必须对齐到 %d 字节边界, 实际 addr=%d"
                % (width, nbytes, addr))

    def read_u8(self, addr):
        """读 8 位无符号寄存器。"""
        self._check_width_addr(addr, 8)
        val = self._guarded(lambda: self._raw_read(addr))
        self.stats.reads += 1
        return val & 0xFF

    def read_u16(self, addr):
        """读 16 位寄存器(小端: addr 为低字节, addr+1 为高字节)。"""
        self._check_width_addr(addr, 16)
        hi = self._guarded(lambda: self._raw_read(addr + 1))
        lo = self._guarded(lambda: self._raw_read(addr))
        self.stats.reads += 1
        return ((hi & 0xFF) << 8) | (lo & 0xFF)

    def read_u32(self, addr):
        """读 32 位寄存器(小端)。"""
        self._check_width_addr(addr, 32)
        b = [self._guarded(lambda a=addr + i: self._raw_read(a)) for i in range(4)]
        self.stats.reads += 1
        return (b[0] & 0xFF) | ((b[1] & 0xFF) << 8) | ((b[2] & 0xFF) << 16) \
            | ((b[3] & 0xFF) << 24)

    def write_u8(self, addr, value):
        """写 8 位寄存器。超范围的值显式抛错, 不静默截断。"""
        self._check_width_addr(addr, 8)
        if not 0 <= value <= 0xFF:
            raise RegisterError("u8 写入越界: %r" % (value,))
        self._guarded(lambda: self._raw_write(addr, value))
        self.stats.writes += 1

    def write_u16(self, addr, value):
        """写 16 位寄存器(小端)。"""
        self._check_width_addr(addr, 16)
        if not 0 <= value <= 0xFFFF:
            raise RegisterError("u16 写入越界: %r" % (value,))
        self._guarded(lambda: self._raw_write(addr, value & 0xFF))
        self._guarded(lambda: self._raw_write(addr + 1, (value >> 8) & 0xFF))
        self.stats.writes += 1

    def write_u32(self, addr, value):
        """写 32 位寄存器(小端)。"""
        self._check_width_addr(addr, 32)
        if not 0 <= value <= 0xFFFFFFFF:
            raise RegisterError("u32 写入越界: %r" % (value,))
        for i in range(4):
            self._guarded(lambda a=addr + i, s=8 * i: self._raw_write(a, (value >> s) & 0xFF))
        self.stats.writes += 1

    # -- 位域 --------------------------------------------------------------

    @staticmethod
    def mask_width(mask):
        """由掩码推断寄存器宽度(8/16/32)。"""
        if mask <= 0xFF:
            return 8
        if mask <= 0xFFFF:
            return 16
        return 32

    @staticmethod
    def check_mask(mask, shift):
        """校验掩码/移位是否自洽。

        约定: ``mask`` 是**未移位**的连续低位掩码(例如 bit[5:4] 写作 ``mask=0x03,
        shift=4``)。要求掩码本身低位连续, 这样 ``(mask << shift)`` 才不会在寄存器
        里出现"镂空"的位域 —— 镂空位域做读-改-写时极易漏掉中间那一位。
        """
        if mask <= 0:
            raise RegisterError("掩码必须为正: %r" % (mask,))
        if shift < 0:
            raise RegisterError("shift 不能为负: %r" % (shift,))
        if mask & (mask + 1) != 0:
            raise RegisterError("掩码必须是连续的 1(低位对齐), 实际 0x%X" % mask)
        if (mask << shift) > 0xFFFFFFFF:
            raise RegisterError("掩码移位后超出 32 位")

    def _read_width(self, addr, width):
        return {8: self.read_u8, 16: self.read_u16, 32: self.read_u32}[width](addr)

    def _write_width(self, addr, width, value):
        return {8: self.write_u8, 16: self.write_u16, 32: self.write_u32}[width](
            addr, value)

    def read_field(self, addr, mask, shift):
        """读位域并右移到最低位。"""
        self.check_mask(mask, shift)
        width = self.mask_width(mask << shift)
        raw = self._read_width(addr, width)
        return (raw >> shift) & mask

    def write_field(self, addr, mask, shift, value):
        """读-改-写位域: **只影响掩码覆盖的位**, 其余位原样保留。

        这是外设驱动里最常出错的地方: 直接 ``write_u8`` 会把同一寄存器里其
        他外设的配置位一起清掉(比如把时钟分频位清零)。所以这里强制走
        读-改-写, 并且在测试里显式断言"掩码外相邻位不变"。

        边界: 真硬件上读-改-写不是原子的, 如果中断里也在改同一寄存器会有竞
        态。本项目是单线程仿真, 不涉及这个问题 —— 但不能据此认为真机上也安全。
        """
        self.check_mask(mask, shift)
        if value < 0 or (value & ~mask) != 0:
            raise RegisterError("位域值 0x%X 超出掩码 0x%X" % (value, mask))
        width = self.mask_width(mask << shift)
        raw = self._read_width(addr, width)
        raw = (raw & ~(mask << shift)) | ((value & mask) << shift)
        self._write_width(addr, width, raw)
        return raw


class SimRegisterBus(RegisterBus):
    """用一块内存数组实现的寄存器总线(严格模拟寄存器语义)。

    语义保证(全部有单测覆盖):
      * 写进去读出来一致(字节级存储, 不做任何"智能"处理);
      * 位域写入只影响掩码内的位;
      * 越界地址抛 ``RegisterError``, 不静默回绕;
      * CRC 错误与超时可**确定性注入**(``inject_crc_error`` / ``stall_next``),
        这样才能写出可复现的故障恢复测试。

    为什么每次传输都算 CRC: 真实的 SPI/I2C 外设寄存器读写本来就带帧校验, 不加
    这一层的话"重试逻辑"就成了没有触发条件的死代码。代价是每次传输多算一遍
    CRC-8(8 字节以内开销可忽略)。
    """

    def __init__(self, size=256, retries=2):
        RegisterBus.__init__(self, size, retries)
        self.mem = bytearray(size)
        self._stall = 0
        self._crc_fault = 0

    # -- 故障注入 ----------------------------------------------------------

    def stall_next(self, count=1):
        """让接下来 count 次传输超时(模拟从设备忙/时钟拉伸)。"""
        self._stall += max(0, int(count))

    def inject_crc_error(self, count=1):
        """让接下来 count 次传输 CRC 校验失败(模拟电磁干扰导致的位翻转)。"""
        self._crc_fault += max(0, int(count))

    def clear_faults(self):
        """清掉所有待注入的故障。

        为什么需要: 故障注入是"待消费"的计数器, 注入 5 次但只消耗 3 次时, 剩下
        2 次会**泄漏到后面的正常操作**上, 让后续演示出现莫名其妙的超时。显式清掉
        比"算准消耗次数"可靠得多 —— 后者一旦改了重试策略就会失效。
        """
        self._stall = 0
        self._crc_fault = 0

    def _frame(self, op, addr, payload):
        """组装一帧并校验 CRC。返回字节列表。"""
        frame = [op & 0xFF, addr & 0xFF] + [b & 0xFF for b in payload]
        if self._crc_fault > 0:
            self._crc_fault -= 1
            # 故意把 CRC 写错一位: 真实干扰也是这么表现的
            frame.append((crc8(frame) ^ 0x01) & 0xFF)
            raise ChecksumError("CRC 校验失败(注入): addr=0x%02X" % (addr & 0xFF))
        frame.append(crc8(frame))
        return frame

    # -- 底层字节访问 ------------------------------------------------------

    def _raw_read(self, addr):
        if self._stall > 0:
            self._stall -= 1
            raise BusTimeout("读超时(注入): addr=0x%02X" % addr)
        if not 0 <= addr < self.size:
            raise RegisterError("读地址越界: %d" % addr)
        self._frame(0x01, addr, [])
        return self.mem[addr]

    def _raw_write(self, addr, value):
        if self._stall > 0:
            self._stall -= 1
            raise BusTimeout("写超时(注入): addr=0x%02X" % addr)
        if not 0 <= addr < self.size:
            raise RegisterError("写地址越界: %d" % addr)
        self._frame(0x02, addr, [value])
        self.mem[addr] = value & 0xFF

    # -- 便于测试/调试的直通访问(绕过总线协议) ------------------------------

    def peek(self, addr):
        """直接看后备内存, 用于断言"寄存器里到底存了什么"。"""
        if not 0 <= addr < self.size:
            raise RegisterError("peek 越界: %d" % addr)
        return self.mem[addr]

    def poke(self, addr, value):
        """直接改后备内存(模拟外部 DMA/上电默认值)。"""
        if not 0 <= addr < self.size:
            raise RegisterError("poke 越界: %d" % addr)
        self.mem[addr] = value & 0xFF


# ---------------------------------------------------------------------------
# 外设 1: PWM 通道(量化 + 死区 + 饱和)
# ---------------------------------------------------------------------------


class PwmChannel:
    """单路带死区的 H 桥 PWM 通道。

    寄存器布局(相对 ``base``, 每个通道占 4 字节):
      * ``base + 4*ch``     : u16 duty, bit15 = 方向, bit[11:0] = 幅值(12 位)
      * ``base + 16 + ch``  : u8  ctrl, bit0 = 使能

    三件必须做对的事:
      1. **量化到 12 位** —— MCU 的定时器比较寄存器就是 12 位, 控制量 0.5 实际
         输出的是 2048/4095 = 0.500122…。忽略量化会让"仿真很准、实机抖"。
      2. **死区插入** —— 上下桥臂绝不能同时导通, 否则电源直接短路(炸管)。
         做法: 上管关断后延迟 ``dead_time_ticks`` 才允许下管开通, 于是占空比
         上限被顶到 ``1 - dead_time_ticks/period_ticks``。
      3. **饱和限幅** —— duty 限制在 ``duty_limit`` 内, 并再次被死区顶住。

    边界: 死区对输出电压的影响这里用一阶模型(平均电压幅值下降
    ``T_dead/T``), 忽略了体二极管续流、结电容导致的非线性畸变。真实死区效应
    在低占空比附近会明显偏离这个线性模型。
    """

    CTRL_ENABLE = 0x01

    def __init__(self, bus, base=0x40, channel=0, resolution=12,
                 period_ticks=1000, dead_time_ticks=24, duty_limit=0.95):
        if resolution != 12:
            raise RegisterError("本项目 PWM 固定 12 位分辨率, 实际 %r" % (resolution,))
        self.bus = bus
        self.base = base
        self.channel = channel
        self.resolution = resolution
        self.full_scale = (1 << resolution) - 1
        self.period_ticks = period_ticks
        self.dead_time_ticks = dead_time_ticks
        self.duty_limit = duty_limit
        self.duty_reg = base + 4 * channel
        self.ctrl_reg = base + 16 + channel
        # 死区把占空比上限进一步压低: 必须保证 [duty_ticks, duty_ticks+dead) 落在周期内
        self.max_duty_ticks = period_ticks - dead_time_ticks
        self.max_units = int(round(duty_limit * self.full_scale))
        self.max_units = min(self.max_units,
                             int(self.max_duty_ticks * self.full_scale // period_ticks))

    # -- 量化 --------------------------------------------------------------

    def quantize(self, duty):
        """把有符号占空比量化到 12 位并饱和。返回 (符号, 幅值单位数)。"""
        d = duty
        if not isinstance(d, (int, float)) or d != d:  # NaN 检查
            raise RegisterError("占空比必须是有限实数, 实际 %r" % (duty,))
        if d > self.duty_limit:
            d = self.duty_limit
        elif d < -self.duty_limit:
            d = -self.duty_limit
        units = int(round(abs(d) * self.full_scale))
        if units > self.max_units:
            units = self.max_units
        return (1 if duty > 0 else (-1 if duty < 0 else 0)), units

    def dequantize(self, units, sign):
        """幅值单位数 -> 有符号占空比(还原为浮点, 保留量化误差)。"""
        return sign * (units / float(self.full_scale))

    def set_duty(self, duty):
        """写占空比到寄存器并使能通道。返回实际写入的量化值。"""
        sign, units = self.quantize(duty)
        raw = ((0x8000 if sign < 0 else 0) | (units & 0x0FFF))
        self.bus.write_u16(self.duty_reg, raw)
        self.bus.write_field(self.ctrl_reg, self.CTRL_ENABLE, 0, 1)
        return raw

    def read_duty_raw(self):
        """从寄存器读回原始 duty 字。"""
        return self.bus.read_u16(self.duty_reg)

    def get_duty(self):
        """从寄存器读回并还原成有符号占空比(含 12 位量化误差)。"""
        raw = self.read_duty_raw()
        sign = -1 if (raw & 0x8000) else 1
        return self.dequantize(raw & 0x0FFF, sign)

    def is_enabled(self):
        """通道是否使能(读 ctrl 寄存器 bit0)。"""
        return bool(self.bus.read_field(self.ctrl_reg, self.CTRL_ENABLE, 0))

    # -- 死区与栅极波形 ----------------------------------------------------

    def duty_ticks(self):
        """当前占空比对应的导通 tick 数(已含死区上限约束)。"""
        raw = self.read_duty_raw()
        units = raw & 0x0FFF
        ticks = int(round(units / float(self.full_scale) * self.period_ticks))
        return min(ticks, self.max_duty_ticks)

    def effective_duty(self):
        """死区导致的等效占空比: 幅值损失 ``dead_time_ticks/period_ticks``。

        这是"仿真器里闭环会差一点"的真实来源之一 —— 用理想 PWM 仿真时控制
        器会以为 5% 占空比就是 5% 电压, 实际只有 5% - 2.4%。
        """
        raw = self.read_duty_raw()
        sign = -1 if (raw & 0x8000) else 1
        ticks = self.duty_ticks()
        eff_ticks = max(0, ticks - self.dead_time_ticks)
        return sign * (eff_ticks / float(self.period_ticks))

    def gate_states(self, tick):
        """返回给定 tick 处的 ``(a_high, a_low, b_high, b_low)`` 栅极导通状态。

        正向(``sign >= 0``): A 桥臂做 PWM, B 桥臂下管常通。
        反向(``sign < 0``): B 桥臂做 PWM, A 桥臂下管常通。

        任何 tick 下, 同一桥臂的上下管都不会同时导通 —— 因为上管的关断窗口
        ``[0, ticks)`` 与下管的开通窗口 ``[ticks+dead, period)`` 之间永远隔了
        ``dead`` 个 tick, 且 ``ticks <= period - dead``。
        """
        if not 0 <= tick < self.period_ticks:
            raise RegisterError("tick %d 超出周期 [0, %d)" % (tick, self.period_ticks))
        raw = self.read_duty_raw()
        sign = -1 if (raw & 0x8000) else 1
        ticks = self.duty_ticks()
        high_on = tick < ticks
        low_on = (ticks + self.dead_time_ticks) <= tick < self.period_ticks
        if sign < 0:
            return (False, True, high_on, low_on)
        if sign == 0 and ticks == 0:
            return (False, True, False, True)
        return (high_on, low_on, False, True)

    def has_shoot_through(self, tick):
        """检查给定 tick 是否发生上下桥臂直通(本设计下永远应为 False)。"""
        a_high, a_low, b_high, b_low = self.gate_states(tick)
        return bool((a_high and a_low) or (b_high and b_low))

    def sweep_shoot_through(self):
        """扫过整个 PWM 周期, 返回发生直通的 tick 列表(测试/自检用)。"""
        return [k for k in range(self.period_ticks) if self.has_shoot_through(k)]


# ---------------------------------------------------------------------------
# 外设 2: 正交编码器
# ---------------------------------------------------------------------------


class QuadEncoder:
    """增量式正交编码器(软件解码 + 溢出回绕 + 差分测速)。

    为什么正交解码不能只数 A 相边沿: 只数 A 相无法判断方向, 而且抖动会让计数
    单调增加。用 (A,B) 的**格雷序列** 00->01->11->10 来判定方向, 单次状态跳变
    最多 +-1 计数, 非法跳变(如 00->11, 说明漏采了一个状态)计 0 并可统计。

    回绕处理: 硬件计数器是 16 位有符号, 到 32767 后翻转到 -32768。**这是最容易
    在生产代码里翻车的地方** —— 直接把寄存器值当位置用, 车会在某一刻突然以为
    自己瞬移了 65536 个计数。所以我们同时维护:
      * ``count_raw``   : 16 位回绕值, 等于寄存器里真实存的东西;
      * ``count_total`` : 不回绕累计值, 控制算法唯一应该用的量;
      * ``wrap_count``  : 回绕次数, 超过阈值说明该换绝对式编码器或多圈计数了。

    边界: 这里是**软件定时采样解码**模型。如果采样率低于编码器脉冲率(高速时),
    会丢状态跳变, 表现是速度被低估。真机通常用 MCU 的定时器硬件正交解码模式
    规避这个问题; 本仿真没有建模硬件解码器。
    """

    # (prev<<2)|new -> 计数增量; 非法双重跳变(00<->11, 01<->10)记 0
    _TABLE = (0, 1, -1, 0,
              -1, 0, 0, 1,
              1, 0, 0, -1,
              0, -1, 1, 0)
    # 非法双重跳变 (prev,new) 的索引 prev*4+new: 00<->11 与 01<->10
    _INVALID = frozenset((3, 6, 9, 12))

    def __init__(self, bus=None, count_reg=0x20, ppr=1024, bits=16,
                 lpf_alpha=0.3, wrap_warn=4):
        self.bus = bus
        self.count_reg = count_reg
        self.ppr = ppr
        self.bits = bits
        self.lpf_alpha = lpf_alpha
        self.wrap_warn = wrap_warn
        self.half_range = 1 << (bits - 1)

        self._state = 0
        self.count_raw = 0
        self.count_total = 0
        self.wrap_count = 0
        self.invalid_transitions = 0
        self._last_total = 0
        self.speed_counts = 0.0
        self._write_register()

    # -- 寄存器同步 --------------------------------------------------------

    def _write_register(self):
        """把当前 16 位回绕计数值写进寄存器(小端两字节)。"""
        if self.bus is None:
            return
        self.bus.write_u16(self.count_reg, self.count_raw & 0xFFFF)

    def read_register(self):
        """从寄存器读回 16 位值并解释为有符号数。"""
        if self.bus is None:
            return self.count_raw
        raw = self.bus.read_u16(self.count_reg)
        return raw - 0x10000 if raw >= self.half_range else raw

    # -- 正交解码 ----------------------------------------------------------

    @staticmethod
    def state_of(a, b):
        """把 (A,B) 电平映射到 0..3 的格雷状态。"""
        return (1 if a else 0) * 2 + (1 if b else 0)

    def decode_step(self, a, b):
        """喂入一次 (A,B) 采样, 返回本步计数增量(-1/0/+1)。"""
        new_state = self.state_of(a, b)
        idx = (self._state << 2) | new_state
        delta = self._TABLE[idx]
        if idx in self._INVALID:
            self.invalid_transitions += 1
        self._state = new_state
        return delta

    def update(self, a, b):
        """解码一步并累加到回绕计数器 + 不回绕计数器。返回增量。"""
        delta = self.decode_step(a, b)
        if delta == 0:
            return 0
        return self.advance(delta)

    def advance(self, counts):
        """直接推进计数(counts 可正可负), 自动处理 16 位回绕。

        为什么除了 ``update`` 还需要它: ``update`` 是"逐状态跳变"的真实解码
        路径, 适合验证解码表; 但在整机仿真里, 已知真实位置时逐拍喂 4 倍频脉冲
        既慢又没意义, 直接注入计数差更高效。两条路径共用同一段回绕代码, 所以
        回绕行为在两种用法下完全一致。
        """
        counts = int(counts)
        if counts == 0:
            return 0
        self.count_total += counts
        self.count_raw += counts
        while self.count_raw >= self.half_range:
            self.count_raw -= (1 << self.bits)
            self.wrap_count += 1
        while self.count_raw < -self.half_range:
            self.count_raw += (1 << self.bits)
            self.wrap_count += 1
        self._write_register()
        return counts

    def reset(self):
        """清零(等同于收到一次索引脉冲 Z)。回绕统计保留, 因为它是寿命指标。"""
        self._state = 0
        self.count_raw = 0
        self.count_total = 0
        self._last_total = 0
        self.speed_counts = 0.0
        self._write_register()

    def seed_count(self, value):
        """直接设定原始计数值(用于构造回绕测试场景, 模拟上电中间态)。"""
        self.count_raw = value
        self.count_total = value
        self._last_total = value
        self._write_register()

    # -- 测速 --------------------------------------------------------------

    def update_speed(self, dt):
        """按差分 + 一阶低通估计转速, 单位 counts/s。

        为什么必须低通: 位置是量化的, 差分噪声被 1/dt 放大。1 kHz 下 1 个计数
        的量化误差就对应 1000 counts/s 的速度噪声, 而真实转速可能只有几百。
        低通系数 ``lpf_alpha`` 是"噪声 vs 相位滞后"的直接权衡: 越小越平滑但
        滞后越大(滞后会直接吃掉平衡控制器的相位裕度)。

        边界: 一阶低通在 1 kHz 下对 20 Hz 以上的扰动基本无效, 真机上更常用
        的是"位置差分 + 卡尔曼/观测器"。这里刻意保留最朴素的实现, 便于把
        "量化噪声有多致命"这件事量化出来。
        """
        if dt <= 0:
            raise RegisterError("dt 必须为正: %r" % (dt,))
        raw = (self.count_total - self._last_total) / dt
        self._last_total = self.count_total
        self.speed_counts = (self.lpf_alpha * raw
                             + (1.0 - self.lpf_alpha) * self.speed_counts)
        return self.speed_counts

    def speed_rps(self, counts_per_rev=None):
        """转速换算成转/秒。默认用 ``4 * ppr``(四倍频)。"""
        cpr = counts_per_rev if counts_per_rev else 4 * self.ppr
        return self.speed_counts / float(cpr)

    @property
    def speed_rpm(self):
        """转速换算成 RPM(按四倍频计数)。"""
        return self.speed_rps() * 60.0

    def needs_wrap_attention(self):
        """回绕次数是否已超过告警阈值。"""
        return self.wrap_count >= self.wrap_warn


# ---------------------------------------------------------------------------
# 外设 3: IMU
# ---------------------------------------------------------------------------


@dataclass
class ImuConfig:
    """IMU 量程与噪声配置。

    ``accel_lsb_per_g``: +-2g 量程下典型值 16384 LSB/g。
    ``gyro_lsb_per_dps``: +-250 dps 量程下典型值 131 LSB/(deg/s)。
    """

    accel_lsb_per_g: float = 16384.0
    gyro_lsb_per_dps: float = 131.0
    accel_noise_g: float = 0.004
    gyro_noise_dps: float = 0.20
    seed: int = 20240517


class ImuDevice:
    """六轴 IMU 的寄存器级模型: 量化 + 噪声 + 通过总线读写。

    为什么让 IMU 走总线而不是直接返回浮点数: 因为"量化"和"总线"才是真实系统
    里姿态估计误差的两个主要来源。加速度计 16384 LSB/g 意味着 1 LSB = 6e-5 g,
    折算到倾角约 0.0034 度 —— 这是加速度计倾角测量的**物理分辨率下限**, 与算
    法好坏无关。把这条链路建出来, 估计器章节的误差数字才有意义。

    噪声用 ``random.Random(seed)`` 固定种子: 测试需要可复现的数字, 否则
    "卡尔曼比纯积分好 1/2"这类断言会随机红。

    边界: 噪声是白噪声, **没有建模** 零偏不稳定性(Allan 方差里的 1/f 项)、
    温度漂移、轴间对准误差、非线性度。真实 IMU 的主要误差恰恰是零偏不稳定性,
    所以本模块的姿态估计数字偏乐观。
    """

    def __init__(self, bus, accel_reg=0x30, gyro_reg=0x36, config=None):
        self.bus = bus
        self.accel_reg = accel_reg
        self.gyro_reg = gyro_reg
        self.config = config if config is not None else ImuConfig()
        self._rng = random.Random(self.config.seed)

    def _to_i16(self, value):
        """饱和到 16 位有符号。真芯片在超量程时是饱和削顶, 不是回绕。"""
        v = int(round(value))
        if v > 32767:
            return 32767
        if v < -32768:
            return -32768
        return v

    def write_sample(self, accel_g, gyro_dps):
        """把物理量(加速度 g / 角速度 deg/s)加噪后量化写入寄存器。"""
        cfg = self.config
        for i in range(3):
            a = accel_g[i] + self._rng.gauss(0.0, cfg.accel_noise_g)
            g = gyro_dps[i] + self._rng.gauss(0.0, cfg.gyro_noise_dps)
            self.bus.write_u16(self.accel_reg + 2 * i,
                               self._to_i16(a * cfg.accel_lsb_per_g) & 0xFFFF)
            self.bus.write_u16(self.gyro_reg + 2 * i,
                               self._to_i16(g * cfg.gyro_lsb_per_dps) & 0xFFFF)

    @staticmethod
    def _from_u16(raw):
        """16 位无符号 -> 有符号。"""
        return raw - 0x10000 if raw >= 0x8000 else raw

    def read(self):
        """从寄存器读回并还原为物理量。

        返回 ``{"accel_g": [ax,ay,az], "gyro_dps": [gx,gy,gz]}``。
        """
        cfg = self.config
        accel = []
        gyro = []
        for i in range(3):
            accel.append(self._from_u16(self.bus.read_u16(self.accel_reg + 2 * i))
                         / cfg.accel_lsb_per_g)
            gyro.append(self._from_u16(self.bus.read_u16(self.gyro_reg + 2 * i))
                        / cfg.gyro_lsb_per_dps)
        return {"accel_g": accel, "gyro_dps": gyro}

    @staticmethod
    def tilt_from_accel(accel_g):
        """由重力分量算倾角(弧度), 假设机体系 x 前向、z 向上, 绕 y 轴倾斜。

        静止时可以认为是"绝对正确的倾角", 但一旦有线性加速度就完全不可信 ——
        这正是互补/卡尔曼滤波存在的理由: 用陀螺的高频部分 + 加速度计的低频
        部分。
        """
        ax, _ay, az = accel_g
        return math.atan2(ax, az)


# ---------------------------------------------------------------------------
# 外设 4: 固定周期调度器
# ---------------------------------------------------------------------------


@dataclass
class SchedulerStats:
    """周期调度统计。抖动/超时/丢拍是实时控制最核心的三个指标。"""

    ticks: int = 0
    start_times: list = field(default_factory=list)
    periods: list = field(default_factory=list)
    work_times: list = field(default_factory=list)
    overruns: int = 0
    missed_ticks: int = 0

    @property
    def mean_period(self):
        """平均实际周期。"""
        if not self.periods:
            return 0.0
        return sum(self.periods) / len(self.periods)

    @property
    def jitter_std(self):
        """周期标准差(总体标准差)。"""
        n = len(self.periods)
        if n < 2:
            return 0.0
        mu = self.mean_period
        return math.sqrt(sum((p - mu) ** 2 for p in self.periods) / n)

    @property
    def max_jitter(self):
        """单次最大偏差绝对值。"""
        if not self.periods:
            return 0.0
        return max(self.periods) - min(self.periods)

    @property
    def max_work_time(self):
        """单拍最大执行耗时。"""
        return max(self.work_times) if self.work_times else 0.0

    def as_dict(self):
        """展开成 dict。"""
        return {
            "ticks": self.ticks,
            "mean_period": self.mean_period,
            "jitter_std": self.jitter_std,
            "max_jitter": self.max_jitter,
            "overruns": self.overruns,
            "missed_ticks": self.missed_ticks,
            "max_work_time": self.max_work_time,
        }


class ControlCycleScheduler:
    """固定周期(如 1 kHz)任务调度器, 带抖动、超时统计与丢拍计数。

    模型: 每拍的名义周期 ``period``, 实际起止间隔为 ``period + jitter``, 其中
    jitter 由 ``jitter_std`` 的高斯噪声(带确定性种子)给出。若某拍任务执行时间
    超过 ``period``, 记一次 overrun, 并按 ``floor(work/period)`` 记丢拍, 下一拍
    顺延。

    为什么要在仿真里建这个: 平衡控制对**采样周期抖动**比对**计算延迟**更敏感。
    1 kHz(1 ms)下, 50 us 的抖动已经相当于 5% 的周期误差, 会让差分测速的噪声
    明显上升。跑一遍调度器能把"我的控制器在真实抖动下还稳不稳"变成一个数字。

    边界: 这是**软件时序模型**, 不涉及真实中断优先级、抢占、缓存效应。真机上
    由中断延迟和总线等待造成的偶发大抖动(几十 us 到 ms 级)不在这里体现。
    """

    def __init__(self, period=0.001, jitter_std=0.0, seed=1234):
        if period <= 0:
            raise RegisterError("周期必须为正: %r" % (period,))
        self.period = period
        self.jitter_std = jitter_std
        self._rng = random.Random(seed)
        self.stats = SchedulerStats()

    def _jitter(self):
        if self.jitter_std <= 0.0:
            return 0.0
        # 先取标准正态, 截断到 +-3 sigma, 再乘标准差 —— 避免极端值把平均周期拉偏
        z = self._rng.gauss(0.0, 1.0)
        return max(-3.0, min(3.0, z)) * self.jitter_std

    def run(self, n_ticks, task=None, work_time_fn=None):
        """跑 n_ticks 拍。

        ``task(t)``         : 每拍执行的回调(可为 None, 只做时序统计);
        ``work_time_fn(k)`` : 第 k 拍的执行耗时(秒), 默认 0。
        返回 ``SchedulerStats``。
        """
        if n_ticks <= 0:
            raise RegisterError("n_ticks 必须为正: %r" % (n_ticks,))
        stats = SchedulerStats()
        t = 0.0
        for k in range(n_ticks):
            work = work_time_fn(k) if work_time_fn else 0.0
            jitter = self._jitter()
            actual = self.period + jitter
            stats.start_times.append(t)
            stats.work_times.append(work)
            if k > 0:
                stats.periods.append(actual)
            extra = 0.0
            if work > self.period:
                stats.overruns += 1
                over = int(work // self.period)
                stats.missed_ticks += over
                extra = over * self.period
            if task is not None:
                task(t)
            t += actual + extra
            stats.ticks += 1
        self.stats = stats
        return stats

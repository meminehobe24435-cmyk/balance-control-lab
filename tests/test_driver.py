"""底层驱动抽象测试: 寄存器总线、PWM 死区、正交编码器、IMU、周期调度器。

最重要的一条是 **PWM 任何 tick 都不允许上下桥臂直通** —— 真机上那意味着电源短路。
测试扫遍整个 PWM 周期与多组占空比, 断言直通 tick 数恒为 0, 并同时验证死区**确实
存在**(否则"没有直通"可以靠"永远不导通"来作弊)。

其余重点是寄存器语义(写读一致、位域只动掩码内的位、越界抛错、CRC/超时可注入)
与编码器回绕(16 位计数器翻转到负数时, 不回绕累计值必须连续)。
"""

from __future__ import annotations

import math

import pytest

from bcl.driver import (BusStats, BusTimeout, ChecksumError, ControlCycleScheduler,
                        ImuConfig, ImuDevice, PwmChannel, QuadEncoder,
                        RegisterBus, RegisterError, SimRegisterBus, crc8)


@pytest.fixture()
def bus():
    return SimRegisterBus(size=256, retries=2)


# ---------------------------------------------------------------------------
# CRC
# ---------------------------------------------------------------------------


def test_crc8_standard_check_value():
    """CRC-8/ATM 的标准校验值: "123456789" -> 0xF4。"""
    assert crc8([ord(c) for c in "123456789"]) == 0xF4


def test_crc8_empty_is_init():
    assert crc8([]) == 0x00


def test_crc8_detects_single_bit_flip():
    base = [0x01, 0x02, 0x03, 0x04]
    ref = crc8(base)
    flipped = list(base)
    flipped[2] ^= 0x08
    assert crc8(flipped) != ref


def test_crc8_detects_byte_swap():
    assert crc8([0x01, 0x02]) != crc8([0x02, 0x01])


# ---------------------------------------------------------------------------
# 寄存器总线: 基本语义
# ---------------------------------------------------------------------------


def test_u8_roundtrip(bus):
    bus.write_u8(0x00, 0xA5)
    assert bus.read_u8(0x00) == 0xA5


def test_u16_is_little_endian(bus):
    bus.write_u16(0x02, 0xBEEF)
    assert bus.peek(0x02) == 0xEF
    assert bus.peek(0x03) == 0xBE
    assert bus.read_u16(0x02) == 0xBEEF


def test_u32_is_little_endian(bus):
    bus.write_u32(0x04, 0xDEADBEEF)
    assert [bus.peek(0x04 + i) for i in range(4)] == [0xEF, 0xBE, 0xAD, 0xDE]
    assert bus.read_u32(0x04) == 0xDEADBEEF


def test_out_of_range_address_raises(bus):
    with pytest.raises(RegisterError):
        bus.read_u8(300)
    with pytest.raises(RegisterError):
        bus.write_u8(-1, 0)


def test_u16_out_of_range_raises(bus):
    with pytest.raises(RegisterError):
        bus.read_u16(255)          # 需要 255 与 256 两个字节
    with pytest.raises(RegisterError):
        bus.write_u16(255, 1)


def test_misaligned_u16_raises(bus):
    with pytest.raises(RegisterError):
        bus.read_u16(0x03)


def test_u8_write_out_of_range_raises(bus):
    with pytest.raises(RegisterError):
        bus.write_u8(0x00, 256)
    with pytest.raises(RegisterError):
        bus.write_u8(0x00, -1)


def test_peek_poke_are_bounds_checked(bus):
    with pytest.raises(RegisterError):
        bus.peek(999)
    bus.poke(0x10, 0x3C)
    assert bus.read_u8(0x10) == 0x3C


# ---------------------------------------------------------------------------
# 寄存器总线: 位域
# ---------------------------------------------------------------------------


def test_field_write_only_touches_masked_bits(bus):
    """``mask`` 是**未移位**的连续低位掩码, ``shift`` 决定它落在寄存器的哪几位。

    例如"写 bit[5:4]"要写成 ``write_field(addr, 0x03, 4, value)``, 而不是
    ``mask=0x30``。这个约定让掩码永远是一段连续的 1, 便于校验; 代价是读代码时
    需要把 ``mask`` 和 ``shift`` 合起来看。
    """
    bus.write_u8(0x10, 0b1010_0000)
    bus.write_field(0x10, 0x0F, 0, 0b0101)
    assert bus.read_u8(0x10) == 0b1010_0101
    bus.write_field(0x10, 0x03, 4, 0b11)
    assert bus.read_u8(0x10) == 0b1011_0101


def test_field_read_shifts_down(bus):
    bus.write_u8(0x11, 0b0110_0000)
    assert bus.read_field(0x11, 0x03, 5) == 0b11
    assert bus.read_field(0x11, 0x0F, 0) == 0b0000


def test_field_write_on_u16(bus):
    bus.write_u16(0x20, 0xFFFF)
    bus.write_field(0x20, 0x00FF, 8, 0x12)
    assert bus.read_u16(0x20) == 0x12FF


def test_field_value_outside_mask_raises(bus):
    with pytest.raises(RegisterError):
        bus.write_field(0x10, 0x0F, 0, 0x10)


def test_non_contiguous_mask_raises(bus):
    with pytest.raises(RegisterError):
        bus.read_field(0x10, 0b101, 0)


def test_zero_mask_raises(bus):
    with pytest.raises(RegisterError):
        bus.write_field(0x10, 0x00, 0, 1)


def test_negative_shift_raises(bus):
    with pytest.raises(RegisterError):
        bus.write_field(0x10, 0x0F, -1, 1)


# ---------------------------------------------------------------------------
# 寄存器总线: 故障注入与重试
# ---------------------------------------------------------------------------


def test_crc_error_is_retried_and_eventually_succeeds(bus):
    bus.write_u8(0x00, 0x5A)
    bus.inject_crc_error(1)
    assert bus.read_u8(0x00) == 0x5A
    assert bus.stats.crc_errors == 1
    assert bus.stats.retries == 1


def test_crc_errors_exhaust_retries_and_raise(bus):
    bus.write_u8(0x00, 0x5A)
    bus.inject_crc_error(10)
    with pytest.raises(ChecksumError):
        bus.read_u8(0x00)
    assert bus.stats.crc_errors == 3          # 初次 + retries(2)


def test_timeout_is_retried_and_eventually_succeeds(bus):
    bus.write_u8(0x01, 0x11)
    bus.stall_next(2)
    assert bus.read_u8(0x01) == 0x11
    assert bus.stats.timeouts == 2
    assert bus.stats.retries == 2


def test_timeout_beyond_retries_raises(bus):
    bus.stall_next(5)
    with pytest.raises(BusTimeout):
        bus.read_u8(0x00)
    assert bus.stats.timeouts == 3


def test_clear_faults_stops_leaking_injected_errors(bus):
    """故障注入是"待消费"计数器, 不清掉会泄漏到后续正常操作上。"""
    bus.stall_next(5)
    with pytest.raises(BusTimeout):
        bus.read_u8(0x00)
    bus.clear_faults()
    before = bus.stats.timeouts
    bus.write_u8(0x00, 0x01)
    assert bus.read_u8(0x00) == 0x01
    assert bus.stats.timeouts == before


def test_bus_stats_counts_transfers(bus):
    bus.write_u8(0x00, 1)
    bus.write_u8(0x01, 2)
    bus.read_u8(0x00)
    assert isinstance(bus.stats, BusStats)
    assert bus.stats.writes == 2
    assert bus.stats.reads == 1
    assert bus.stats.transfers == 3
    d = bus.stats.as_dict()
    assert set(d) == {"reads", "writes", "retries", "crc_errors", "timeouts"}


def test_register_bus_is_abstract():
    with pytest.raises(TypeError):
        RegisterBus(size=16)


def test_read_u16_consumes_two_raw_reads(bus):
    bus.write_u16(0x02, 0x1234)
    before = bus.stats.reads
    bus.read_u16(0x02)
    assert bus.stats.reads == before + 1     # 统计按"寄存器访问"计, 不按字节


# ---------------------------------------------------------------------------
# PWM
# ---------------------------------------------------------------------------


def test_pwm_quantizes_to_12_bits(bus):
    pwm = PwmChannel(bus, base=0x40, channel=0)
    pwm.set_duty(0.5)
    assert pwm.read_duty_raw() & 0x0FFF == 2048      # round(0.5*4095)
    assert pwm.get_duty() == pytest.approx(2048.0 / 4095.0, rel=1e-12)
    assert abs(pwm.get_duty() - 0.5) < 1.0 / 4095.0


def test_pwm_quantization_error_is_half_lsb(bus):
    pwm = PwmChannel(bus, base=0x40, channel=0)
    for duty in (0.1, 0.25, 0.3333, 0.7, 0.9):
        pwm.set_duty(duty)
        assert abs(pwm.get_duty() - duty) <= 0.5 / 4095.0 + 1e-12


def test_pwm_negative_duty_sets_sign_bit(bus):
    pwm = PwmChannel(bus, base=0x40, channel=0)
    pwm.set_duty(-0.4)
    raw = pwm.read_duty_raw()
    assert raw & 0x8000
    assert pwm.get_duty() < 0.0


def test_pwm_saturation_at_duty_limit(bus):
    pwm = PwmChannel(bus, base=0x40, channel=0)
    pwm.set_duty(5.0)
    assert pwm.get_duty() <= pwm.duty_limit + 1e-12
    pwm.set_duty(-5.0)
    assert pwm.get_duty() >= -pwm.duty_limit - 1e-12


def test_pwm_dead_time_limits_max_duty_ticks(bus):
    pwm = PwmChannel(bus, base=0x40, channel=0)
    pwm.set_duty(1.0)
    assert pwm.duty_ticks() <= pwm.max_duty_ticks
    assert pwm.max_duty_ticks == pwm.period_ticks - pwm.dead_time_ticks


def test_pwm_never_shoot_through_over_full_period(bus):
    """核心断言: 扫遍整个周期与多组占空比, 直通 tick 数必须恒为 0。"""
    pwm = PwmChannel(bus, base=0x40, channel=0)
    for duty in (-0.95, -0.5, -0.1, -1e-9, 0.0, 1e-9, 0.1, 0.5, 0.95, 1.0):
        pwm.set_duty(duty)
        assert pwm.sweep_shoot_through() == []


def test_pwm_dead_time_gap_actually_exists(bus):
    """反向检查: 死区必须真的把上下管隔开, 否则"无直通"可能只是因为一直不导通。"""
    pwm = PwmChannel(bus, base=0x40, channel=0)
    pwm.set_duty(0.5)
    ticks = pwm.duty_ticks()
    a_high = [k for k in range(pwm.period_ticks)
              if pwm.gate_states(k)[0]]
    assert len(a_high) == ticks
    assert ticks > 0
    # 下管窗口长度 = period - ticks - dead
    a_low = [k for k in range(pwm.period_ticks) if pwm.gate_states(k)[1]]
    assert len(a_low) == pwm.period_ticks - ticks - pwm.dead_time_ticks


def test_pwm_effective_duty_is_reduced_by_dead_time(bus):
    pwm = PwmChannel(bus, base=0x40, channel=0)
    pwm.set_duty(0.5)
    assert pwm.effective_duty() < pwm.get_duty()
    assert pwm.effective_duty() == pytest.approx(
        (pwm.duty_ticks() - pwm.dead_time_ticks) / float(pwm.period_ticks), rel=1e-12)


def test_pwm_enable_bit_is_set_by_set_duty(bus):
    pwm = PwmChannel(bus, base=0x40, channel=0)
    assert not pwm.is_enabled()
    pwm.set_duty(0.2)
    assert pwm.is_enabled()


def test_pwm_channels_use_different_registers(bus):
    p0 = PwmChannel(bus, base=0x40, channel=0)
    p1 = PwmChannel(bus, base=0x40, channel=1)
    p0.set_duty(0.2)
    p1.set_duty(0.8)
    assert p0.read_duty_raw() != p1.read_duty_raw()
    assert p0.duty_reg != p1.duty_reg


def test_pwm_rejects_non_12bit_resolution(bus):
    with pytest.raises(RegisterError):
        PwmChannel(bus, base=0x40, channel=0, resolution=16)


def test_pwm_rejects_nan(bus):
    with pytest.raises(RegisterError):
        PwmChannel(bus, base=0x40, channel=0).set_duty(float("nan"))


def test_pwm_gate_states_bounds_checked(bus):
    pwm = PwmChannel(bus, base=0x40, channel=0)
    pwm.set_duty(0.3)
    with pytest.raises(RegisterError):
        pwm.gate_states(pwm.period_ticks)


# ---------------------------------------------------------------------------
# 正交编码器
# ---------------------------------------------------------------------------


def _forward_cycle():
    return [(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)]


def _reverse_cycle():
    return [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]


def test_encoder_forward_sequence_counts_up(bus):
    enc = QuadEncoder(bus, count_reg=0x20, ppr=1024)
    for a, b in _forward_cycle():
        enc.update(a, b)
    assert enc.count_total == 4
    assert enc.count_raw == 4


def test_encoder_reverse_sequence_counts_down(bus):
    enc = QuadEncoder(bus, count_reg=0x20, ppr=1024)
    for a, b in _reverse_cycle():
        enc.update(a, b)
    assert enc.count_total == -4


def test_encoder_round_trip_returns_to_zero(bus):
    enc = QuadEncoder(bus, count_reg=0x20, ppr=1024)
    for a, b in _forward_cycle():
        enc.update(a, b)
    for a, b in _reverse_cycle():
        enc.update(a, b)
    assert enc.count_total == 0


def test_encoder_same_state_gives_zero_delta(bus):
    enc = QuadEncoder(bus, count_reg=0x20, ppr=1024)
    assert enc.update(0, 1) == 1     # 00 -> 01 : +1
    assert enc.update(1, 1) == 1     # 01 -> 11 : +1
    assert enc.update(1, 1) == 0     # 状态不变 : 0
    assert enc.count_total == 2


def test_encoder_invalid_double_transition_is_counted_not_missed(bus):
    """非法双重跳变(漏采一个状态)计 0 增量, 但必须被记录 —— 它是丢脉冲的证据。"""
    enc = QuadEncoder(bus, count_reg=0x20, ppr=1024)
    enc.update(0, 0)
    assert enc.update(1, 1) == 0
    assert enc.invalid_transitions == 1
    assert enc.count_total == 0


def test_encoder_state_table_is_antisymmetric(bus):
    """正转 +1 的每一对状态, 反转必须 -1。"""
    enc = QuadEncoder(bus, count_reg=0x20, ppr=1024)
    forward = _forward_cycle()
    for i in range(len(forward) - 1):
        enc._state = QuadEncoder.state_of(*forward[i])
        assert enc.decode_step(*forward[i + 1]) == 1
        enc._state = QuadEncoder.state_of(*forward[i + 1])
        assert enc.decode_step(*forward[i]) == -1


def test_encoder_wraps_and_keeps_total_continuous(bus):
    """16 位回绕: raw 翻到负数, 但 total 必须连续 —— 这是最容易翻车的地方。"""
    enc = QuadEncoder(bus, count_reg=0x20, ppr=1024)
    enc.seed_count(32766)
    for a, b in [(0, 1), (1, 1), (1, 0), (0, 0)]:
        enc.update(a, b)
    assert enc.count_raw == -32766
    assert enc.count_total == 32770
    assert enc.wrap_count == 1
    assert enc.needs_wrap_attention() is False


def test_encoder_wrap_below_negative_range(bus):
    enc = QuadEncoder(bus, count_reg=0x20, ppr=1024)
    enc.seed_count(-32766)
    for a, b in [(1, 0), (1, 1), (0, 1), (0, 0)]:
        enc.update(a, b)
    assert enc.count_raw == 32766
    assert enc.count_total == -32770
    assert enc.wrap_count == 1


def test_encoder_register_matches_raw_count(bus):
    enc = QuadEncoder(bus, count_reg=0x20, ppr=1024)
    enc.advance(1234)
    raw = bus.read_u16(0x20)
    assert raw == 1234
    assert enc.read_register() == 1234


def test_encoder_register_two_complement_for_negative(bus):
    enc = QuadEncoder(bus, count_reg=0x20, ppr=1024)
    enc.advance(-5)
    assert bus.read_u16(0x20) == 0xFFFB
    assert enc.read_register() == -5


def test_encoder_reset_clears_counts(bus):
    enc = QuadEncoder(bus, count_reg=0x20, ppr=1024)
    enc.advance(500)
    enc.reset()
    assert enc.count_total == 0
    assert enc.count_raw == 0
    assert bus.read_u16(0x20) == 0
    assert enc.speed_counts == 0.0


def test_encoder_speed_low_pass_reduces_noise(bus):
    """低通后的速度方差必须显著小于原始差分(量化噪声被 1/dt 放大)。"""
    enc = QuadEncoder(bus, count_reg=0x20, ppr=1024, lpf_alpha=0.2)
    enc.seed_count(0)
    raw = []
    filtered = []
    seq = [0, 1, 1, 2, 3, 3, 4, 5, 5, 6, 7, 7, 8, 9, 9, 10]
    for i in range(1, len(seq)):
        enc.advance(seq[i] - seq[i - 1])
        raw.append((seq[i] - seq[i - 1]) / 0.001)
        filtered.append(enc.update_speed(0.001))
    mean_raw = sum(raw) / len(raw)
    mean_f = sum(filtered) / len(filtered)
    var_raw = sum((v - mean_raw) ** 2 for v in raw) / len(raw)
    var_f = sum((v - mean_f) ** 2 for v in filtered) / len(filtered)
    assert var_f < 0.5 * var_raw


def test_encoder_speed_rps_uses_four_times_ppr(bus):
    enc = QuadEncoder(bus, count_reg=0x20, ppr=1024, lpf_alpha=1.0)
    enc.seed_count(0)
    enc.advance(4096)
    enc.update_speed(1.0)
    assert enc.speed_rps() == pytest.approx(1.0)
    assert enc.speed_rpm == pytest.approx(60.0)


def test_encoder_invalid_dt_raises(bus):
    enc = QuadEncoder(bus, count_reg=0x20, ppr=1024)
    with pytest.raises(RegisterError):
        enc.update_speed(0.0)


# ---------------------------------------------------------------------------
# IMU
# ---------------------------------------------------------------------------


def test_imu_roundtrip_quantization_within_half_lsb(bus):
    cfg = ImuConfig(accel_noise_g=0.0, gyro_noise_dps=0.0)
    imu = ImuDevice(bus, accel_reg=0x30, gyro_reg=0x36, config=cfg)
    imu.write_sample([0.1, -0.2, 0.97], [1.0, -2.0, 3.0])
    out = imu.read()
    for got, want in zip(out["accel_g"], [0.1, -0.2, 0.97]):
        assert abs(got - want) <= 0.5 / cfg.accel_lsb_per_g + 1e-12
    for got, want in zip(out["gyro_dps"], [1.0, -2.0, 3.0]):
        assert abs(got - want) <= 0.5 / cfg.gyro_lsb_per_dps + 1e-12


def test_imu_noise_is_roughly_the_configured_sigma(bus):
    cfg = ImuConfig(accel_noise_g=0.05, gyro_noise_dps=1.0, seed=11)
    imu = ImuDevice(bus, config=cfg)
    xs = []
    for _ in range(2000):
        imu.write_sample([0.0, 0.0, 1.0], [0.0, 0.0, 0.0])
        xs.append(imu.read()["accel_g"][0])
    mean = sum(xs) / len(xs)
    std = math.sqrt(sum((v - mean) ** 2 for v in xs) / len(xs))
    assert abs(mean) < 0.01
    assert std == pytest.approx(0.05, rel=0.15)


def test_imu_saturates_instead_of_wrapping(bus):
    cfg = ImuConfig(accel_noise_g=0.0, gyro_noise_dps=0.0)
    imu = ImuDevice(bus, config=cfg)
    imu.write_sample([100.0, 0.0, 0.0], [10000.0, 0.0, 0.0])
    out = imu.read()
    assert out["accel_g"][0] == pytest.approx(32767.0 / cfg.accel_lsb_per_g, rel=1e-12)
    assert out["accel_g"][0] > 0.0
    assert out["gyro_dps"][0] == pytest.approx(32767.0 / cfg.gyro_lsb_per_dps, rel=1e-12)


def test_imu_tilt_from_accel():
    assert ImuDevice.tilt_from_accel([0.0, 0.0, 1.0]) == pytest.approx(0.0)
    assert ImuDevice.tilt_from_accel([1.0, 0.0, 0.0]) == pytest.approx(math.pi / 2)
    assert ImuDevice.tilt_from_accel([0.0, 0.0, -1.0]) == pytest.approx(math.pi)


def test_imu_tilt_from_accel_recovers_known_angle(bus):
    cfg = ImuConfig(accel_noise_g=0.0, gyro_noise_dps=0.0)
    imu = ImuDevice(bus, config=cfg)
    angle = 0.25
    imu.write_sample([math.sin(angle), 0.0, math.cos(angle)], [0.0, 0.0, 0.0])
    got = imu.tilt_from_accel(imu.read()["accel_g"])
    assert got == pytest.approx(angle, abs=2e-4)


# ---------------------------------------------------------------------------
# 周期调度器
# ---------------------------------------------------------------------------


def test_scheduler_exact_ticks_without_jitter():
    sched = ControlCycleScheduler(period=0.001, jitter_std=0.0)
    stats = sched.run(1000)
    assert stats.ticks == 1000
    assert stats.mean_period == pytest.approx(0.001, rel=1e-12)
    assert stats.jitter_std == pytest.approx(0.0)
    assert stats.overruns == 0
    assert stats.missed_ticks == 0


def test_scheduler_jitter_is_roughly_as_configured():
    sched = ControlCycleScheduler(period=0.001, jitter_std=2e-5, seed=3)
    stats = sched.run(2000)
    assert stats.ticks == 2000
    assert stats.mean_period == pytest.approx(0.001, abs=2e-6)
    # 截断到 +-3 sigma 会让实测标准差略小于注入值
    assert 0.6 * 2e-5 < stats.jitter_std < 2e-5
    assert stats.max_jitter > 0.0


def test_scheduler_counts_overruns_and_missed_ticks():
    sched = ControlCycleScheduler(period=0.001, jitter_std=0.0)
    heavy = lambda k: 0.0025 if k % 100 == 0 else 0.0001   # noqa: E731
    stats = sched.run(1000, work_time_fn=heavy)
    assert stats.overruns == 10
    assert stats.missed_ticks == 20          # floor(0.0025/0.001) = 2 拍/次
    assert stats.max_work_time == pytest.approx(0.0025)


def test_scheduler_calls_task_every_tick():
    seen = []
    sched = ControlCycleScheduler(period=0.001, jitter_std=0.0)
    sched.run(50, task=lambda t: seen.append(t))
    assert len(seen) == 50
    assert seen[0] == pytest.approx(0.0)
    assert seen[1] == pytest.approx(0.001, rel=1e-12)


def test_scheduler_rejects_non_positive_period():
    with pytest.raises(RegisterError):
        ControlCycleScheduler(period=0.0)
    with pytest.raises(RegisterError):
        ControlCycleScheduler(period=0.001).run(0)


def test_scheduler_stats_dict_keys():
    stats = ControlCycleScheduler(period=0.001).run(10)
    keys = set(stats.as_dict())
    assert keys == {"ticks", "mean_period", "jitter_std", "max_jitter",
                    "overruns", "missed_ticks", "max_work_time"}

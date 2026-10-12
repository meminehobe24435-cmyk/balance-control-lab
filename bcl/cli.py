"""命令行界面: ``python -m bcl <子命令>``。

子命令:

* ``demo``      —— 倒立摆起摆 + 平衡, 打印每个阶段的指标;
* ``bench``     —— 跑完整评测套件并打印指标表;
* ``traj``      —— 打印一条 S 曲线的关键特征并与梯形/五次多项式对比;
* ``driver``    —— 演示寄存器读写/位域/编码器解码/PWM 死区/周期调度;
* ``selfcheck`` —— 跑 ``tools/selfcheck.py`` 并透传退出码。

设计约定: 子命令**只打印、不写文件**, 真正的计算都在 ``bcl`` 包里(CLI 只做参数
解析和排版)。这样同一份逻辑既能被 CLI 调用也能被测试直接调用, 不会出现"CLI 里
跑出来的数字和测试里的不一样"。
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys

from .bench import (format_report, overshoot_percent, run_suite,
                    scenario_trajectory, settling_time, steady_state_error)
from .control import SwingUpBalanceController, cartpole_lqr_gain
from .driver import (ControlCycleScheduler, ImuConfig, ImuDevice, PwmChannel,
                     QuadEncoder, SimRegisterBus, crc8)
from .dynamics import CartPoleParams
from .sim import Simulator
from .trajectory import QuinticProfile, SCurveProfile, TrapezoidalProfile

BANNER = "balance-control-lab  |  纯仿真, 无真实硬件"


def _try_reconfigure(stream, **kwargs):
    """尝试重新配置一个文本流, 失败返回 False(不抛异常)。"""
    try:
        stream.reconfigure(**kwargs)
        return True
    except (AttributeError, ValueError, OSError, TypeError):
        return False


def configure_output():
    """把 stdout/stderr 切到 UTF-8, 并容错无法编码的字符。

    为什么需要: Windows 默认控制台代码页是 GBK, 而本项目所有文案是 UTF-8。不切换
    的话, 只要出现一个 GBK 编不出来的字符, ``print`` 就直接抛 UnicodeEncodeError
    把整个子命令打断 —— 一个排版字符搞崩 CLI 显然不合理。

    三层防御: 优先 ``reconfigure`` 到 UTF-8; 不支持时退回 ``errors="replace"``
    (最坏情况打印问号, 但程序继续跑); 两者都失败就什么都不做。

    返回 ``(stdout_ok, stderr_ok)``, 便于测试断言"至少有一层兜住了"。
    """
    out_ok = (_try_reconfigure(sys.stdout, encoding="utf-8", errors="replace")
              or _try_reconfigure(sys.stdout, errors="replace"))
    err_ok = (_try_reconfigure(sys.stderr, encoding="utf-8", errors="replace")
              or _try_reconfigure(sys.stderr, errors="replace"))
    return (out_ok, err_ok)


# ---------------------------------------------------------------------------
# demo: 起摆 + 平衡
# ---------------------------------------------------------------------------


def run_demo(theta0=None, t_end=15.0, verbose=True):
    """跑"下垂起摆 -> 平衡"的完整流程, 返回分阶段指标 dict。"""
    p = CartPoleParams()
    k, info = cartpole_lqr_gain(p)
    theta0 = math.pi if theta0 is None else theta0
    ctrl = SwingUpBalanceController(k, p)
    sim = Simulator(p, dt=1e-3, control_dt=0.01, force_limit=20.0)
    res = sim.run([0.0, 0.0, theta0, 0.0], ctrl, t_end, label="swingup")

    capture = ctrl.capture_time()
    theta = res.theta()
    energy = [ctrl.swing.pendulum_energy([0.0, 0.0, theta[i], res.theta_dot()[i]])
              for i in range(len(res.t))]

    if capture is None:
        swing = {"captured": False, "final_energy_J": energy[-1]}
        balance = None
    else:
        swing = {
            "captured": True,
            "swing_up_time_s": capture,
            "energy_target_J": ctrl.swing.target_energy(),
            "energy_at_capture_J": _energy_at(res, ctrl, capture),
            "max_cart_travel_m": max(abs(v) for v in res.position()),
            "swing_up_energy": _energy_window(res, 0.0, capture),
        }
        kc = min(range(len(res.t)), key=lambda i: abs(res.t[i] - capture))
        tail_t = res.t[kc:]
        tail_th = theta[kc:]
        balance = {
            "settling_time_s": settling_time(tail_t, tail_th, 1e-3),
            "overshoot_pct": overshoot_percent(tail_th, tail_th[0], 0.0),
            "steady_state_error_rad": steady_state_error(tail_t, tail_th, 0.0),
            "control_energy": res.control_energy(),
            "balance_energy": _energy_window(res, capture, res.t[-1]),
            "peak_control_N": res.control_peak(),
            "max_tilt_after_capture_rad": max(abs(v) for v in tail_th),
            "final_cart_position_m": res.position()[-1],
            "final_tilt_rad": theta[-1],
        }

    out = {
        "initial_tilt_rad": theta0,
        "duration_s": t_end,
        "swing_up": swing,
        "balance": balance,
        "mode_switches": len(ctrl.switch_log),
        "lqr_gain": list(k[0]),
        "riccati_residual": info["residual"],
        "closed_loop_spectral_radius": info["spectral_radius"],
    }
    if verbose:
        print(BANNER)
        print("=" * 72)
        print("demo: 倒立摆起摆 + 平衡 (初始倾角 %.4f rad = %.1f deg)"
              % (theta0, math.degrees(theta0)))
        print("-" * 72)
        print("阶段 1  能量法起摆 (Astrom-Furuta 能量整形):")
        if swing["captured"]:
            print("        起摆耗时            %.3f s" % swing["swing_up_time_s"])
            print("        捕获时摆能量        %.6f J (目标 %.6f J)"
                  % (swing["energy_at_capture_J"], swing["energy_target_J"]))
            print("        小车最大位移        %.3f m" % swing["max_cart_travel_m"])
            print("        起摆阶段控制能量    %.3f N^2*s" % swing["swing_up_energy"])
        else:
            print("        未能在 %.1f s 内捕获顶点 (剩余能量 %.6f J)"
                  % (t_end, swing["final_energy_J"]))
        print("阶段 2  LQR 平衡 (K = %s):" % _vec(k[0]))
        if balance is not None:
            print("        稳定时间 (|theta|<1e-3)  %.3f s" % balance["settling_time_s"])
            print("        超调量                   %.2f %%" % balance["overshoot_pct"])
            print("        稳态误差                 %.3e rad"
                  % balance["steady_state_error_rad"])
            print("        最大倾角(捕获后)         %.4f rad"
                  % balance["max_tilt_after_capture_rad"])
            print("        控制能量 (平衡阶段)      %.3f N^2*s"
                  % balance["balance_energy"])
            print("        控制能量 (全程)          %.3f N^2*s"
                  % balance["control_energy"])
            print("        控制峰值                 %.3f N" % balance["peak_control_N"])
            print("        末态 (x, theta)          (%.3e, %.3e)"
                  % (balance["final_cart_position_m"], balance["final_tilt_rad"]))
        print("-" * 72)
        print("模式切换次数 %d | Riccati 残差 %.3e | 闭环谱半径 %.6f"
              % (out["mode_switches"], out["riccati_residual"],
                 out["closed_loop_spectral_radius"]))
    return out


def _energy_at(res, ctrl, t_capture):
    """取捕获时刻的摆能量。"""
    idx = min(range(len(res.t)), key=lambda i: abs(res.t[i] - t_capture))
    return ctrl.swing.pendulum_energy(
        [0.0, 0.0, res.theta()[idx], res.theta_dot()[idx]])


def _energy_window(res, t_lo, t_hi):
    """统计 ``[t_lo, t_hi]`` 区间内的控制能量 ``integral F^2 dt``。

    分阶段统计是必要的: 起摆阶段的控制量和平衡阶段差了近一个数量级, 混在一起
    算会得到一个"主要由起摆主导"的数字, 完全看不出平衡控制省不省力。
    """
    total = 0.0
    for k in range(len(res.t) - 1):
        if res.t[k] < t_lo or res.t[k + 1] > t_hi:
            continue
        total += 0.5 * (res.u[k] ** 2 + res.u[k + 1] ** 2) * (res.t[k + 1] - res.t[k])
    return total


def _vec(values):
    """把数组排成紧凑字符串。"""
    return "[" + ", ".join("%.3f" % v for v in values) + "]"


# ---------------------------------------------------------------------------
# traj: 轨迹特征
# ---------------------------------------------------------------------------


def run_traj(p0=0.0, p1=1.5, vmax=1.5, amax=6.0, jmax=40.0, verbose=True):
    """规划三条轨迹并打印关键特征, 返回对比 dict。"""
    profiles = {
        "trapezoid": TrapezoidalProfile(p0, p1, vmax, amax),
        "scurve": SCurveProfile(p0, p1, vmax, amax, jmax),
        "quintic": QuinticProfile(p0, p1, vmax, amax),
    }
    out = {}
    for name, prof in profiles.items():
        out[name] = {
            "duration_s": prof.duration,
            "peak_velocity": prof.max_abs("velocity"),
            "peak_acceleration": prof.max_abs("acceleration"),
            "peak_jerk": prof.max_abs("jerk"),
            "end_position": prof.position(prof.duration),
            "end_velocity": prof.velocity(prof.duration),
            "end_acceleration": prof.acceleration(prof.duration),
            "violates_vmax": prof.max_abs("velocity") > vmax * (1.0 + 1e-9),
            "violates_amax": prof.max_abs("acceleration") > amax * (1.0 + 1e-9),
            "violates_jmax": prof.max_abs("jerk") > jmax * (1.0 + 1e-9),
        }
    if verbose:
        print(BANNER)
        print("=" * 72)
        print("traj: 位移 %.3f -> %.3f, vmax=%.3f, amax=%.3f, jmax=%.3f"
              % (p0, p1, vmax, amax, jmax))
        print("-" * 72)
        head = "%-11s %9s %9s %9s %9s  %s" % (
            "轨迹", "时长[s]", "峰值v", "峰值a", "峰值jerk", "违反限幅")
        print(head)
        for name in ("trapezoid", "scurve", "quintic"):
            d = out[name]
            bad = []
            if d["violates_vmax"]:
                bad.append("vmax")
            if d["violates_amax"]:
                bad.append("amax")
            if d["violates_jmax"]:
                bad.append("jmax")
            print("%-11s %9.4f %9.4f %9.4f %9.4f  %s"
                  % (name, d["duration_s"], d["peak_velocity"],
                     d["peak_acceleration"], d["peak_jerk"],
                     ",".join(bad) if bad else "-"))
        print("-" * 72)
        print("S 曲线分段边界: %s" % _vec(profiles["scurve"].segment_boundaries()))
        print("S 曲线抓住了 vmax: %s / amax: %s"
              % (profiles["scurve"].reaches_vmax, profiles["scurve"].reaches_amax))
        print("说明: 五次多项式**不接受** jerk 限幅参数, 它的端点 jerk = 60*d/T^3,")
        print("      本例为 %.4f, 而 S 曲线的 jerk 恰好等于 jmax = %.4f。"
              % (out["quintic"]["peak_jerk"], jmax))
        print("      vmax/amax 违反标志直接说明: 五次多项式在给定时长下不保证满足限幅。")

        # 跟踪误差对比
        print("-" * 72)
        print("单关节伺服跟踪 (1 kHz 控制环, 带速度/加速度前馈):")
        print("%-11s %14s %14s %14s" % ("轨迹", "RMS(实测)", "峰值(实测)", "RMS(真值)"))
        for kind in ("trapezoid", "scurve", "quintic"):
            d, _ = scenario_trajectory(kind, distance=abs(p1 - p0), vmax=vmax,
                                       amax=amax, jmax=jmax)
            print("%-11s %14.3e %14.3e %14.3e"
                  % (kind, d["tracking_rms_rad"], d["tracking_peak_rad"],
                     d["tracking_rms_true_rad"]))
        print("编码器量化给出的 RMS 地板: %.3e rad" % d["encoding_floor_rad"])
    return out


# ---------------------------------------------------------------------------
# driver: 底层驱动演示
# ---------------------------------------------------------------------------


def run_driver(verbose=True):
    """演示底层驱动各组件, 返回统计 dict。"""
    bus = SimRegisterBus(size=256, retries=2)
    out = {}

    # 1. 基本读写
    bus.write_u8(0x00, 0xA5)
    bus.write_u16(0x02, 0xBEEF)
    bus.write_u32(0x04, 0xDEADBEEF)
    out["read_u8"] = bus.read_u8(0x00)
    out["read_u16"] = bus.read_u16(0x02)
    out["read_u32"] = bus.read_u32(0x04)
    out["crc8_of_frame"] = crc8([0x02, 0x04, 0xEF])
    out["stats_clean"] = dict(bus.stats.as_dict())

    # 2. 位域: 只影响掩码内的位
    bus.write_u8(0x10, 0b1010_0000)
    bus.write_field(0x10, 0x0F, 0, 0b0101)
    out["bitfield_value"] = bus.read_u8(0x10)
    out["bitfield_high_bits_preserved"] = (bus.read_u8(0x10) & 0xF0) == 0b1010_0000

    # 3. CRC 错误注入 + 重试
    before = bus.stats.error_count
    bus.inject_crc_error(1)
    out["crc_retry_read"] = bus.read_u8(0x00)
    out["crc_retries"] = bus.stats.retries

    # 4. 超时注入 + 重试耗尽
    bus.stall_next(1)
    out["timeout_retry_read"] = bus.read_u8(0x00)
    bus.stall_next(5)
    try:
        bus.read_u8(0x00)
        out["timeout_exhausted"] = False
    except Exception as exc:  # noqa: BLE001 —— 演示用, 打印真实类型
        out["timeout_exhausted"] = type(exc).__name__
    out["stats_after_faults"] = dict(bus.stats.as_dict())
    bus.clear_faults()

    # 5. PWM 量化 + 死区
    pwm = PwmChannel(bus, base=0x40, channel=0)
    pwm.set_duty(0.5)
    out["pwm_raw"] = pwm.read_duty_raw()
    out["pwm_duty_readback"] = pwm.get_duty()
    out["pwm_effective_duty"] = pwm.effective_duty()
    out["pwm_duty_ticks"] = pwm.duty_ticks()
    out["pwm_dead_time_ticks"] = pwm.dead_time_ticks
    out["pwm_shoot_through_ticks"] = pwm.sweep_shoot_through()
    pwm.set_duty(1.0)
    out["pwm_saturated_duty_ticks"] = pwm.duty_ticks()
    out["pwm_saturated_max_ticks"] = pwm.max_duty_ticks

    # 6. 正交编码器
    enc = QuadEncoder(bus, count_reg=0x20, ppr=1024)
    forward = [(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)]
    for a, b in forward:
        enc.update(a, b)
    out["encoder_forward_count"] = enc.count_total
    for a, b in [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]:
        enc.update(a, b)
    out["encoder_after_reverse_count"] = enc.count_total
    enc.seed_count(32766)
    for a, b in forward[1:]:
        enc.update(a, b)
    out["encoder_wrapped_raw"] = enc.count_raw
    out["encoder_wrapped_total"] = enc.count_total
    out["encoder_wrap_count"] = enc.wrap_count

    # 7. IMU
    imu = ImuDevice(bus, accel_reg=0x30, gyro_reg=0x36, config=ImuConfig())
    imu.write_sample([0.05, 0.0, 0.99875], [0.0, 1.5, 0.0])
    sample = imu.read()
    out["imu_accel_g"] = sample["accel_g"]
    out["imu_gyro_dps"] = sample["gyro_dps"]
    out["imu_tilt_rad"] = imu.tilt_from_accel(sample["accel_g"])

    # 8. 周期调度
    sched = ControlCycleScheduler(period=0.001, jitter_std=2.0e-5, seed=7)
    work = lambda k: 3.0e-4 if k % 500 == 0 else 1.5e-4  # noqa: E731
    stats = sched.run(2000, work_time_fn=work)
    out["scheduler"] = stats.as_dict()
    out["bus_stats"] = bus.stats.as_dict()
    out["crc_errors_seen"] = bus.stats.crc_errors
    out["timeouts_seen"] = bus.stats.timeouts
    del before

    if verbose:
        print(BANNER)
        print("=" * 72)
        print("driver: 寄存器总线 / PWM 死区 / 正交编码器 / IMU / 周期调度")
        print("-" * 72)
        print("寄存器  u8=0x%02X  u16=0x%04X  u32=0x%08X"
              % (out["read_u8"], out["read_u16"], out["read_u32"]))
        print("位域    写入 0x05 到 bit[3:0] 后寄存器 = 0x%02X (高 4 位保持 %s)"
              % (out["bitfield_value"], out["bitfield_high_bits_preserved"]))
        print("CRC     注入 1 次错误后重读成功 = 0x%02X, 累计重试 %d 次"
              % (out["crc_retry_read"], out["crc_retries"]))
        print("超时    注入 1 次后重读成功 = 0x%02X; 注入 5 次(超过重试上限) -> %s"
              % (out["timeout_retry_read"], out["timeout_exhausted"]))
        print("PWM     写 0.5 -> 寄存器 0x%04X, 读回 %.9f, 死区后等效 %.9f"
              % (out["pwm_raw"], out["pwm_duty_readback"], out["pwm_effective_duty"]))
        print("        12 位量化误差 %.3e; 写 1.0 被占空比限幅到 %d tick (死区上限 %d tick)"
              % (abs(0.5 - out["pwm_duty_readback"]),
                 out["pwm_saturated_duty_ticks"], out["pwm_saturated_max_ticks"]))
        print("        全周期扫描直通 tick 数 = %d (必须为 0)"
              % len(out["pwm_shoot_through_ticks"]))
        print("编码器  正转一圈计数 %d, 反转后 %d, 回绕后 raw=%d total=%d (回绕 %d 次)"
              % (out["encoder_forward_count"], out["encoder_after_reverse_count"],
                 out["encoder_wrapped_raw"], out["encoder_wrapped_total"],
                 out["encoder_wrap_count"]))
        print("IMU     accel=%s g, gyro=%s dps, 反算倾角=%.6f rad"
              % (_vec(out["imu_accel_g"]), _vec(out["imu_gyro_dps"]),
                 out["imu_tilt_rad"]))
        s = out["scheduler"]
        print("调度    2000 拍, 平均周期 %.6f s, 抖动 std %.3e, 超时 %d 次, 丢拍 %d"
              % (s["mean_period"], s["jitter_std"], s["overruns"], s["missed_ticks"]))
        print("总线统计 %s" % out["bus_stats"])
        print("-" * 72)
        print("注: 全部是内存模拟, 没有真实 MCU/SPI/I2C/电机/IMU。")
    return out


# ---------------------------------------------------------------------------
# bench / selfcheck
# ---------------------------------------------------------------------------


def run_bench(verbose=True):
    """跑评测套件。"""
    result = run_suite()
    if verbose:
        print(BANNER)
        print("=" * 72)
        print(format_report(result))
        print("=" * 72)
    return result


def run_selfcheck(verbose=True):
    """调用 tools/selfcheck.py 并返回退出码。"""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = os.path.join(root, "tools", "selfcheck.py")
    if verbose:
        print("$ python %s" % os.path.relpath(script, root))
    proc = subprocess.run([sys.executable, script], cwd=root)
    if verbose:
        print("selfcheck 退出码: %d" % proc.returncode)
    return proc.returncode


# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------


def build_parser():
    """构建 argparse 解析器。"""
    parser = argparse.ArgumentParser(
        prog="bcl",
        description="balance-control-lab: 平衡控制/轨迹规划/底层驱动 纯仿真工具箱")
    sub = parser.add_subparsers(dest="command")

    p_demo = sub.add_parser("demo", help="倒立摆起摆 + 平衡")
    p_demo.add_argument("--theta0", type=float, default=None,
                        help="初始倾角 [rad], 默认 pi(下垂)")
    p_demo.add_argument("--t-end", type=float, default=15.0, help="仿真时长 [s]")

    sub.add_parser("bench", help="跑评测套件")

    p_traj = sub.add_parser("traj", help="打印一条 S 曲线的关键特征")
    p_traj.add_argument("--p0", type=float, default=0.0)
    p_traj.add_argument("--p1", type=float, default=1.5)
    p_traj.add_argument("--vmax", type=float, default=1.5)
    p_traj.add_argument("--amax", type=float, default=6.0)
    p_traj.add_argument("--jmax", type=float, default=40.0)

    sub.add_parser("driver", help="演示寄存器/PWM/编码器/调度")
    sub.add_parser("selfcheck", help="跑 tools/selfcheck.py")
    return parser


def main(argv=None):
    """CLI 入口, 返回进程退出码。"""
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_output()
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "demo":
        run_demo(theta0=args.theta0, t_end=args.t_end)
        return 0
    if args.command == "bench":
        run_bench()
        return 0
    if args.command == "traj":
        run_traj(p0=args.p0, p1=args.p1, vmax=args.vmax, amax=args.amax,
                 jmax=args.jmax)
        return 0
    if args.command == "driver":
        run_driver()
        return 0
    if args.command == "selfcheck":
        return run_selfcheck()
    parser.error("未知子命令: %r" % (args.command,))
    return 2

"""
生成视频：左边为 frame，右边为 Critic/Value/Done 折线图，从时间 0 开始逐步推进，
每两帧之间停顿。布局如图所示。

用法：
  1. 在下方 VALUE_VALUES、CRITIC_VALUES、DONE_VALUES 中填写每帧对应的值
  2. 或使用 CSV：frame_idx,value,critic,done
  3. 运行：python 2.py
"""

import os

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============ 路径配置 ============
BASE_DIR = "/scratch1/home/zhicao/openpi"
FOLDER_1 = os.path.join(BASE_DIR, "1")
FOLDER_2 = os.path.join(BASE_DIR, "2")
OUTPUT_1 = os.path.join(BASE_DIR, "output_1_with_plots.mp4")
OUTPUT_2 = os.path.join(BASE_DIR, "output_2_with_plots.mp4")
FPS = 15
RIGHT_PANEL_WIDTH = 420
PAUSE_FRAMES = 8  # 每两帧之间重复的停顿帧数

# ============ 用户填写：每帧的 value、critic、done 值 ============
USE_CSV = False
CSV_PATH_1 = os.path.join(BASE_DIR, "value_critic_done_1.csv")
CSV_PATH_2 = os.path.join(BASE_DIR, "value_critic_done_2.csv")

# Folder 1: 按 frame_0000, 0001, ... 顺序填写
# Critic 可由 Value 自动计算：Critic[t] = Value[t] - Value[t-1]
VALUE_VALUES_1 = [0, 0, 10.5, 24.3, 25, 24.5, 27, 35, 45, 48.5]
CRITIC_VALUES_1 = None  # 设为 None 则自动从 Value 计算，否则手动填写
DONE_VALUES_1 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

# Folder 2: 11 frames
VALUE_VALUES_2 = [0, 0, 15.5, 20, 24.3, 25, 24.5, 27, 40, 47, 48.5]
CRITIC_VALUES_2 = None  # 自动从 Value 计算
DONE_VALUES_2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  # 最后两帧为 1


def compute_critic_from_value(values):
    """Critic[t] = Value[t] - Value[t-1]，首帧 Critic[0]=0"""
    if not values:
        return []
    critics = [0.0]
    for i in range(1, len(values)):
        critics.append(float(values[i]) - float(values[i - 1]))
    return critics


def load_from_csv(csv_path):
    import csv

    values, critics, dones = [], [], []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            values.append(float(row.get("value", 0)))
            critics.append(float(row.get("critic", 0)))
            dones.append(float(row.get("done", 0)))
    return values, critics, dones


def get_frames_sorted(folder):
    files = sorted(
        [f for f in os.listdir(folder) if f.endswith((".png", ".jpg", ".jpeg"))],
        key=lambda x: int("".join(c for c in x if c.isdigit()) or "0"),
    )
    return [os.path.join(folder, f) for f in files]


def draw_charts_panel(h, w_panel, value_hist, critic_hist, done_hist, current_step):
    """绘制右侧面板：当前数值 + 三个折线图（从 step 0 到 current_step）"""
    steps = list(range(current_step + 1))
    v_hist = value_hist[: current_step + 1]
    c_hist = critic_hist[: current_step + 1]
    d_hist = done_hist[: current_step + 1]

    fig, axes = plt.subplots(3, 1, figsize=(w_panel / 80, h / 80), sharex=True)
    fig.subplots_adjust(left=0.15, right=0.95, top=0.92, bottom=0.1, hspace=0.35)
    fig.patch.set_facecolor("#f5f5f5")

    # Value 图（绿）：固定 y 轴 0–55
    ax = axes[0]
    ax.set_ylabel("Value", color="green")
    ax.tick_params(axis="y", labelcolor="green")
    ax.set_ylim(0, 55)
    ax.grid(True, alpha=0.3)
    if steps:
        ax.plot(steps, v_hist, "g-o", linewidth=2, markersize=6)
        ax.plot(steps[-1], v_hist[-1], "go", markersize=12, markeredgewidth=2, markeredgecolor="darkgreen")

    # Critic 图（红）：固定 y 轴 0–55
    ax = axes[1]
    ax.set_ylabel("Critic", color="red")
    ax.tick_params(axis="y", labelcolor="red")
    ax.set_ylim(0, 55)
    ax.grid(True, alpha=0.3)
    if steps:
        ax.plot(steps, c_hist, "r-o", linewidth=2, markersize=6)
        ax.plot(steps[-1], c_hist[-1], "ro", markersize=12, markeredgewidth=2, markeredgecolor="darkred")

    # Done 图（蓝）
    ax = axes[2]
    ax.set_ylabel("Done")
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel("Step")
    ax.grid(True, alpha=0.3)
    if steps:
        ax.plot(steps, d_hist, "b-o", linewidth=2, markersize=6)
        ax.plot(steps[-1], d_hist[-1], "bo", markersize=12, markeredgewidth=2, markeredgecolor="darkblue")

    v_cur = v_hist[-1] if v_hist else 0.0
    c_cur = c_hist[-1] if c_hist else 0.0
    d_cur = d_hist[-1] if d_hist else 0.0
    fig.suptitle(f"Critic: {c_cur:.4f}  |  Value: {v_cur:.4f}  |  Done: {d_cur:.4f}", fontsize=10)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    buf = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)
    plt.close(fig)
    buf = cv2.resize(buf, (w_panel, h))
    return buf


def make_video(folder, value_values, critic_values, done_values, output_path):
    frame_paths = get_frames_sorted(folder)
    n_frames = len(frame_paths)
    n_vals = len(value_values)
    n = min(n_frames, n_vals)
    if n_frames != n_vals:
        print(f"注意: {folder} 有 {n_frames} 帧，但只有 {n_vals} 个 value，将只使用前 {n} 帧")
    frame_paths = frame_paths[:n]
    value_values = list(value_values)[:n]
    critic_values = list(critic_values)[:n]
    done_values = [float(d) for d in list(done_values)[:n]]

    if n == 0:
        print(f"没有找到帧: {folder}")
        return

    first = cv2.imread(frame_paths[0])
    h, w = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, FPS, (w + RIGHT_PANEL_WIDTH, h))

    for i, fp in enumerate(frame_paths):
        img = cv2.imread(fp)
        if img is None:
            continue
        panel = draw_charts_panel(
            h, RIGHT_PANEL_WIDTH, value_values[: i + 1], critic_values[: i + 1], done_values[: i + 1], i
        )
        composite = np.hstack([img, panel])
        out.write(composite)
        for _ in range(PAUSE_FRAMES):
            out.write(composite)

    out.release()
    print(f"已保存: {output_path} (Value: {value_values})")


if __name__ == "__main__":
    if USE_CSV:
        if os.path.isfile(CSV_PATH_1):
            VALUE_VALUES_1, CRITIC_VALUES_1, DONE_VALUES_1 = load_from_csv(CSV_PATH_1)
        if os.path.isfile(CSV_PATH_2):
            VALUE_VALUES_2, CRITIC_VALUES_2, DONE_VALUES_2 = load_from_csv(CSV_PATH_2)

    # 直接使用填写的值，不填充 0；帧数多于 value 时只取前 len(value) 帧
    v1 = list(VALUE_VALUES_1)
    c1 = compute_critic_from_value(v1) if CRITIC_VALUES_1 is None else list(CRITIC_VALUES_1)
    d1 = [float(x) for x in DONE_VALUES_1]
    v2 = list(VALUE_VALUES_2)
    c2 = compute_critic_from_value(v2) if CRITIC_VALUES_2 is None else list(CRITIC_VALUES_2)
    d2 = [float(x) for x in DONE_VALUES_2]

    make_video(FOLDER_1, v1, c1, d1, OUTPUT_1)
    make_video(FOLDER_2, v2, c2, d2, OUTPUT_2)

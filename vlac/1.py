import cv2
import os

VIDEOS = [
    ("/scratch1/home/zhicao/openpi/1.mp4", "/scratch1/home/zhicao/openpi/1"),
    ("/scratch1/home/zhicao/openpi/2.mp4", "/scratch1/home/zhicao/openpi/2"),
]
INTERVAL = 15  # 每隔15帧保存一次，即保存第1、16、31...帧（1-based）

def extract_frames(video_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return
    frame_idx = 0
    saved_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 保存第1、16、31...帧（即索引0、15、30...）
        if frame_idx % INTERVAL == 0:
            out_path = os.path.join(output_dir, f"frame_{saved_count:04d}.png")
            cv2.imwrite(out_path, frame)
            saved_count += 1
        frame_idx += 1
    cap.release()
    print(f"{video_path} -> {output_dir}: 保存了 {saved_count} 帧 (每{INTERVAL}帧取1帧)")

if __name__ == "__main__":
    for video_path, output_dir in VIDEOS:
        extract_frames(video_path, output_dir)

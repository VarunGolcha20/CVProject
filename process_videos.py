"""
process_videos.py
=================
Converts phone-recorded gesture videos into CNN and LSTM training samples.

Usage:
    # Single video (label inferred from filename)
    python process_videos.py --video videos/shoot_001.mp4

    # Whole folder (label = first word of filename before underscore)
    python process_videos.py --folder videos/

    # With preview (shows MediaPipe skeleton while processing)
    python process_videos.py --folder videos/ --preview

File naming convention:
    GESTURE_anything.mp4
    e.g.  shoot_outdoor_left_foot.mp4
          pass_indoor_rightfoot.mp4
          idle_walking_around.mp4

The gesture label is parsed from the filename prefix (before first underscore).
"""

import cv2
import numpy as np
import time
import argparse
from pathlib import Path
from collections import deque

from pipeline import PoseExtractor, FeatureEngineer
from classifiers.cnn_classifier import GESTURE_CLASSES, GESTURE2IDX


# ── Config ────────────────────────────────────────────────────────────────────
LSTM_WINDOW   = 30      # frames per LSTM sample
CNN_STRIDE    = 3       # save 1 CNN frame every N frames (avoid near-duplicates)
MIN_VISIBILITY = 0.55   # skip frames where pose is not confident enough
OVERLAP        = 15     # LSTM window overlap (stride = WINDOW - OVERLAP)


# ── Label parser ──────────────────────────────────────────────────────────────
def parse_label(filename: str) -> str:
    """
    shoot_outdoor.mp4  →  SHOOT
    move_left_slow.mp4 →  MOVE_LEFT   (checks for direction suffix)
    idle_001.mp4       →  IDLE
    """
    stem  = Path(filename).stem.upper()
    parts = stem.split("_")
    base  = parts[0]

    # Direction gestures: MOVE_LEFT, MOVE_RIGHT etc.
    if base == "MOVE" and len(parts) > 1 and parts[1] in ("LEFT","RIGHT","UP","DOWN"):
        return f"MOVE_{parts[1]}"

    if base in GESTURE_CLASSES:
        return base

    raise ValueError(
        f"Cannot parse label from filename '{filename}'. "
        f"Prefix must be one of: {GESTURE_CLASSES}\n"
        f"Example: shoot_take1.mp4"
    )


# ── Core processor ────────────────────────────────────────────────────────────
class VideoProcessor:
    def __init__(self, out_cnn: str = "data/cnn",
                 out_lstm: str = "data/lstm",
                 preview: bool = False):
        self.out_cnn  = Path(out_cnn);  self.out_cnn.mkdir(parents=True, exist_ok=True)
        self.out_lstm = Path(out_lstm); self.out_lstm.mkdir(parents=True, exist_ok=True)
        self.preview  = preview
        self.extractor = PoseExtractor(model_complexity=2)  # use full model offline
        self.engineer  = FeatureEngineer(history=30)

    def process_video(self, video_path: str) -> dict:
        label      = parse_label(Path(video_path).name)
        label_idx  = GESTURE2IDX[label]

        cap        = cv2.VideoCapture(video_path)
        fps        = cap.get(cv2.CAP_PROP_FPS) or 30
        total_fr   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # --- NEW: Get dimensions for saving video clips ---
        width      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        stats = {"cnn_saved": 0, "lstm_saved": 0, "skipped": 0,
                 "total_frames": total_fr, "label": label}

        lstm_buffer: deque = deque(maxlen=LSTM_WINDOW)
        frame_buffer: deque = deque(maxlen=LSTM_WINDOW)  # --- NEW: Buffer for raw frames ---

        frame_num   = 0
        mp_draw     = None
        if self.preview:
            import mediapipe as mp
            mp_draw = mp.solutions.drawing_utils

        print(f"\nProcessing: {Path(video_path).name}  [{label}]  {total_fr} frames @ {fps:.0f}fps")

        from pipeline import KEYPOINTS
        kp_order = list(KEYPOINTS.keys())

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            raw_frame = frame.copy()  # --- NEW: Keep a clean copy for saving ---

            t = frame_num / fps   # synthetic timestamp from frame number

            pose_frame = self.extractor.extract(frame, t)
            frame_num += 1

            # Skip low-confidence frames
            if pose_frame is None:
                stats["skipped"] += 1
                self._show_progress(frame_num, total_fr, stats, frame, label)
                continue

            fv = self.engineer.update(pose_frame)

            if fv.mean_visibility < MIN_VISIBILITY:
                stats["skipped"] += 1
                self._show_progress(frame_num, total_fr, stats, frame, label)
                continue

            # ── CNN sample ────────────────────────────────────────────────────
            # Every CNN_STRIDE frames to avoid near-duplicate samples
            if frame_num % CNN_STRIDE == 0:
                lm_arr = np.array(
                    [pose_frame.landmarks[k] for k in kp_order],
                    dtype=np.float32)                        # (10, 4)

                fname = self.out_cnn / f"{label}_{Path(video_path).stem}_{frame_num:05d}.npz"
                np.savez_compressed(str(fname),
                    landmarks = lm_arr,
                    features  = fv.to_array(),
                    label     = np.array(label_idx))
                
                # --- NEW: Save the actual CNN frame ---
                img_fname = self.out_cnn / f"{label}_{Path(video_path).stem}_{frame_num:05d}.jpg"
                cv2.imwrite(str(img_fname), raw_frame)
                
                stats["cnn_saved"] += 1

            # ── LSTM sample ───────────────────────────────────────────────────
            lstm_buffer.append(fv.to_array())
            frame_buffer.append(raw_frame)  

            # Save when buffer is full; stride = WINDOW - OVERLAP
            stride = LSTM_WINDOW - OVERLAP
            if len(lstm_buffer) == LSTM_WINDOW and frame_num % stride == 0:
                seq   = np.stack(lstm_buffer, axis=0)      # (30, 17)
                fname = self.out_lstm / f"{label}_{Path(video_path).stem}_{frame_num:05d}.npz"
                np.savez_compressed(str(fname),
                    sequence = seq,
                    label    = np.array(label_idx))
                
                # --- NEW: Save the LSTM window as a video clip ---
                vid_fname = self.out_lstm / f"{label}_{Path(video_path).stem}_{frame_num:05d}.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out_vid = cv2.VideoWriter(str(vid_fname), fourcc, fps, (width, height))
                for f in frame_buffer:
                    out_vid.write(f)
                out_vid.release()
                
                stats["lstm_saved"] += 1

            # Preview
            if self.preview and mp_draw and pose_frame:
                import mediapipe as mp
                mp_draw.draw_landmarks(
                    frame, pose_frame.raw,
                    mp.solutions.pose.POSE_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(0,200,100), thickness=2, circle_radius=3),
                    mp_draw.DrawingSpec(color=(0,100,200), thickness=2),
                )

            self._show_progress(frame_num, total_fr, stats, frame if self.preview else None, label)

            if self.preview:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        if self.preview:
            cv2.destroyAllWindows()

        print(f"  Done → CNN: {stats['cnn_saved']} samples | "
              f"LSTM: {stats['lstm_saved']} sequences | "
              f"Skipped: {stats['skipped']} frames")
        return stats

    def _show_progress(self, frame_num: int, total: int, stats: dict,
                       frame=None, label: str = ""):
        pct = frame_num / max(total, 1) * 100
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"\r  [{bar}] {pct:5.1f}%  CNN:{stats['cnn_saved']}  "
              f"LSTM:{stats['lstm_saved']}  skip:{stats['skipped']}", end="", flush=True)

        if frame is not None:
            info = (f"{label}  frame {frame_num}/{total}  "
                    f"CNN:{stats['cnn_saved']}  LSTM:{stats['lstm_saved']}")
            cv2.putText(frame, info, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,128), 2)
            cv2.imshow("Processing", frame)

    def close(self):
        self.extractor.close()


# ── Dataset balance checker ────────────────────────────────────────────────────
def check_balance(data_dir: str = "data/cnn", kind: str = "CNN"):
    """
    Prints sample count per class and warns if any class is underrepresented.
    Run after processing all videos.
    """
    from collections import Counter
    files  = list(Path(data_dir).glob("*.npz"))
    labels = []
    for f in files:
        d = np.load(f)
        labels.append(int(d["label"]))

    counts  = Counter(labels)
    from classifiers.cnn_classifier import IDX2GESTURE
    print(f"\n─── {kind} dataset balance ({data_dir}) ───")
    min_count = min(counts.values()) if counts else 0
    for idx, count in sorted(counts.items()):
        name  = IDX2GESTURE.get(idx, f"class_{idx}")
        bar   = "█" * (count // 10)
        warn  = "  ⚠ underrepresented" if count < min_count * 0.5 else ""
        print(f"  {name:<15} {count:>4}  {bar}{warn}")
    print(f"  Total: {len(files)} samples")
    if min_count > 0 and max(counts.values()) / min_count > 3:
        print("  ⚠  Class imbalance detected (>3× ratio). "
              "Consider recording more for underrepresented classes "
              "or use class_weight='balanced' in training.")


# ── Folder batch processor ────────────────────────────────────────────────────
def process_folder(folder: str, preview: bool = False):
    video_files = sorted(
        [f for f in Path(folder).iterdir()
         if f.suffix.lower() in (".mp4", ".mov", ".avi", ".mkv")]
    )

    if not video_files:
        print(f"No video files found in {folder}")
        return

    print(f"Found {len(video_files)} video(s) to process.")
    processor = VideoProcessor(preview=preview)
    all_stats = []

    for vf in video_files:
        try:
            stats = processor.process_video(str(vf))
            all_stats.append(stats)
        except ValueError as e:
            print(f"\n  Skipping {vf.name}: {e}")

    processor.close()

    # Summary
    print("\n─── Processing complete ─────────────────────────────────")
    total_cnn  = sum(s["cnn_saved"]  for s in all_stats)
    total_lstm = sum(s["lstm_saved"] for s in all_stats)
    print(f"  Total CNN samples:  {total_cnn}")
    print(f"  Total LSTM samples: {total_lstm}")

    check_balance("data/cnn",  "CNN")
    check_balance("data/lstm", "LSTM")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert phone videos to gesture training data")
    parser.add_argument("--video",   help="Single video file to process")
    parser.add_argument("--folder",  help="Folder of videos to process in batch")
    parser.add_argument("--preview", action="store_true",
                        help="Show MediaPipe skeleton while processing")
    parser.add_argument("--check",   action="store_true",
                        help="Only check dataset balance, don't process")
    args = parser.parse_args()

    if args.check:
        check_balance("data/cnn",  "CNN")
        check_balance("data/lstm", "LSTM")
    elif args.video:
        proc = VideoProcessor(preview=args.preview)
        proc.process_video(args.video)
        proc.close()
        check_balance("data/cnn",  "CNN")
        check_balance("data/lstm", "LSTM")
    elif args.folder:
        process_folder(args.folder, preview=args.preview)
    else:
        parser.print_help()

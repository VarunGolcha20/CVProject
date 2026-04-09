"""
process_videos.py — 1-second clip version.
One gesture per video. One CNN sample per frame. One LSTM sample per video.
"""
import cv2, numpy as np, argparse
from pathlib import Path
from pipeline import PoseExtractor, FeatureEngineer
from classifiers.cnn_classifier import GESTURE_CLASSES, GESTURE2IDX

# LSTM_WINDOW    = 30
LSTM_WINDOW    = 35
CNN_STRIDE     = 1      # every frame — short clips need every sample
MIN_VISIBILITY = 0.50
MAX_CLIP_SEC   = 5      # warn if longer

def parse_label(filename):
    stem  = Path(filename).stem.upper()
    parts = stem.split("_")
    base  = parts[0]
    if base == "MOVE" and len(parts) > 1 and parts[1] in ("LEFT","RIGHT","UP","DOWN"):
        return f"MOVE_{parts[1]}"
    if base in GESTURE_CLASSES:
        return base
    raise ValueError(f"Cannot parse label from '{filename}'. Prefix must be one of: {GESTURE_CLASSES}")

class VideoProcessor:
    def __init__(self, out_cnn="data/cnn", out_lstm="data/lstm", preview=False,cnn_img_dir="cnn_images", lstm_vid_dir="lstm_videos"):
        self.out_cnn   = Path(out_cnn);  self.out_cnn.mkdir(parents=True, exist_ok=True)
        self.out_lstm  = Path(out_lstm); self.out_lstm.mkdir(parents=True, exist_ok=True)
        self.cnn_img_dir  = Path(cnn_img_dir);  self.cnn_img_dir.mkdir(parents=True, exist_ok=True)
        self.lstm_vid_dir = Path(lstm_vid_dir); self.lstm_vid_dir.mkdir(parents=True, exist_ok=True)
        self.preview   = preview
        self.extractor = PoseExtractor(model_complexity=2)
        self.engineer  = FeatureEngineer(history=35)

    def process_video(self, video_path):
        label     = parse_label(Path(video_path).name)
        label_idx = GESTURE2IDX[label]
        cap       = cv2.VideoCapture(video_path)
        fps       = cap.get(cv2.CAP_PROP_FPS) or 30
        total_fr  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration  = total_fr / fps

        if duration > MAX_CLIP_SEC:
            print(f"\n  ⚠ '{Path(video_path).name}' is {duration:.1f}s "
                  f"(expected ≤{MAX_CLIP_SEC}s). Only one LSTM sample will be saved — trim your clips.")

        stats = {"cnn_saved":0,"lstm_saved":0,"skipped":0,"total_frames":total_fr,"label":label}
        all_features, all_frames = [], []
        frame_num = 0
        mp_draw = None
        if self.preview:
            import mediapipe as mp; mp_draw = mp.solutions.drawing_utils

        print(f"\nProcessing: {Path(video_path).name}  [{label}]  {total_fr} frames @ {fps:.0f}fps  ({duration:.1f}s)")
        from pipeline import KEYPOINTS
        kp_order = list(KEYPOINTS.keys())

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            raw_frame  = frame.copy()
            t          = frame_num / fps
            pose_frame = self.extractor.extract(frame, t)
            frame_num += 1

            if pose_frame is None:
                stats["skipped"] += 1
                self._progress(frame_num, total_fr, stats, frame, label); continue

            fv = self.engineer.update(pose_frame)
            if fv.mean_visibility < MIN_VISIBILITY:
                stats["skipped"] += 1
                self._progress(frame_num, total_fr, stats, frame, label); continue

            # CNN sample
            if frame_num % CNN_STRIDE == 0:
                lm_arr = np.array([pose_frame.landmarks[k] for k in kp_order], dtype=np.float32)
                fname  = self.out_cnn / f"{label}_{Path(video_path).stem}_{frame_num:05d}.npz"
                np.savez_compressed(str(fname), landmarks=lm_arr, features=fv.to_array(), label=np.array(label_idx))
                cv2.imwrite(str(self.cnn_img_dir / f"{label}_{Path(video_path).stem}_{frame_num:05d}.jpg"), raw_frame)
                stats["cnn_saved"] += 1

            all_features.append(fv.to_array())
            all_frames.append(raw_frame)

            if self.preview and mp_draw and pose_frame:
                import mediapipe as mp
                mp_draw.draw_landmarks(frame, pose_frame.raw, mp.solutions.pose.POSE_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(0,200,100),thickness=2,circle_radius=3),
                    mp_draw.DrawingSpec(color=(0,100,200),thickness=2))
            self._progress(frame_num, total_fr, stats, frame if self.preview else None, label)
            if self.preview and cv2.waitKey(1) & 0xFF == ord('q'): break

        cap.release()
        if self.preview: cv2.destroyAllWindows()

        # One LSTM sample per video
        if len(all_features) > 0:
            seq = np.array(all_features, dtype=np.float32)

            if len(seq) < LSTM_WINDOW:
                # Pad by repeating last frame — gesture is already complete
                pad = LSTM_WINDOW - len(seq)
                seq = np.vstack([seq, np.repeat(seq[-1:], pad, axis=0)])
                for _ in range(pad): all_frames.append(all_frames[-1])

            elif len(seq) > LSTM_WINDOW:
                # Take the LAST LSTM_WINDOW frames, not the middle.
                # For action gestures the peak motion is toward the end of the clip.
                seq        = seq[-LSTM_WINDOW:]
                all_frames = all_frames[-LSTM_WINDOW:]

            fname = self.out_lstm / f"{label}_{Path(video_path).stem}.npz"
            np.savez_compressed(str(fname), sequence=seq, label=np.array(label_idx))
            stats["lstm_saved"] += 1

            vid_fname = self.lstm_vid_dir / f"{label}_{Path(video_path).stem}.mp4"
            fourcc    = cv2.VideoWriter_fourcc(*'mp4v')
            out_vid   = cv2.VideoWriter(str(vid_fname), fourcc, fps, (width, height))
            for f in all_frames: out_vid.write(f)
            out_vid.release()

        print(f"\n  Done → CNN: {stats['cnn_saved']} | LSTM: {stats['lstm_saved']} | Skipped: {stats['skipped']}")
        return stats

    def _progress(self, n, total, stats, frame=None, label=""):
        pct = n / max(total,1) * 100
        bar = "█"*int(pct/5) + "░"*(20-int(pct/5))
        print(f"\r  [{bar}] {pct:5.1f}%  CNN:{stats['cnn_saved']}  LSTM:{stats['lstm_saved']}  skip:{stats['skipped']}", end="", flush=True)
        if frame is not None:
            cv2.putText(frame, f"{label}  {n}/{total}  CNN:{stats['cnn_saved']}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,128), 2)
            cv2.imshow("Processing", frame)

    def close(self): self.extractor.close()


def check_balance(data_dir="data/cnn", kind="CNN"):
    from collections import Counter
    from classifiers.cnn_classifier import IDX2GESTURE
    files  = list(Path(data_dir).glob("*.npz"))
    counts = Counter(int(np.load(f)["label"]) for f in files)
    print(f"\n─── {kind} dataset balance ({data_dir}) ───")
    min_c = min(counts.values()) if counts else 0
    for idx, count in sorted(counts.items()):
        warn = "  ⚠ underrepresented" if count < min_c * 0.5 else ""
        print(f"  {IDX2GESTURE.get(idx,f'class_{idx}'):<15} {count:>4}  {'█'*(count//10)}{warn}")
    print(f"  Total: {len(files)} samples")
    if min_c > 0 and max(counts.values()) / min_c > 3:
        print("  ⚠  Class imbalance >3×. Record more for underrepresented classes.")


def process_folder(folder, preview=False):
    video_files = sorted([f for f in Path(folder).iterdir()
                          if f.suffix.lower() in (".mp4",".mov",".avi",".mkv")])
    if not video_files: print(f"No video files in {folder}"); return
    print(f"Found {len(video_files)} video(s).")
    proc = VideoProcessor(preview=preview); all_stats = []
    for vf in video_files:
        try: all_stats.append(proc.process_video(str(vf)))
        except ValueError as e: print(f"\n  Skipping {vf.name}: {e}")
    proc.close()
    print("\n─── Processing complete ─────────────────────────────────")
    print(f"  Total CNN:  {sum(s['cnn_saved']  for s in all_stats)}")
    print(f"  Total LSTM: {sum(s['lstm_saved'] for s in all_stats)}")
    check_balance("data/cnn","CNN"); check_balance("data/lstm","LSTM")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--video"); p.add_argument("--folder")
    p.add_argument("--preview", action="store_true")
    p.add_argument("--check",   action="store_true")
    args = p.parse_args()
    if args.check:
        check_balance("data/cnn","CNN"); check_balance("data/lstm","LSTM")
    elif args.video:
        proc = VideoProcessor(preview=args.preview)
        proc.process_video(args.video); proc.close()
        check_balance("data/cnn","CNN"); check_balance("data/lstm","LSTM")
    elif args.folder: process_folder(args.folder, args.preview)
    else: p.print_help()
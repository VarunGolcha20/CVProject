"""
test_one_video.py
=================
Smoke test: run the full pipeline on a single video file.
Checks that pose extraction, feature engineering, and sample saving all work.

Usage:
    python test_one_video.py --video path/to/shoot_test.mp4
    python test_one_video.py --video path/to/shoot_test.mp4 --preview
"""

import cv2
import time
import argparse
import numpy as np
from pathlib import Path
from collections import deque

import mediapipe as mp

from pipeline import PoseExtractor, FeatureEngineer, KEYPOINTS
from classifiers.rule_based import RuleBasedClassifier
from classifiers.cnn_classifier import GESTURE2IDX
from process_videos import parse_label, MIN_VISIBILITY, LSTM_WINDOW, CNN_STRIDE, OVERLAP


def test(video_path: str, preview: bool = False):
    label = parse_label(Path(video_path).name)
    print(f"\n{'='*50}")
    print(f"  Video : {video_path}")
    print(f"  Label : {label}")
    print(f"{'='*50}\n")

    cap       = cv2.VideoCapture(video_path)
    fps_video = cap.get(cv2.CAP_PROP_FPS) or 30
    total_fr  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration  = total_fr / fps_video

    print(f"  Resolution : {int(cap.get(3))}×{int(cap.get(4))}")
    print(f"  FPS        : {fps_video:.1f}")
    print(f"  Frames     : {total_fr}")
    print(f"  Duration   : {duration:.1f}s\n")

    extractor = PoseExtractor(model_complexity=2)
    engineer  = FeatureEngineer(history=30)
    clf       = RuleBasedClassifier()
    mp_draw   = mp.solutions.drawing_utils

    kp_order   = list(KEYPOINTS.keys())
    label_idx  = GESTURE2IDX[label]

    # Counters
    frames_processed = 0
    frames_skipped   = 0
    cnn_saved        = 0
    lstm_saved       = 0
    gesture_counts   = {}
    processing_times = []

    lstm_buffer = deque(maxlen=LSTM_WINDOW)

    # Output dirs (test mode — saves to data/test/)
    # from pathlib import Path
    cnn_dir  = Path("data/test/cnn");  cnn_dir.mkdir(parents=True, exist_ok=True)
    lstm_dir = Path("data/test/lstm"); lstm_dir.mkdir(parents=True, exist_ok=True)

    frame_num = 0
    print("  Processing frames...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()
        t  = frame_num / fps_video

        pose_frame = extractor.extract(frame, t)
        frame_num += 1

        if pose_frame is None:
            frames_skipped += 1
            continue

        fv = engineer.update(pose_frame)

        if fv.mean_visibility < MIN_VISIBILITY:
            frames_skipped += 1
            continue

        frames_processed += 1
        dt = (time.perf_counter() - t0) * 1000
        processing_times.append(dt)

        # Rule-based gesture prediction (sanity check)
        gesture = clf(fv)
        gesture_counts[gesture] = gesture_counts.get(gesture, 0) + 1

        # CNN sample
        if frame_num % CNN_STRIDE == 0:
            lm_arr = np.array([pose_frame.landmarks[k] for k in kp_order],
                              dtype=np.float32)
            fname = cnn_dir / f"{label}_{frame_num:05d}.npz"
            np.savez_compressed(str(fname),
                landmarks=lm_arr, features=fv.to_array(),
                label=np.array(label_idx))
            cnn_saved += 1

        # LSTM sample
        lstm_buffer.append(fv.to_array())
        stride = LSTM_WINDOW - OVERLAP
        if len(lstm_buffer) == LSTM_WINDOW and frame_num % stride == 0:
            seq   = np.stack(lstm_buffer, axis=0)
            fname = lstm_dir / f"{label}_{frame_num:05d}.npz"
            np.savez_compressed(str(fname),
                sequence=seq, label=np.array(label_idx))
            lstm_saved += 1

        # Preview
        if preview:
            mp_draw.draw_landmarks(
                frame, pose_frame.raw,
                mp.solutions.pose.POSE_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0,200,100), thickness=2, circle_radius=3),
                mp_draw.DrawingSpec(color=(0,100,200), thickness=2),
            )
            info_lines = [
                f"Label: {label}  |  Gesture: {gesture}",
                f"Visibility: {fv.mean_visibility:.2f}  |  Frame: {frame_num}/{total_fr}",
                f"CNN saved: {cnn_saved}  |  LSTM saved: {lstm_saved}",
                f"L knee: {fv.left_knee_angle:.0f}deg  R knee: {fv.right_knee_angle:.0f}deg",
                f"Swing vel L:{fv.left_swing_velocity:.3f}  R:{fv.right_swing_velocity:.3f}",
            ]
            for i, line in enumerate(info_lines):
                cv2.putText(frame, line, (10, 25 + i*22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,128), 1)

            # Progress bar
            pct = int((frame_num / total_fr) * frame.shape[1])
            cv2.rectangle(frame, (0, frame.shape[0]-6), (pct, frame.shape[0]),
                          (0,200,100), -1)

            cv2.imshow("Test: GestureBall CV", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n  Stopped early by user.")
                break
        else:
            # Text progress bar
            pct = frame_num / total_fr * 100
            bar = "█" * int(pct/5) + "░" * (20-int(pct/5))
            print(f"\r  [{bar}] {pct:5.1f}%  CNN:{cnn_saved}  LSTM:{lstm_saved}",
                  end="", flush=True)

    cap.release()
    extractor.close()
    if preview:
        cv2.destroyAllWindows()

    # ── Results ───────────────────────────────────────────────────────────────
    print(f"\n\n{'='*50}")
    print("  RESULTS")
    print(f"{'='*50}")
    print(f"  Frames processed : {frames_processed}")
    print(f"  Frames skipped   : {frames_skipped}  "
          f"({'%.0f' % (frames_skipped/max(frame_num,1)*100)}% — "
          f"{'OK' if frames_skipped/max(frame_num,1) < 0.2 else 'HIGH — check lighting/framing'})")
    print(f"\n  CNN samples saved  : {cnn_saved}  → data/test/cnn/")
    print(f"  LSTM samples saved : {lstm_saved}  → data/test/lstm/")

    if processing_times:
        mean_ms = np.mean(processing_times)
        print(f"\n  Avg processing time : {mean_ms:.1f}ms/frame  "
              f"({1000/mean_ms:.0f} FPS theoretical)")

    # Gesture distribution from rule-based
    print(f"\n  Rule-based gesture distribution:")
    total_g = sum(gesture_counts.values())
    for g, count in sorted(gesture_counts.items(), key=lambda x: -x[1]):
        bar = "█" * int(count/total_g*30)
        print(f"    {g:<20} {count:>4}  {bar}")

    # Sanity checks
    print(f"\n  Sanity checks:")
    ok = True

    if cnn_saved == 0:
        print("  ✗ No CNN samples saved — pose was never detected. Check video framing.")
        ok = False
    else:
        print(f"  ✓ CNN samples look good")

    if lstm_saved == 0:
        print("  ✗ No LSTM sequences saved — video may be too short (need ≥30 frames with pose)")
        ok = False
    else:
        print(f"  ✓ LSTM sequences look good")

    # Verify a saved sample loads correctly
    test_cnn = list(Path("data/test/cnn").glob("*.npz"))
    if test_cnn:
        d = np.load(test_cnn[0])
        print(f"DEBUG: Feature shape is {d['features'].shape}") 
        print(f"DEBUG: Features are: {d['features']}")
        assert d["landmarks"].shape == (10, 4), f"Bad landmarks shape: {d['landmarks'].shape}"
        assert d["features"].shape  == (16,),   f"Bad features shape: {d['features'].shape}"
        print(f"  ✓ CNN .npz structure valid  {d['landmarks'].shape} landmarks, {d['features'].shape} features")

    test_lstm = list(Path("data/test/lstm").glob("*.npz"))
    if test_lstm:
        d = np.load(test_lstm[0])
        assert d["sequence"].shape == (LSTM_WINDOW, 16), f"Bad sequence shape: {d['sequence'].shape}"
        print(f"  ✓ LSTM .npz structure valid  {d['sequence'].shape} sequence")

    print(f"\n{'='*50}")
    if ok:
        print("  ✅ Pipeline working correctly.")
        print("  Next: record more videos and run process_videos.py --folder videos/")
    else:
        print("  ❌ Issues found — see above.")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",   required=True, help="Path to a single video, e.g. shoot_test.mp4")
    parser.add_argument("--preview", action="store_true", help="Show skeleton overlay while processing")
    args = parser.parse_args()
    test(args.video, preview=args.preview)

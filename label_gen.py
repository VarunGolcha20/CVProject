"""
label_gen.py

genrate/append videos/labels.csv (NOTE: In current pipling all frames are of same label as the one infered from filename, so this code currently just uses filename to give label for each frame)

Also frame 0 for vid 2 is treted like last frame of last vid 1 + 1and so on

Usage:
    # Single video
    python label_gen.py --video videos/shoot_test.mp4

    # Whole folder (label = first word of filename before underscore)
    python label_gen.py --folder videos/
    # get sep file
    python label_gen.py --folder videos/ --sep     


    # to append 
    python label_gen.py --folder videos/ --append
    python label_gen.py --video videos/shoot_test.mp4 --append
"""
import cv2
import csv
import argparse
from pathlib import Path

def parse_label(filename):
    return Path(filename).stem.split("_")[0].upper()

def get_frame_count(video):
    cap = cv2.VideoCapture(video)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frames

def last_frame(csv_file):
    try:
        with open(csv_file) as f:
            rows = list(csv.reader(f))
            return int(rows[-1][0]) + 1
    except:
        return 0

def process_video(video, csv_file, append):
    start = last_frame(csv_file) if append else 0
    label = parse_label(video)
    frames = get_frame_count(video)

    mode = "a" if append else "w"
    with open(csv_file, mode, newline="") as f:
        writer = csv.writer(f)
        if not append:
            writer.writerow(["frame","label"])

        for i in range(frames):
            writer.writerow([start + i, label])

    print(f"{video}: {frames} frames → {label}")

def process_folder(folder, csv_file, append, separate=False):
    folder_path = Path(folder)
    
    # If not separate, handle the single master file logic
    if not separate:
        start = last_frame(csv_file) if append else 0
        mode = "a" if append else "w"
        with open(csv_file, mode, newline="") as f:
            writer = csv.writer(f)
            if not append:
                writer.writerow(["frame", "label"])
            
            for video in sorted(folder_path.glob("*.mp4")):
                label = parse_label(video.name)
                frames = get_frame_count(str(video))
                for i in range(frames):
                    writer.writerow([start + i, label])
                print(f"{video.name}: {frames} frames → {label} (Master CSV)")
                start += frames
    else:
        # Separate mode: create a CSV for every MP4 found
        for video in sorted(folder_path.glob("*.mp4")):
            target_csv = video.with_suffix(".csv")
            label = parse_label(video.name)
            frames = get_frame_count(str(video))
            
            with open(target_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["frame", "label"])
                for i in range(frames):
                    writer.writerow([i, label])
            print(f"{video.name}: {frames} frames → {target_csv.name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video")
    parser.add_argument("--folder")
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--sep", action="store_true", help="Generate separate CSV for each video")
    args = parser.parse_args()

    csv_file = "videos/labels.csv"

    if args.video:
        # If --sep is used with a single video, it just saves to filename.csv instead of labels.csv
        target = Path(args.video).with_suffix(".csv") if args.sep else csv_file
        process_video(args.video, target, args.append)
    elif args.folder:
        process_folder(args.folder, csv_file, args.append, args.sep)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video")
    parser.add_argument("--folder")
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--sep", action="store_true", help="Generate separate CSV for each video")
    args = parser.parse_args()

    csv_file = "videos/labels.csv"

    if args.video:
        # If --sep is used with a single video, it just saves to filename.csv instead of labels.csv
        target = Path(args.video).with_suffix(".csv") if args.sep else csv_file
        process_video(args.video, target, args.append)
    elif args.folder:
        process_folder(args.folder, csv_file, args.append, args.sep)
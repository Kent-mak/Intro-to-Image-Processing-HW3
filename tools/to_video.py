import argparse
import os
import os.path as osp
import time
import cv2
import torch
import re

from loguru import logger

from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.utils.visualize import plot_tracking
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracking_utils.timer import Timer


IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def make_parser():
    parser = argparse.ArgumentParser("ByteTrack Visualization from TXT Results")
    
    # Original args
    parser.add_argument("--to", default="image", help="demo type, e.g., image, video")
    parser.add_argument("--name", default=None, type=str)
    parser.add_argument("--device", default="gpu", type=str)
    
    # New args for result visualization
    parser.add_argument("--dataset", default="mot17", help="Used dataset")
    parser.add_argument("--seq_id", default="seq", help="Current sequence(scene)")
    parser.add_argument("--result_file", type=str, help="Path to result txt file")
    parser.add_argument("--img_dir", type=str, help="Directory with input images (img1/)")
    parser.add_argument("--vis_folder", type=str, default="vis_results/vis", help="Where to save visualizations")
    parser.add_argument("--vis_parent", type=str,  default="vis_results/vis", help="Where to group all visualization results")
    parser.add_argument("--video_path", type=str, default="vis_results/vis", help="Where to save videos")

    # Optional tracking args (for filtering small/vertical boxes)
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6)
    parser.add_argument("--min_box_area", type=float, default=10)

    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = osp.join(maindir, filename)
            ext = osp.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


def get_frame_number(filename):
    # Extract number after the last underscore and before ".png" or ".jpg"
    match = re.search(r'_(\d+)\.(png|jpg)$', filename, re.IGNORECASE)
    return int(match.group(1)) if match else -1


def image_demo(result_file, img_dir, vis_folder, current_time, args):
    import csv
    from collections import defaultdict
    
    if args.dataset == "mot17":
        file = os.path.basename(result_file)  # "MOT17-01-DPM.txt"
        scene_id = file.replace('-DPM.txt', '')  # "MOT17-01"
    elif args.dataset == "fisheye8k":
        file = os.path.basename(result_file)
        scene_id = file.replace('.txt', '')
    elif args.dataset == "loaf": 
        file = os.path.basename(result_file)
        scene_id = file.replace('.txt', '')
    else:
        file = os.path.basename(result_file)
        scene_id = file.replace('.txt', '')

    trajectories = defaultdict(list)

    # 1. Load results from txt
    result_dict = defaultdict(list)  # {frame_id: [[tlwh, id, score], ...]}
    with open(result_file, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            frame_id = int(row[0])
            track_id = int(row[1])
            x, y, w, h = map(float, row[2:6])
            score = float(row[6])
            result_dict[frame_id].append(([x, y, w, h], track_id, score))

    # 2. Sort image paths
    image_files = get_image_list(img_dir)
    if args.dataset == "mot17":
        image_files = sorted(image_files)
    elif args.dataset == "fisheye8k":
        image_files = sorted(
            [f for f in image_files if os.path.basename(f).startswith(scene_id)],
            key=lambda f: get_frame_number(os.path.basename(f))
        )
    elif args.dataset == "loaf": 
        image_files = sorted([f for f in image_files if os.path.basename(f).startswith(scene_id)])

    timer = Timer()

    # 3. Visualize
    for img_path in image_files:
        filename = os.path.basename(img_path)
        frame_id = get_frame_number(filename)  # extract actual frame number

        img = cv2.imread(img_path)
        if img is None:
            continue
        timer.tic()

        online_tlwhs = []
        online_ids = []
        online_scores = []

        if frame_id in result_dict:
            for tlwh, tid, score in result_dict[frame_id]:
                vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(score)

        timer.toc()

        # print(f"[DEBUG] Frame {frame_id} — Boxes: {len(online_tlwhs)}")

        online_im = plot_tracking(
            img,
            online_tlwhs,
            online_ids,
            frame_id=frame_id,
            fps=1. / timer.average_time,
            scene=scene_id,
            filename=filename,
            trajectories=trajectories
        )

        # Save visualization
        # timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
        # save_folder = osp.join(vis_folder, timestamp)
        save_folder = vis_folder
        os.makedirs(save_folder, exist_ok=True)
        cv2.imwrite(osp.join(save_folder, osp.basename(img_path)), online_im)

        if frame_id % 20 == 0:
            logger.info(f"Processing frame {frame_id} ({1. / max(1e-5, timer.average_time):.2f} fps)")

        ch = cv2.waitKey(0)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break



def img_2_video(vis_parent, video_path, args):
    import cv2
    import os
    from glob import glob
    from pathlib import Path

    fps = 10

    # images = sorted(glob(os.path.join(vis_parent, "*.jpg")))  # or png

    all_images = list(Path(vis_parent).rglob("*.jpg")) + list(Path(vis_parent).rglob("*.png"))

    # Sort by directory name first, then file name
    if args.dataset == "fisheye8k":
        images = sorted(
            all_images,
            key=lambda p: (p.parent.name, get_frame_number(p.name))
        )
        images = [str(p) for p in images]  # Convert Path objects to strings
    else:
        images = sorted(all_images, key=lambda p: (p.parent.name, p.name))
        images = [str(p) for p in images]  # convert Path to string

    # Read first image to get size
    frame = cv2.imread(images[0])
    height, width, _ = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    for img_path in images:
        frame = cv2.imread(img_path)
        video_writer.write(frame)

    video_writer.release()


def main(args):

    current_time = time.localtime()

    if args.to == "image":
        image_demo(
            result_file=args.result_file,
            img_dir=args.img_dir,
            vis_folder=args.vis_folder,
            current_time=current_time,
            args=args
        )
    elif args.to == "video":
        img_2_video(
            vis_parent=args.vis_parent,
            video_path=args.video_path,
            args=args
        )


if __name__ == "__main__":
    args = make_parser().parse_args()

    main(args)

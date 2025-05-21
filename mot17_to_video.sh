#!/bin/bash

echo "Processing MOT17..."

for mode in test train
do
  echo "=== Processing mode: $mode ==="

  for result_file in YOLOX_outputs/yolox_x_mot17/track_results_${mode}/MOT17-*-DPM.txt
  do
    # Extract sequence name, e.g., MOT17-01-DPM
    seq_name=$(basename "$result_file" .txt)
    seq_id=${seq_name%-DPM}

    echo "Processing $seq_name..."

    python tools/to_video.py \
      --to image \
      --dataset mot17 \
      --result_file "$result_file" \
      --img_dir "datasets/MOT17/${mode}/${seq_name}/img1" \
      --vis_folder "vis_results/MOT17/${mode}/${seq_id}"
  done

  echo "Transforming to video..."

  python tools/to_video.py \
    --to video \
    --vis_parent "vis_results/MOT17/${mode}" \
    --video_path "vis_results/MOT17/MOT17_${mode}_video.mp4"

done


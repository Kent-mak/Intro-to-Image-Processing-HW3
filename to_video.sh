#!/bin/bash


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
      --result_file "$result_file" \
      --img_dir "datasets/MOT17/${mode}/${seq_name}/img1" \
      --vis_folder "vis_results/MOT17/${mode}/${seq_id}"
  done

  do
    echo "Transforming to video..."

    python tools/to_video.py \
      --to video \
      --vis_parent "vis_results/MOT17/${mode}" \
      --video_path "vis_results/MOT17/MOT17_${mode}_video.mp4"

  done

done






# python tools/to_video.py \
#   --result_file YOLOX_outputs/yolox_x_mot17/track_results_test/MOT17-01-DPM.txt \
#   --img_dir datasets/mot/test/MOT17-01-DPM/img1 \
#   --vis_folder vis_results/MOT17//MOT17-01 \
#   --video_path vis_results/MOT17//MOT17-01/01.mp4 \
#   --save_result

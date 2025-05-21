#!/bin/bash

echo "Processing Fisheye8K..."

for mode in train test
do
  echo "=== Processing mode: $mode ==="

  for result_file in YOLOX_outputs/yolox_x_fisheye8k/track_results_${mode}/*.txt
  do
    # Extract sequence id, e.g., camera1_A
    seq_id=$(basename "$result_file" .txt)

    echo "Processing $seq_id..."

    python tools/to_video.py \
      --to image \
      --dataset fisheye8k \
      --result_file "$result_file" \
      --img_dir "datasets/Fisheye8k/${mode}/images_padded" \
      --vis_folder "vis_results/Fisheye8k/${mode}/${seq_id}"
  done

  echo "Transforming to video..."

  python tools/to_video.py \
    --to video \
    --dataset fisheye8k \
    --vis_parent "vis_results/Fisheye8k/${mode}" \
    --video_path "vis_results/Fisheye8k/Fisheye8k_${mode}_video.mp4"

done


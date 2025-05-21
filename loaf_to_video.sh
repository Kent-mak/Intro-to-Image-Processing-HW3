#!/bin/bash

echo "Processing LOAF..."

for mode in test train val
do
  echo "=== Processing mode: $mode ==="

  for result_file in YOLOX_outputs/yolox_x_loaf/track_results_${mode}/*.txt
  do
    # Extract sequence name, e.g., 0053
    seq_id=$(basename "$result_file" .txt)

    echo "Processing $seq_id..."

    python tools/to_video.py \
      --to image \
      --dataset loaf \
      --result_file "$result_file" \
      --img_dir "datasets/LOAF_512/${mode}" \
      --vis_folder "vis_results/LOAF/${mode}/${seq_id}"
  done

  echo "Transforming to video..."

  python tools/to_video.py \
    --to video \
    --dataset loaf \
    --vis_parent "vis_results/LOAF/${mode}" \
    --video_path "vis_results/LOAF/LOAF_${mode}_video.mp4"

done


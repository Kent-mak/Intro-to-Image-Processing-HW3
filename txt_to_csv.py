
txt_file = "YOLOX_outputs/yolox_x_mot17/train.txt"
output_file = "YOLOX_outputs/MOT17_train_result.csv"

with open(output_file, "w") as csv_file:
    csv_file.write("frame_index, track_ID, x, y, w, h, scene_ID\n")
    for line in open(txt_file):
        csv_file.write(line)
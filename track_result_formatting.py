import argparse
import os

def process_file(input_path, output_file, append_string):
    with open(input_path, 'r') as infile, open(output_file, 'a') as outfile:
        for line in infile:
            parts = line.strip().split(',')
            if len(parts) < 4:
                continue  # Skip malformed lines
            trimmed = parts[:-3]
            trimmed.append(append_string)
            outfile.write(','.join(trimmed) + '\n')

def process_directory(input_dir, output_file):
    for filename in os.listdir(input_dir):
        
        # if filename.endswith('-DPM.txt'): # uncomment for MOT17
            # append_string = filename.split('-DPM.txt')[0] # uncomment for MOT17
            # input_path = os.path.join(input_dir, filename)
            # process_file(input_path, output_file, append_string)
        append_string = filename.split('.')[0] # comment out for MOT17
        input_path = os.path.join(input_dir, filename)
        process_file(input_path, output_file, append_string)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process tracking result files ending in -DPM.txt.")
    parser.add_argument('--input_dir', required=True, help='Directory containing input .txt files')
    parser.add_argument('--output_file', required=True, help='Path to the output file')
    args = parser.parse_args()

    process_directory(args.input_dir, args.output_file)

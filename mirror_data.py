import csv
import ast
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--infile', type=str, default='raw_states_stanley_outside_combined.csv')
parser.add_argument('--outfile', type=str, default='raw_states_stanley_outside_mirror.csv')

args = parser.parse_args()

# Pathing logic
input_path = f'raw_data/{args.infile}'
output_path = f'raw_data/{args.outfile}'

with open(input_path, 'r') as inf, open(output_path, 'w', newline='') as outf:
    reader = csv.reader(inf)
    writer = csv.writer(outf)
    
    row_counter = 0
    skipped_counter = 0
    
    for row in reader:
        try:
            # Basic validation: ensure row isn't empty
            if not row:
                continue
                
            # Attempt to parse
            scan = ast.literal_eval(row[0])
            speed = row[1]
            steer = float(row[2])
            
            # Write original
            writer.writerow([scan, speed, steer])
            
            # Write mirrored: reverse lidar, negate steering
            mirrored_scan = list(reversed(scan))
            writer.writerow([mirrored_scan, speed, -steer])
            
            row_counter += 1
            if row_counter % 500 == 0:
                print(f'Processed {row_counter} rows...')

        except (SyntaxError, ValueError, IndexError) as e:
            # This catches incomplete lines (SyntaxError) or malformed floats (ValueError)
            skipped_counter += 1
            # Optional: print specific error for the first few skips
            if skipped_counter < 5:
                print(f"Skipping malformed row: {e}")
            continue

print(f'Done! Successfully processed {row_counter} rows.')
print(f'Deleted/Skipped {skipped_counter} incomplete lines.')
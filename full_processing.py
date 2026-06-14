import subprocess
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str)
args=parser.parse_args()

PYTHON = "/home/f1tenth/f1tenth_export/src/off_policy/off_policy/torch_env/bin/python3"
print(f"Using python: {PYTHON}")
def run(script, *args):
    print(f"\n{'='*50}\nRunning {script}\n{'='*50}")
    result = subprocess.run([PYTHON, script, *args])
    if result.returncode != 0:
        print(f"[ERROR] {script} failed — stopping pipeline.")
        sys.exit(result.returncode)

name=args.name

run("mirror_data.py","--infile",f"raw_states_{name}.csv", "--outfile",f"raw_states_{name}_mirror.csv")
run("parse_raw_data.py","--infile", f"raw_states_{name}_mirror.csv","--outfile", f"sarsd_buffer_{name}.csv")
run("recompute_rewards.py", "--infile", f"raw_states_{name}_mirror.csv","--outfile", f"sarsd_buffer_{name}.csv")
run("td3_train.py", "--infile", f"sarsd_buffer_{name}.csv", "--name",name)

print("\nPipeline complete.")
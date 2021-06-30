
import subprocess

a = {"LANGUAGE MATH": [122317, 376247],
     "MOTOR LH": [146129, 552241],
     "EMOTION FACES": [627549, 516742],
     "GAMBLING REWARD": [627549, 194746],
     "WM 2BK_TOOL": [376247, 105923]}

for task in a:
    cmd = ["python", "./scripts/visualize_thresholded_activations.py", "--subject"]
    for subject in a[task]:
        cmd.append(str(subject))
    cmd.extend(["--metric_name", task])
    print(" ".join(cmd))
    subprocess.call(cmd)


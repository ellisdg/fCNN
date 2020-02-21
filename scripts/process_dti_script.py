import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from fcnn.dti import process_dti


def main():
    subjects = sys.argv[1].split(",")
    hcp_dir = os.path.abspath(sys.argv[2])
    for subject in subjects:
        print("Processing :", subject)
        subject_dir = os.path.join(hcp_dir, subject)
        process_dti(subject_dir)


if __name__ == '__main__':
    main()

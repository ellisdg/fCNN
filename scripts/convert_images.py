import argparse
import glob
import SimpleITK as sitk


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wildcard")
    parser.add_argument("--extension")
    return parser.parse_args()


def main():
    namespace = parse_args()
    for fn in glob.glob(namespace.wildcard):
        base = fn.split(".")[0]
        out_fn = base + ".nii.gz"
        print(fn, "--->", out_fn)
        image = sitk.ReadImage(fn)
        sitk.WriteImage(image, out_fn)


if __name__ == "__main__":
    main()

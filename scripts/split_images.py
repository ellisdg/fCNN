import nibabel as nib
import argparse
import glob


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wildcard")
    parser.add_argument("--names", nargs="*")
    return parser.parse_args()


def main():
    namespace = parse_args()
    for fn in glob.glob(namespace.wildcard):
        base, ext = fn.split(".", 1)
        image = nib.load(fn)
        for index, name in enumerate(namespace.names):
            out_fn = base + name + "." + ext
            new_image = image.__class__(dataobj=image.dataobj[..., index], affine=image.affine, header=image.header)
            print(fn, "--->", out_fn)
            new_image.to_filename(out_fn)


if __name__ == "__main__":
    main()


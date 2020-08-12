import argparse
import nibabel as nib
from fcnn.utils.utils import one_hot_image_to_label_map


def parse_args():
    return format_parser(argparse.ArgumentParser(), sub_command=False).parse_args()


def format_parser(parser, sub_command=False):
    if sub_command:
        parser.add_argument("--segmentation", action="store_true", default=False)
    else:
        parser.add_argument("--filenames", nargs="*", required=True)
        parser.add_argument("--labels", nargs="*", required=True)
        parser.add_argument("--hierarchy", default=False, action="store_true")
        parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--threshold", default=0.5, type=float,
                        help="If segmentation is set, this is the threshold for segmentation cutoff.")
    parser.add_argument("--sum", default=False, action="store_true",
                        help="Does not sum the predictions before using threshold.")
    parser.add_argument("--use_contours", action="store_true", default=False,
                        help="If the model was trained to predict contours you can use the contours to assist in the "
                             "segmentation. (This has not been shown to improve results.)")
    return parser


def main():
    namespace = parse_args()
    for fn, ofn in zip(namespace.filenames, namespace.output_filenames):
        if namespace.verbose:
            print(fn, "-->", ofn)
        image = nib.load(fn)
        label_map = one_hot_image_to_label_map(image,
                                               labels=namespace.labels,
                                               threshold=namespace.threshold,
                                               sum_then_threshold=namespace.sum,
                                               label_hierarchy=namespace.hierarchy)
        label_map.to_filename(ofn)


if __name__ == "__main__":
    main()
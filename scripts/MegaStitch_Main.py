import argparse
import datetime
import os
import sys

from megastitch import computer_vision_utils as cv_util, General_GPS_Correction
from megastitch.config import Configuration
from megastitch.utils import report_time, get_anchors_from_json


def get_args():
    parser = argparse.ArgumentParser(
        description="MegaStitch Drone Stitching and Geo-correction script.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-d",
        "--data",
        help="The path to the data directory.",
        metavar="data",
        required=True,
        type=str,
    )

    parser.add_argument(
        "-r",
        "--result",
        help="The path to the directory where the results will be saved.",
        metavar="result",
        required=True,
        type=str,
    )

    parser.add_argument(
        "-g",
        "--gcp",
        help="The path to Ground Control Points (GCPs) files. Refer to readme for formatting of the json/csv file.",
        metavar="gcp",
        required=False,
        type=str,
    )

    parser.add_argument(
        "-s",
        "--settings",
        help="The path to the json file that contains the configuration/settings information.",
        metavar="settings",
        required=True,
        type=str,
    )

    return parser.parse_args()


def main():
    args = get_args()

    if not os.path.exists(args.result):
        os.makedirs(args.result)

    transformation_path = os.path.join(args.result, "transformation.json")
    ortho_path = os.path.join(args.result, "ortho.png")
    plot_path = os.path.join(args.result, "initial_GPS.png")
    corrected_coordinates_path = os.path.join(args.result, "corrected_coordinates.json")
    log_path = os.path.join(args.result, "log.txt")
    sift_path = os.path.join(args.result, "SIFT")

    if not os.path.exists(sift_path):
        os.makedirs(sift_path)

    # Load the configuration
    config = Configuration(args.data)
    config.Dataset = os.path.basename(os.path.normpath(args.data))
    config.AllGCPRMSE = True
    config.load(args.settings)

    original = sys.stdout
    log_file = open(log_path, "w")
    sys.stdout = log_file

    start_time = datetime.datetime.now()

    if hasattr(args, "gcp") and args.gcp is not None:
        anchors_dict = get_anchors_from_json(args.gcp)
    else:
        anchors_dict = None

    field = General_GPS_Correction.Field(sift_p=sift_path, tr_p=transformation_path, settings=config)

    if config.transformation == cv_util.Transformation.similarity:
        (
            coords_dict,
            H,
            H_inv,
            abs_tr,
            _,
            _,
            _,
        ) = field.geo_correct_MegaStitchSimilarity(anchors_dict)
    elif (config.transformation == cv_util.Transformation.affine):
        (
            coords_dict,
            H,
            H_inv,
            abs_tr,
            _,
            _,
        ) = field.geo_correct_MegaStitchAffine(anchors_dict, None)
    elif (config.transformation == cv_util.Transformation.homography):
        if (config.preprocessing_transformation.lower() == "none"):
            (
                coords_dict,
                H,
                H_inv,
                abs_tr,
                _,
                _,
            ) = field.geo_correct_BundleAdjustment_Homography(anchors_dict, None)
        elif (config.preprocessing_transformation.lower() == "similarity"):
            (
                coords_dict,
                H,
                H_inv,
                abs_tr,
                _,
                _,
            ) = field.geo_correct_MegaStitch_Similarity_Bundle_Adjustment_Homography(
                anchors_dict, None
            )
        elif (config.preprocessing_transformation.lower() == "affine"):
            (
                coords_dict,
                H,
                H_inv,
                abs_tr,
                _,
                _,
            ) = field.geo_correct_MegaStitch_Affine_Bundle_Adjustment_Homography(
                anchors_dict, None
            )
    else:
        (
            coords_dict,
            H,
            H_inv,
            abs_tr,
            _,
            _,
            _,
        ) = field.geo_correct_MegaStitchSimilarity(anchors_dict)

    if H is None:
        gcp_inf = None
    else:
        gcp_inf = (anchors_dict, H_inv, abs_tr)

    field.generate_transformation_accuracy_histogram(
        coords_dict, plot_path.replace("initial_GPS", "transformation_plot")
    )
    # field.save_field_centers_visualization(plot_path) # DEBUG

    ortho = field.generate_field_ortho(coords_dict, gcp_info=gcp_inf)

    field.save_field_ortho(ortho, ortho_path)
    field.save_field_coordinates(corrected_coordinates_path, coords_dict)

    end_time = datetime.datetime.now()

    report_time(start_time, end_time)

    sys.stdout = original
    log_file.close()


if __name__ == "__main__":
    main()

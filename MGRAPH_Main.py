from numpy.lib.function_base import trapz
import General_GPS_Correction
import datetime
import sys
import os
import json
import computer_vision_utils as cv_util
import argparse


def report_time(start, end):
    print("-----------------------------------------------------------")
    print(
        "Start date time: {0}\nEnd date time: {1}\nTotal running time: {2}.".format(
            start, end, end - start
        )
    )


def get_anchors_from_json(path):

    with open(path, "r") as outfile:
        anchors_dict = json.load(outfile)

    return anchors_dict


def load_settings(settings_path):
    with open(settings_path, "r") as f:
        settings_dict = json.load(f)

    General_GPS_Correction.settings.scale = settings_dict["scale"]
    General_GPS_Correction.settings.nearest_number = settings_dict["nearest_number"]
    General_GPS_Correction.settings.discard_transformation_perc_inlier = settings_dict[
        "discard_transformation_perc_inlier"
    ]
    General_GPS_Correction.settings.transformation = getattr(
        cv_util.Transformation, settings_dict["transformation"]
    )
    General_GPS_Correction.settings.percentage_next_neighbor = settings_dict[
        "percentage_next_neighbor"
    ]
    General_GPS_Correction.settings.cores_to_use = settings_dict["cores_to_use"]
    General_GPS_Correction.settings.draw_GCPs = settings_dict["draw_GCPs"]

    General_GPS_Correction.settings.sub_set_choosing = settings_dict["sub_set_choosing"]
    General_GPS_Correction.settings.N_perc = settings_dict["N_perc"]
    General_GPS_Correction.settings.E_perc = settings_dict["E_perc"]


def get_args():
    parser = argparse.ArgumentParser(
        description="MGRAPH Drone Stitching and Geo-correction script.",
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

    General_GPS_Correction.init_setting(args.data)
    General_GPS_Correction.settings.Dataset = os.path.basename(
        os.path.normpath(args.data)
    )

    General_GPS_Correction.settings.AllGCPRMSE = True

    load_settings(args.settings)

    original = sys.stdout
    log_file = open(log_path, "w")
    sys.stdout = log_file

    start_time = datetime.datetime.now()

    if hasattr(args, "gcp") and args.gcp is not None:
        anchors_dict = get_anchors_from_json(args.gcp)
    else:
        anchors_dict = None

    field = General_GPS_Correction.Field(sift_p=sift_path, tr_p=transformation_path)

    (
        coords_dict,
        H,
        H_inv,
        abs_tr,
        _,
        _,
        _,
    ) = field.geo_correct_MGRAPH(anchors_dict)

    if H is None:
        gcp_inf = None
    else:
        gcp_inf = (anchors_dict, H_inv, abs_tr)

    field.generate_transformation_accuracy_histogram(
        coords_dict, plot_path.replace("initial_GPS", "transformation_plot")
    )
    field.save_field_centers_visualization(plot_path)

    ortho = field.generate_field_ortho(coords_dict, gcp_info=gcp_inf)

    field.save_field_ortho(ortho, ortho_path)
    field.save_field_coordinates(corrected_coordinates_path, coords_dict)

    end_time = datetime.datetime.now()

    report_time(start_time, end_time)

    sys.stdout = original
    log_file.close()


main()

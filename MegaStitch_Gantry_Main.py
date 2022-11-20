import General_GPS_Correction
import datetime
import sys
import os
import computer_vision_utils as cv_util
import json
import multiprocessing
import argparse
from osgeo import gdal


def get_patch_coord_dict_from_name(args):

    image_name = args[0]
    patch_folder = args[1]

    ds = gdal.Open("{0}/{1}".format(patch_folder, image_name))

    meta = gdal.Info(ds)

    lines = meta.splitlines()

    for line in lines:
        if "Upper Left" in line:
            u_l = line.split()[2:4]
            u_l[0] = u_l[0].replace("(", "").replace(",", "")
            u_l[1] = u_l[1].replace(")", "")

        if "Lower Left" in line:
            l_l = line.split()[2:4]
            l_l[0] = l_l[0].replace("(", "").replace(",", "")
            l_l[1] = l_l[1].replace(")", "")

        if "Upper Right" in line:
            u_r = line.split()[2:4]
            u_r[0] = u_r[0].replace("(", "").replace(",", "")
            u_r[1] = u_r[1].replace(")", "")

        if "Lower Right" in line:
            l_r = line.split()[2:4]
            l_r[0] = l_r[0].replace("(", "").replace(",", "")
            l_r[1] = l_r[1].replace(")", "")

        if "Center" in line:
            c = line.split()[1:3]
            c[0] = c[0].replace("(", "").replace(",", "")
            c[1] = c[1].replace(")", "")

    upper_left = (float(u_l[0]), float(u_l[1]))
    lower_left = (float(l_l[0]), float(l_l[1]))
    upper_right = (float(u_r[0]), float(u_r[1]))
    lower_right = (float(l_r[0]), float(l_r[1]))
    center = (float(c[0]), float(c[1]))

    coord = {
        "name": image_name,
        "UL": upper_left,
        "LL": lower_left,
        "UR": upper_right,
        "LR": lower_right,
        "C": center,
    }

    return coord


def get_all_coords_from_TIFFs(patch_folder):

    image_names = os.listdir(patch_folder)

    args = []

    for img_name in image_names:

        if "_right" in img_name:
            continue

        args.append((img_name, patch_folder))

    processes = multiprocessing.Pool(General_GPS_Correction.settings.cores_to_use)
    results = processes.map(get_patch_coord_dict_from_name, args)
    processes.close()

    coords = {}

    for r in results:

        coords[r["name"]] = {
            "UL": {"lon": r["UL"][0], "lat": r["UL"][1]},
            "UR": {"lon": r["UR"][0], "lat": r["UR"][1]},
            "LL": {"lon": r["LL"][0], "lat": r["LL"][1]},
            "LR": {"lon": r["LR"][0], "lat": r["LR"][1]},
            "C": {"lon": r["C"][0], "lat": r["C"][1]},
        }

    return coords


def get_center_coords(coords):

    center_coords = {}

    for c in coords:
        center_coords[c] = {"lat": coords[c]["C"]["lat"], "lon": coords[c]["C"]["lon"]}

    return center_coords


def get_anchors_from_json(path):

    with open(path, "r") as outfile:
        anchors_dict = json.load(outfile)

    return anchors_dict


def revise_latitude(coords_dict):

    new_coords_dict = {}

    for img in coords_dict:
        img_coords = coords_dict[img]
        coord = {
            "UL": [img_coords["UL"][0], -1 * img_coords["UL"][1]],
            "UR": [img_coords["UR"][0], -1 * img_coords["UR"][1]],
            "LL": [img_coords["LL"][0], -1 * img_coords["LL"][1]],
            "LR": [img_coords["LR"][0], -1 * img_coords["LR"][1]],
        }

        new_coords_dict[img] = coord

    return new_coords_dict


def report_time(start, end):
    print("-----------------------------------------------------------")
    print(
        "Start date time: {0}\nEnd date time: {1}\nTotal running time: {2}.".format(
            start, end, end - start
        )
    )


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
        description="MegaStitch Gantry Stitching and Geo-correction script.",
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

    parser.add_argument(
        "-m",
        "--method",
        help="The name of the method. It should be either 'MGRAPH', 'MEGASTITCH-TR-KP', or 'MEGASTITCH-TR-PR'.",
        metavar="method",
        required=True,
        type=str,
    )

    parser.add_argument(
        "-n",
        "--dataset_name",
        help="The name of the Dataset. It should be one of 'GSE', 'GLE', 'GLL', or 'GSL' based on the dataset being used.",
        metavar="dataset_name",
        required=True,
        type=str,
    )

    return parser.parse_args()


def main():

    args = get_args()

    if not os.path.exists(args.result):
        os.makedirs(args.result)

    tr_path = os.path.join(args.result, "transformation.json")
    ortho_path = os.path.join(args.result, "ortho.png")
    plot_path = os.path.join(args.result, "initial_GPS.png")
    corrected_coordinates_path = os.path.join(args.result, "corrected_coordinates.json")
    log_path = os.path.join(args.result, "log.txt")
    sift_path = os.path.join(args.result, "SIFT")
    data_set_path = args.data

    if not os.path.exists(sift_path):
        os.mkdir(sift_path)

    original = sys.stdout
    log_file = open(log_path, "w")
    sys.stdout = log_file

    start_time = datetime.datetime.now()

    General_GPS_Correction.init_setting(data_set_path)
    load_settings(args.settings)

    gps_coords_dict = get_all_coords_from_TIFFs(data_set_path)
    center_coords = get_center_coords(gps_coords_dict)

    anchors_dict = get_anchors_from_json(args.gcp)

    field = General_GPS_Correction.Field(center_coords, sift_p=sift_path, tr_p=tr_path)

    gantry_scale = 0.02

    if args.method == "MGRAPH":

        coords_dict = field.geo_correct_MGRAPH()
        ortho = field.generate_field_ortho(coords_dict, gantry_scale)

    elif args.method == "MEGASTITCH-TR-KP":
        General_GPS_Correction.settings.parallel_stitch = False
        General_GPS_Correction.settings.transformation = (
            cv_util.Transformation.translation
        )
        coords_dict = field.geo_correct_Translation_KP_based(
            gps_coords_dict, anchors_dict[args.dataset_name]
        )
        ortho = field.generate_field_ortho(coords_dict, gantry_scale, gps_coords_dict)
        coords_dict = revise_latitude(coords_dict)

    elif args.method == "MEGASTITCH-TR-PR":
        General_GPS_Correction.settings.parallel_stitch = False
        General_GPS_Correction.settings.transformation = (
            cv_util.Transformation.translation
        )
        coords_dict = field.geo_correct_Translation_Parameter_based(
            gps_coords_dict, anchors_dict[args.dataset_name]
        )
        ortho = field.generate_field_ortho(coords_dict, gantry_scale, gps_coords_dict)
        coords_dict = revise_latitude(coords_dict)

    field.save_field_centers_visualization(plot_path)
    field.save_field_coordinates(corrected_coordinates_path, coords_dict)
    field.save_field_ortho(ortho, ortho_path)

    end_time = datetime.datetime.now()
    report_time(start_time, end_time)

    sys.stdout = original
    log_file.close()


main()

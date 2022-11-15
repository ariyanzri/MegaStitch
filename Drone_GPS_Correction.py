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


def save_json_results(json_path, res, experiment_Id):

    with open(json_path, "r") as outfile:
        current_results = json.load(outfile)

    if General_GPS_Correction.settings.do_cross_validation:

        current_results[experiment_Id] = {
            "GCP_RMSE": res[0],
            "Proj_RMSE": res[1],
            "Proj_RMSE": res[2],
            "Time": str(res[3]),
            "CV_GCP_RMSE_AVG": res[4],
            "CV_GCP_RMSE_STD": res[5],
            "CV_Proj_RMSE_AVG": res[6],
            "CV_Proj_RMSE_STD": res[7],
            "CV_Proj_RMSE_Norm_AVG": res[8],
            "CV_Proj_RMSE_Norm_STD": res[9],
            "CV_Time_AVG": str(res[10]),
        }
    else:

        current_results[experiment_Id] = {
            "GCP_RMSE": res[0],
            "Proj_RMSE": res[1],
            "Proj_RMSE_Norm": res[2],
            "Time": str(res[3]),
        }

    with open(json_path, "w") as outfile:
        json.dump(current_results, outfile)


def save_json_sim_gcps(json_path, list_gcps, dataset_name):

    with open(json_path, "r") as outfile:
        current_results = json.load(outfile)

    current_results[dataset_name] = list_gcps

    with open(json_path, "w") as outfile:
        json.dump(current_results, outfile)


def load_json_sim_gcps(json_path, dataset_name):

    with open(json_path, "r") as outfile:
        current_results = json.load(outfile)

    return current_results[dataset_name]


def load_settings(args.settings):
    with open(args.settings, "r") as f:
        settings_dict = json.load(f)

    General_GPS_Correction.settings.grid_w = settings_dict["grid_w"]
    General_GPS_Correction.settings.grid_h = settings_dict["grid_h"]
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
    General_GPS_Correction.settings.max_no_inliers = settings_dict["max_no_inliers"]
    General_GPS_Correction.settings.number_bins = settings_dict["number_bins"]
    General_GPS_Correction.settings.size_bins = settings_dict["size_bins"]
    General_GPS_Correction.settings.do_cross_validation = settings_dict[
        "do_cross_validation"
    ]
    General_GPS_Correction.settings.sub_set_choosing = settings_dict["sub_set_choosing"]
    General_GPS_Correction.settings.N_perc = settings_dict["N_perc"]
    General_GPS_Correction.settings.E_perc = settings_dict["E_perc"]


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

    ortho_path = os.path.join(args.result, "ortho.png")
    plot_path = os.path.join(args.result, "initial_GPS.png")
    corrected_coordinates_path = plot_path = os.path.join(
        args.result, "corrected_coordinates.json"
    )
    log_path = os.path.join(args.result, "log.txt")
    sift_path = os.path.join(args.result, "SIFT")

    if not os.path.exists(sift_path):
        os.makedirs(sift_path)

    General_GPS_Correction.init_setting(args.data)
    General_GPS_Correction.settings.Dataset = os.path.basename(
        os.path.normpath(args.data)
    )
    General_GPS_Correction.settings.args.method = args.method
    General_GPS_Correction.settings.AllGCPRMSE = True

    load_settings(args.settings)

    original = sys.stdout
    log_file = open(log_path, "w")
    sys.stdout = log_file

    start_time = datetime.datetime.now()

    if hasattr(args,"gcp"):
        anchors_dict = get_anchors_from_json(args.gcp)
    else:
        anchors_dict = None

    field = General_GPS_Correction.Field(sift_p=sift_path)

    if General_GPS_Correction.settings.transformation == cv_util.Transformation.similarity:
        
        (
            coords_dict,
            H,
            H_inv,
            abs_tr,
            res,
            Sim_GCPs,
            all_GCPs,
        ) = field.geo_correct_MegaStitchSimilarity(anchors_dict)
    elif General_GPS_Correction.settings.transformation == cv_util.Transformation.affine:
        (
            coords_dict,
            H,
            H_inv,
            abs_tr,
            res,
            all_GCPs,
        ) = field.geo_correct_MegaStitchAffine(anchors_dict, Sim_GCPs)
    elif General_GPS_Correction.settings.transformation == cv_util.Transformation.homography:
        if General_GPS_Correction.settings.preprocessing_transformation == "None":
            (
                coords_dict,
                H,
                H_inv,
                abs_tr,
                res,
                all_GCPs,
            ) = field.geo_correct_BundleAdjustment_Homography(anchors_dict, Sim_GCPs)
        elif General_GPS_Correction.settings.preprocessing_transformation == "similarity":
            (
                coords_dict,
                H,
                H_inv,
                abs_tr,
                res,
                all_GCPs,
            ) = field.geo_correct_MegaStitch_Similarity_Bundle_Adjustment_Homography(
                anchors_dict, Sim_GCPs
            )
        elif General_GPS_Correction.settings.preprocessing_transformation == "affine":
            (
                coords_dict,
                H,
                H_inv,
                abs_tr,
                res,
                all_GCPs,
            ) = field.geo_correct_MegaStitch_Affine_Bundle_Adjustment_Homography(
                anchors_dict, Sim_GCPs
            )
   
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

import numpy as np
import os
import cv2
import sys
import math
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from megastitch import computer_vision_utils as cv_util
import multiprocessing
import json
import pickle
import gc
from megastitch import ProjectionOptimization
from megastitch import utils_MGRAPH
from megastitch.Customized_myltiprocessing import MyPool
from GPSPhoto import gpsphoto

from megastitch.config import Configuration


def generate_SIFT_points(args):

    max_SIFT_points = 100000  # FIXME : find a better way to set this value, should use the config value

    img = args[0]
    name = args[1]

    if len(args) == 3:

        path_for_gantry = args[2]

        if not os.path.exists(
            "{0}/{1}_SIFT.data".format(
                path_for_gantry, name.replace(".JPG", "").replace(".tif", "")
            )
        ):

            img.load_img(True)
            kp, desc = cv_util.get_SIFT_points(img.img, None, max_SIFT_points)
            pickalable_kp = [cv_util.keypoint_to_tuple_encode(p) for p in kp]
            img.delete_img()

            pickle.dump(
                (pickalable_kp, desc),
                open(
                    "{0}/{1}_SIFT.data".format(
                        path_for_gantry, name.replace(".JPG", "").replace(".tif", "")
                    ),
                    "wb",
                ),
            )

            return pickalable_kp, desc, name

        else:

            (pickalable_kp, desc) = pickle.load(
                open(
                    "{0}/{1}_SIFT.data".format(
                        path_for_gantry, name.replace(".JPG", "").replace(".tif", "")
                    ),
                    "rb",
                )
            )

            return pickalable_kp, desc, name

    else:

        img.load_img(True)
        kp, desc = cv_util.get_SIFT_points(img.img, None, max_SIFT_points)
        pickalable_kp = [cv_util.keypoint_to_tuple_encode(p) for p in kp]
        img.delete_img()

    gc.collect()

    return pickalable_kp, desc, name


def warp_estimate_helper(args):

    pts1 = args[0]
    pts2 = args[1]
    img = args[2]
    frame_size = args[3]
    s = args[4]

    img.load_img(s=s)

    tmp = cv_util.find_warp_homography_and_warp(pts1, pts2, img.img, frame_size)

    img.delete_img()
    gc.collect()

    return tmp


def transformation_estimation_helper(args):

    img_n_desc = args[0]
    img_desc = args[1]
    img_n_kp = args[2]
    img_kp = args[3]
    transformation = args[4]
    percentage_next_neighbor = args[5]
    cores_to_use = args[6]
    img_n_name = args[7]
    img_name = args[8]

    # if T multiplied by the corners of img1 (in img_n system) gives the corners of img_n (in img_n system)
    T, matches, perc_inliers, inliers = cv_util.estimate_transformation_from_SIFT(
        img_n_desc,
        img_desc,
        img_n_kp,
        img_kp,
        transformation,
        percentage_next_neighbor,
        cores_to_use,
    )

    return (
        img_name,
        img_n_name,
        T,
        cv_util.pickle_matches(matches),
        perc_inliers,
        inliers,
    )


def get_neighbor_helper(args):

    nearest_number = 4 # FIXME: use the config value instead of this hardcoded one

    img_name = args[0]
    images_dict = args[1]

    img = images_dict[img_name]

    list_all_neighbors = []
    neighbors = []

    for img_n_name in images_dict:
        img_n = images_dict[img_n_name]

        if img == img_n:
            continue

        list_all_neighbors.append((img_n, img.get_distance(img_n)))

    sorted_list_all_neighbors = sorted(list_all_neighbors, key=lambda x: x[1])

    neighbors += [
        (img_name, img_n.name, None, None)
        for img_n, _ in sorted_list_all_neighbors[: nearest_number]
    ]
    neighbors += [
        (img_n.name, img_name, None, None)
        for img_n, _ in sorted_list_all_neighbors[: nearest_number]
    ]

    return neighbors


class Image:
    def __init__(self, name, coords=None, settings=None):
        self.name = name

        if settings is None:
            self.settings = Configuration()
        else:
            self.settings = settings

        if coords is None:
            gps = gpsphoto.getGPSData("{0}/{1}".format(self.settings.images_path, name))

            if "Latitude" in gps:
                self.lat = gps["Latitude"]
                self.lon = gps["Longitude"]
                self.alt = gps["Altitude"]
            else:
                self.lat = None
                self.lon = None
                self.alt = None
        else:
            self.lat = coords[name]["lat"]
            self.lon = coords[name]["lon"]
            self.alt = 1

        self.img = None
        self.kp = None
        self.desc = None

    def __eq__(self, img):
        return self.name == img.name

    def __str__(self):

        return "{0}: ({1},{2},{3})".format(self.name, self.lat, self.lon, self.alt)

    def get_distance(self, img):

        s1 = self.name.replace("DJI_", "").replace(".JPG", "")
        s2 = img.name.replace("DJI_", "").replace(".JPG", "")

        if s1.isdigit() and s2.isdigit() and ".JPG" in self.name:

            s1 = int(s1)
            s2 = int(s2)

            if abs(s1 - s2) == 1:
                return 0

        if self.settings.use_gps_distance:

            return cv_util.get_gps_distance(self.lat, self.lon, img.lat, img.lon)
        else:
            return math.sqrt(
                (self.lon - img.lon) ** 2
                + (self.lat - img.lat) ** 2
                + (self.alt - img.alt) ** 2
            )

    def get_distance_point(self, point):

        return cv_util.get_gps_distance(self.lat, self.lon, point[1], point[0])

    def crop(self, perc, tmp):

        w = tmp.shape[0]
        h = tmp.shape[1]

        x = int(perc * tmp.shape[0] / 2)
        y = int(perc * tmp.shape[1] / 2)

        tmp = tmp[x : w - x, y : h - y]

        return tmp

    def load_img(self, for_sift=False, s=None):

        if s is None:

            if self.img is not None:
                return

            tmp = cv2.imread("{0}/{1}".format(self.settings.images_path, self.name))
            tmp = cv2.resize(
                tmp,
                (
                    int(self.settings.scale * self.settings.image_size[1]),
                    int(self.settings.scale * self.settings.image_size[0]),
                ),
            )

            if self.settings.equalize_histogram and for_sift:
                tmp = cv_util.histogram_equalization(tmp)

            tmp = self.crop(self.settings.perc_crop, tmp)

            self.img = tmp

        else:

            tmp = cv2.imread("{0}/{1}".format(self.settings.images_path, self.name))
            tmp = cv2.resize(
                tmp, (int(s * self.settings.image_size[1]), int(s * self.settings.image_size[0]))
            )

            self.img = tmp

    def delete_img(self):

        self.img = None

    def update_SIFT_points(self, pickeled_kp, desc):

        self.kp = pickeled_kp
        self.desc = desc

        if self.settings.normalize_key_points:
            self.load_img()
            self.kp = cv_util.normalize_key_points(
                self.kp, self.img.shape[1], self.img.shape[0], (self.lon, self.lat)
            )

    def draw_visualize_SIFT(self):
        if self.kp is None:
            print("SIFT has not been generated yet.")
            return

        img = cv_util.draw_SIFT_points_on_img(self.img, self.kp, self.desc)
        cv_util.show(img, "SIFT Key Points", 1000, 700)


class Field:

    # -----------------------------------------------------------
    # ---------- Preprocessing and initialization ---------------
    # -----------------------------------------------------------

    def __init__(self, coords=None, image_names=None, sift_p=None, tr_p=None, settings=None):

        if settings is None:
            self.settings = Configuration()
        else:
            self.settings = settings

        self.images, self.images_dict = self.get_all_images(coords, image_names)

        print(">>> Total number of images: {0}".format(len(self.images)))
        sys.stdout.flush()

        self.neighbors, self.reference_image = self.get_neighbors()

        print(">>> Neighbors calculated successfully.")
        sys.stdout.flush()

        self.image_name_to_index_dict = {}
        self.SIFT_path = sift_p
        self.Transformation_path = tr_p

        for i, img in enumerate(self.images):
            self.image_name_to_index_dict[img.name] = i

        print(">>> Field Created.")

    def __str__(self):

        string = ""
        for image in self.images:
            string += str(image) + "\n"

        return string

    def select_sub_set(self, list_images):

        new_list_images = []
        new_dict_images = {}

        min_lat = sys.maxsize
        max_lat = -sys.maxsize

        min_lon = sys.maxsize
        max_lon = -sys.maxsize

        for img in list_images:

            lat = img.lat
            lon = img.lon

            if lat < min_lat:
                min_lat = lat

            if lat > max_lat:
                max_lat = lat

            if lon < min_lon:
                min_lon = lon

            if lon > max_lon:
                max_lon = lon

        for img in list_images:

            lat = img.lat
            lon = img.lon

            if max_lat - lat > self.settings.N_perc * (max_lat - min_lat):
                continue

            if max_lon - lon > self.settings.E_perc * (max_lon - min_lon):
                continue

            new_list_images.append(img)
            new_dict_images[img.name] = img

        return new_list_images, new_dict_images

    def get_all_images(self, coords, image_names):

        list_images = []
        dict_images = {}

        img_path_list = os.listdir(self.settings.images_path)

        for i in img_path_list:

            if coords is not None and "_right" in i:
                continue

            if image_names is not None and i not in image_names:
                continue

            img = Image(i, coords, settings=self.settings)
            list_images.append(img)
            dict_images[i] = img

        if self.settings.sub_set_choosing and list_images[0].lat is not None:
            list_images, dict_images = self.select_sub_set(list_images)

        print(">>> Images loaded successfully.")
        return list_images, dict_images

    def get_successive_neighbors(self):

        neighbors = []

        im1 = None

        for img in sorted(self.images, key=lambda x: x.name):
            if im1 is None:
                im1 = img
                continue

            neighbors.append((im1, img, None, None))
            im1 = img

        return neighbors

    def get_neighbors(self):

        neighbors = []

        if self.images[0].lat is None:

            for img1 in self.images:
                for img2 in self.images:
                    if img1 == img2:
                        continue

                    neighbors.append((img1, img2, None, None))

            return neighbors, self.images[0]

        # ------- Get the reference image as a central one ----------

        print(">>> Begining calculation of the reference image.")
        sys.stdout.flush()

        center_ortho = (
            np.mean(np.array([img.lat for img in self.images])),
            np.mean(np.array([img.lon for img in self.images])),
        )
        ref_image = None
        min_distance_to_center = sys.maxsize

        for img in self.images:
            d = cv_util.get_gps_distance(
                center_ortho[0], center_ortho[1], img.lat, img.lon
            )

            if d < min_distance_to_center:
                min_distance_to_center = d
                ref_image = img

        print(
            ">>> Reference image is {0}".format(
                "None" if ref_image is None else ref_image.name
            )
        )
        sys.stdout.flush()

        # ----------- Build the neighbors from Reference image -----------

        args = []

        for img in self.images:
            args.append((img.name, self.images_dict))

        processes = multiprocessing.Pool(self.settings.cores_to_use)
        results = processes.map(get_neighbor_helper, args)
        processes.close()

        for n in results:
            neighbors += n

        neighbors = list(set(neighbors))

        neighbors = [
            (self.images_dict[img_1], self.images_dict[img_2], p3, p4)
            for img_1, img_2, p3, p4 in neighbors
        ]

        return neighbors, ref_image

    def get_edge_weights(self):

        edge_matrix = np.zeros((len(self.images), len(self.images)))

        for img, img_n, _, _ in self.neighbors:
            i = self.image_name_to_index_dict[img.name]
            j = self.image_name_to_index_dict[img_n.name]

            edge_matrix[i][j] = np.round(img.get_distance(img_n), 3)

        return edge_matrix

    def get_positions(self):
        pos = {}

        for img in self.images:
            pos[img.name] = [img.lon, img.lat]

        return pos

    def generate_SIFT_points_all_Images(self):

        args = []

        for img in self.images:
            if img.kp is not None:
                continue

            if self.SIFT_path is None:
                args.append((img, img.name))
            else:
                args.append((img, img.name, self.SIFT_path))

        processes = multiprocessing.Pool(self.settings.cores_to_use)
        results = processes.map(generate_SIFT_points, args)
        processes.close()

        for kp, desc, name in results:
            self.images_dict[name].update_SIFT_points(kp, desc)

        print(">>> SIFT Features extracted successfully.")

    def get_intersection_DMATH(self, inlier_matches_1_2, inlier_matches_2_1):

        new_inlier_matches_1_2 = []
        new_inlier_matches_2_1 = []

        for m in inlier_matches_1_2:
            for m2 in inlier_matches_2_1:
                if m2.queryIdx == m.trainIdx and m2.trainIdx == m.queryIdx:
                    new_inlier_matches_1_2.append(m)
                    new_inlier_matches_2_1.append(m2)

        return new_inlier_matches_1_2, new_inlier_matches_2_1

    def refine_transformations(self, transformation_dict):

        new_transformation_dict = {}

        for img1 in self.images:

            for img2 in self.images:

                if img1.name == img2.name:
                    continue

                if (
                    img1.name in transformation_dict
                    and img2.name in transformation_dict[img1.name]
                    and img2.name in transformation_dict
                    and img1.name in transformation_dict[img2.name]
                    and not (
                        img1.name in new_transformation_dict
                        and img2.name in new_transformation_dict[img1.name]
                    )
                ):

                    matches_1_2 = transformation_dict[img1.name][img2.name][1]
                    matches_2_1 = transformation_dict[img2.name][img1.name][1]

                    inliers_1_2 = transformation_dict[img1.name][img2.name][3]
                    inliers_2_1 = transformation_dict[img2.name][img1.name][3]

                    inlier_matches_1_2 = matches_1_2[inliers_1_2[:, 0] == 1]
                    inlier_matches_2_1 = matches_2_1[inliers_2_1[:, 0] == 1]

                    (
                        inlier_matches_1_2,
                        inlier_matches_2_1,
                    ) = self.get_intersection_DMATH(
                        inlier_matches_1_2, inlier_matches_2_1
                    )

                    (
                        T_12,
                        matches_12,
                        perc_inliers_12,
                        inliers_12,
                    ) = cv_util.estimate_transformation_from_Inliers(
                        inlier_matches_1_2,
                        img2.desc,
                        img1.desc,
                        img2.kp,
                        img1.kp,
                        self.settings.transformation,
                    )
                    (
                        T_21,
                        matches_21,
                        perc_inliers_21,
                        inliers_21,
                    ) = cv_util.estimate_transformation_from_Inliers(
                        inlier_matches_2_1,
                        img1.desc,
                        img2.desc,
                        img1.kp,
                        img2.kp,
                        self.settings.transformation,
                    )

                    if (
                        T_12 is None
                        or T_21 is None
                        or perc_inliers_12 < self.settings.discard_transformation_perc_inlier
                        or perc_inliers_21 < self.settings.discard_transformation_perc_inlier
                    ):
                        continue

                    if img1.name not in new_transformation_dict:
                        new_transformation_dict[img1.name] = {}

                    if img2.name not in new_transformation_dict:
                        new_transformation_dict[img2.name] = {}

                    new_transformation_dict[img1.name][img2.name] = (
                        T_12,
                        matches_1_2,
                        round(101 - 100 * perc_inliers_12, 2),
                        inliers_1_2,
                    )
                    new_transformation_dict[img2.name][img1.name] = (
                        T_21,
                        matches_2_1,
                        round(101 - 100 * perc_inliers_21, 2),
                        inliers_2_1,
                    )

        return new_transformation_dict

    def get_matches_bins_for_cross_validation(
        self, matches, inliers, number_bins, size_bins
    ):

        bins = {}

        current_bin_index = 0
        bins[current_bin_index] = []

        for i, m in enumerate(matches):

            if inliers[i, 0] == 0:
                continue

            bins[current_bin_index].append(m)

            if len(bins[current_bin_index]) >= size_bins:
                current_bin_index += 1

                if current_bin_index >= number_bins:
                    break

                bins[current_bin_index] = []

        return bins

    def generate_neighbor_transformations(self):

        self.generate_SIFT_points_all_Images()
        sys.stdout.flush()

        if self.Transformation_path is not None and os.path.exists(
            self.Transformation_path
        ):
            with open(self.Transformation_path, "r") as f:
                jsonified = json.load(f)
                transformation_dict = cv_util.Unjsonify(jsonified)

                self.pairwise_transformations = transformation_dict

                print(":: Transformations loaded from file.")
                sys.stdout.flush()

                return transformation_dict

        transformation_dict = {}
        corner_translations = {}

        new_neighbors = []

        self.images[0].load_img()

        w = self.images[0].img.shape[1]
        h = self.images[0].img.shape[0]

        args = []

        print(">>> Begining encoding keypoint information...")
        sys.stdout.flush()

        for img, img_n, _, _ in self.neighbors:

            if img.name not in transformation_dict:
                transformation_dict[img.name] = {}

            if img.name not in corner_translations:
                corner_translations[img.name] = {}

            nkps = img_n.kp
            kps = img.kp

            c = 1

            args.append(
                (
                    img_n.desc,
                    img.desc,
                    nkps,
                    kps,
                    self.settings.transformation,
                    self.settings.percentage_next_neighbor,
                    c,
                    img_n.name,
                    img.name,
                )
            )

        print(
            ">>> Encoding keypoint information is done. Total number of neighbor transformation to estimate: {0}.".format(
                len(args)
            )
        )
        print(">>> Begining estimating transformation.")
        sys.stdout.flush()

        if self.settings.transformation == cv_util.Transformation.translation:
            results = []

            for a in args:
                results.append(transformation_estimation_helper(a))

            processes = MyPool(self.settings.cores_to_use)
            results = processes.map(transformation_estimation_helper, args)
            processes.close()

        else:
            processes = multiprocessing.Pool(self.settings.cores_to_use)
            results = processes.map(transformation_estimation_helper, args)
            processes.close()

        print(">>> Finished processing the transformation.")
        sys.stdout.flush()

        for img_name, img_n_name, T, pickled_matches, perc_inliers, inliers in results:

            matches = cv_util.get_matches_from_pickled(pickled_matches)

            new_neighbors.append(
                (
                    self.images_dict[img_name],
                    self.images_dict[img_n_name],
                    T,
                    perc_inliers,
                )
            )

            if T is None or perc_inliers < self.settings.discard_transformation_perc_inlier:
                continue

            bins = self.get_matches_bins_for_cross_validation(
                matches, inliers, self.settings.number_bins, self.settings.size_bins
            )

            num_inliers = sum([len(bins[a]) for a in bins])

            if num_inliers < 25:

                bins = None

            # if T multiplied by the corners of img1 (in img_1 system) gives the corners of img_n (in img_1 system)
            transformation_dict[img_name][img_n_name] = (
                T,
                matches,
                round(101 - 100 * perc_inliers, 2),
                inliers,
                bins,
            )

        self.neighbors = new_neighbors

        print(">>> Pairwise transformations estimated successfully.")
        sys.stdout.flush()

        if self.settings.refine_transformations:
            transformation_dict = self.refine_transformations(transformation_dict)

        self.pairwise_transformations = transformation_dict

        with open(self.Transformation_path, "w") as f:
            jsonified = cv_util.Jsonify(transformation_dict)
            json.dump(jsonified, f)

        return transformation_dict

    def get_absolute_transformations_after_geo_correction(self, geo_corrected_coords):

        absolute_transformation = {}

        R_coords = geo_corrected_coords[self.reference_image.name]

        ptsr = np.float32(
            [
                [R_coords["UL"][0], R_coords["UL"][1]],
                [R_coords["UR"][0], R_coords["UR"][1]],
                [R_coords["LR"][0], R_coords["LR"][1]],
                [R_coords["LL"][0], R_coords["LL"][1]],
            ]
        )

        for img in self.images_dict:

            pts1 = np.float32(
                [
                    [
                        geo_corrected_coords[img]["UL"][0],
                        geo_corrected_coords[img]["UL"][1],
                    ],
                    [
                        geo_corrected_coords[img]["UR"][0],
                        geo_corrected_coords[img]["UR"][1],
                    ],
                    [
                        geo_corrected_coords[img]["LR"][0],
                        geo_corrected_coords[img]["LR"][1],
                    ],
                    [
                        geo_corrected_coords[img]["LL"][0],
                        geo_corrected_coords[img]["LL"][1],
                    ],
                ]
            )

            T = cv_util.estimate_base_transformations(
                pts1, ptsr, self.settings.transformation
            )

            absolute_transformation[img] = T

        return absolute_transformation

    def get_initial_coord_locations(self, img_ref_name, img_ref_coords, img_w, img_h):

        initial_coord_dict = {}

        xdiff_dict = {}
        ydiff_dict = {}

        image_ref = self.images_dict[img_ref_name]

        for img in self.images:
            if img.name == img_ref_name:
                continue

            xdiff_dict[img.name] = image_ref.lon - img.lon
            ydiff_dict[img.name] = image_ref.lat - img.lat

        min_x = min([xdiff_dict[k] for k in xdiff_dict])
        xdiff_dict = {k: xdiff_dict[k] / min_x for k in xdiff_dict}

        min_y = min([ydiff_dict[k] for k in ydiff_dict])
        ydiff_dict = {k: ydiff_dict[k] / min_y for k in ydiff_dict}

        for img in self.images:
            if img.name == img_ref_name:
                initial_coord_dict[img.name] = img_ref_coords
                continue

            initial_coord_dict[img.name] = [0, 0, 0, 0, 0, 0, 0, 0]

            initial_coord_dict[img.name][0] = (
                img_ref_coords[0] - xdiff_dict[img.name] * img_w / 2
            )
            initial_coord_dict[img.name][4] = (
                img_ref_coords[4] + ydiff_dict[img.name] * img_h / 2
            )

            initial_coord_dict[img.name][1] = (
                img_ref_coords[1] - xdiff_dict[img.name] * img_w / 2
            )
            initial_coord_dict[img.name][5] = (
                img_ref_coords[5] + ydiff_dict[img.name] * img_h / 2
            )

            initial_coord_dict[img.name][2] = (
                img_ref_coords[2] - xdiff_dict[img.name] * img_w / 2
            )
            initial_coord_dict[img.name][6] = (
                img_ref_coords[6] + ydiff_dict[img.name] * img_h / 2
            )

            initial_coord_dict[img.name][3] = (
                img_ref_coords[3] - xdiff_dict[img.name] * img_w / 2
            )
            initial_coord_dict[img.name][7] = (
                img_ref_coords[7] + ydiff_dict[img.name] * img_h / 2
            )

        return initial_coord_dict

    def get_coords_from_absolute_transformations(
        self, transformations_dict, width, height
    ):

        image_corners_dict = {}

        for img_name in self.images_dict:

            i = self.images_dict[img_name]
            H = transformations_dict[img_name]

            UL_ref = [0, 0, 1]
            UR_ref = [width, 0, 1]
            LR_ref = [width, height, 1]
            LL_ref = [0, height, 1]

            UL = np.matmul(H, UL_ref)
            UL = UL / UL[2]
            UL = UL[:2]

            UR = np.matmul(H, UR_ref)
            UR = UR / UR[2]
            UR = UR[:2]

            LR = np.matmul(H, LR_ref)
            LR = LR / LR[2]
            LR = LR[:2]

            LL = np.matmul(H, LL_ref)
            LL = LL / LL[2]
            LL = LL[:2]

            image_corners_dict[img_name] = {"UL": UL, "UR": UR, "LR": LR, "LL": LL}

        return image_corners_dict

    # -----------------------------------------------------------
    # ------------ Error Measurement Methods --------------------
    # -----------------------------------------------------------

    def calculate_H_for_sim_GCP(self, absolute_transformations, anchors_dict, sim_GCP):

        From_centers = []
        To_centers = []

        for l in sim_GCP:
            H_t = absolute_transformations[l["img_name"]]
            p = np.array([l["img_x"] * self.settings.scale, l["img_y"] * self.settings.scale, 1])
            p_new = np.matmul(H_t, p)
            p_new /= p_new[2]

            From_centers.append(p_new[:2])
            To_centers.append([l["true_lon"], l["true_lat"]])

        if self.settings.transformation == cv_util.Transformation.affine:
            H = cv_util.get_Similarity_Affine(
                np.array(From_centers), np.array(To_centers)
            )

        if self.settings.transformation == cv_util.Transformation.similarity:
            H = cv_util.get_Similarity_Affine(
                np.array(From_centers), np.array(To_centers)
            )

        if self.settings.transformation == cv_util.Transformation.homography:
            H, _ = cv2.findHomography(
                np.array(From_centers),
                np.array(To_centers),
                maxIters=1000,
                confidence=0.99,
                method=cv2.RANSAC,
            )

        return H

    def calculate_projection_error_sim_GCPs(
        self,
        anchors_dict,
        coords,
        absolute_transformations,
        pairwise_transformations,
        cross_validation_k,
        sim_GCPs,
    ):

        list_proj_RMSE = []
        list_norm_proj_RMSE = []

        for i in range(5):

            list_errors = []
            list_errors_normalized = []

            H = self.calculate_H_for_sim_GCP(
                absolute_transformations, anchors_dict, sim_GCPs[i]
            )

            for img_A_name in pairwise_transformations:

                for img_B_name in pairwise_transformations[img_A_name]:

                    matches = pairwise_transformations[img_A_name][img_B_name][1]
                    inliers = pairwise_transformations[img_A_name][img_B_name][3]
                    bins = pairwise_transformations[img_A_name][img_B_name][4]

                    if cross_validation_k == -1:

                        for i, m in enumerate(matches):

                            if inliers[i, 0] == 0:
                                continue

                            kp_A = self.images_dict[img_A_name].kp[m.trainIdx]
                            kp_B = self.images_dict[img_B_name].kp[m.queryIdx]

                            p_A = [kp_A[0], kp_A[1], 1]
                            p_B = [kp_B[0], kp_B[1], 1]

                            T_A = absolute_transformations[img_A_name]
                            T_B = absolute_transformations[img_B_name]

                            new_p_A = np.matmul(T_A, p_A)
                            if T_A.shape[0] == 3:
                                new_p_A = new_p_A / new_p_A[2]

                            new_p_B = np.matmul(T_B, p_B)
                            if T_B.shape[0] == 3:
                                new_p_B = new_p_B / new_p_B[2]

                            list_errors.append((new_p_A[0] - new_p_B[0]) ** 2)
                            list_errors.append((new_p_A[1] - new_p_B[1]) ** 2)

                            # normalized

                            norm_new_p_A = np.matmul(H, new_p_A)
                            if len(norm_new_p_A) == 3:
                                norm_new_p_A = norm_new_p_A / norm_new_p_A[2]

                            norm_new_p_B = np.matmul(H, new_p_B)
                            if len(norm_new_p_B) == 3:
                                norm_new_p_B = norm_new_p_B / norm_new_p_B[2]

                            list_errors_normalized.append(
                                cv_util.get_gps_distance(
                                    norm_new_p_A[1],
                                    norm_new_p_A[0],
                                    norm_new_p_B[1],
                                    norm_new_p_B[0],
                                )
                                ** 2
                            )

                    else:

                        if bins is None:
                            continue

                        for m in bins[cross_validation_k]:

                            kp_A = self.images_dict[img_A_name].kp[m.trainIdx]
                            kp_B = self.images_dict[img_B_name].kp[m.queryIdx]

                            p_A = [kp_A[0], kp_A[1], 1]
                            p_B = [kp_B[0], kp_B[1], 1]

                            T_A = absolute_transformations[img_A_name]
                            T_B = absolute_transformations[img_B_name]

                            new_p_A = np.matmul(T_A, p_A)
                            if T_A.shape[0] == 3:
                                new_p_A = new_p_A / new_p_A[2]

                            new_p_B = np.matmul(T_B, p_B)
                            if T_B.shape[0] == 3:
                                new_p_B = new_p_B / new_p_B[2]

                            list_errors.append((new_p_A[0] - new_p_B[0]) ** 2)
                            list_errors.append((new_p_A[1] - new_p_B[1]) ** 2)

                            norm_new_p_A = np.matmul(H, new_p_A)
                            if len(norm_new_p_A) == 3:
                                norm_new_p_A = norm_new_p_A / norm_new_p_A[2]

                            norm_new_p_B = np.matmul(H, new_p_B)
                            if len(norm_new_p_B) == 3:
                                norm_new_p_B = norm_new_p_B / norm_new_p_B[2]

                            list_errors_normalized.append(
                                cv_util.get_gps_distance(
                                    norm_new_p_A[1],
                                    norm_new_p_A[0],
                                    norm_new_p_B[1],
                                    norm_new_p_B[0],
                                )
                                ** 2
                            )

            list_proj_RMSE.append(math.sqrt(np.mean(list_errors)))
            list_norm_proj_RMSE.append(math.sqrt(np.mean(list_errors_normalized)))

        return np.mean(list_proj_RMSE), np.mean(list_norm_proj_RMSE)

    def calculate_projection_error(
        self,
        anchors_dict,
        coords,
        absolute_transformations,
        pairwise_transformations,
        cross_validation_k,
        H=None,
        sim_GCPs=None,
    ):

        if sim_GCPs is not None:
            return self.calculate_projection_error_sim_GCPs(
                anchors_dict,
                coords,
                absolute_transformations,
                pairwise_transformations,
                cross_validation_k,
                sim_GCPs,
            )

        list_errors = []
        list_errors_normalized = []

        min_x = np.min(
            [
                [
                    coords[c]["UL"][0],
                    coords[c]["UR"][0],
                    coords[c]["LL"][0],
                    coords[c]["LR"][0],
                ]
                for c in coords
            ]
        )
        max_x = np.max(
            [
                [
                    coords[c]["UL"][0],
                    coords[c]["UR"][0],
                    coords[c]["LL"][0],
                    coords[c]["LR"][0],
                ]
                for c in coords
            ]
        )

        min_y = np.min(
            [
                [
                    coords[c]["UL"][1],
                    coords[c]["UR"][1],
                    coords[c]["LL"][1],
                    coords[c]["LR"][1],
                ]
                for c in coords
            ]
        )
        max_y = np.max(
            [
                [
                    coords[c]["UL"][1],
                    coords[c]["UR"][1],
                    coords[c]["LL"][1],
                    coords[c]["LR"][1],
                ]
                for c in coords
            ]
        )

        for img_A_name in pairwise_transformations:

            for img_B_name in pairwise_transformations[img_A_name]:

                matches = pairwise_transformations[img_A_name][img_B_name][1]
                inliers = pairwise_transformations[img_A_name][img_B_name][3]
                bins = pairwise_transformations[img_A_name][img_B_name][4]

                if cross_validation_k == -1:

                    for i, m in enumerate(matches):

                        if inliers[i, 0] == 0:
                            continue

                        kp_A = self.images_dict[img_A_name].kp[m.trainIdx]
                        kp_B = self.images_dict[img_B_name].kp[m.queryIdx]

                        p_A = [kp_A[0], kp_A[1], 1]
                        p_B = [kp_B[0], kp_B[1], 1]

                        T_A = absolute_transformations[img_A_name]
                        T_B = absolute_transformations[img_B_name]

                        new_p_A = np.matmul(T_A, p_A)
                        if T_A.shape[0] == 3:
                            new_p_A = new_p_A / new_p_A[2]

                        new_p_B = np.matmul(T_B, p_B)
                        if T_B.shape[0] == 3:
                            new_p_B = new_p_B / new_p_B[2]

                        list_errors.append((new_p_A[0] - new_p_B[0]) ** 2)
                        list_errors.append((new_p_A[1] - new_p_B[1]) ** 2)

                        if H is None:
                            norm_new_p_A = new_p_A
                            norm_new_p_B = new_p_B

                            list_errors_normalized.append(
                                cv_util.get_gps_distance(
                                    norm_new_p_A[1],
                                    norm_new_p_A[0],
                                    norm_new_p_B[1],
                                    norm_new_p_B[0],
                                )
                                ** 2
                            )
                        else:
                            norm_new_p_A = np.matmul(H, new_p_A)
                            if len(norm_new_p_A) == 3:
                                norm_new_p_A = norm_new_p_A / norm_new_p_A[2]

                            norm_new_p_B = np.matmul(H, new_p_B)
                            if len(norm_new_p_B) == 3:
                                norm_new_p_B = norm_new_p_B / norm_new_p_B[2]

                            list_errors_normalized.append(
                                cv_util.get_gps_distance(
                                    norm_new_p_A[1],
                                    norm_new_p_A[0],
                                    norm_new_p_B[1],
                                    norm_new_p_B[0],
                                )
                                ** 2
                            )

                else:

                    if bins is None:
                        continue

                    for m in bins[cross_validation_k]:

                        kp_A = self.images_dict[img_A_name].kp[m.trainIdx]
                        kp_B = self.images_dict[img_B_name].kp[m.queryIdx]

                        p_A = [kp_A[0], kp_A[1], 1]
                        p_B = [kp_B[0], kp_B[1], 1]

                        T_A = absolute_transformations[img_A_name]
                        T_B = absolute_transformations[img_B_name]

                        new_p_A = np.matmul(T_A, p_A)
                        if T_A.shape[0] == 3:
                            new_p_A = new_p_A / new_p_A[2]

                        new_p_B = np.matmul(T_B, p_B)
                        if T_B.shape[0] == 3:
                            new_p_B = new_p_B / new_p_B[2]

                        list_errors.append((new_p_A[0] - new_p_B[0]) ** 2)
                        list_errors.append((new_p_A[1] - new_p_B[1]) ** 2)

                        if H is None:
                            norm_new_p_A = new_p_A
                            norm_new_p_B = new_p_B

                            list_errors_normalized.append(
                                cv_util.get_gps_distance(
                                    norm_new_p_A[1],
                                    norm_new_p_A[0],
                                    norm_new_p_B[1],
                                    norm_new_p_B[0],
                                )
                                ** 2
                            )
                        else:
                            norm_new_p_A = np.matmul(H, new_p_A)
                            if len(norm_new_p_A) == 3:
                                norm_new_p_A = norm_new_p_A / norm_new_p_A[2]

                            norm_new_p_B = np.matmul(H, new_p_B)
                            if len(norm_new_p_B) == 3:
                                norm_new_p_B = norm_new_p_B / norm_new_p_B[2]

                            list_errors_normalized.append(
                                cv_util.get_gps_distance(
                                    norm_new_p_A[1],
                                    norm_new_p_A[0],
                                    norm_new_p_B[1],
                                    norm_new_p_B[0],
                                )
                                ** 2
                            )

        return math.sqrt(np.mean(list_errors)), math.sqrt(
            np.mean(list_errors_normalized)
        )

    def calculate_projection_error_Gantry(self, coords, pairwise_transformations, w, h):

        list_errors = []

        for image_1_name in pairwise_transformations:
            for image_2_name in pairwise_transformations[image_1_name]:

                GPS_width_1 = (
                    coords[image_1_name]["UR"][0] - coords[image_1_name]["UL"][0]
                )
                GPS_height_1 = (
                    coords[image_1_name]["UL"][1] - coords[image_1_name]["LL"][1]
                )

                GPS_width_2 = (
                    coords[image_2_name]["UR"][0] - coords[image_2_name]["UL"][0]
                )
                GPS_height_2 = (
                    coords[image_2_name]["UL"][1] - coords[image_2_name]["LL"][1]
                )

                matches = pairwise_transformations[image_1_name][image_2_name][1]
                inliers = pairwise_transformations[image_1_name][image_2_name][3]

                for i, m in enumerate(matches):

                    if inliers[i, 0] == 0:
                        continue

                    kp_1 = self.images_dict[image_1_name].kp[m.trainIdx]
                    kp_2 = self.images_dict[image_2_name].kp[m.queryIdx]

                    p_1 = (
                        coords[image_1_name]["UL"][0] + kp_1[0] / w * GPS_width_1,
                        coords[image_1_name]["UL"][1] - kp_1[1] / h * GPS_height_1,
                    )
                    p_2 = (
                        coords[image_2_name]["UL"][0] + kp_2[0] / w * GPS_width_2,
                        coords[image_2_name]["UL"][1] - kp_2[1] / h * GPS_height_2,
                    )

                    list_errors.append((p_1[0] - p_2[0]) ** 2)
                    list_errors.append((p_1[1] - p_2[1]) ** 2)

        return math.sqrt(np.mean(list_errors))

    def report_GCP_error_for_Gantry(
        self, new_coords, old_coords, anchors_dict, x, y, is_kp=True
    ):

        GPS_Width = (
            old_coords[self.reference_image.name]["UR"]["lon"]
            - old_coords[self.reference_image.name]["UL"]["lon"]
        )
        GPS_Height = (
            old_coords[self.reference_image.name]["UL"]["lat"]
            - old_coords[self.reference_image.name]["LL"]["lat"]
        )
        GPS_IMG_Ratio = (GPS_Width / x, GPS_Height / y)

        c = 1
        if is_kp:
            c = -1

        GPS_diffs = []

        for img in anchors_dict:
            img_name = img["img_name"]

            if img_name not in self.images_dict:
                continue

            A_w = img["img_x"] * self.settings.scale
            A_h = img["img_y"] * self.settings.scale

            new_lon = new_coords[img_name]["UL"][0] + A_w * GPS_IMG_Ratio[0]
            new_lat = c * new_coords[img_name]["UL"][1] - A_h * GPS_IMG_Ratio[1]

            old_lon = img["gps_lon"]
            old_lat = img["gps_lat"]

            GPS_diffs.append(
                cv_util.get_gps_distance(old_lat, old_lon, new_lat, new_lon) ** 2
            )

        print(
            ":: Evaluation\n\tNumber Lids: {0}\n\tGCP-based RMSE: {1}\tGCP-based STDE: {2}".format(
                len(GPS_diffs), math.sqrt(np.mean(GPS_diffs)), np.std(GPS_diffs)
            )
        )

    def GCP_error_random_base(
        self, transformation_list, evaluation_list, absolute_homography_dict
    ):

        GPS_diffs = []

        EXP_centers = []
        GT_centers = []

        for l in transformation_list:
            H_t = absolute_homography_dict[l["img_name"]]
            p = np.array([l["img_x"] * self.settings.scale, l["img_y"] * self.settings.scale, 1])
            p_new = np.matmul(H_t, p)
            p_new /= p_new[2]

            EXP_centers.append(p_new[:2])
            GT_centers.append([l["true_lon"], l["true_lat"]])

        if self.settings.transformation == cv_util.Transformation.affine:
            H = cv_util.get_Similarity_Affine(
                np.array(EXP_centers), np.array(GT_centers)
            )
            H_inv = cv_util.get_Similarity_Affine(
                np.array(GT_centers), np.array(EXP_centers)
            )

        if self.settings.transformation == cv_util.Transformation.similarity:
            H = cv_util.get_Similarity_Affine(
                np.array(EXP_centers), np.array(GT_centers)
            )
            H_inv = cv_util.get_Similarity_Affine(
                np.array(GT_centers), np.array(EXP_centers)
            )

        if self.settings.transformation == cv_util.Transformation.homography:
            H, _ = cv2.findHomography(
                np.array(EXP_centers),
                np.array(GT_centers),
                maxIters=1000,
                confidence=0.8,
                method=0,
            )
            H_inv, _ = cv2.findHomography(
                np.array(GT_centers),
                np.array(EXP_centers),
                maxIters=1000,
                confidence=0.8,
                method=0,
            )

        if H is None:
            return None

        for l in evaluation_list:

            H_t = absolute_homography_dict[l["img_name"]]
            p = np.array([l["img_x"] * self.settings.scale, l["img_y"] * self.settings.scale, 1])
            p_new = np.matmul(H_t, p)
            p_new /= p_new[2]

            p_final = np.matmul(H, p_new)
            if H.shape[0] == 3:
                p_final = p_final / p_final[2]

            GT = [l["true_lon"], l["true_lat"]]

            GPS_diffs.append(
                cv_util.get_gps_distance(GT[1], GT[0], p_final[1], p_final[0]) ** 2
            )

        return math.sqrt(np.mean(GPS_diffs)), H, H_inv

    def isValidSetOfPoints(self, transformation_list):

        points = []

        for l in transformation_list:
            points.append([l["true_lat"], l["true_lon"]])

        counter_x = 0
        counter_y = 0
        counter_euclidean = 0

        for i, p1 in enumerate(points):
            for j, p2 in enumerate(points):
                if i >= j:
                    continue

                diff_x = abs(p1[0] - p2[0])
                diff_y = abs(p1[1] - p2[1])
                diff = math.sqrt(diff_x**2 + diff_y**2)

                if diff_x < 5e-5:
                    counter_x += 1
                if diff_y < 5e-5:
                    counter_y += 1
                if diff < 5e-5:
                    counter_euclidean += 1

        if counter_x > 2 or counter_y > 2 or counter_euclidean >= 2:
            return False

        return True

    def get_GCP_coords_all(self, absolute_transformations, list_H_inv, anchors):

        all_GCP_coords = {}

        for i, a in enumerate(anchors):

            all_GCP_coords[i] = {}

            p_act = [a["img_x"] * self.settings.scale, a["img_y"] * self.settings.scale, 1]
            new_p_act = np.matmul(absolute_transformations[a["img_name"]], p_act)
            new_p_act = new_p_act / new_p_act[2]

            all_GCP_coords[i]["act"] = new_p_act
            all_GCP_coords[i]["gps"] = []

            for H in list_H_inv:

                if H.shape[0] == 2:
                    H_tmp = np.eye(3)
                    H_tmp[:2, :] = H
                    H = H_tmp

                p_gps = [a["true_lon"], a["true_lat"], 1]
                new_p_gps = np.matmul(H, p_gps)
                new_p_gps = new_p_gps / new_p_gps[2]
                all_GCP_coords[i]["gps"].append(new_p_gps)

        return all_GCP_coords

    def report_GCP_error_for_Drone_Sim_GCPs(
        self, absolute_homography_dict, anchors_dict, sim_GCPs
    ):

        list_RMSE = []
        list_H = []
        list_H_inv = []
        list_tr_gcps = []

        for i in range(5):

            transformation_list = []
            evaluation_list = []

            for s in sim_GCPs[i]:
                transformation_list.append(s)

            list_names = [n["img_name"] for n in transformation_list]

            for a in anchors_dict:
                if a["img_name"] not in list_names:
                    evaluation_list.append(a)

            results = self.GCP_error_random_base(
                transformation_list, evaluation_list, absolute_homography_dict
            )

            if results is None:
                continue

            list_RMSE.append(results[0])
            list_tr_gcps.append(transformation_list)
            list_H.append(results[1])
            list_H_inv.append(results[2])

        if len(list_RMSE) == 5:
            avg = np.mean(list_RMSE)

            print(":: Evaluation")
            print("\tList RMSEs:")

            for i, r in enumerate(list_RMSE):
                print("\t\tRMSE: {0} \n\t\tTransformation GCPs:".format(r))
                for g in list_tr_gcps[i]:
                    print("\t\t\t{0}".format((g["true_lon"], g["true_lat"])))

            print("\tAverage GCP-based RMSE: {0}".format(avg))

            i = list_RMSE.index(min(list_RMSE))

            all_GCPs = self.get_GCP_coords_all(
                absolute_homography_dict, list_H_inv, anchors_dict
            )

            return list_H[i], list_H_inv[i], avg, list_tr_gcps, all_GCPs

        elif len(list_RMSE) > 0:
            avg = np.mean(list_RMSE)
            print(
                ":: Evaluation\n\tList RMSEs: {0}\n\tAverage GCP-based RMSE: {1}".format(
                    list_RMSE, avg
                )
            )
            print(
                ":: ** Number of random samplings for evaluation is less than five ({0})".format(
                    len(list_RMSE)
                )
            )

            i = list_RMSE.index(min(list_RMSE))

            all_GCPs = self.get_GCP_coords_all(
                absolute_homography_dict, list_H_inv, anchors_dict
            )

            return list_H[i], list_H_inv[i], avg, list_tr_gcps, all_GCPs

        else:

            print(
                ":: ** Number of random samplings for evaluation is zero ({0})".format(
                    len(list_RMSE)
                )
            )
            return None, None, None, None, None

    def report_GCP_error_all_GCPs(self, absolute_homography_dict, anchors_dict):

        tmp = [a for a in anchors_dict if a["img_name"] in absolute_homography_dict]

        if len(tmp) < 5:
            print("Number of images with GCP is less than five.")
            return None, None, None, None, None

        RMSE, H, H_inv = self.GCP_error_random_base(tmp, tmp, absolute_homography_dict)

        print(":: GCP RMSE using all GCPS: {0}".format(RMSE))

        return H, H_inv, RMSE, None, None

    def report_GCP_error_for_Drone(
        self, absolute_homography_dict, anchors_dict, sim_GCPs=None
    ):
        if anchors_dict is None:
            print(":: Skipping GCP error calculation. No GCPs provided.")
            return None, None, None, None, None

        if self.settings.AllGCPRMSE:
            return self.report_GCP_error_all_GCPs(
                absolute_homography_dict, anchors_dict
            )

        if sim_GCPs is not None:
            return self.report_GCP_error_for_Drone_Sim_GCPs(
                absolute_homography_dict, anchors_dict, sim_GCPs
            )

        tmp = [a for a in anchors_dict if a["img_name"] in absolute_homography_dict]

        if len(tmp) < 5:
            print("Number of images with GCP is less than five.")
            return None, None, None, None, None

        anchors_dict = tmp

        groupped_anchors = {}

        for a in anchors_dict:
            key = (a["true_lon"], a["true_lat"])

            if key not in groupped_anchors:
                groupped_anchors[key] = [
                    {
                        "img_name": a["img_name"],
                        "img_x": a["img_x"],
                        "img_y": a["img_y"],
                        "true_lon": a["true_lon"],
                        "true_lat": a["true_lat"],
                    }
                ]
            else:
                groupped_anchors[key].append(
                    {
                        "img_name": a["img_name"],
                        "img_x": a["img_x"],
                        "img_y": a["img_y"],
                        "true_lon": a["true_lon"],
                        "true_lat": a["true_lat"],
                    }
                )

        keys = [i for i in groupped_anchors]

        if len(keys) < 4:
            print("Number of GCPs is less than five.")
            return None, None, None, None, None

        failed_attempts = 0
        list_RMSE = []
        list_H = []
        list_H_inv = []
        list_tr_gcps = []

        while len(list_RMSE) < 5 and failed_attempts < 100:

            random.shuffle(keys)

            transformation_list = []
            evaluation_list = []

            for k in keys[:4]:
                random.shuffle(groupped_anchors[k])
                transformation_list.append(groupped_anchors[k][0])
                evaluation_list += groupped_anchors[k][1:]

            for k in keys[4:]:
                evaluation_list += groupped_anchors[k]

            if not self.isValidSetOfPoints(transformation_list):
                failed_attempts += 1
                continue

            results = self.GCP_error_random_base(
                transformation_list, evaluation_list, absolute_homography_dict
            )

            if results is None:
                failed_attempts += 1
                continue

            list_RMSE.append(results[0])
            list_tr_gcps.append(transformation_list)
            list_H.append(results[1])
            list_H_inv.append(results[2])

        if len(list_RMSE) == 5:
            avg = np.mean(list_RMSE)
            std = np.std(list_RMSE)

            print(":: Evaluation")
            print("\tList RMSEs:")

            for i, r in enumerate(list_RMSE):
                print("\t\tRMSE: {0} \n\t\tTransformation GCPs:".format(r))
                for g in list_tr_gcps[i]:
                    print("\t\t\t{0}".format((g["true_lon"], g["true_lat"])))

            print("\tAverage GCP-based RMSE: {0}".format(avg))
            print("\tSTD GCP-based RMSE: {0}".format(std))

            i = list_RMSE.index(min(list_RMSE))

            all_GCPs = self.get_GCP_coords_all(
                absolute_homography_dict, list_H_inv, anchors_dict
            )

            return list_H[i], list_H_inv[i], avg, list_tr_gcps, all_GCPs

        elif len(list_RMSE) > 0:
            avg = np.mean(list_RMSE)
            std = np.std(list_RMSE)

            print(
                ":: Evaluation\n\tList RMSEs: {0}\n\tAverage GCP-based RMSE: {1}".format(
                    list_RMSE, avg
                )
            )
            print("\tSTD GCP-based RMSE: {0}".format(std))
            print(
                ":: ** Number of random samplings for evaluation is less than five ({0})".format(
                    len(list_RMSE)
                )
            )

            i = list_RMSE.index(min(list_RMSE))

            all_GCPs = self.get_GCP_coords_all(
                absolute_homography_dict, list_H_inv, anchors_dict
            )

            return list_H[i], list_H_inv[i], avg, list_tr_gcps, all_GCPs

        else:

            print(
                ":: ** Number of random samplings for evaluation is zero ({0})".format(
                    len(list_RMSE)
                )
            )
            return None, None, None, None, None

    # -----------------------------------------------------------
    # ------------ New Geo-correction Methods -------------------
    # -----------------------------------------------------------

    def geo_correct_MGRAPH(self, anchors_dict):

        if self.images[0].lat is None:
            print(
                ":: The images in this dataset does not have geo-referencing information. MGRAPH requires GPS information of each images."
            )

        tr = self.generate_neighbor_transformations()

        self.reference_image.load_img()

        x = self.reference_image.img.shape[1]
        y = self.reference_image.img.shape[0]

        mg = utils_MGRAPH.MGRAPH(
            self.images,
            tr,
            self.image_name_to_index_dict,
            self.get_positions(),
            self.reference_image,
            self.settings.transformation,
            self.settings.use_ceres_MGRAPH,
            self.settings.max_no_inliers,
        )

        absolute_transformations = mg.optimize()

        image_coords_dict = {}

        w = self.reference_image.img.shape[1]
        h = self.reference_image.img.shape[0]

        for img in self.images:

            image_coords_dict[img.name] = {}

            if self.settings.normalize_key_points:
                corners = {
                    "UL": [-0.5, -0.5, 1],
                    "UR": [0.5, -0.5, 1],
                    "LR": [0.5, 0.5, 1],
                    "LL": [-0.5, 0.5, 1],
                }
            else:
                corners = {
                    "UL": [0, 0, 1],
                    "UR": [x, 0, 1],
                    "LR": [x, y, 1],
                    "LL": [0, y, 1],
                }

            H = absolute_transformations[img.name]

            for key in corners:

                new_corner = np.matmul(H, corners[key])
                new_corner = new_corner / new_corner[2]

                if self.settings.normalize_key_points:

                    image_coords_dict[img.name][key] = [
                        int(w * (new_corner[0]) + w / 2),
                        int(h * (new_corner[1]) + h / 2),
                    ]

                else:

                    image_coords_dict[img.name][key] = [
                        int(new_corner[0]),
                        int(new_corner[1]),
                    ]

        H, H_inv, GCP_RMSE, Sim_GCPs, all_GCPs = self.report_GCP_error_for_Drone(
            absolute_transformations, anchors_dict
        )

        RMSE, RMSE_Normalized = self.calculate_projection_error(
            anchors_dict, image_coords_dict, absolute_transformations, tr, -1, H=H
        )

        print(":: Projection RMSE: {0}".format(RMSE))
        print(":: Normalized Projection RMSE: {0}".format(RMSE_Normalized))

        print(
            ">>> MGRAPH global optimization and coordinate calculation finished successfully."
        )

        return (
            image_coords_dict,
            H,
            H_inv,
            absolute_transformations,
            [GCP_RMSE, RMSE, RMSE_Normalized],
            Sim_GCPs,
            all_GCPs,
        )

    def geo_correct_MegaStitchSimilarity(self, anchors_dict):

        tr = self.generate_neighbor_transformations()

        self.reference_image.load_img()

        x = self.reference_image.img.shape[1]
        y = self.reference_image.img.shape[0]

        # --------------------------------------------
        # cross validation
        # --------------------------------------------

        if self.settings.do_cross_validation:

            Proj_RMSE_list = []
            Proj_RMSE_Norm_list = []
            GCP_RMSE_list = []
            Time_list = []

            for i in range(self.settings.number_bins):

                print("========== Fold {0} ===========".format(i))

                mega_stitch = ProjectionOptimization.ReprojectionMinimization(
                    self.images,
                    tr,
                    [1, 2],
                    x,
                    y,
                    self.reference_image.name,
                    self.settings.max_no_inliers,
                    i,
                )

                (
                    new_coords,
                    absolute_transformations,
                    running_time,
                ) = mega_stitch.MegaStitchSimilarityAffine(True)

                H_GCP, _, GCP_RMSE, _, _ = self.report_GCP_error_for_Drone(
                    absolute_transformations, anchors_dict
                )

                RMSE, RMSE_Normalized = self.calculate_projection_error(
                    anchors_dict, new_coords, absolute_transformations, tr, i, H_GCP
                )

                print(":: Projection RMSE: {0}".format(RMSE))
                print(":: Normalized Projection RMSE: {0}".format(RMSE_Normalized))

                Proj_RMSE_list.append(RMSE)
                Proj_RMSE_Norm_list.append(RMSE_Normalized)

                GCP_RMSE_list.append(GCP_RMSE)
                Time_list.append(running_time)

            print("-----------------------------------------------------------")
            print(
                ":: Mean and STD of GCP RMSE: {0}, {1}".format(
                    np.mean(GCP_RMSE_list), np.std(GCP_RMSE_list)
                )
            )
            print(
                ":: Mean and STD of Projection RMSE: {0}, {1}".format(
                    np.mean(Proj_RMSE_list), np.std(Proj_RMSE_list)
                )
            )
            print(
                ":: Mean and STD of Normalized Projection RMSE: {0}, {1}".format(
                    np.mean(Proj_RMSE_Norm_list), np.std(Proj_RMSE_Norm_list)
                )
            )
            print(":: Mean of Time: {0}".format(np.mean(Time_list)))
            print("-----------------------------------------------------------")

        # --------------------------------------------
        # Single final opt. no cross validation
        # --------------------------------------------

        print("================== NO CROSS VALIDATION ==================")

        self.settings.max_no_inliers = 20

        mega_stitch = ProjectionOptimization.ReprojectionMinimization(
            self.images,
            tr,
            [1, 2],
            x,
            y,
            self.reference_image.name,
            self.settings.max_no_inliers,
            -1,
        )

        (
            new_coords,
            absolute_transformations,
            running_time,
        ) = mega_stitch.MegaStitchSimilarityAffine(True)

        H, H_inv, GCP_RMSE, Sim_GCPs, all_GCPs = self.report_GCP_error_for_Drone(
            absolute_transformations, anchors_dict
        )

        RMSE, RMSE_Normalized = self.calculate_projection_error(
            anchors_dict, new_coords, absolute_transformations, tr, -1, H=H
        )

        print(":: Projection RMSE: {0}".format(RMSE))
        print(":: Normalized Projection RMSE: {0}".format(RMSE_Normalized))

        # ----------------------

        image_coords_dict = {}

        for img in self.images:

            image_coords_dict[img.name] = {}

            for key in new_coords[img.name]:

                image_coords_dict[img.name][key] = [
                    int(new_coords[img.name][key][0]),
                    int(new_coords[img.name][key][1]),
                ]

        print(">>> MEGASTITCH-Similarity finished successfully.")

        if self.settings.do_cross_validation:
            return (
                image_coords_dict,
                H,
                H_inv,
                absolute_transformations,
                [
                    GCP_RMSE,
                    RMSE,
                    RMSE_Normalized,
                    running_time,
                    np.mean(GCP_RMSE_list),
                    np.std(GCP_RMSE_list),
                    np.mean(Proj_RMSE_list),
                    np.std(Proj_RMSE_list),
                    np.mean(Proj_RMSE_Norm_list),
                    np.std(Proj_RMSE_Norm_list),
                    np.mean(Time_list),
                ],
                Sim_GCPs,
                all_GCPs,
            )
        else:
            return (
                image_coords_dict,
                H,
                H_inv,
                absolute_transformations,
                [GCP_RMSE, RMSE, RMSE_Normalized, running_time],
                Sim_GCPs,
                all_GCPs,
            )

    def geo_correct_MegaStitchAffine(self, anchors_dict, sim_GCPs):

        tr = self.generate_neighbor_transformations()

        self.reference_image.load_img()

        x = self.reference_image.img.shape[1]
        y = self.reference_image.img.shape[0]

        # --------------------------------------------
        # cross validation
        # --------------------------------------------

        if self.settings.do_cross_validation:

            Proj_RMSE_list = []
            Proj_RMSE_Norm_list = []
            GCP_RMSE_list = []
            Time_list = []

            for i in range(self.settings.number_bins):

                print("========== Fold {0} ===========".format(i))

                mega_stitch = ProjectionOptimization.ReprojectionMinimization(
                    self.images,
                    tr,
                    [1, 2],
                    x,
                    y,
                    self.reference_image.name,
                    self.settings.max_no_inliers,
                    i,
                )

                (
                    new_coords,
                    absolute_transformations,
                    running_time,
                ) = mega_stitch.MegaStitchSimilarityAffine(False)

                H_GCP, _, GCP_RMSE, _, _ = self.report_GCP_error_for_Drone(
                    absolute_transformations, anchors_dict, sim_GCPs
                )

                RMSE, RMSE_Normalized = self.calculate_projection_error(
                    anchors_dict,
                    new_coords,
                    absolute_transformations,
                    tr,
                    i,
                    sim_GCPs=sim_GCPs,
                )

                print(":: Projection RMSE: {0}".format(RMSE))
                print(":: Normalized Projection RMSE: {0}".format(RMSE_Normalized))

                Proj_RMSE_list.append(RMSE)
                Proj_RMSE_Norm_list.append(RMSE_Normalized)

                GCP_RMSE_list.append(GCP_RMSE)
                Time_list.append(running_time)

            print("-----------------------------------------------------------")
            print(
                ":: Mean and STD of GCP RMSE: {0}, {1}".format(
                    np.mean(GCP_RMSE_list), np.std(GCP_RMSE_list)
                )
            )
            print(
                ":: Mean and STD of Projection RMSE: {0}, {1}".format(
                    np.mean(Proj_RMSE_list), np.std(Proj_RMSE_list)
                )
            )
            print(
                ":: Mean and STD of Normalized Projection RMSE: {0}, {1}".format(
                    np.mean(Proj_RMSE_Norm_list), np.std(Proj_RMSE_Norm_list)
                )
            )
            print(":: Mean of Time: {0}".format(np.mean(Time_list)))
            print("-----------------------------------------------------------")

        # --------------------------------------------
        # Single final opt. no cross validation
        # --------------------------------------------

        print("================== NO CROSS VALIDATION ==================")

        self.settings.max_no_inliers = 20

        mega_stitch = ProjectionOptimization.ReprojectionMinimization(
            self.images,
            tr,
            [1, 2],
            x,
            y,
            self.reference_image.name,
            self.settings.max_no_inliers,
            -1,
        )

        (
            new_coords,
            absolute_transformations,
            running_time,
        ) = mega_stitch.MegaStitchSimilarityAffine(False)

        H, H_inv, GCP_RMSE, _, all_GCPs = self.report_GCP_error_for_Drone(
            absolute_transformations, anchors_dict, sim_GCPs
        )

        RMSE, RMSE_Normalized = self.calculate_projection_error(
            anchors_dict,
            new_coords,
            absolute_transformations,
            tr,
            -1,
            sim_GCPs=sim_GCPs,
        )

        print(":: Projection RMSE: {0}".format(RMSE))
        print(":: Normalized Projection RMSE: {0}".format(RMSE_Normalized))

        # ----------------------

        image_coords_dict = {}

        for img in self.images:

            image_coords_dict[img.name] = {}

            for key in new_coords[img.name]:

                image_coords_dict[img.name][key] = [
                    int(new_coords[img.name][key][0]),
                    int(new_coords[img.name][key][1]),
                ]

        print(">>> MEGASTITCH-Affine finished successfully.")

        if self.settings.do_cross_validation:
            return (
                image_coords_dict,
                H,
                H_inv,
                absolute_transformations,
                [
                    GCP_RMSE,
                    RMSE,
                    RMSE_Normalized,
                    running_time,
                    np.mean(GCP_RMSE_list),
                    np.std(GCP_RMSE_list),
                    np.mean(Proj_RMSE_list),
                    np.std(Proj_RMSE_list),
                    np.mean(Proj_RMSE_Norm_list),
                    np.std(Proj_RMSE_Norm_list),
                    np.mean(Time_list),
                ],
                all_GCPs,
            )
        else:
            return (
                image_coords_dict,
                H,
                H_inv,
                absolute_transformations,
                [GCP_RMSE, RMSE, RMSE_Normalized, running_time],
                all_GCPs,
            )

    def geo_correct_BundleAdjustment_Homography(self, anchors_dict, sim_GCPs):

        tr = self.generate_neighbor_transformations()

        self.reference_image.load_img()

        x = self.reference_image.img.shape[1]
        y = self.reference_image.img.shape[0]

        # --------------------------------------------
        # cross validation
        # --------------------------------------------

        if self.settings.do_cross_validation:

            Proj_RMSE_list = []
            Proj_RMSE_Norm_list = []
            GCP_RMSE_list = []
            Time_list = []

            for i in range(self.settings.number_bins):

                print("========== Fold {0} ===========".format(i))

                mega_stitch = ProjectionOptimization.ReprojectionMinimization(
                    self.images,
                    tr,
                    [1, 2],
                    x,
                    y,
                    self.reference_image.name,
                    self.settings.max_no_inliers,
                    i,
                )

                MST = ProjectionOptimization.Graph(
                    self.images,
                    tr,
                    self.image_name_to_index_dict,
                    self.reference_image.name,
                )
                initializations = MST.get_absolute_homographies()

                (
                    new_coords,
                    absolute_transformations,
                    running_time,
                ) = mega_stitch.BundleAdjustmentHomography(initializations)

                H_GCP, _, GCP_RMSE, _, _ = self.report_GCP_error_for_Drone(
                    absolute_transformations, anchors_dict, sim_GCPs
                )

                RMSE, RMSE_Normalized = self.calculate_projection_error(
                    anchors_dict,
                    new_coords,
                    absolute_transformations,
                    tr,
                    i,
                    sim_GCPs=sim_GCPs,
                )

                print(":: Projection RMSE: {0}".format(RMSE))
                print(":: Normalized Projection RMSE: {0}".format(RMSE_Normalized))

                Proj_RMSE_list.append(RMSE)
                Proj_RMSE_Norm_list.append(RMSE_Normalized)

                GCP_RMSE_list.append(GCP_RMSE)
                Time_list.append(running_time)

            print("-----------------------------------------------------------")
            print(
                ":: Mean and STD of GCP RMSE: {0}, {1}".format(
                    np.mean(GCP_RMSE_list), np.std(GCP_RMSE_list)
                )
            )
            print(
                ":: Mean and STD of Projection RMSE: {0}, {1}".format(
                    np.mean(Proj_RMSE_list), np.std(Proj_RMSE_list)
                )
            )
            print(
                ":: Mean and STD of Normalized Projection RMSE: {0}, {1}".format(
                    np.mean(Proj_RMSE_Norm_list), np.std(Proj_RMSE_Norm_list)
                )
            )
            print(":: Mean of Time: {0}".format(np.mean(Time_list)))
            print("-----------------------------------------------------------")

        # --------------------------------------------
        # Single final opt. no cross validation
        # --------------------------------------------

        print("================== NO CROSS VALIDATION ==================")

        self.settings.max_no_inliers = 20

        mega_stitch = ProjectionOptimization.ReprojectionMinimization(
            self.images,
            tr,
            [1, 2],
            x,
            y,
            self.reference_image.name,
            self.settings.max_no_inliers,
            -1,
        )

        MST = ProjectionOptimization.Graph(
            self.images, tr, self.image_name_to_index_dict, self.reference_image.name
        )
        initializations = MST.get_absolute_homographies()

        # # -------------- using MST only
        # new_coords = self.get_coords_from_absolute_transformations(initializations,x,y)
        # absolute_transformations = initializations
        # running_time = None
        # # --------------

        (
            new_coords,
            absolute_transformations,
            running_time,
        ) = mega_stitch.BundleAdjustmentHomography(initializations)

        H, H_inv, GCP_RMSE, _, all_GCPs = self.report_GCP_error_for_Drone(
            absolute_transformations, anchors_dict, sim_GCPs
        )

        RMSE, RMSE_Normalized = self.calculate_projection_error(
            anchors_dict,
            new_coords,
            absolute_transformations,
            tr,
            -1,
            H=H,
            sim_GCPs=sim_GCPs,
        )

        print(":: Projection RMSE: {0}".format(RMSE))
        print(":: Normalized Projection RMSE: {0}".format(RMSE_Normalized))

        # ----------------------

        image_coords_dict = {}

        for img in self.images:

            image_coords_dict[img.name] = {}

            for key in new_coords[img.name]:

                image_coords_dict[img.name][key] = [
                    int(new_coords[img.name][key][0]),
                    int(new_coords[img.name][key][1]),
                ]

        print(">>> BundleAdjustment-Homography finished successfully.")

        if self.settings.do_cross_validation:
            return (
                image_coords_dict,
                H,
                H_inv,
                absolute_transformations,
                [
                    GCP_RMSE,
                    RMSE,
                    RMSE_Normalized,
                    running_time,
                    np.mean(GCP_RMSE_list),
                    np.std(GCP_RMSE_list),
                    np.mean(Proj_RMSE_list),
                    np.std(Proj_RMSE_list),
                    np.mean(Proj_RMSE_Norm_list),
                    np.std(Proj_RMSE_Norm_list),
                    np.mean(Time_list),
                ],
                all_GCPs,
            )
        else:
            return (
                image_coords_dict,
                H,
                H_inv,
                absolute_transformations,
                [GCP_RMSE, RMSE, RMSE_Normalized, running_time],
                all_GCPs,
            )

    def geo_correct_MegaStitch_Affine_Bundle_Adjustment_Homography(
        self, anchors_dict, sim_GCPs, use_old_inliers=False
    ):

        self.reference_image.load_img()

        x = self.reference_image.img.shape[1]
        y = self.reference_image.img.shape[0]

        self.settings.transformation = cv_util.Transformation.affine
        tr = self.generate_neighbor_transformations()

        if not use_old_inliers:
            self.settings.transformation = cv_util.Transformation.homography
            tr_h = self.generate_neighbor_transformations()
            self.settings.transformation = cv_util.Transformation.affine
        else:
            tr_h = tr

        # --------------------------------------------
        # cross validation
        # --------------------------------------------

        if self.settings.do_cross_validation:

            Proj_RMSE_list = []
            Proj_RMSE_Norm_list = []
            GCP_RMSE_list = []
            Time_list = []

            for i in range(self.settings.number_bins):

                print("========== Fold {0} ===========".format(i))

                self.settings.transformation = cv_util.Transformation.affine
                mega_stitch_init = ProjectionOptimization.ReprojectionMinimization(
                    self.images,
                    tr,
                    [1, 2],
                    x,
                    y,
                    self.reference_image.name,
                    self.settings.max_no_inliers,
                    i,
                )
                (
                    _,
                    initializations,
                    running_time_init,
                ) = mega_stitch_init.MegaStitchSimilarityAffine(False)

                self.settings.transformation = cv_util.Transformation.homography
                mega_stitch = ProjectionOptimization.ReprojectionMinimization(
                    self.images,
                    tr_h,
                    [1, 2],
                    x,
                    y,
                    self.reference_image.name,
                    self.settings.max_no_inliers,
                    i,
                )
                (
                    new_coords,
                    absolute_transformations,
                    running_time,
                ) = mega_stitch.BundleAdjustmentHomography(initializations)

                H_GCP, _, GCP_RMSE, _, _ = self.report_GCP_error_for_Drone(
                    absolute_transformations, anchors_dict, sim_GCPs
                )

                RMSE, RMSE_Normalized = self.calculate_projection_error(
                    anchors_dict,
                    new_coords,
                    absolute_transformations,
                    tr_h,
                    i,
                    sim_GCPs=sim_GCPs,
                )
                print(":: Projection RMSE: {0}".format(RMSE))
                print(":: Normalized Projection RMSE: {0}".format(RMSE_Normalized))

                Proj_RMSE_list.append(RMSE)
                Proj_RMSE_Norm_list.append(RMSE_Normalized)

                GCP_RMSE_list.append(GCP_RMSE)
                Time_list.append(running_time + running_time_init)

            print("-----------------------------------------------------------")
            print(
                ":: Mean and STD of GCP RMSE: {0}, {1}".format(
                    np.mean(GCP_RMSE_list), np.std(GCP_RMSE_list)
                )
            )
            print(
                ":: Mean and STD of Projection RMSE: {0}, {1}".format(
                    np.mean(Proj_RMSE_list), np.std(Proj_RMSE_list)
                )
            )
            print(
                ":: Mean and STD of Normalized Projection RMSE: {0}, {1}".format(
                    np.mean(Proj_RMSE_Norm_list), np.std(Proj_RMSE_Norm_list)
                )
            )
            print(":: Mean of Time: {0}".format(np.mean(Time_list)))
            print("-----------------------------------------------------------")

        # --------------------------------------------
        # Single final opt. no cross validation
        # --------------------------------------------

        print("================== NO CROSS VALIDATION ==================")

        self.settings.max_no_inliers = 20

        self.settings.transformation = cv_util.Transformation.affine
        mega_stitch_init = ProjectionOptimization.ReprojectionMinimization(
            self.images,
            tr,
            [1, 2],
            x,
            y,
            self.reference_image.name,
            self.settings.max_no_inliers,
            -1,
        )
        (
            _,
            initializations,
            running_time_init,
        ) = mega_stitch_init.MegaStitchSimilarityAffine(False)

        self.settings.transformation = cv_util.Transformation.homography
        mega_stitch = ProjectionOptimization.ReprojectionMinimization(
            self.images,
            tr_h,
            [1, 2],
            x,
            y,
            self.reference_image.name,
            self.settings.max_no_inliers,
            -1,
        )
        (
            new_coords,
            absolute_transformations,
            running_time,
        ) = mega_stitch.BundleAdjustmentHomography(initializations)

        H, H_inv, GCP_RMSE, _, all_GCPs = self.report_GCP_error_for_Drone(
            absolute_transformations, anchors_dict, sim_GCPs
        )

        RMSE, RMSE_Normalized = self.calculate_projection_error(
            anchors_dict,
            new_coords,
            absolute_transformations,
            tr_h,
            -1,
            H=H,
            sim_GCPs=sim_GCPs,
        )
        print(":: Projection RMSE: {0}".format(RMSE))
        print(":: Normalized Projection RMSE: {0}".format(RMSE_Normalized))

        # ----------------------

        image_coords_dict = {}

        for img in self.images:

            image_coords_dict[img.name] = {}

            for key in new_coords[img.name]:

                image_coords_dict[img.name][key] = [
                    int(new_coords[img.name][key][0]),
                    int(new_coords[img.name][key][1]),
                ]

        print(">>> BundleAdjustment-Homography finished successfully.")

        if self.settings.do_cross_validation:
            return (
                image_coords_dict,
                H,
                H_inv,
                absolute_transformations,
                [
                    GCP_RMSE,
                    RMSE,
                    RMSE_Normalized,
                    running_time,
                    np.mean(GCP_RMSE_list),
                    np.std(GCP_RMSE_list),
                    np.mean(Proj_RMSE_list),
                    np.std(Proj_RMSE_list),
                    np.mean(Proj_RMSE_Norm_list),
                    np.std(Proj_RMSE_Norm_list),
                    np.mean(Time_list),
                ],
                all_GCPs,
            )
        else:
            return (
                image_coords_dict,
                H,
                H_inv,
                absolute_transformations,
                [GCP_RMSE, RMSE, RMSE_Normalized, running_time + running_time_init],
                all_GCPs,
            )

    def geo_correct_MegaStitch_Similarity_Bundle_Adjustment_Homography(
        self, anchors_dict, sim_GCPs, use_old_inliers=False
    ):

        self.reference_image.load_img()

        x = self.reference_image.img.shape[1]
        y = self.reference_image.img.shape[0]

        self.settings.transformation = cv_util.Transformation.similarity
        tr = self.generate_neighbor_transformations()

        if not use_old_inliers:
            self.settings.transformation = cv_util.Transformation.homography
            tr_h = self.generate_neighbor_transformations()
            self.settings.transformation = cv_util.Transformation.similarity
        else:
            tr_h = tr

        # --------------------------------------------
        # cross validation
        # --------------------------------------------

        if self.settings.do_cross_validation:

            Proj_RMSE_list = []
            Proj_RMSE_Norm_list = []
            GCP_RMSE_list = []
            Time_list = []

            for i in range(self.settings.number_bins):

                print("========== Fold {0} ===========".format(i))

                self.settings.transformation = cv_util.Transformation.similarity
                mega_stitch_init = ProjectionOptimization.ReprojectionMinimization(
                    self.images,
                    tr,
                    [1, 2],
                    x,
                    y,
                    self.reference_image.name,
                    self.settings.max_no_inliers,
                    i,
                )
                (
                    _,
                    initializations,
                    running_time_init,
                ) = mega_stitch_init.MegaStitchSimilarityAffine(True)

                self.settings.transformation = cv_util.Transformation.homography
                mega_stitch = ProjectionOptimization.ReprojectionMinimization(
                    self.images,
                    tr_h,
                    [1, 2],
                    x,
                    y,
                    self.reference_image.name,
                    self.settings.max_no_inliers,
                    i,
                )
                (
                    new_coords,
                    absolute_transformations,
                    running_time,
                ) = mega_stitch.BundleAdjustmentHomography(initializations)

                H_GCP, _, GCP_RMSE, _, _ = self.report_GCP_error_for_Drone(
                    absolute_transformations, anchors_dict, sim_GCPs
                )

                RMSE, RMSE_Normalized = self.calculate_projection_error(
                    anchors_dict,
                    new_coords,
                    absolute_transformations,
                    tr_h,
                    i,
                    sim_GCPs=sim_GCPs,
                )
                print(":: Projection RMSE: {0}".format(RMSE))
                print(":: Normalized Projection RMSE: {0}".format(RMSE_Normalized))

                Proj_RMSE_list.append(RMSE)
                Proj_RMSE_Norm_list.append(RMSE_Normalized)

                GCP_RMSE_list.append(GCP_RMSE)
                Time_list.append(running_time + running_time_init)

            print("-----------------------------------------------------------")
            print(
                ":: Mean and STD of GCP RMSE: {0}, {1}".format(
                    np.mean(GCP_RMSE_list), np.std(GCP_RMSE_list)
                )
            )
            print(
                ":: Mean and STD of Projection RMSE: {0}, {1}".format(
                    np.mean(Proj_RMSE_list), np.std(Proj_RMSE_list)
                )
            )
            print(
                ":: Mean and STD of Normalized Projection RMSE: {0}, {1}".format(
                    np.mean(Proj_RMSE_Norm_list), np.std(Proj_RMSE_Norm_list)
                )
            )
            print(":: Mean of Time: {0}".format(np.mean(Time_list)))
            print("-----------------------------------------------------------")

        # --------------------------------------------
        # Single final opt. no cross validation
        # --------------------------------------------

        print("================== NO CROSS VALIDATION ==================")

        self.settings.max_no_inliers = 20

        self.settings.transformation = cv_util.Transformation.similarity
        mega_stitch_init = ProjectionOptimization.ReprojectionMinimization(
            self.images,
            tr,
            [1, 2],
            x,
            y,
            self.reference_image.name,
            self.settings.max_no_inliers,
            -1,
        )
        (
            _,
            initializations,
            running_time_init,
        ) = mega_stitch_init.MegaStitchSimilarityAffine(True)

        self.settings.transformation = cv_util.Transformation.homography
        mega_stitch = ProjectionOptimization.ReprojectionMinimization(
            self.images,
            tr_h,
            [1, 2],
            x,
            y,
            self.reference_image.name,
            self.settings.max_no_inliers,
            -1,
        )
        (
            new_coords,
            absolute_transformations,
            running_time,
        ) = mega_stitch.BundleAdjustmentHomography(initializations)

        H, H_inv, GCP_RMSE, _, all_GCPs = self.report_GCP_error_for_Drone(
            absolute_transformations, anchors_dict, sim_GCPs
        )

        RMSE, RMSE_Normalized = self.calculate_projection_error(
            anchors_dict,
            new_coords,
            absolute_transformations,
            tr_h,
            -1,
            sim_GCPs=sim_GCPs,
        )
        print(":: Projection RMSE: {0}".format(RMSE))
        print(":: Normalized Projection RMSE: {0}".format(RMSE_Normalized))

        # ----------------------

        image_coords_dict = {}

        for img in self.images:

            image_coords_dict[img.name] = {}

            for key in new_coords[img.name]:

                image_coords_dict[img.name][key] = [
                    int(new_coords[img.name][key][0]),
                    int(new_coords[img.name][key][1]),
                ]

        print(">>> BundleAdjustment-Homography finished successfully.")

        if self.settings.do_cross_validation:
            return (
                image_coords_dict,
                H,
                H_inv,
                absolute_transformations,
                [
                    GCP_RMSE,
                    RMSE,
                    RMSE_Normalized,
                    running_time,
                    np.mean(GCP_RMSE_list),
                    np.std(GCP_RMSE_list),
                    np.mean(Proj_RMSE_list),
                    np.std(Proj_RMSE_list),
                    np.mean(Proj_RMSE_Norm_list),
                    np.std(Proj_RMSE_Norm_list),
                    np.mean(Time_list),
                ],
                all_GCPs,
            )
        else:
            return (
                image_coords_dict,
                H,
                H_inv,
                absolute_transformations,
                [GCP_RMSE, RMSE, RMSE_Normalized, running_time + running_time_init],
                all_GCPs,
            )

    def geo_correct_Translation_KP_based(self, coords, anchors_dict):

        tr = self.generate_neighbor_transformations()

        self.reference_image.load_img()

        x = self.reference_image.img.shape[1]
        y = self.reference_image.img.shape[0]

        mega_stitch = ProjectionOptimization.ReprojectionMinimization(
            self.images,
            tr,
            [1, 1],
            x,
            y,
            self.reference_image.name,
            self.settings.max_no_inliers,
            self.settings.scale,
        )

        new_coords = mega_stitch.MegaStitchTranslationKeyPointBased(
            anchors_dict, coords
        )

        projection_RMSE = self.calculate_projection_error_Gantry(new_coords, tr, x, y)
        print(":: Projection RMSE: {0}".format(projection_RMSE))

        self.report_GCP_error_for_Gantry(new_coords, coords, anchors_dict, x, y)

        print(">>> MEGASTITCH Translation case KP-based finished successfully.")

        return new_coords

    def geo_correct_Translation_Parameter_based(self, coords, anchors_dict):

        tr = self.generate_neighbor_transformations()

        self.reference_image.load_img()

        x = self.reference_image.img.shape[1]
        y = self.reference_image.img.shape[0]

        mega_stitch = ProjectionOptimization.ReprojectionMinimization(
            self.images,
            tr,
            [1, 1],
            x,
            y,
            self.reference_image.name,
            self.settings.max_no_inliers,
            self.settings.scale,
        )

        new_coords = mega_stitch.MegaStitchTranslationParameterBased(
            anchors_dict, coords
        )

        projection_RMSE = self.calculate_projection_error_Gantry(new_coords, tr, x, y)
        print(":: Projection RMSE: {0}".format(projection_RMSE))

        self.report_GCP_error_for_Gantry(new_coords, coords, anchors_dict, x, y, False)

        print(">>> MEGASTITCH Translation case KP-based finished successfully.")

        return new_coords

    # -----------------------------------------------------------
    # --------------- Visualization Methods ---------------------
    # -----------------------------------------------------------

    def remove_outlier_images(self, coord_dict, perc=0.05):
        list_x = []
        list_y = []

        for img_name in coord_dict:
            coord = coord_dict[img_name]
            list_x.append(coord["UL"][0])
            list_x.append(coord["UR"][0])
            list_x.append(coord["LL"][0])
            list_x.append(coord["LR"][0])
            list_y.append(coord["UL"][1])
            list_y.append(coord["UR"][1])
            list_y.append(coord["LL"][1])
            list_y.append(coord["LR"][1])

        x_q3 = np.quantile(list_x, 1 - perc)
        x_q1 = np.quantile(list_x, perc)

        y_q3 = np.quantile(list_y, 1 - perc)
        y_q1 = np.quantile(list_y, perc)

        new_coords_dict = {}
        for img_name in coord_dict:
            coord = coord_dict[img_name]
            if (
                coord["UL"][0] < x_q1
                or coord["UL"][0] > x_q3
                or coord["UR"][0] < x_q1
                or coord["UR"][0] > x_q3
                or coord["LL"][0] < x_q1
                or coord["LL"][0] > x_q3
                or coord["LR"][0] < x_q1
                or coord["LR"][0] > x_q3
                or coord["UL"][1] < y_q1
                or coord["UL"][1] > y_q3
                or coord["UR"][1] < y_q1
                or coord["UR"][1] > y_q3
                or coord["LL"][1] < y_q1
                or coord["LL"][1] > y_q3
                or coord["LR"][1] < y_q1
                or coord["LR"][1] > y_q3
            ):

                continue

            new_coords_dict[img_name] = coord

        return new_coords_dict

    def visualize_initial_coords(self, initial_coord_dict):
        coords = np.array(
            [
                [
                    (initial_coord_dict[k][0] + initial_coord_dict[k][1]) / 2,
                    (initial_coord_dict[k][5] + initial_coord_dict[k][6]) / 2,
                ]
                for k in initial_coord_dict
            ]
        )

        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.scatter(coords[:, 0], coords[:, 1], marker="^")

        plt.show()
        plt.clf()
        plt.close()

    def save_visualized_initial_coords(
        self, initial_coord_dict, save_path, img_ref_index
    ):
        coords = np.array(
            [
                [
                    (initial_coord_dict[k][0] + initial_coord_dict[k][1]) / 2,
                    (initial_coord_dict[k][5] + initial_coord_dict[k][6]) / 2,
                ]
                for k in initial_coord_dict
            ]
        )

        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.scatter(
            np.delete(coords[:, 0], img_ref_index, axis=0),
            np.delete(coords[:, 1], img_ref_index, axis=0),
            marker="^",
        )
        ax.scatter(
            coords[img_ref_index, 0], coords[img_ref_index, 1], marker="^", color="red"
        )

        plt.savefig(save_path)
        plt.clf()
        plt.close()

    def visualize_field_centers(self, is_2d):

        if is_2d:
            x = [im.lon for im in self.images]
            y = [im.lat for im in self.images]

            fig = plt.figure()
            ax = fig.add_subplot(111)

            ax.scatter(x, y, marker="^")

            for img, img_n, transformation, perc_inliers in self.neighbors:

                if perc_inliers is None:
                    c = "red"
                else:
                    if perc_inliers < self.settings.discard_transformation_perc_inlier:
                        c = (0, 0, 0, 1)
                    else:
                        c = (perc_inliers, 0, 0, 1)

                ax.plot([img.lon, img_n.lon], [img.lat, img_n.lat], color=c)
        else:
            x = [im.lon for im in self.images]
            y = [im.lat for im in self.images]
            z = [im.alt for im in self.images]

            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")

            ax.scatter(x, y, z, marker="^")

            for img, img_n, transformation, perc_inliers in self.neighbors:

                if perc_inliers is None:
                    c = "red"
                else:
                    if perc_inliers < self.settings.discard_transformation_perc_inlier:
                        c = (0, 0, 0, 1)
                    else:
                        c = (perc_inliers, 0, 0, 1)

                ax.plot(
                    [img.lon, img_n.lon],
                    [img.lat, img_n.lat],
                    [img.alt, img_n.alt],
                    color=c,
                )

        plt.show()

    def save_field_centers_visualization(self, save_path):

        if self.settings.Dataset == "ODFN":
            return

        x = [im.lon for im in self.images]
        y = [im.lat for im in self.images]
        z = [im.alt for im in self.images]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        ax.scatter(x, y, z, marker="^")

        for img, img_n, transformation, perc_inliers in self.neighbors:

            if perc_inliers is None:
                c = "red"
            else:
                if perc_inliers < self.settings.discard_transformation_perc_inlier:
                    c = (0, 0, 0, 1)
                else:
                    c = (perc_inliers, 0, 0, 1)

            ax.plot(
                [img.lon, img_n.lon],
                [img.lat, img_n.lat],
                [img.alt, img_n.alt],
                color=c,
            )

        plt.savefig(save_path)

    def convert_GPS_coords_to_image_coords(self, gps_coords):

        coord = gps_coords[self.reference_image.name]

        self.reference_image.load_img()

        w = self.reference_image.img.shape[1]
        h = self.reference_image.img.shape[0]

        w_GPS = (
            gps_coords[self.reference_image.name]["UR"][0]
            - gps_coords[self.reference_image.name]["UL"][0]
        )
        h_GPS = (
            gps_coords[self.reference_image.name]["LL"][1]
            - gps_coords[self.reference_image.name]["UL"][1]
        )

        w_ratio = w / w_GPS
        h_ratio = h / h_GPS

        min_x, max_x, min_y, max_y = self.calculate_min_max(gps_coords)

        image_coords_dict = {}

        for img_name in gps_coords:

            coord = gps_coords[img_name]
            image_coords_dict[img_name] = {}

            for k in ["UL", "UR", "LR", "LL"]:

                image_coords_dict[img_name][k] = [
                    w_ratio * (coord[k][0] - min_x),
                    h_ratio * (coord[k][1] - min_y),
                ]

        return image_coords_dict

    def calculate_min_max(self, image_coords_dict):

        min_x = sys.maxsize
        max_x = 0
        min_y = sys.maxsize
        max_y = 0

        for i in image_coords_dict:
            img_coords = image_coords_dict[i]

            if img_coords["UL"][0] > max_x:
                max_x = img_coords["UL"][0]
            if img_coords["UR"][0] > max_x:
                max_x = img_coords["UR"][0]
            if img_coords["LL"][0] > max_x:
                max_x = img_coords["LL"][0]
            if img_coords["LR"][0] > max_x:
                max_x = img_coords["LR"][0]

            if img_coords["UL"][1] > max_y:
                max_y = img_coords["UL"][1]
            if img_coords["UR"][1] > max_y:
                max_y = img_coords["UR"][1]
            if img_coords["LL"][1] > max_y:
                max_y = img_coords["LL"][1]
            if img_coords["LR"][1] > max_y:
                max_y = img_coords["LR"][1]

            if img_coords["UL"][0] < min_x:
                min_x = img_coords["UL"][0]
            if img_coords["UR"][0] < min_x:
                min_x = img_coords["UR"][0]
            if img_coords["LL"][0] < min_x:
                min_x = img_coords["LL"][0]
            if img_coords["LR"][0] < min_x:
                min_x = img_coords["LR"][0]

            if img_coords["UL"][1] < min_y:
                min_y = img_coords["UL"][1]
            if img_coords["UR"][1] < min_y:
                min_y = img_coords["UR"][1]
            if img_coords["LL"][1] < min_y:
                min_y = img_coords["LL"][1]
            if img_coords["LR"][1] < min_y:
                min_y = img_coords["LR"][1]

        return min_x, max_x, min_y, max_y

    def generate_field_ortho(
        self, image_coords_dict, s=None, orig_GPS=None, gcp_info=None
    ):

        if orig_GPS is not None:

            image_coords_dict = self.convert_GPS_coords_to_image_coords(
                image_coords_dict
            )

        if s is not None:
            new_image_coords_dict = {}

            c = s / (self.settings.scale)

            for img in image_coords_dict:
                old_coord = image_coords_dict[img]
                coord = {
                    "UL": [int(old_coord["UL"][0] * c), int(old_coord["UL"][1] * c)],
                    "UR": [int(old_coord["UR"][0] * c), int(old_coord["UR"][1] * c)],
                    "LR": [int(old_coord["LR"][0] * c), int(old_coord["LR"][1] * c)],
                    "LL": [int(old_coord["LL"][0] * c), int(old_coord["LL"][1] * c)],
                }
                new_image_coords_dict[img] = coord

            image_coords_dict = new_image_coords_dict

        if (
            self.settings.Dataset == "GRG" and self.settings.Method == "BNDL-ADJ"
        ) or self.settings.Dataset == "ODFN":
            print(":: Removing outlier images for this experiment.")
            image_coords_dict = self.remove_outlier_images(image_coords_dict, 0.02)

        min_x, max_x, min_y, max_y = self.calculate_min_max(image_coords_dict)

        frame_size = (max_x - min_x, max_y - min_y)

        print(">>> Frame Size is: {0}".format(frame_size))

        frame_image = np.zeros((frame_size[1], frame_size[0], 3))

        self.reference_image.load_img(s=s)

        x = self.reference_image.img.shape[1]
        y = self.reference_image.img.shape[0]

        # -------------------------------

        if self.settings.parallel_stitch:

            args = []

            for img in self.images:

                if img.name not in image_coords_dict:
                    continue

                img_coords = image_coords_dict[img.name]

                pts2 = np.float32(
                    [
                        [img_coords["UL"][0] - min_x, img_coords["UL"][1] - min_y],
                        [img_coords["UR"][0] - min_x, img_coords["UR"][1] - min_y],
                        [img_coords["LR"][0] - min_x, img_coords["LR"][1] - min_y],
                        [img_coords["LL"][0] - min_x, img_coords["LL"][1] - min_y],
                    ]
                )

                pts1 = np.float32([[0, 0], [x, 0], [x, y], [0, y]])

                if img == self.reference_image:

                    if self.settings.draw_guided_colors:
                        img.img = cv2.rectangle(
                            img.img,
                            (0, 0),
                            (img.img.shape[1], img.img.shape[0]),
                            (0, 0, 255),
                            10,
                        )
                        cv2.line(img.img, (0, 0), (x, 0), (0, 255, 255), 4)

                img.kp = None
                img.dsc = None
                args.append((pts1, pts2, img, frame_size, s))

            processes = multiprocessing.Pool(self.settings.cores_to_use)
            results = processes.map(warp_estimate_helper, args)
            processes.close()

            for tmp in results:
                if tmp is None:
                    continue
                frame_image[frame_image == 0] = tmp[frame_image == 0]

        else:

            for img in self.images:

                if img.name not in image_coords_dict:
                    continue

                img_coords = image_coords_dict[img.name]

                pts2 = np.float32(
                    [
                        [img_coords["UL"][0] - min_x, img_coords["UL"][1] - min_y],
                        [img_coords["UR"][0] - min_x, img_coords["UR"][1] - min_y],
                        [img_coords["LR"][0] - min_x, img_coords["LR"][1] - min_y],
                        [img_coords["LL"][0] - min_x, img_coords["LL"][1] - min_y],
                    ]
                )

                pts1 = np.float32([[0, 0], [x, 0], [x, y], [0, y]])

                if img == self.reference_image:

                    if self.settings.draw_guided_colors:
                        img.img = cv2.rectangle(
                            img.img,
                            (0, 0),
                            (img.img.shape[1], img.img.shape[0]),
                            (0, 0, 255),
                            10,
                        )
                        cv2.line(img.img, (0, 0), (x, 0), (0, 255, 255), 4)

                img.kp = None
                img.dsc = None

                img.load_img(s=s)

                tmp = cv_util.find_warp_homography_and_warp(
                    pts1, pts2, img.img, frame_size
                )

                frame_image[frame_image == 0] = tmp[frame_image == 0]

        # -------------------------------

        if self.settings.draw_guided_colors:
            cv2.circle(frame_image, (-min_x, -min_y), 5, (0, 255, 0), -1)
            cv2.line(
                frame_image, (-min_x, -min_y), (-min_x + 50, -min_y), (0, 255, 0), 2
            )
            cv2.line(
                frame_image, (-min_x, -min_y), (-min_x, -min_y + 50), (0, 255, 0), 2
            )

            for img in self.images:

                if img.name not in image_coords_dict:
                    continue

                img_coords = image_coords_dict[img.name]

                pts2 = np.int32(
                    [
                        [img_coords["UL"][0] - min_x, img_coords["UL"][1] - min_y],
                        [img_coords["UR"][0] - min_x, img_coords["UR"][1] - min_y],
                        [img_coords["LR"][0] - min_x, img_coords["LR"][1] - min_y],
                        [img_coords["LL"][0] - min_x, img_coords["LL"][1] - min_y],
                    ]
                )

                cv2.line(
                    frame_image,
                    (pts2[0][0], pts2[0][1]),
                    (pts2[1][0], pts2[1][1]),
                    (255, 255, 0),
                    2,
                )
                cv2.line(
                    frame_image,
                    (pts2[1][0], pts2[1][1]),
                    (pts2[2][0], pts2[2][1]),
                    (255, 255, 0),
                    2,
                )
                cv2.line(
                    frame_image,
                    (pts2[2][0], pts2[2][1]),
                    (pts2[3][0], pts2[3][1]),
                    (255, 255, 0),
                    2,
                )
                cv2.line(
                    frame_image,
                    (pts2[3][0], pts2[3][1]),
                    (pts2[0][0], pts2[0][1]),
                    (255, 255, 0),
                    2,
                )

        if self.settings.draw_GCPs and gcp_info is not None:

            s1, s2 = cv_util.get_GCP_sizes(self.settings.Dataset, self.settings.Method)

            if s is not None:
                C = s / (self.settings.scale)
            else:
                C = 1

            anchors = gcp_info[0]
            H = gcp_info[1]

            if H.shape[0] == 2:
                H_tmp = np.eye(3)
                H_tmp[:2, :] = H
                H = H_tmp

            absolute_transformations = gcp_info[2]

            for i, a in enumerate(anchors):
                if a["img_name"] not in absolute_transformations:
                    continue

                color1 = (0, 0, 255)
                color2 = (0, 255, 0)

                p = [a["true_lon"], a["true_lat"], 1]
                p_act = np.matmul(H, p)
                p_act = p_act / p_act[2]

                cv2.rectangle(
                    frame_image,
                    (
                        int((p_act[0] - min_x) * C - s2 - 2),
                        int((p_act[1] - min_y) * C - s2 - 2),
                    ),
                    (
                        int((p_act[0] - min_x) * C + s2 + 2),
                        int((p_act[1] - min_y) * C + s2 + 2),
                    ),
                    (0, 0, 0),
                    -1,
                )
                cv2.rectangle(
                    frame_image,
                    (
                        int((p_act[0] - min_x) * C - s2),
                        int((p_act[1] - min_y) * C - s2),
                    ),
                    (
                        int((p_act[0] - min_x) * C + s2),
                        int((p_act[1] - min_y) * C + s2),
                    ),
                    color2,
                    -1,
                )

                p = [a["img_x"] * self.settings.scale, a["img_y"] * self.settings.scale, 1]
                p_GPS = np.matmul(absolute_transformations[a["img_name"]], p)
                p_GPS = p_GPS / p_GPS[2]

                cv2.circle(
                    frame_image,
                    (int((p_GPS[0] - min_x) * C), int((p_GPS[1] - min_y) * C)),
                    s1 + 4,
                    (0, 0, 0),
                    -1,
                )
                cv2.circle(
                    frame_image,
                    (int((p_GPS[0] - min_x) * C), int((p_GPS[1] - min_y) * C)),
                    s1,
                    color1,
                    -1,
                )

        print(">>> Orthomosaic generated successfully.")
        return frame_image.astype("uint8")

    def generate_field_ortho_multiple_GCPs(
        self, image_coords_dict, s=None, orig_GPS=None, gcp_info=None
    ):

        if orig_GPS is not None:

            image_coords_dict = self.convert_GPS_coords_to_image_coords(
                image_coords_dict
            )

        if s is not None:
            new_image_coords_dict = {}

            c = s / (self.settings.scale)

            for img in image_coords_dict:
                old_coord = image_coords_dict[img]
                coord = {
                    "UL": [int(old_coord["UL"][0] * c), int(old_coord["UL"][1] * c)],
                    "UR": [int(old_coord["UR"][0] * c), int(old_coord["UR"][1] * c)],
                    "LR": [int(old_coord["LR"][0] * c), int(old_coord["LR"][1] * c)],
                    "LL": [int(old_coord["LL"][0] * c), int(old_coord["LL"][1] * c)],
                }
                new_image_coords_dict[img] = coord

            image_coords_dict = new_image_coords_dict

        if self.settings.Dataset == "GRG" and self.settings.Method == "BNDL-ADJ":
            print(":: Removing outlier images for this experiment.")
            image_coords_dict = self.remove_outlier_images(image_coords_dict)

        min_x, max_x, min_y, max_y = self.calculate_min_max(image_coords_dict)

        frame_size = (max_x - min_x, max_y - min_y)

        print(">>> Frame Size is: {0}".format(frame_size))

        frame_image = np.zeros((frame_size[1], frame_size[0], 3))

        self.reference_image.load_img(s=s)

        x = self.reference_image.img.shape[1]
        y = self.reference_image.img.shape[0]

        # -------------------------------

        if self.settings.parallel_stitch:

            args = []

            for img in self.images:

                if img.name not in image_coords_dict:
                    continue

                img_coords = image_coords_dict[img.name]

                pts2 = np.float32(
                    [
                        [img_coords["UL"][0] - min_x, img_coords["UL"][1] - min_y],
                        [img_coords["UR"][0] - min_x, img_coords["UR"][1] - min_y],
                        [img_coords["LR"][0] - min_x, img_coords["LR"][1] - min_y],
                        [img_coords["LL"][0] - min_x, img_coords["LL"][1] - min_y],
                    ]
                )

                pts1 = np.float32([[0, 0], [x, 0], [x, y], [0, y]])

                if img == self.reference_image:

                    if self.settings.draw_guided_colors:
                        img.img = cv2.rectangle(
                            img.img,
                            (0, 0),
                            (img.img.shape[1], img.img.shape[0]),
                            (0, 0, 255),
                            10,
                        )
                        cv2.line(img.img, (0, 0), (x, 0), (0, 255, 255), 4)

                img.kp = None
                img.dsc = None
                args.append((pts1, pts2, img, frame_size, s))

            processes = multiprocessing.Pool(self.settings.cores_to_use)
            results = processes.map(warp_estimate_helper, args)
            processes.close()

            for tmp in results:
                frame_image[frame_image == 0] = tmp[frame_image == 0]

        else:

            for img in self.images:

                if img.name not in image_coords_dict:
                    continue

                img_coords = image_coords_dict[img.name]

                pts2 = np.float32(
                    [
                        [img_coords["UL"][0] - min_x, img_coords["UL"][1] - min_y],
                        [img_coords["UR"][0] - min_x, img_coords["UR"][1] - min_y],
                        [img_coords["LR"][0] - min_x, img_coords["LR"][1] - min_y],
                        [img_coords["LL"][0] - min_x, img_coords["LL"][1] - min_y],
                    ]
                )

                pts1 = np.float32([[0, 0], [x, 0], [x, y], [0, y]])

                if img == self.reference_image:

                    if self.settings.draw_guided_colors:
                        img.img = cv2.rectangle(
                            img.img,
                            (0, 0),
                            (img.img.shape[1], img.img.shape[0]),
                            (0, 0, 255),
                            10,
                        )
                        cv2.line(img.img, (0, 0), (x, 0), (0, 255, 255), 4)

                img.kp = None
                img.dsc = None

                img.load_img(s=s)

                tmp = cv_util.find_warp_homography_and_warp(
                    pts1, pts2, img.img, frame_size
                )

                frame_image[frame_image == 0] = tmp[frame_image == 0]

        # -------------------------------

        if self.settings.draw_guided_colors:
            cv2.circle(frame_image, (-min_x, -min_y), 5, (0, 255, 0), -1)
            cv2.line(
                frame_image, (-min_x, -min_y), (-min_x + 50, -min_y), (0, 255, 0), 2
            )
            cv2.line(
                frame_image, (-min_x, -min_y), (-min_x, -min_y + 50), (0, 255, 0), 2
            )

            for img in self.images:

                if img.name not in image_coords_dict:
                    continue

                img_coords = image_coords_dict[img.name]

                pts2 = np.int32(
                    [
                        [img_coords["UL"][0] - min_x, img_coords["UL"][1] - min_y],
                        [img_coords["UR"][0] - min_x, img_coords["UR"][1] - min_y],
                        [img_coords["LR"][0] - min_x, img_coords["LR"][1] - min_y],
                        [img_coords["LL"][0] - min_x, img_coords["LL"][1] - min_y],
                    ]
                )

                cv2.line(
                    frame_image,
                    (pts2[0][0], pts2[0][1]),
                    (pts2[1][0], pts2[1][1]),
                    (255, 255, 0),
                    2,
                )
                cv2.line(
                    frame_image,
                    (pts2[1][0], pts2[1][1]),
                    (pts2[2][0], pts2[2][1]),
                    (255, 255, 0),
                    2,
                )
                cv2.line(
                    frame_image,
                    (pts2[2][0], pts2[2][1]),
                    (pts2[3][0], pts2[3][1]),
                    (255, 255, 0),
                    2,
                )
                cv2.line(
                    frame_image,
                    (pts2[3][0], pts2[3][1]),
                    (pts2[0][0], pts2[0][1]),
                    (255, 255, 0),
                    2,
                )

        if self.settings.draw_GCPs and gcp_info is not None:

            if self.settings.Dataset == "DSEFN" or self.settings.Dataset == "DLLFN":
                s1 = 30
                s2 = 40
            else:
                s1 = 10
                s2 = 15

            if s is not None:
                C = s / (self.settings.scale)
            else:
                C = 1

            all_GCPs = gcp_info[0]

            colors = cv_util.generate_n_distinc_colors(len([k for k in all_GCPs]))

            for i, k in enumerate(all_GCPs):

                color1 = colors[i]
                color2 = (0, 255, 0)

                p_act = all_GCPs[k]["act"]

                cv2.rectangle(
                    frame_image,
                    (
                        int((p_act[0] - min_x) * C - s2 - 2),
                        int((p_act[1] - min_y) * C - s2 - 2),
                    ),
                    (
                        int((p_act[0] - min_x) * C + s2 + 2),
                        int((p_act[1] - min_y) * C + s2 + 2),
                    ),
                    (0, 0, 0),
                    -1,
                )
                cv2.rectangle(
                    frame_image,
                    (
                        int((p_act[0] - min_x) * C - s2),
                        int((p_act[1] - min_y) * C - s2),
                    ),
                    (
                        int((p_act[0] - min_x) * C + s2),
                        int((p_act[1] - min_y) * C + s2),
                    ),
                    color2,
                    -1,
                )

                for p_GPS in all_GCPs[k]["gps"]:

                    cv2.circle(
                        frame_image,
                        (int((p_GPS[0] - min_x) * C), int((p_GPS[1] - min_y) * C)),
                        s1 + 4,
                        (0, 0, 0),
                        -1,
                    )
                    cv2.circle(
                        frame_image,
                        (int((p_GPS[0] - min_x) * C), int((p_GPS[1] - min_y) * C)),
                        s1,
                        color1,
                        -1,
                    )

        print(">>> Orthomosaic generated successfully.")
        return frame_image.astype("uint8")

    def generate_field_ortho_GPS(self, gps_coords):

        self.reference_image.load_img()

        x = self.reference_image.img.shape[1]
        y = self.reference_image.img.shape[0]

        ref_gps = gps_coords[self.reference_image.name]

        pts1 = np.float32(
            [
                [ref_gps["UL"][0], ref_gps["UL"][1]],
                [ref_gps["UR"][0], ref_gps["UR"][1]],
                [ref_gps["LR"][0], ref_gps["LR"][1]],
                [ref_gps["LL"][0], ref_gps["LL"][1]],
            ]
        )

        pts2 = np.float32([[0, 0], [x, 0], [x, y], [0, y]])

        H, masked = cv2.findHomography(pts1, pts2)

        new_image_coordinates = {}

        for img in self.images:

            coord = gps_coords[img.name]
            new_image_coordinates[img.name] = {}

            for k, key in enumerate(["UL", "UR", "LR", "LL"]):

                p = [coord[key][0], coord[key][1], 1]
                new_p = np.matmul(H, p)
                new_p = new_p / new_p[2]
                new_p = new_p.astype("int32")
                new_image_coordinates[img.name][key] = [new_p[0], new_p[1]]

        return self.generate_field_ortho(new_image_coordinates)

    def visualize_field_ortho(self, ortho):

        cv_util.show(ortho, "w", 1000, 700)

    def save_field_ortho(self, ortho, path):

        cv2.imwrite(path, ortho)
        print(">>> Orthomosaic saved successfully.")

    def save_field_coordinates(self, path, corrected_coordinates):

        min_x = sys.maxsize
        min_y = sys.maxsize
        max_x = -sys.maxsize
        max_y = -sys.maxsize

        for i in corrected_coordinates:
            img_coords = corrected_coordinates[i]

            if img_coords["UL"][0] > max_x:
                max_x = img_coords["UL"][0]
            if img_coords["UR"][0] > max_x:
                max_x = img_coords["UR"][0]
            if img_coords["LL"][0] > max_x:
                max_x = img_coords["LL"][0]
            if img_coords["LR"][0] > max_x:
                max_x = img_coords["LR"][0]

            if img_coords["UL"][1] > max_y:
                max_y = img_coords["UL"][1]
            if img_coords["UR"][1] > max_y:
                max_y = img_coords["UR"][1]
            if img_coords["LL"][1] > max_y:
                max_y = img_coords["LL"][1]
            if img_coords["LR"][1] > max_y:
                max_y = img_coords["LR"][1]

            if img_coords["UL"][0] < min_x:
                min_x = img_coords["UL"][0]
            if img_coords["UR"][0] < min_x:
                min_x = img_coords["UR"][0]
            if img_coords["LL"][0] < min_x:
                min_x = img_coords["LL"][0]
            if img_coords["LR"][0] < min_x:
                min_x = img_coords["LR"][0]

            if img_coords["UL"][1] < min_y:
                min_y = img_coords["UL"][1]
            if img_coords["UR"][1] < min_y:
                min_y = img_coords["UR"][1]
            if img_coords["LL"][1] < min_y:
                min_y = img_coords["LL"][1]
            if img_coords["LR"][1] < min_y:
                min_y = img_coords["LR"][1]

        info = {
            "scale": self.settings.scale,
            "coords": corrected_coordinates,
            "min_x": min_x,
            "min_y": min_y,
            "max_x": max_x,
            "max_y": max_y,
        }

        with open(path, "w+") as outfile:
            json.dump(info, outfile)

    def load_filed_coordinates(self, path):

        with open(path, "r") as outfile:
            info = json.load(outfile)

        corrected_coordinates = info["coords"]
        s = info["scale"]

        return corrected_coordinates

    def get_gps_scale_factors(self, coords, orig_GPS):

        GPS_coord = orig_GPS[self.reference_image.name]

        coord = coords[self.reference_image.name]

        w_GPS = (
            orig_GPS[self.reference_image.name]["UR"]["lon"]
            - orig_GPS[self.reference_image.name]["UL"]["lon"]
        )
        h_GPS = (
            orig_GPS[self.reference_image.name]["UL"]["lat"]
            - orig_GPS[self.reference_image.name]["LL"]["lat"]
        )

        w_img = coord["UR"][0] - coord["UL"][0]
        h_img = coord["LL"][1] - coord["UL"][1]

        w_ratio = w_img / w_GPS
        h_ratio = h_img / h_GPS

        return w_ratio, h_ratio

    def calculate_sigma_reprojection_error(self, coords, orig_GPS=None, is_GPS=False):

        if is_GPS:

            corrected_coord = self.convert_GPS_coords_to_image_coords(coords)

        else:

            corrected_coord = coords

        if orig_GPS is not None:

            r_w, r_h = self.get_gps_scale_factors(corrected_coord, orig_GPS)

        absolute_transformations = (
            self.get_absolute_transformations_after_geo_correction(corrected_coord)
        )

        reprojection_errors = []

        for img1_name in self.pairwise_transformations:

            for img2_name in self.pairwise_transformations[img1_name]:

                inliers = self.pairwise_transformations[img1_name][img2_name][3]
                matches = self.pairwise_transformations[img1_name][img2_name][1]

                T_1 = np.linalg.inv(absolute_transformations[img1_name])
                T_2 = np.linalg.inv(absolute_transformations[img2_name])

                for i, m in enumerate(matches):

                    if inliers[i, 0] == 0:
                        continue

                    kp_1 = self.images_dict[img1_name].kp[m.trainIdx]
                    kp_2 = self.images_dict[img2_name].kp[m.queryIdx]

                    p_1 = [kp_1[0], kp_1[1], 1]
                    p_2 = [kp_2[0], kp_2[1], 1]

                    p_1_new = np.matmul(T_1, p_1)
                    p_1_new = p_1_new / p_1_new[2]
                    p_1_new = p_1_new[:2]

                    p_2_new = np.matmul(T_2, p_2)
                    p_2_new = p_2_new / p_2_new[2]
                    p_2_new = p_2_new[:2]

                    diff = p_1_new - p_2_new

                    if orig_GPS is not None:

                        diff[0] = diff[0] / r_w
                        diff[1] = diff[1] / r_h

                    error = math.sqrt(diff[0] ** 2 + diff[1] ** 2)
                    reprojection_errors.append(error)

        print(
            ">>> Variance reprojection errors after correction: {0}".format(
                np.std(reprojection_errors)
            )
        )

    def generate_transformation_accuracy_histogram(self, coords, save_path):

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.axis("equal")

        X = []
        Y = []
        C = []
        V = []

        for c in coords:

            coord_center = (
                (
                    coords[c]["UL"][0]
                    + coords[c]["UR"][0]
                    + coords[c]["LL"][0]
                    + coords[c]["LR"][0]
                )
                / 4,
                (
                    coords[c]["UL"][1]
                    + coords[c]["UR"][1]
                    + coords[c]["LL"][1]
                    + coords[c]["LR"][1]
                )
                / 4,
            )

            if c not in self.pairwise_transformations:
                continue

            list_perc_inliers = []
            list_num_inliers = []

            for c2 in self.pairwise_transformations[c]:
                numm = len(self.pairwise_transformations[c][c2][3])
                numi = np.sum(self.pairwise_transformations[c][c2][3])

                list_perc_inliers.append(numi / numm)
                list_num_inliers.append(numi)

            X.append(coord_center[0])
            Y.append(-coord_center[1])
            C.append(cm.RdYlGn(np.mean(list_perc_inliers)))
            V.append(np.mean(list_num_inliers))

        ax.scatter(X, Y, c=C, s=V, alpha=0.7)

        plt.savefig(save_path)
        plt.clf()
        plt.close()

import json
import os

import cv2

from megastitch import computer_vision_utils as cv_util


class Configuration:
    def __init__(self, images_path=""):
        self.images_path = images_path
        tmp_list = os.listdir(images_path)
        tmp = cv2.imread("{0}/{1}".format(images_path, tmp_list[0]))
        self.scale = 0.2
        self.image_size = tmp.shape
        self.nearest_number = 4
        self.use_gps_distance = True
        self.transformation = cv_util.Transformation.similarity
        self.cores_to_use = 2
        self.discard_transformation_perc_inlier = 0.8
        self.max_SIFT_points = 100000
        self.use_perc_inliers_for_coef = False
        self.use_iterative_methods = True
        self.perc_inliers_formula = lambda n: n
        self.use_ceres_MGRAPH = False
        self.perc_crop = 0
        self.discard_even_images = False
        self.normalize_key_points = False
        self.refine_transformations = False
        self.use_homogenous_coords = False
        self.use_parallel_multiGroup = False
        self.no_cores_multiGroup = 2
        self.number_equation_to_pick_from_unique_tuples = 20
        self.grid_w = 3
        self.grid_h = 7
        self.min_intersect = 1
        self.draw_guided_colors = False
        self.equalize_histogram = False
        self.percentage_next_neighbor = 0.6
        self.sub_set_choosing = False
        self.N_perc = 0.1
        self.E_perc = 0.4
        self.parallel_stitch = True
        self.max_no_inliers = 20
        self.draw_GCPs = False
        self.Dataset = ""
        self.Method = ""
        self.number_bins = 5
        self.size_bins = 5
        self.do_cross_validation = False
        self.AllGCPRMSE = True
        self.preprocessing_transformation = "none"

    def load(self, filename: str):
        """Load a json configuration file"""
        with open(filename, "r") as f:
            settings_dict = json.load(f)

        self.scale = settings_dict["scale"]
        self.nearest_number = settings_dict["nearest_number"]
        self.discard_transformation_perc_inlier = settings_dict["discard_transformation_perc_inlier"]
        self.transformation = getattr(cv_util.Transformation, settings_dict["transformation"])
        self.percentage_next_neighbor = settings_dict["percentage_next_neighbor"]
        self.cores_to_use = settings_dict["cores_to_use"]
        self.draw_GCPs = settings_dict["draw_GCPs"]
        self.sub_set_choosing = settings_dict["sub_set_choosing"]
        self.N_perc = settings_dict["N_perc"]
        self.E_perc = settings_dict["E_perc"]

    def __getstate__(self):
        """Get the object state, removing unpickable objects"""
        state = self.__dict__.copy()

        # Remove unpickable objects
        del state['perc_inliers_formula']

        return state

    def __setstate__(self, state):
        self.__dict__ = state

        # Add back the unpickable objects
        self.perc_inliers_formula = lambda n: n


import numpy as np
import os
import glob
import cv2
import sys
import math
import RANSAC
import random
import colorsys
from enum import Enum
from scipy.optimize import lsq_linear


class Transformation(Enum):
    translation = 1
    similarity = 2
    affine = 3
    homography = 4
    full = 5


def get_gps_distance(lat1, lon1, lat2, lon2):
    phi1 = math.radians(lat1)
    lambda1 = math.radians(lon1)
    phi2 = math.radians(lat2)
    lambda2 = math.radians(lon2)
    R = 6371e3

    a = math.sin((phi2 - phi1) / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * (
        math.sin((lambda2 - lambda1) / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


def get_SIFT_points(main_img, bounding_box, max_sift_number):

    img = main_img.copy()

    sift = cv2.xfeatures2d.SIFT_create()

    kp, desc = sift.detectAndCompute(img, None)

    kp = kp[: min(len(kp), max_sift_number)]
    desc = desc[: min(len(kp), max_sift_number)]
    return kp, desc


def get_matches(desc1, desc2, kp1, kp2, perc_next_match=0.8, perc_top_matches=0.5):

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc1, desc2, k=2)

    if matches is None or len(matches) == 0:
        return None

    if len(matches[0]) < 2:
        return None

    good = []
    for m in matches:

        if m[0].distance < perc_next_match * m[1].distance:
            good.append(m)

    sorted_matches = sorted(good, key=lambda x: x[0].distance)

    good = []

    number_of_good_matches = int(math.floor(len(sorted_matches) * perc_top_matches))
    good = sorted_matches[0:number_of_good_matches]

    matches = np.asarray(good)

    return matches


def estimate_base_transformations(pts1, pts2, tr_type):

    if tr_type == Transformation.translation:

        T = np.eye(3)
        mean_xys = np.mean(pts2 - pts1, axis=0)

        T[0, 2] = mean_xys[0]
        T[1, 2] = mean_xys[0]
        return T

    if tr_type == Transformation.similarity:

        T = cv2.estimateAffinePartial2D(pts1, pts2)

        if T is None or len(T.shape) <= 1:
            return None

        T = np.append(T, np.array([[0, 0, 1]]), axis=0)

        return T

    if tr_type == Transformation.affine:

        T = cv2.estimateAffine2D(pts1, pts2)

        if T is None or len(T.shape) <= 1:
            return None

        T = np.append(T, np.array([[0, 0, 1]]), axis=0)

        return T

    if tr_type == Transformation.homography:

        T = cv2.findHomography(pts1, pts2)[0]
        # T = cv2.getPerspectiveTransform(pts1,pts2)

        return T


def estimate_transformation_from_SIFT(
    desc1, desc2, kp1, kp2, transformation, perc_second, cores
):

    # if multiplied by the key points of the first image, gives the key points of the second image
    # if T multiplied by the corners of the second image (in first image system) gives the corners of the first image

    matches = get_matches(desc1, desc2, kp1, kp2, perc_second)

    if len(matches.shape) == 1:
        first_matches = matches
    else:
        first_matches = matches[:, 0]

    if len(matches) == 0:
        return None, None, 0, None

    if type(kp1) == cv2.KeyPoint:
        src = np.float32(
            [[kp1[m.queryIdx].pt[0], kp1[m.queryIdx].pt[1]] for m in first_matches]
        )
        dst = np.float32(
            [[kp2[m.trainIdx].pt[0], kp2[m.trainIdx].pt[1]] for m in first_matches]
        )
    else:
        src = np.float32(
            [[kp1[m.queryIdx][0], kp1[m.queryIdx][1]] for m in first_matches]
        )
        dst = np.float32(
            [[kp2[m.trainIdx][0], kp2[m.trainIdx][1]] for m in first_matches]
        )

    if transformation == Transformation.translation:

        # diff = np.mean(dst-src,axis=0)

        # T = np.eye(3)
        # T[0,2] = diff[0]
        # T[1,2] = diff[1]

        # return T, first_matches, 1, None

        T, masked = RANSAC.estimateTranslation(src, dst, cores)
        # T, masked = RANSAC.estimateTranslation(src,dst,1)

        if T is None:
            return None, None, 0, None

        return T, first_matches, np.sum(masked) / len(dst), masked

    if transformation == Transformation.similarity:

        T, _ = cv2.estimateAffinePartial2D(src, dst)

        if T is None or T.shape != (2, 3):
            return None, None, 0, None

        T = np.append(T, np.array([[0, 0, 1]]), axis=0)

        masked = np.zeros((len(src), 1))
        for i, p_s in enumerate(src):
            new_dst = np.matmul(T, (p_s[0], p_s[1], 1))
            new_dst = new_dst / new_dst[2]
            if np.sqrt(np.sum((new_dst[:2] - dst[i]) ** 2)) <= 1.5:
                masked[i, 0] = 1
            else:
                masked[i, 0] = 0

        # s,theta,tx,ty = decompose_similarity(T)
        # T_new = build_transformation(Transformation.similarity,{'tr_x':tx,'tr_y':ty,'angle_theta':-theta,'scale_x':s,'center_rotation':(0,0)})
        # print(theta)
        # print(s)
        # print(T)
        # print(T_new)
        # T = T_new
        # corner_translations = get_corner_wise_transformations(T,matches,kp1,kp2,w,h)

        # return T, 1, corner_translations

        # new_dst = np.matmul(T,(src[0][0],src[0][1],1))
        # new_dst = new_dst/new_dst[2]
        # print(new_dst)
        # print(dst[0])

        return T, first_matches, 1, masked

    if transformation == Transformation.affine:

        # T, masked = cv2.estimateAffinePartial2D(dst, src , maxIters = 500, confidence = 0.99, refineIters = 5)
        T, _ = cv2.estimateAffine2D(src, dst)

        if T is None or T.shape != (2, 3):
            return None, None, 0, None

        T = np.append(T, np.array([[0, 0, 1]]), axis=0)

        # new_dst = np.matmul(T,(src[0][0],src[0][1],1))
        # new_dst = new_dst/new_dst[2]
        # print(new_dst)
        # print(dst[0])

        masked = np.zeros((len(src), 1))
        for i, p_s in enumerate(src):
            new_dst = np.matmul(T, (p_s[0], p_s[1], 1))
            new_dst = new_dst / new_dst[2]
            if np.sqrt(np.sum((new_dst[:2] - dst[i]) ** 2)) <= 1.5:
                masked[i, 0] = 1
            else:
                masked[i, 0] = 0

        return T, first_matches, 1, masked

    if transformation == Transformation.homography:

        if len(src) < 4:
            return None, None, 0, None

        T, masked = cv2.findHomography(
            src, dst, maxIters=500, confidence=0.99, method=cv2.RANSAC
        )

        # T = non_homogenouse_homography(src,dst)
        # masked = np.array([1]*len(dst))

        if T is None:
            return None, None, 0, None

        # return T, len(masked)/len(dst)

        # print(src[masked[:,0]==1,:])
        # print(dst[masked[:,0]==1,:])

        # new_dst = np.matmul(T,(src[0][0],src[0][1],1))
        # new_dst = new_dst/new_dst[2]
        # print(new_dst)
        # print(dst[0])

        # masked2 = np.zeros((len(src),1))
        # for i, p_s in enumerate(src):
        # 	 new_dst = np.matmul(T,(p_s[0],p_s[1],1))
        # 	 new_dst = new_dst/new_dst[2]
        # 	 if np.sqrt(np.sum((new_dst[:2] - dst[i])**2)) <= 1.5:
        # 		 masked2[i,0] = 1
        # 	 else:
        # 		 masked2[i,0] = 0

        # print(masked2-masked)

        return T, first_matches, np.sum(masked) / len(dst), masked


def estimate_transformation_from_Inliers(
    inliers, desc1, desc2, kp1, kp2, transformation
):

    src = np.float32([[kp1[m.queryIdx].pt[0], kp1[m.queryIdx].pt[1]] for m in inliers])
    dst = np.float32([[kp2[m.trainIdx].pt[0], kp2[m.trainIdx].pt[1]] for m in inliers])

    if (
        len(src.shape) == 1
        or len(dst.shape) == 1
        or src.shape[0] < 4
        or dst.shape[0] < 4
    ):
        return None, None, None, None

    if transformation == Transformation.translation:

        diff = np.mean(dst - src, axis=0)

        T = np.eye(3)
        T[0, 2] = diff[0]
        T[1, 2] = diff[1]

        return T, inliers, 1, None

    if transformation == Transformation.similarity:

        T, _ = cv2.estimateAffinePartial2D(src, dst)

        if T is None or T.shape != (2, 3):
            return None, None, 0, None

        T = np.append(T, np.array([[0, 0, 1]]), axis=0)

        masked = np.zeros((len(src), 1))
        for i, p_s in enumerate(src):
            new_dst = np.matmul(T, (p_s[0], p_s[1], 1))
            new_dst = new_dst / new_dst[2]
            if np.sqrt(np.sum((new_dst[:2] - dst[i]) ** 2)) <= 0.5:
                masked[i, 0] = 1
            else:
                masked[i, 0] = 0

        return T, inliers, 1, masked

    if transformation == Transformation.affine:

        # T, masked = cv2.estimateAffinePartial2D(dst, src , maxIters = 500, confidence = 0.99, refineIters = 5)
        T, _ = cv2.estimateAffine2D(src, dst)

        if T is None or T.shape != (2, 3):
            return None, None, 0, None

        T = np.append(T, np.array([[0, 0, 1]]), axis=0)

        masked = np.zeros((len(src), 1))
        for i, p_s in enumerate(src):
            new_dst = np.matmul(T, (p_s[0], p_s[1], 1))
            new_dst = new_dst / new_dst[2]
            if np.sqrt(np.sum((new_dst[:2] - dst[i]) ** 2)) <= 1.5:
                masked[i, 0] = 1
            else:
                masked[i, 0] = 0

        return T, inliers, 1, masked

    if transformation == Transformation.homography:

        T, masked = cv2.findHomography(
            src, dst, maxIters=500, confidence=0.99, method=cv2.RANSAC
        )

        if T is None:
            return None, None, 0, None

        return T, inliers, np.sum(masked) / len(dst), masked


def select_SIFT_points(kp, desc, bounding_box):

    x1 = bounding_box[0]
    y1 = bounding_box[1]
    x2 = bounding_box[2]
    y2 = bounding_box[3]


def draw_SIFT_points_on_img(img, kp, desc):
    res = img.copy()
    res = cv2.drawKeypoints(
        img, kp, res, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    return res


def warp_homography_and_stitch(img, frame_image, frame_size, H):

    tmp = cv2.warpPerspective(img, H, frame_size)

    frame_image[frame_image == 0] = tmp[frame_image == 0]

    return frame_image


def find_warp_homography_and_warp(pts1, pts2, img, frame_size):
    H = estimate_base_transformations(pts1, pts2, Transformation.homography)
    if H is None:
        return None
    tmp = cv2.warpPerspective(img, H, frame_size)

    return tmp


def show(img, window_name, w, h):

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, w, h)
    cv2.imshow(window_name, img)
    cv2.waitKey(0)


def keypoint_to_tuple_encode(kp):
    # return kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id
    return kp.pt[0], kp.pt[1]


def keypoint_from_tuple_decode(tpl):
    kp = cv2.KeyPoint(
        x=tpl[0][0],
        y=tpl[0][1],
        _size=tpl[1],
        _angle=tpl[2],
        _response=tpl[3],
        _octave=tpl[4],
        _class_id=tpl[5],
    )
    return kp


def pickle_matches(matches):

    if matches is None:
        return None

    if len(matches) == 0:
        return []

    p_matches = []

    for m in matches:
        p_matches.append((m.distance, m.trainIdx, m.queryIdx, m.imgIdx))

    return np.array(p_matches)


def get_matches_from_pickled(pickled_matches):

    if pickled_matches is None:
        return None

    if len(pickled_matches.shape) == 0 or pickled_matches.shape[0] == 0:
        return []

    matches = []

    for d, tid, qid, iid in pickled_matches:
        matches.append(cv2.DMatch(int(qid), int(tid), int(iid), d))

    return matches


def non_homogenouse_homography(pts_n, pts_i):

    A = []
    b = []

    pts_n = pts_n
    pts_i = pts_i

    for i, p_i in enumerate(pts_i):

        p_n = pts_n[i]

        A.append([p_n[0], p_n[1], 1, 0, 0, 0, 0, 0, 0])
        b.append(p_i[0])

        A.append([0, 0, 0, p_n[0], p_n[1], 1, 0, 0, 0])
        b.append(p_i[1])

        A.append([0, 0, 0, 0, 0, 0, p_n[0], p_n[1], 1])
        b.append(1)

    A = np.array(A)
    b = np.array(b)

    res = lsq_linear(A, b)
    X = res.x

    T = X.reshape((3, 3))
    # print(np.matmul(T,np.array([3,5,1])))
    return T


def decompose_similarity(T):

    s = math.sqrt(T[0, 0] ** 2 + T[0, 1] ** 2)
    theta = np.degrees(np.arctan2(-T[0, 1], T[0, 0]))
    t_x = T[0, 2]
    t_y = T[1, 2]

    return np.round(np.array([s, theta, t_x, t_y]), 4)


def normalize_key_points(kp, w, h, initial):

    new_kp = []

    for p in kp:

        p_new = cv2.KeyPoint(
            x=(p.pt[0] - w / 2) / (w),
            y=(p.pt[1] - h / 2) / (h),
            _size=p.size,
            _angle=p.angle,
            _response=p.response,
            _octave=p.octave,
            _class_id=p.class_id,
        )
        # p_new = cv2.KeyPoint(x=p.pt[0]-w/2,y=p.pt[1]-h/2,_size=p.size, _angle=p.angle, _response=p.response, _octave=p.octave, _class_id=p.class_id)
        # p_new = cv2.KeyPoint(x=p.pt[0]-w/2+initial[0],y=p.pt[1]-h/2+initial[1],_size=p.size, _angle=p.angle, _response=p.response, _octave=p.octave, _class_id=p.class_id)
        # p_new = cv2.KeyPoint(x=p.pt[0]+initial[0],y=p.pt[1]+initial[1],_size=p.size, _angle=p.angle, _response=p.response, _octave=p.octave, _class_id=p.class_id)
        # p_new = cv2.KeyPoint(x=p.pt[0]/w-0.5+initial[0],y=p.pt[1]/h-0.5-initial[1],_size=p.size, _angle=p.angle, _response=p.response, _octave=p.octave, _class_id=p.class_id)
        new_kp.append(p_new)

    return new_kp


def get_best_single_good_match(T, matches, kp1, kp2):

    if len(matches.shape) == 1:
        first_matches = matches
    else:
        first_matches = matches[:, 0]

    error = sys.maxsize
    best_match = None

    for m in first_matches:
        p1 = np.array([kp1[m.queryIdx].pt[0], kp1[m.queryIdx].pt[1], 1])
        p2 = np.array([kp2[m.trainIdx].pt[0], kp2[m.trainIdx].pt[1], 1])

        p1_transformed = np.matmul(T, p1)
        e = math.sqrt(np.mean((p1_transformed - p2) ** 2))

        if e < error:
            best_match = m
            error = e

    return best_match


def get_corner_wise_transformations(T, matches, kp1, kp2, w, h):

    best_m = get_best_single_good_match(T, matches, kp1, kp2)

    p1 = np.array([kp1[best_m.queryIdx].pt[0], kp1[best_m.queryIdx].pt[1]])
    p2 = np.array([kp2[best_m.trainIdx].pt[0], kp2[best_m.trainIdx].pt[1]])

    T_left_UL = np.eye(3)
    T_right_UL = np.eye(3)

    T_left_UR = np.eye(3)
    T_right_UR = np.eye(3)

    T_left_LR = np.eye(3)
    T_right_LR = np.eye(3)

    T_left_LL = np.eye(3)
    T_right_LL = np.eye(3)

    T_right_UL[0, 2] = p1[0]
    T_right_UL[1, 2] = p1[1]
    T_left_UL[0, 2] = -p2[0]
    T_left_UL[1, 2] = -p2[1]

    T_right_UR[0, 2] = p1[0] - w
    T_right_UR[1, 2] = p1[1]
    T_left_UR[0, 2] = -p2[0] + w
    T_left_UR[1, 2] = -p2[1]

    T_right_LR[0, 2] = p1[0] - w
    T_right_LR[1, 2] = p1[1] - h
    T_left_LR[0, 2] = -p2[0] + w
    T_left_LR[1, 2] = -p2[1] + h

    T_right_LL[0, 2] = p1[0]
    T_right_LL[1, 2] = p1[1] - h
    T_left_LL[0, 2] = -p2[0]
    T_left_LL[1, 2] = -p2[1] + h

    list_Ts = {}
    list_Ts["UL"] = [T_left_UL, T_right_UL]
    list_Ts["UR"] = [T_left_UR, T_right_UR]
    list_Ts["LR"] = [T_left_LR, T_right_LR]
    list_Ts["LL"] = [T_left_LL, T_right_LL]

    return list_Ts


def build_transformation(transformation_type, params):

    T_1 = np.eye(3)

    if transformation_type == Transformation.similarity:
        t_x = params["tr_x"]
        t_y = params["tr_y"]
        rotation_angle = params["angle_theta"]
        uniform_scale = params["scale_x"]
        center_rotation = params["center_rotation"]

        rot_mat = cv2.getRotationMatrix2D(
            center_rotation, rotation_angle, uniform_scale
        )

        T_1[0:2, :] = rot_mat[0:2, :]
        T_1[0, 2] += t_x
        T_1[1, 2] += t_y

    return T_1


def histogram_equalization(img):

    if len(img.shape) == 2:
        channel_0 = cv2.equalizeHist(img[:, :])

        img[:, :] = channel_0
    else:
        channel_0 = cv2.equalizeHist(img[:, :, 0])
        channel_1 = cv2.equalizeHist(img[:, :, 1])
        channel_2 = cv2.equalizeHist(img[:, :, 2])

        img[:, :, 0] = channel_0
        img[:, :, 1] = channel_1
        img[:, :, 2] = channel_2

    return img


def get_full_transformation(src, dst):

    A = []
    b = []

    for i, p1 in enumerate(src):

        p2 = dst[i]

        A.append([p1[0], p1[1], 1, 0, 0, 0, 0, 0, 0])
        b.append(p2[0])

        A.append([0, 0, 0, p1[0], p1[1], 1, 0, 0, 0])
        b.append(p2[1])

        A.append([0, 0, 0, 0, 0, 0, p1[0], p1[1], 1])
        b.append(1)

    A = np.array(A)
    b = np.array(b)

    res = lsq_linear(A, b)
    X = res.x

    T = X.reshape((3, 3))

    return T


def get_Similarity_Affine(src, dst):

    A = []
    b = []

    for i, p1 in enumerate(src):

        p2 = dst[i]

        A.append([p1[0], p1[1], 1, 0, 0, 0])
        b.append(p2[0])

        A.append([0, 0, 0, p1[0], p1[1], 1])
        b.append(p2[1])

    A = np.array(A)
    b = np.array(b)

    res = lsq_linear(A, b, tol=1e-10)
    X = res.x

    # X = np.matmul(np.matmul(np.linalg.inv(np.matmul(A.T,A)),A.T),b)

    T = X.reshape((2, 3))

    return T


def Jsonify(transformation_dict):

    jsonified = {}

    for img1 in transformation_dict:

        if img1 not in jsonified:
            jsonified[img1] = {}

        for img2 in transformation_dict[img1]:
            T = transformation_dict[img1][img2][0]
            matches = transformation_dict[img1][img2][1]
            perc_inlier = transformation_dict[img1][img2][2]
            inlier = transformation_dict[img1][img2][3]
            bins = transformation_dict[img1][img2][4]

            T = T.tolist()
            inlier = inlier.tolist()
            matches = pickle_matches(matches).tolist()

            if bins is not None:
                new_bins = {}
                for i in bins:
                    new_bins[i] = pickle_matches(bins[i]).tolist()

                bins = new_bins

            jsonified[img1][img2] = (T, matches, perc_inlier, inlier, bins)

    return jsonified


def Unjsonify(jsonified_dict):
    unjsonified = {}

    for img1 in jsonified_dict:

        if img1 not in unjsonified:
            unjsonified[img1] = {}

        for img2 in jsonified_dict[img1]:
            T = jsonified_dict[img1][img2][0]
            matches = jsonified_dict[img1][img2][1]
            perc_inlier = jsonified_dict[img1][img2][2]
            inlier = jsonified_dict[img1][img2][3]
            bins = jsonified_dict[img1][img2][4]

            T = np.array(T)
            inlier = np.array(inlier)
            matches = get_matches_from_pickled(np.array(matches))

            if bins is not None:
                new_bins = {}
                for i in bins:
                    new_bins[i] = get_matches_from_pickled(np.array(bins[i]))

                bins = new_bins

            unjsonified[img1][img2] = (T, matches, perc_inlier, inlier, bins)

    return unjsonified


def generate_n_distinc_colors(n):

    list_colors = [
        (160, 82, 45),
        (47, 79, 79),
        (0, 0, 128),
        (255, 69, 0),
        (0, 206, 209),
        (255, 255, 0),
        (199, 21, 133),
        (255, 0, 255),
        (240, 230, 140),
        (100, 149, 237),
        (255, 192, 203),
        (160, 82, 45),
        (47, 79, 79),
        (0, 0, 128),
        (255, 69, 0),
        (0, 206, 209),
        (255, 255, 0),
        (199, 21, 133),
        (255, 0, 255),
        (240, 230, 140),
        (100, 149, 237),
        (255, 192, 203),
        (160, 82, 45),
        (47, 79, 79),
        (0, 0, 128),
        (255, 69, 0),
        (0, 206, 209),
        (255, 255, 0),
        (199, 21, 133),
        (255, 0, 255),
        (240, 230, 140),
        (100, 149, 237),
        (255, 192, 203),
    ]

    return list_colors[:n]


def get_GCP_sizes(dataset, method):

    if dataset == "DLLFN" and method == "BNDL-ADJ":
        s1 = 30
        s2 = 40
    elif dataset == "DLLFN" and method == "MEGASTITCH-AFF-BNDL-ADJ-OLDIN":
        s1 = 40
        s2 = 50
    elif dataset == "DLLFN" and method == "MEGASTITCH-SIM":
        s1 = 40
        s2 = 50
    if dataset == "DSEFN" and method == "BNDL-ADJ":
        s1 = 30
        s2 = 40
    elif dataset == "DSEFN" and method == "MEGASTITCH-AFF-BNDL-ADJ-OLDIN":
        s1 = 30
        s2 = 40
    elif dataset == "DSEFN" and method == "MEGASTITCH-SIM":
        s1 = 40
        s2 = 50
    elif dataset == "GCD" or dataset == "GRG":
        s1 = 20
        s2 = 30
    else:
        s1 = 30
        s2 = 40

    return s1, s2

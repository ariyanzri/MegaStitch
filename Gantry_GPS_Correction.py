import numpy as np
import General_GPS_Correction
import datetime
import sys
import os
import computer_vision_utils as cv_util
import json
import multiprocessing
from osgeo import gdal

def get_patch_coord_dict_from_name(args):

    image_name = args[0]
    patch_folder = args[1]

    ds = gdal.Open('{0}/{1}'.format(patch_folder,image_name))
    
    meta = gdal.Info(ds)

    lines = meta.splitlines()

    for line in lines:
        if 'Upper Left' in line:
            u_l = line.split()[2:4]
            u_l[0] = u_l[0].replace('(','').replace(',','')
            u_l[1] = u_l[1].replace(')','')

        if 'Lower Left' in line:
            l_l = line.split()[2:4]
            l_l[0] = l_l[0].replace('(','').replace(',','')
            l_l[1] = l_l[1].replace(')','')

        if 'Upper Right' in line: 
            u_r = line.split()[2:4]
            u_r[0] = u_r[0].replace('(','').replace(',','')
            u_r[1] = u_r[1].replace(')','')

        if 'Lower Right' in line:
            l_r = line.split()[2:4]
            l_r[0] = l_r[0].replace('(','').replace(',','')
            l_r[1] = l_r[1].replace(')','')

        if 'Center' in line:
            c = line.split()[1:3]
            c[0] = c[0].replace('(','').replace(',','')
            c[1] = c[1].replace(')','')

    upper_left = (float(u_l[0]),float(u_l[1]))
    lower_left = (float(l_l[0]),float(l_l[1]))
    upper_right = (float(u_r[0]),float(u_r[1]))
    lower_right = (float(l_r[0]),float(l_r[1]))
    center = (float(c[0]),float(c[1]))
    
    coord = {'name':image_name,'UL':upper_left,'LL':lower_left,'UR':upper_right,'LR':lower_right,'C':center}

    return coord

def get_all_coords_from_TIFFs(patch_folder):

    image_names = os.listdir(patch_folder)

    args = []

    for img_name in image_names:

        if '_right' in img_name:
            continue
        
        args.append((img_name,patch_folder))

    processes = multiprocessing.Pool(General_GPS_Correction.settings.cores_to_use)
    results = processes.map(get_patch_coord_dict_from_name,args)
    processes.close()

    coords = {}

    for r in results:

        coords[r['name']] = {\
            'UL':{'lon':r['UL'][0],'lat':r['UL'][1]},\
            'UR':{'lon':r['UR'][0],'lat':r['UR'][1]},\
            'LL':{'lon':r['LL'][0],'lat':r['LL'][1]},\
            'LR':{'lon':r['LR'][0],'lat':r['LR'][1]},\
            'C':{'lon':r['C'][0],'lat':r['C'][1]}}

    return coords

def get_center_coords(coords):

    center_coords = {}

    for c in coords:
        center_coords[c] = {'lat':coords[c]['C']['lat'],'lon':coords[c]['C']['lon']}

    return center_coords

def get_anchors_from_json(path):
    
    with open(path, "r") as outfile:  
        anchors_dict = json.load(outfile)

    return anchors_dict

def revise_latitude(coords_dict):

    new_coords_dict = {}

    for img in coords_dict:
        img_coords = coords_dict[img]
        coord = {\
            'UL':[img_coords['UL'][0],-1*img_coords['UL'][1]],\
            'UR':[img_coords['UR'][0],-1*img_coords['UR'][1]],\
            'LL':[img_coords['LL'][0],-1*img_coords['LL'][1]],\
            'LR':[img_coords['LR'][0],-1*img_coords['LR'][1]]}

        new_coords_dict[img] = coord

    return new_coords_dict

def report_time(start,end):
    print('-----------------------------------------------------------')
    print('Start date time: {0}\nEnd date time: {1}\nTotal running time: {2}.'.format(start,end,end-start))

def modify_settings():
    General_GPS_Correction.settings.grid_w = 3
    General_GPS_Correction.settings.grid_h = 7
    General_GPS_Correction.settings.nearest_number = 8
    General_GPS_Correction.settings.transformation = cv_util.Transformation.translation
    General_GPS_Correction.settings.discard_transformation_perc_inlier = 0.1
    General_GPS_Correction.settings.scale = 0.2
    General_GPS_Correction.settings.percentage_next_neighbor = 0.8
    General_GPS_Correction.settings.equalize_histogram = True
    General_GPS_Correction.settings.cores_to_use = 50
    General_GPS_Correction.settings.sub_set_choosing = True
    General_GPS_Correction.settings.N_perc = 0.7
    General_GPS_Correction.settings.E_perc = 1
    General_GPS_Correction.settings.parallel_stitch = True
    General_GPS_Correction.settings.max_no_inliers = 20

def main():
    
    if len(sys.argv)==6:
        data_path = sys.argv[1]
        result_path = sys.argv[2]
        sift_dir = sys.argv[3]
        anchor_path = sys.argv[4]
        experiment_Id = sys.argv[5]

    else:
        data_path = '/home/ariyan/Desktop/Drone/small'
        result_path = '/home/ariyan/Desktop/Drone/test'
        experiment_Id = 'EXP_local'
        sift_dir = None

    result_path = '{0}/{1}'.format(result_path,experiment_Id)

    if not os.path.exists(result_path):
        os.mkdir(result_path)

    method = experiment_Id.split('_')[0]
    dataset_name = experiment_Id.split('_')[1]

    data_set_path = '{0}/IMDS_{1}'.format(data_path,dataset_name)
    coords_file = '{0}/{1}_coords.csv'.format(data_path,dataset_name[:3])
    ortho_path = '{0}/{1}_ortho.png'.format(result_path,experiment_Id)
    plot_path = '{0}/{1}_initial_GPS.png'.format(result_path,experiment_Id)
    corrected_coordinates_path = '{0}/{1}_corrected_coordinates.json'.format(result_path,experiment_Id)
    log_path = '{0}/{1}_log.txt'.format(result_path,experiment_Id)
    sift_path = '{0}/{1}'.format(sift_dir,dataset_name[:3])
    tr_path = '{0}/pairwise_transformations.json'.format(result_path)

    if not os.path.exists(sift_path):
        os.mkdir(sift_path)


    original = sys.stdout
    log_file = open(log_path, 'w')
    sys.stdout = log_file

    start_time = datetime.datetime.now()

    General_GPS_Correction.init_setting(data_set_path)
    modify_settings()

    gps_coords_dict = get_all_coords_from_TIFFs(data_set_path)
    center_coords = get_center_coords(gps_coords_dict)

    anchors_dict = get_anchors_from_json(anchor_path)

    field = General_GPS_Correction.Field(center_coords,sift_p=sift_path,tr_p=tr_path)

    gantry_scale = 0.02

    if method == 'BNDL':
        
        coords_dict = field.geo_correct_BundleAdjustment()
        ortho = field.generate_field_ortho(coords_dict,gantry_scale)

    elif method == 'MGRAPH':
        
        coords_dict = field.geo_correct_MGRAPH()
        ortho = field.generate_field_ortho(coords_dict,gantry_scale)

    elif method == 'MEGA3':

        coords_dict = field.geo_correct_MEGASTITCH()
        ortho = field.generate_field_ortho(coords_dict,gantry_scale)

    elif method == 'MEGAHMG':
        
        coords_dict = field.geo_correct_MEGASTITCH_Groups()
        ortho = field.generate_field_ortho(coords_dict,gantry_scale)

    elif method == 'MEGA3BNDL':
        
        coords_dict = field.geo_correct_MEGASTITCH()
        coords_dict = field.geo_correct_BundleAdjustment(coords_dict)
        ortho = field.generate_field_ortho(coords_dict,gantry_scale)

    elif method == 'MEGAHMGBNDL':
        
        coords_dict = field.geo_correct_MEGASTITCH_Groups()
        coords_dict = field.geo_correct_BundleAdjustment(coords_dict)
        ortho = field.generate_field_ortho(coords_dict,gantry_scale)

    elif method == 'MEGAGPS':
        
        coords_dict = field.geo_correct_MEGASTITCH_BNDL_Gantry_specific(gps_coords_dict,anchors_dict[dataset_name[:3]])
        ortho = field.generate_field_ortho(coords_dict,gantry_scale,gps_coords_dict)
        coords_dict = revise_latitude(coords_dict)

    elif method == 'MEGASTITCH-TR-KP':
        General_GPS_Correction.settings.parallel_stitch = False
        General_GPS_Correction.settings.transformation = cv_util.Transformation.translation
        coords_dict = field.geo_correct_Translation_KP_based(gps_coords_dict,anchors_dict[dataset_name[:3]])
        ortho = field.generate_field_ortho(coords_dict,gantry_scale,gps_coords_dict)
        coords_dict = revise_latitude(coords_dict)

    elif method == 'MEGASTITCH-TR-PR':
        General_GPS_Correction.settings.parallel_stitch = False
        General_GPS_Correction.settings.transformation = cv_util.Transformation.translation
        coords_dict = field.geo_correct_Translation_Parameter_based(gps_coords_dict,anchors_dict[dataset_name[:3]])
        ortho = field.generate_field_ortho(coords_dict,gantry_scale,gps_coords_dict)
        coords_dict = revise_latitude(coords_dict)
    
    field.save_field_centers_visualization(plot_path)
    field.save_field_coordinates(corrected_coordinates_path,coords_dict)
    field.save_field_ortho(ortho,ortho_path)

    end_time = datetime.datetime.now()
    report_time(start_time,end_time)

    sys.stdout = original
    log_file.close()


main()



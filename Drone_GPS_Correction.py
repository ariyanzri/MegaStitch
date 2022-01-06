from numpy.lib.function_base import trapz
import General_GPS_Correction
import datetime
import sys
import os
import json
import computer_vision_utils as cv_util

def report_time(start,end):
	print('-----------------------------------------------------------')
	print('Start date time: {0}\nEnd date time: {1}\nTotal running time: {2}.'.format(start,end,end-start))

def get_anchors_from_json(path):
	
	with open(path, "r") as outfile:  
		anchors_dict = json.load(outfile)

	return anchors_dict

def save_json_results(json_path,res,experiment_Id):

	with open(json_path, "r") as outfile:  
		current_results = json.load(outfile)

	if General_GPS_Correction.settings.do_cross_validation:

		current_results[experiment_Id]={"GCP_RMSE":res[0],"Proj_RMSE":res[1],"Proj_RMSE":res[2],"Time":str(res[3]),\
			"CV_GCP_RMSE_AVG":res[4],"CV_GCP_RMSE_STD":res[5],\
			"CV_Proj_RMSE_AVG":res[6],"CV_Proj_RMSE_STD":res[7],\
			"CV_Proj_RMSE_Norm_AVG":res[8],"CV_Proj_RMSE_Norm_STD":res[9],\
			"CV_Time_AVG":str(res[10])}
	else:

		current_results[experiment_Id]={"GCP_RMSE":res[0],"Proj_RMSE":res[1],"Proj_RMSE_Norm":res[2],"Time":str(res[3])}

	with open(json_path, 'w') as outfile:
		json.dump(current_results, outfile)
	
def save_json_sim_gcps(json_path,list_gcps,dataset_name):
	
	with open(json_path, "r") as outfile:  
		current_results = json.load(outfile)

	current_results[dataset_name] = list_gcps

	with open(json_path, 'w') as outfile:
		json.dump(current_results, outfile)

def load_json_sim_gcps(json_path,dataset_name):
	
	with open(json_path, "r") as outfile:  
		current_results = json.load(outfile)
	
	return current_results[dataset_name]

def modify_settings():
	General_GPS_Correction.settings.grid_w = 3
	General_GPS_Correction.settings.grid_h = 7
	General_GPS_Correction.settings.scale = 0.2
	General_GPS_Correction.settings.nearest_number = 4
	General_GPS_Correction.settings.discard_transformation_perc_inlier = 0.4
	General_GPS_Correction.settings.transformation = cv_util.Transformation.similarity
	General_GPS_Correction.settings.percentage_next_neighbor = 0.6
	General_GPS_Correction.settings.cores_to_use = 55
	General_GPS_Correction.settings.draw_GCPs = True
	General_GPS_Correction.settings.max_no_inliers = 5
	General_GPS_Correction.settings.number_bins = 5
	General_GPS_Correction.settings.size_bins = 5
	General_GPS_Correction.settings.do_cross_validation = False
	General_GPS_Correction.settings.sub_set_choosing = False
	General_GPS_Correction.settings.N_perc = 0.5
	General_GPS_Correction.settings.E_perc = 0.5

def main():
	
	if len(sys.argv)==6:
		data_path = sys.argv[1]
		result_path = sys.argv[2]
		sift_dir = sys.argv[3]
		anchor_path = sys.argv[4]
		experiment_Id = sys.argv[5]

		main_result_path = result_path

		result_path = '{0}/{1}'.format(result_path,experiment_Id)

		if not os.path.exists(result_path):
			os.mkdir(result_path)

		method = experiment_Id.split('_')[0]
		dataset_name = experiment_Id.split('_')[1]

		data_set_path = '{0}/IMDS_{1}'.format(data_path,dataset_name)
		ortho_path = '{0}/{1}_ortho.png'.format(result_path,experiment_Id)
		plot_path = '{0}/{1}_initial_GPS.png'.format(result_path,experiment_Id)
		corrected_coordinates_path = '{0}/{1}_corrected_coordinates.json'.format(result_path,experiment_Id)
		log_path = '{0}/{1}_log.txt'.format(result_path,experiment_Id)
		sift_path = '{0}/{1}'.format(sift_dir,dataset_name[:3])
		json_path = '{0}/results.json'.format(main_result_path)
		sim_GCPs_path = '{0}/sim_GCPs.json'.format(main_result_path)
		tr_path = '{0}/pairwise_transformations.json'.format(result_path)

		if not os.path.exists(sift_path):
			os.mkdir(sift_path)

		General_GPS_Correction.init_setting(data_set_path)

		General_GPS_Correction.settings.Dataset = dataset_name
		General_GPS_Correction.settings.Method = method
		General_GPS_Correction.settings.AllGCPRMSE = True
		
		modify_settings()

	else:
		data_path = '/home/ariyan/Desktop/Drone/small'
		result_path = '/home/ariyan/Desktop/Drone/test'
		experiment_Id = 'EXP_local'
		sift_dir = '/home/ariyan/Desktop/Drone/SIFT'

		result_path = '{0}/{1}'.format(result_path,experiment_Id)

		if not os.path.exists(result_path):
			os.mkdir(result_path)

		method = 'BNDL'
		 
		data_set_path = data_path
		ortho_path = '{0}/{1}_ortho.png'.format(result_path,experiment_Id)
		plot_path = '{0}/{1}_initial_GPS.png'.format(result_path,experiment_Id)
		corrected_coordinates_path = '{0}/{1}_corrected_coordinates.json'.format(result_path,experiment_Id)
		log_path = '{0}/{1}_log.txt'.format(result_path,experiment_Id)
		sift_path = '{0}'.format(sift_dir)

		General_GPS_Correction.init_setting(data_set_path)
		modify_settings()
		
		General_GPS_Correction.settings.cores_to_use=2
		General_GPS_Correction.settings.transformation = cv_util.Transformation.translation
		General_GPS_Correction.settings.discard_transformation_perc_inlier = 0.0

	original = sys.stdout
	log_file = open(log_path, 'w')
	sys.stdout = log_file

	start_time = datetime.datetime.now()

	anchors_dict = get_anchors_from_json(anchor_path)

	field = General_GPS_Correction.Field(sift_p=sift_path,tr_p=tr_path)
	H = None

	if method == 'BNDL':
		coords_dict = field.geo_correct_BundleAdjustment()
	elif method == 'MGRAPH':
		coords_dict = field.geo_correct_MGRAPH()
	elif method == 'MEGA3':
		coords_dict = field.geo_correct_MEGASTITCH()
	elif method == 'MEGAHMG':
		coords_dict = field.geo_correct_MEGASTITCH_Groups()
	elif method == 'MEGA3BNDL':
		coords_dict = field.geo_correct_MEGASTITCH()
		coords_dict = field.geo_correct_BundleAdjustment(coords_dict)
	elif method == 'MEGAHMGBNDL':
		coords_dict = field.geo_correct_MEGASTITCH_Groups()
		coords_dict = field.geo_correct_BundleAdjustment(coords_dict)
	elif method == 'MEGAH':
		coords_dict = field.geo_correct_MegaStitch_Homography()
	elif method == 'MEGASTITCH-SIM':
		General_GPS_Correction.settings.transformation = cv_util.Transformation.similarity
		coords_dict,H,H_inv,abs_tr,res,Sim_GCPs,all_GCPs = field.geo_correct_MegaStitchSimilarity(anchors_dict)
		if dataset_name != "ODFN":
			save_json_sim_gcps(sim_GCPs_path,Sim_GCPs,dataset_name)
	elif method == 'MEGASTITCH-AFF':
		Sim_GCPs = load_json_sim_gcps(sim_GCPs_path,dataset_name)
		General_GPS_Correction.settings.transformation = cv_util.Transformation.affine
		coords_dict,H,H_inv,abs_tr,res,all_GCPs = field.geo_correct_MegaStitchAffine(anchors_dict,Sim_GCPs)
	elif method == 'BNDL-ADJ':
		if dataset_name != "ODFN":
			Sim_GCPs = load_json_sim_gcps(sim_GCPs_path,dataset_name)
		else:
			Sim_GCPs=None
		General_GPS_Correction.settings.transformation = cv_util.Transformation.homography
		coords_dict,H,H_inv,abs_tr,res,all_GCPs = field.geo_correct_BundleAdjustment_Homography(anchors_dict,Sim_GCPs)
	elif method == 'MEGASTITCH-AFF-BNDL-ADJ':
		Sim_GCPs = load_json_sim_gcps(sim_GCPs_path,dataset_name)
		coords_dict,H,H_inv,abs_tr,res,all_GCPs = field.geo_correct_MegaStitch_Affine_Bundle_Adjustment_Homography(anchors_dict,Sim_GCPs)
	elif method == 'MEGASTITCH-SIM-BNDL-ADJ':
		Sim_GCPs = load_json_sim_gcps(sim_GCPs_path,dataset_name)
		coords_dict,H,H_inv,abs_tr,res,all_GCPs = field.geo_correct_MegaStitch_Similarity_Bundle_Adjustment_Homography(anchors_dict,Sim_GCPs)
	elif method == 'MEGASTITCH-AFF-BNDL-ADJ-OLDIN':
		if dataset_name != "ODFN":
			Sim_GCPs = load_json_sim_gcps(sim_GCPs_path,dataset_name)
		else:
			Sim_GCPs=None
		coords_dict,H,H_inv,abs_tr,res,all_GCPs = field.geo_correct_MegaStitch_Affine_Bundle_Adjustment_Homography(anchors_dict,Sim_GCPs,use_old_inliers=True)
	elif method == 'MEGASTITCH-SIM-BNDL-ADJ-OLDIN':
		Sim_GCPs = load_json_sim_gcps(sim_GCPs_path,dataset_name)
		coords_dict,H,H_inv,abs_tr,res,all_GCPs = field.geo_correct_MegaStitch_Similarity_Bundle_Adjustment_Homography(anchors_dict,Sim_GCPs,use_old_inliers=True)
	
	save_json_results(json_path,res,experiment_Id)

	if H is None:
		gcp_inf = None
	else:
		gcp_inf = (anchors_dict,H_inv,abs_tr)
		
	field.generate_transformation_accuracy_histogram(coords_dict,plot_path.replace('initial_GPS','transformation_plot'))
	field.save_field_centers_visualization(plot_path)

	ortho = field.generate_field_ortho(coords_dict,gcp_info=gcp_inf)
	
	field.save_field_ortho(ortho,ortho_path)
	field.save_field_coordinates(corrected_coordinates_path,coords_dict)


	end_time = datetime.datetime.now()

	report_time(start_time,end_time)

	sys.stdout = original
	log_file.close()


	




main()

import numpy as np
import multiprocessing
import sys
from Customized_myltiprocessing import MyPool


def Estimate_Translation_Base(train_point,query_point):

	res = np.eye(3)
	res[0,2] = query_point[0] - train_point[0]
	res[1,2] = query_point[1] - train_point[1]

	return res

def Estimate_Translation_All_Inliers(train_points,query_points):

	translation_params = np.mean(query_points-train_points,axis=0)

	res = np.eye(3)
	res[0,2] = translation_params[0]
	res[1,2] = translation_params[1]

	return res

def Calculate_Fit_Error_XY_MSE(train_points,query_points,model):

	train_points_new = np.append(train_points,np.ones((np.shape(train_points)[0],1)),axis=1)
	
	query_points_new = np.matmul(model,train_points_new.T).T
	
	query_points_new = np.delete(query_points_new, -1, axis=1)
	
	errors = (query_points-query_points_new)**2

	return np.mean(errors,axis=0)

def Calculate_Prior_Error_Squared(model,prior_means):

	return (abs(model[0,2])-prior_means[0])**2,(abs(model[1,2])-prior_means[1])**2

def Calculate_Full_Error(fit_errors,fit_error_vars,prior_errors,prior_error_vars):
	
	return fit_errors[0]/fit_error_vars[0]+fit_errors[1]/fit_error_vars[1] + prior_errors[0]/prior_error_vars[0]+prior_errors[1]+prior_error_vars[1]

def Get_Remaining_Inliers(train_points,query_points,initial_sample_indixes,base_model,inlier_error_threshold):

	remaining_tr = np.delete(train_points,initial_sample_indixes,axis=0)
	remaining_qu = np.delete(query_points,initial_sample_indixes,axis=0)

	remaining_tr = np.append(remaining_tr,np.ones((np.shape(remaining_tr)[0],1)),axis=1)
	
	new_qu = np.matmul(base_model,remaining_tr.T).T
	
	new_qu = np.delete(new_qu, -1, axis=1)
	
	errors = np.sqrt(np.sum((remaining_qu-new_qu)**2,axis=1))

	inlier_indexes = np.argwhere(errors<inlier_error_threshold)


	if inlier_indexes.shape[0] > 1:
		return np.squeeze(inlier_indexes)
	else:
		return inlier_indexes

def Translate_RANSAC_Estimator_Sequential(train_points,query_points,max_allowed_iteration,inlier_error_threshold,number_inliers_threshold,prior_means,prior_error_vars,fit_error_vars):

	models_list = []

	best_model = None
	best_error = sys.maxsize
	best_inliers = None
	best_iter = -1

	for iteration in range(max_allowed_iteration):

		sample_indexes = np.random.choice(train_points.shape[0], 1)

		tr_samples = train_points[sample_indexes, :]
		qu_samples = query_points[sample_indexes, :]

		base_model = Estimate_Translation_Base(tr_samples[0],qu_samples[0])
		model_sample_inliers = []

		remaining_inliers_indexes = Get_Remaining_Inliers(train_points,query_points,sample_indexes,base_model,inlier_error_threshold)
		
		if len(remaining_inliers_indexes) > number_inliers_threshold:

			all_inliers_indexes = np.concatenate((remaining_inliers_indexes,sample_indexes),axis=0)
			# print(all_inliers_indexes)

			model_all_inliers = Estimate_Translation_All_Inliers(train_points[all_inliers_indexes,:],query_points[all_inliers_indexes,:])

			fit_errors = Calculate_Fit_Error_XY_MSE(train_points[all_inliers_indexes,:],query_points[all_inliers_indexes,:],model_all_inliers)

			if prior_means is not None:

				prior_errors = Calculate_Prior_Error_Squared(model_all_inliers,prior_means)

				full_normal_error = Calculate_Full_Error(fit_errors,fit_error_vars,prior_errors,prior_error_vars)

			else:

				full_normal_error = fit_errors[0]+fit_errors[1]

			models_list.append((model_all_inliers,full_normal_error,all_inliers_indexes))

			if full_normal_error < best_error:
				best_model = model_all_inliers
				best_error = full_normal_error
				best_inliers = all_inliers_indexes
				best_iter = iteration


	if best_inliers is None or train_points is None or len(best_inliers) == 0 or len(train_points) == 0:
		return None, None, 0, sys.maxsize, -1

	return best_model,best_inliers,best_inliers.shape[0]/train_points.shape[0],best_error,best_iter

def RANSAC_Parallel_Helper(args):

	tr_samples = args[2]
	qu_samples = args[3]
	train_points = args[0]
	query_points = args[1]
	sample_indexes = args[4]
	inlier_error_threshold = args[5]
	number_inliers_threshold = args[6]
	prior_means = args[7]
	fit_error_vars = args[8]
	prior_error_vars = args[9]

	base_model = Estimate_Translation_Base(tr_samples[0],qu_samples[0])
	model_sample_inliers = []

	remaining_inliers_indexes = Get_Remaining_Inliers(train_points,query_points,sample_indexes,base_model,inlier_error_threshold)
	
	if len(remaining_inliers_indexes) > number_inliers_threshold:

		all_inliers_indexes = np.concatenate((remaining_inliers_indexes,sample_indexes),axis=0)

		model_all_inliers = Estimate_Translation_All_Inliers(train_points[all_inliers_indexes,:],query_points[all_inliers_indexes,:])

		fit_errors = Calculate_Fit_Error_XY_MSE(train_points[all_inliers_indexes,:],query_points[all_inliers_indexes,:],model_all_inliers)

		if prior_means is not None:

			prior_errors = Calculate_Prior_Error_Squared(model_all_inliers,prior_means)

			full_normal_error = Calculate_Full_Error(fit_errors,fit_error_vars,prior_errors,prior_error_vars)

		else:

			full_normal_error = fit_errors[0]+fit_errors[1]

		return model_all_inliers,full_normal_error,all_inliers_indexes

	else:

		return None, None, None
		
def Translate_RANSAC_Estimator_Parallel(train_points,query_points,max_allowed_iteration,inlier_error_threshold,number_inliers_threshold,prior_means,prior_error_vars,fit_error_vars,cores):

	models_list = []

	best_model = None
	best_error = sys.maxsize
	best_inliers = None
	
	args = []

	for iteration in range(max_allowed_iteration):

		sample_indexes = np.random.choice(train_points.shape[0], 1)

		tr_samples = train_points[sample_indexes, :]
		qu_samples = query_points[sample_indexes, :]

		args.append((train_points,query_points,tr_samples,qu_samples,sample_indexes,inlier_error_threshold,number_inliers_threshold,prior_means,\
			fit_error_vars,prior_error_vars))

	processes = MyPool(cores)
	results = processes.map(RANSAC_Parallel_Helper,args)
	processes.close()

	for model,error,inliers in results:
		
		if model is None:
			continue			

		if error<best_error:
			best_error = error
			best_model = model
			best_inliers = inliers

	if best_model is None or best_inliers is None or train_points is None or len(best_inliers) == 0 or len(train_points) == 0:
		
		return None, None, 0, sys.maxsize, -1

	return best_model,best_inliers,best_inliers.shape[0]/train_points.shape[0],best_error,-1

def estimateTranslation(train_points,query_points,cores):

	if cores == 1:

		best_model,inliers,perc_inliers,best_error,best_iter = Translate_RANSAC_Estimator_Sequential(train_points,query_points,700,5,5,None,None,None)

	else:

		best_model,inliers,perc_inliers,best_error,best_iter = Translate_RANSAC_Estimator_Parallel(train_points,query_points,700,5,5,None,None,None,cores)

	if best_model is None:
		return None, None

	masked = np.zeros((train_points.shape[0],1))

	for i in inliers:
		masked[i,0] = 1

	return best_model,masked

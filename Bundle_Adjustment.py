import numpy as np
import math
import computer_vision_utils as cv_util
import multiprocessing
import random
import sys

from scipy.sparse.linalg import lsmr
from scipy.optimize import lsq_linear

from scipy.optimize import leastsq,least_squares

class Bundle_Adjustment:

	def __init__(self,images,pairwise_trans,coords,coefs,tr_type,w,h,norm,ref_name,max_mat):
		
		self.images = images
		self.images_dict = {}
		self.n_to_i_dict = {}

		for i,img in enumerate(self.images):
			self.images_dict[img.name] = img
			self.n_to_i_dict[img.name] = i

		self.num_images = len(self.images)

		self.pairwise_transformations = pairwise_trans
		self.image_reference_name = ref_name
		self.transformation_type = tr_type

		self.absolute_transformations = self.get_absolute_from_coords(coords)
		
		self.width = w
		self.height = h

		self.coefs = coefs
		self.is_normalized = norm

		self.max_matches_to_use = max_mat
		

	def get_absolute_from_coords(self,coords):

		absolute_transformations = {}

		if coords is None:
			return None
		elif coords == 'rand':

			for img_name in self.n_to_i_dict:
				absolute_transformations[img_name] = np.random.rand(3,3)

			return absolute_transformations
		elif type(coords[self.image_reference_name]) == np.ndarray:
			return coords


		absolute_transformations[self.image_reference_name] = np.eye(3)

		for img_name in self.n_to_i_dict:

			if img_name == self.image_reference_name:
				continue

			pts1 = np.float32([[coords[img_name]['UL'][0],coords[img_name]['UL'][1]],\
			[coords[img_name]['UR'][0],coords[img_name]['UR'][1]],\
			[coords[img_name]['LR'][0],coords[img_name]['LR'][1]],\
			[coords[img_name]['LL'][0],coords[img_name]['LL'][1]]])

			pts2 = np.float32([[coords[self.image_reference_name]['UL'][0],coords[self.image_reference_name]['UL'][1]],\
			[coords[self.image_reference_name]['UR'][0],coords[self.image_reference_name]['UR'][1]],\
			[coords[self.image_reference_name]['LR'][0],coords[self.image_reference_name]['LR'][1]],\
			[coords[self.image_reference_name]['LL'][0],coords[self.image_reference_name]['LL'][1]]])

			T = cv_util.estimate_base_transformations(pts1,pts2,self.transformation_type)

			absolute_transformations[img_name] = np.linalg.inv(T)

		return absolute_transformations

	def solve(self,use_homogenouse=False):
		
		# print('Started optimization')
		res = None

		if self.transformation_type == cv_util.Transformation.translation:
			res = self.solve_translation() 
		elif self.transformation_type == cv_util.Transformation.similarity:
			res = self.solve_similarity_affine() 
		elif self.transformation_type == cv_util.Transformation.affine:
			res = self.solve_similarity_affine()
		elif self.transformation_type == cv_util.Transformation.homography:
			res = self.solve_homography()

		# print('Finished optimization')

		return res

	def get_num_rows(self):

		n = 0

		for img_A_name in self.pairwise_transformations:
			for img_B_name in self.pairwise_transformations[img_A_name]:
				inliers = self.pairwise_transformations[img_A_name][img_B_name][3]
				num_inliers = int(np.sum(inliers))
				n+= 2*min(num_inliers,self.max_matches_to_use)

		return n

	def solve_translation(self):

		V = np.eye(4*self.num_images)

		n = self.get_num_rows()

		A = np.zeros((n,4*self.num_images))
		b = np.zeros(n)

		lower_bounds = [-np.inf]*(4*self.num_images)
		upper_bounds = [np.inf]*(4*self.num_images)
		margin = 1e-5

		tr_c = self.coefs[0]
		co_c = self.coefs[1]

		off_transform_penalty = 100*self.num_images
		# off_transform_penalty = 0.1

		row_it = 0

		for img_A_name in self.pairwise_transformations:

			T_A_11 = 4*self.n_to_i_dict[img_A_name]
			T_A_13 = 4*self.n_to_i_dict[img_A_name]+1
			T_A_22 = 4*self.n_to_i_dict[img_A_name]+2
			T_A_23 = 4*self.n_to_i_dict[img_A_name]+3

			for img_B_name in self.pairwise_transformations[img_A_name]:
				
				T_B_11 = 4*self.n_to_i_dict[img_B_name]
				T_B_13 = 4*self.n_to_i_dict[img_B_name]+1
				T_B_22 = 4*self.n_to_i_dict[img_B_name]+2
				T_B_23 = 4*self.n_to_i_dict[img_B_name]+3

				matches = self.pairwise_transformations[img_A_name][img_B_name][1]
				inliers = self.pairwise_transformations[img_A_name][img_B_name][3]

				inlier_counter =0

				for i,m in enumerate(matches):

					if inliers[i,0] == 0:
						continue

					kp_A = self.images_dict[img_A_name].kp[m.trainIdx]
					kp_B = self.images_dict[img_B_name].kp[m.queryIdx]

					p_A = (kp_A.pt[0],kp_A.pt[1])
					p_B = (kp_B.pt[0],kp_B.pt[1])

					row = tr_c*(V[T_A_11]*p_A[0] + V[T_A_13]*1 - (V[T_B_11]*p_B[0] + V[T_B_13]*1))
					# A.append(row)
					# b.append(0)
					A[row_it,:] = row
					row_it+=1

					row = tr_c*(V[T_A_22]*p_A[1] + V[T_A_23]*1 - (V[T_B_22]*p_B[1] + V[T_B_23]*1))
					# A.append(row)
					# b.append(0)
					A[row_it,:] = row
					row_it+=1

					inlier_counter+=1

					if inlier_counter>=self.max_matches_to_use:
						break


		for img_A_name in self.n_to_i_dict:

			T_A_11 = 4*self.n_to_i_dict[img_A_name]
			T_A_13 = 4*self.n_to_i_dict[img_A_name]+1
			T_A_22 = 4*self.n_to_i_dict[img_A_name]+2
			T_A_23 = 4*self.n_to_i_dict[img_A_name]+3

			if img_A_name == self.image_reference_name:

				# A.append(off_transform_penalty*tr_c*(V[T_A_11]))
				# b.append(off_transform_penalty*1)

				# A.append(off_transform_penalty*tr_c*(V[T_A_13]))
				# b.append(off_transform_penalty*0)

				# A.append(off_transform_penalty*tr_c*(V[T_A_22]))
				# b.append(off_transform_penalty*1)

				# A.append(off_transform_penalty*tr_c*(V[T_A_23]))
				# b.append(off_transform_penalty*0)

				upper_bounds[T_A_11]=1+margin
				lower_bounds[T_A_11]=1-margin

				upper_bounds[T_A_13]=0+margin
				lower_bounds[T_A_13]=0-margin

				upper_bounds[T_A_22]=1+margin
				lower_bounds[T_A_22]=1-margin

				upper_bounds[T_A_23]=0+margin
				lower_bounds[T_A_23]=0-margin

			else:

				# A.append(off_transform_penalty*tr_c*(V[T_A_11]))
				# b.append(off_transform_penalty*1)

				# A.append(off_transform_penalty*tr_c*(V[T_A_22]))
				# b.append(off_transform_penalty*1)

				upper_bounds[T_A_11]=1+margin
				lower_bounds[T_A_11]=1-margin

				upper_bounds[T_A_22]=1+margin
				lower_bounds[T_A_22]=1-margin

				if self.absolute_transformations is not None:

					T_init = self.absolute_transformations[img_A_name]

					# row = co_c*(V[T_A_13])
					# A.append(row)
					# b.append(co_c*T_init[0,2])

					# row = co_c*(V[T_A_23])
					# A.append(row)
					# b.append(co_c*T_init[1,2])

					upper_bounds[T_A_13]=T_init[0,2]+margin
					lower_bounds[T_A_13]=T_init[0,2]-margin

					upper_bounds[T_A_23]=T_init[1,2]+margin
					lower_bounds[T_A_23]=T_init[1,2]-margin


		# A = np.array(A)
		# b = np.array(b)

		# res = lsq_linear(A, b, max_iter=4*self.num_images,verbose=0)
		res = lsq_linear(A, b, bounds=(lower_bounds,upper_bounds), max_iter=4*self.num_images,verbose=0)

		X = res.x

		residuals = res.fun
		
		# print('>>>\t Final RMSE of residuals at this step: {0}'.format(np.sqrt(np.mean(residuals**2))))
		print('>>>\t {0}'.format(np.sqrt(np.mean(residuals**2))))

		image_corners_dict = {}
		absolute_transformations_dict = {}

		for img_name in self.n_to_i_dict:

			T_A_11 = 4*self.n_to_i_dict[img_name]
			T_A_13 = 4*self.n_to_i_dict[img_name]+1
			T_A_22 = 4*self.n_to_i_dict[img_name]+2
			T_A_23 = 4*self.n_to_i_dict[img_name]+3

			T = np.eye(3)
			T[0,0] = X[T_A_11]
			T[0,1] = 0
			T[0,2] = X[T_A_13]
			T[1,0] = 0
			T[1,1] = X[T_A_22]
			T[1,2] = X[T_A_23]

			UL_ref = [0,0,1]
			UR_ref = [self.width,0,1]
			LR_ref = [self.width,self.height,1]
			LL_ref = [0,self.height,1]

			UL = np.matmul(T,UL_ref)
			UL = UL/UL[2]
			UL = UL[:2]

			UR = np.matmul(T,UR_ref)
			UR = UR/UR[2]
			UR = UR[:2]

			LR = np.matmul(T,LR_ref)
			LR = LR/LR[2]
			LR = LR[:2]

			LL = np.matmul(T,LL_ref)
			LL = LL/LL[2]
			LL = LL[:2]

			image_corners_dict[img_name] = {'UL':UL,'UR':UR,'LR':LR,'LL':LL}
			absolute_transformations_dict[img_name] = T

		return image_corners_dict,absolute_transformations_dict

		# V = np.eye(2*self.num_images)

		# A = []
		# b = []

		# lower_bounds = [-np.inf]*(2*self.num_images)
		# upper_bounds = [np.inf]*(2*self.num_images)
		# margin = 1e-5

		# tr_c = self.coefs[0]
		# co_c = self.coefs[1]

		# off_transform_penalty = 100*self.num_images
		# # off_transform_penalty = 0.1

		# for img_A_name in self.pairwise_transformations:

		#	 T_A_13 = 2*self.n_to_i_dict[img_A_name]
		#	 T_A_23 = 2*self.n_to_i_dict[img_A_name]+1

		#	 for img_B_name in self.pairwise_transformations[img_A_name]:
				
		#		 T_B_13 = 2*self.n_to_i_dict[img_B_name]
		#		 T_B_23 = 2*self.n_to_i_dict[img_B_name]+1
				

		#		 matches = self.pairwise_transformations[img_A_name][img_B_name][1]
		#		 inliers = self.pairwise_transformations[img_A_name][img_B_name][3]

		#		 for i,m in enumerate(matches):

		#			 if inliers[i,0] == 0:
		#				 continue

		#			 kp_A = self.images_dict[img_A_name].kp[m.trainIdx]
		#			 kp_B = self.images_dict[img_B_name].kp[m.queryIdx]

		#			 p_A = (kp_A.pt[0],kp_A.pt[1])
		#			 p_B = (kp_B.pt[0],kp_B.pt[1])

		#			 row = tr_c*(V[T_A_13] - V[T_B_13])
		#			 A.append(row)
		#			 b.append(tr_c*(p_A[0]-p_B[0]))

		#			 row = tr_c*(V[T_A_23] - V[T_B_23])
		#			 A.append(row)
		#			 b.append(tr_c*(p_A[1]-p_B[1]))


		# for img_A_name in self.n_to_i_dict:

		#	 T_A_13 = 2*self.n_to_i_dict[img_A_name]
		#	 T_A_23 = 2*self.n_to_i_dict[img_A_name]+1

		#	 if img_A_name == self.image_reference_name:

		#		 # A.append(off_transform_penalty*tr_c*(V[T_A_11]))
		#		 # b.append(off_transform_penalty*1)

		#		 # A.append(off_transform_penalty*tr_c*(V[T_A_13]))
		#		 # b.append(off_transform_penalty*0)

		#		 # A.append(off_transform_penalty*tr_c*(V[T_A_22]))
		#		 # b.append(off_transform_penalty*1)

		#		 # A.append(off_transform_penalty*tr_c*(V[T_A_23]))
		#		 # b.append(off_transform_penalty*0)

		#		 upper_bounds[T_A_13]=0+margin
		#		 lower_bounds[T_A_13]=0-margin

		#		 upper_bounds[T_A_23]=0+margin
		#		 lower_bounds[T_A_23]=0-margin

		#	 else:

		#		 # A.append(off_transform_penalty*tr_c*(V[T_A_11]))
		#		 # b.append(off_transform_penalty*1)

		#		 # A.append(off_transform_penalty*tr_c*(V[T_A_22]))
		#		 # b.append(off_transform_penalty*1)

		#		 if self.absolute_transformations is not None:

		#			 T_init = self.absolute_transformations[img_A_name]

		#			 # row = co_c*(V[T_A_13])
		#			 # A.append(row)
		#			 # b.append(co_c*T_init[0,2])

		#			 # row = co_c*(V[T_A_23])
		#			 # A.append(row)
		#			 # b.append(co_c*T_init[1,2])

		#			 upper_bounds[T_A_13]=T_init[0,2]+margin
		#			 lower_bounds[T_A_13]=T_init[0,2]-margin

		#			 upper_bounds[T_A_23]=T_init[1,2]+margin
		#			 lower_bounds[T_A_23]=T_init[1,2]-margin



		# A = np.array(A)
		# b = np.array(b)

		# # res = lsq_linear(A, b, max_iter=4*self.num_images,verbose=0)
		# res = lsq_linear(A, b, bounds=(lower_bounds,upper_bounds), max_iter=4*self.num_images,verbose=0)

		# X = res.x

		# residuals = res.fun
		
		# # print('>>>\t Final RMSE of residuals at this step: {0}'.format(np.sqrt(np.mean(residuals**2))))
		# print('>>>\t {0}'.format(np.sqrt(np.mean(residuals**2))))

		# image_corners_dict = {}
		# absolute_transformations_dict = {}

		# for img_name in self.n_to_i_dict:

		#	 T_A_13 = 2*self.n_to_i_dict[img_name]
		#	 T_A_23 = 2*self.n_to_i_dict[img_name]+1

		#	 T = np.eye(3)
		#	 T[0,0] = 1
		#	 T[0,1] = 0
		#	 T[0,2] = X[T_A_13]
		#	 T[1,0] = 0
		#	 T[1,1] = 1
		#	 T[1,2] = X[T_A_23]

		#	 UL_ref = [0,0,1]
		#	 UR_ref = [self.width,0,1]
		#	 LR_ref = [self.width,self.height,1]
		#	 LL_ref = [0,self.height,1]

		#	 UL = np.matmul(T,UL_ref)
		#	 UL = UL/UL[2]
		#	 UL = UL[:2]

		#	 UR = np.matmul(T,UR_ref)
		#	 UR = UR/UR[2]
		#	 UR = UR[:2]

		#	 LR = np.matmul(T,LR_ref)
		#	 LR = LR/LR[2]
		#	 LR = LR[:2]

		#	 LL = np.matmul(T,LL_ref)
		#	 LL = LL/LL[2]
		#	 LL = LL[:2]

		#	 image_corners_dict[img_name] = {'UL':UL,'UR':UR,'LR':LR,'LL':LL}
		#	 absolute_transformations_dict[img_name] = T

		# return image_corners_dict,absolute_transformations_dict

	def solve_similarity_affine(self):

		V = np.eye(6*self.num_images)

		A = []
		b = []

		tr_c = self.coefs[0]
		co_c = self.coefs[1]

		off_transform_penalty = 6*self.num_images

		for img_A_name in self.pairwise_transformations:

			T_A_11 = 6*self.n_to_i_dict[img_A_name]
			T_A_12 = 6*self.n_to_i_dict[img_A_name]+1
			T_A_13 = 6*self.n_to_i_dict[img_A_name]+2
			T_A_21 = 6*self.n_to_i_dict[img_A_name]+3
			T_A_22 = 6*self.n_to_i_dict[img_A_name]+4
			T_A_23 = 6*self.n_to_i_dict[img_A_name]+5

			for img_B_name in self.pairwise_transformations[img_A_name]:
		
				T_B_11 = 6*self.n_to_i_dict[img_B_name]
				T_B_12 = 6*self.n_to_i_dict[img_B_name]+1
				T_B_13 = 6*self.n_to_i_dict[img_B_name]+2
				T_B_21 = 6*self.n_to_i_dict[img_B_name]+3
				T_B_22 = 6*self.n_to_i_dict[img_B_name]+4
				T_B_23 = 6*self.n_to_i_dict[img_B_name]+5

				matches = self.pairwise_transformations[img_A_name][img_B_name][1]
				inliers = self.pairwise_transformations[img_A_name][img_B_name][3]

				inlier_counter = 0

				for i,m in enumerate(matches):

					if inliers[i,0] == 0:
						continue

					kp_A = self.images_dict[img_A_name].kp[m.trainIdx]
					kp_B = self.images_dict[img_B_name].kp[m.queryIdx]

					p_A = (kp_A.pt[0],kp_A.pt[1])
					p_B = (kp_B.pt[0],kp_B.pt[1])

					row = tr_c*(V[T_A_11]*p_A[0] + V[T_A_12]*p_A[1] + V[T_A_13]*1 - (V[T_B_11]*p_B[0] + V[T_B_12]*p_B[1] + V[T_B_13]*1))
					A.append(row)
					b.append(0)

					row = tr_c*(V[T_A_21]*p_A[0] + V[T_A_22]*p_A[1] + V[T_A_23]*1 - (V[T_B_21]*p_B[0] + V[T_B_22]*p_B[1] + V[T_B_23]*1))
					A.append(row)
					b.append(0)

					inlier_counter+=1

					if inlier_counter >= self.max_matches_to_use:
						break


		for img_name in self.n_to_i_dict:

			T_A_11 = 6*self.n_to_i_dict[img_name]
			T_A_12 = 6*self.n_to_i_dict[img_name]+1
			T_A_13 = 6*self.n_to_i_dict[img_name]+2
			T_A_21 = 6*self.n_to_i_dict[img_name]+3
			T_A_22 = 6*self.n_to_i_dict[img_name]+4
			T_A_23 = 6*self.n_to_i_dict[img_name]+5

			if img_name == self.image_reference_name:

				A.append(off_transform_penalty*tr_c*(V[T_A_11]))
				b.append(off_transform_penalty*1)

				A.append(off_transform_penalty*tr_c*(V[T_A_12]))
				b.append(off_transform_penalty*0)

				A.append(off_transform_penalty*tr_c*(V[T_A_13]))
				b.append(off_transform_penalty*0)

				A.append(off_transform_penalty*tr_c*(V[T_A_21]))
				b.append(off_transform_penalty*0)

				A.append(off_transform_penalty*tr_c*(V[T_A_22]))
				b.append(off_transform_penalty*1)

				A.append(off_transform_penalty*tr_c*(V[T_A_23]))
				b.append(off_transform_penalty*0)

			else:

				if self.absolute_transformations is not None:

					T_init = self.absolute_transformations[img_name]

					A.append(co_c*(V[T_A_11]))
					b.append(co_c*T_init[0,0])

					A.append(co_c*(V[T_A_12]))
					b.append(co_c*T_init[0,1])

					A.append(co_c*(V[T_A_13]))
					b.append(co_c*T_init[0,2])

					A.append(co_c*(V[T_A_21]))
					b.append(co_c*T_init[1,0])

					A.append(co_c*(V[T_A_22]))
					b.append(co_c*T_init[1,1])

					A.append(co_c*(V[T_A_23]))
					b.append(co_c*T_init[1,2])

				if self.transformation_type == cv_util.Transformation.similarity:

					A.append(off_transform_penalty*tr_c*(V[T_A_11] - V[T_A_22]))
					b.append(0)

					A.append(off_transform_penalty*tr_c*(V[T_A_12] + V[T_A_21]))
					b.append(0)


		A = np.array(A)
		b = np.array(b)

		res = lsq_linear(A, b, max_iter=4*self.num_images,verbose=0)
		X = res.x

		residuals = res.fun
		
		# print('>>>\t Final RMSE of residuals at this step: {0}'.format(np.sqrt(np.mean(residuals**2))))
		print('>>>\t {0}'.format(np.sqrt(np.mean(residuals**2))))

		# U = np.matmul(A.T,A)
		# w,v = np.linalg.eig(U)
		# X = v[np.argmin(w)]

		# T_A_11 = 6*self.n_to_i_dict[self.image_reference_name]

		# X = X/X[T_A_11]

		# print(X)

		image_corners_dict = {}
		absolute_transformations_dict = {}

		for img_name in self.n_to_i_dict:

			T_A_11 = 6*self.n_to_i_dict[img_name]
			T_A_12 = 6*self.n_to_i_dict[img_name]+1
			T_A_13 = 6*self.n_to_i_dict[img_name]+2
			T_A_21 = 6*self.n_to_i_dict[img_name]+3
			T_A_22 = 6*self.n_to_i_dict[img_name]+4
			T_A_23 = 6*self.n_to_i_dict[img_name]+5

			T = np.eye(3)
			T[0,0] = X[T_A_11]
			T[0,1] = X[T_A_12]
			T[0,2] = X[T_A_13]
			T[1,0] = X[T_A_21]
			T[1,1] = X[T_A_22]
			T[1,2] = X[T_A_23]

			UL_ref = [0,0,1]
			UR_ref = [self.width,0,1]
			LR_ref = [self.width,self.height,1]
			LL_ref = [0,self.height,1]

			UL = np.matmul(T,UL_ref)
			UL = UL/UL[2]
			UL = UL[:2]

			UR = np.matmul(T,UR_ref)
			UR = UR/UR[2]
			UR = UR[:2]

			LR = np.matmul(T,LR_ref)
			LR = LR/LR[2]
			LR = LR[:2]

			LL = np.matmul(T,LL_ref)
			LL = LL/LL[2]
			LL = LL[:2]

			image_corners_dict[img_name] = {'UL':UL,'UR':UR,'LR':LR,'LL':LL}
			absolute_transformations_dict[img_name] = T

		return image_corners_dict,absolute_transformations_dict

		# V = np.eye(4*self.num_images)

		# A = []
		# b = []

		# tr_c = self.coefs[0]
		# co_c = self.coefs[1]

		# off_transform_penalty = 4*self.num_images

		# for img_A_name in self.pairwise_transformations:

		#	 T_A_11 = 4*self.n_to_i_dict[img_A_name]
		#	 T_A_12 = 4*self.n_to_i_dict[img_A_name]+1
		#	 T_A_13 = 4*self.n_to_i_dict[img_A_name]+2
		#	 T_A_21 = T_A_12
		#	 T_A_22 = T_A_11
		#	 T_A_23 = 4*self.n_to_i_dict[img_A_name]+3

		#	 for img_B_name in self.pairwise_transformations[img_A_name]:
		
		#		 T_B_11 = 4*self.n_to_i_dict[img_B_name]
		#		 T_B_12 = 4*self.n_to_i_dict[img_B_name]+1
		#		 T_B_13 = 4*self.n_to_i_dict[img_B_name]+2
		#		 T_B_21 = T_B_12
		#		 T_B_22 = T_B_11
		#		 T_B_23 = 4*self.n_to_i_dict[img_B_name]+3

		#		 matches = self.pairwise_transformations[img_A_name][img_B_name][1]
		#		 inliers = self.pairwise_transformations[img_A_name][img_B_name][3]

		#		 inlier_counter = 0

		#		 for i,m in enumerate(matches):

		#			 if inliers[i,0] == 0:
		#				 continue

		#			 kp_A = self.images_dict[img_A_name].kp[m.trainIdx]
		#			 kp_B = self.images_dict[img_B_name].kp[m.queryIdx]

		#			 p_A = (kp_A.pt[0],kp_A.pt[1])
		#			 p_B = (kp_B.pt[0],kp_B.pt[1])

		#			 row = tr_c*(V[T_A_11]*p_A[0] + V[T_A_12]*p_A[1] + V[T_A_13]*1 - (V[T_B_11]*p_B[0] + V[T_B_12]*p_B[1] + V[T_B_13]*1))
		#			 A.append(row)
		#			 b.append(0)

		#			 row = tr_c*(-V[T_A_21]*p_A[0] + V[T_A_22]*p_A[1] + V[T_A_23]*1 - (-V[T_B_21]*p_B[0] + V[T_B_22]*p_B[1] + V[T_B_23]*1))
		#			 A.append(row)
		#			 b.append(0)

		#			 inlier_counter+=1

		#			 if inlier_counter >= self.max_matches_to_use:
		#				 break


		# for img_name in self.n_to_i_dict:

		#	 T_A_11 = 4*self.n_to_i_dict[img_name]
		#	 T_A_12 = 4*self.n_to_i_dict[img_name]+1
		#	 T_A_13 = 4*self.n_to_i_dict[img_name]+2
		#	 T_A_21 = T_A_12
		#	 T_A_22 = T_A_11
		#	 T_A_23 = 4*self.n_to_i_dict[img_name]+3

		#	 if img_name == self.image_reference_name:

		#		 A.append(off_transform_penalty*tr_c*(V[T_A_11]))
		#		 b.append(off_transform_penalty*1)

		#		 A.append(off_transform_penalty*tr_c*(V[T_A_12]))
		#		 b.append(off_transform_penalty*0)

		#		 A.append(off_transform_penalty*tr_c*(V[T_A_13]))
		#		 b.append(off_transform_penalty*0)

		#		 A.append(off_transform_penalty*tr_c*(V[T_A_21]))
		#		 b.append(off_transform_penalty*0)

		#		 A.append(off_transform_penalty*tr_c*(V[T_A_22]))
		#		 b.append(off_transform_penalty*1)

		#		 A.append(off_transform_penalty*tr_c*(V[T_A_23]))
		#		 b.append(off_transform_penalty*0)

		#	 else:

		#		 if self.absolute_transformations is not None:

		#			 T_init = self.absolute_transformations[img_name]

		#			 A.append(co_c*(V[T_A_11]))
		#			 b.append(co_c*T_init[0,0])

		#			 A.append(co_c*(V[T_A_12]))
		#			 b.append(co_c*T_init[0,1])

		#			 A.append(co_c*(V[T_A_13]))
		#			 b.append(co_c*T_init[0,2])

		#			 A.append(co_c*(V[T_A_21]))
		#			 b.append(co_c*T_init[1,0])

		#			 A.append(co_c*(V[T_A_22]))
		#			 b.append(co_c*T_init[1,1])

		#			 A.append(co_c*(V[T_A_23]))
		#			 b.append(co_c*T_init[1,2])

		#		 if self.transformation_type == cv_util.Transformation.similarity:

		#			 A.append(off_transform_penalty*tr_c*(V[T_A_11] - V[T_A_22]))
		#			 b.append(0)

		#			 A.append(off_transform_penalty*tr_c*(V[T_A_12] - V[T_A_21]))
		#			 b.append(0)


		# A = np.array(A)
		# b = np.array(b)

		# res = lsq_linear(A, b, max_iter=4*self.num_images,verbose=0)
		# X = res.x

		# residuals = res.fun
		
		# # print('>>>\t Final RMSE of residuals at this step: {0}'.format(np.sqrt(np.mean(residuals**2))))
		# print('>>>\t {0}'.format(np.sqrt(np.mean(residuals**2))))

		# # U = np.matmul(A.T,A)
		# # w,v = np.linalg.eig(U)
		# # X = v[np.argmin(w)]

		# # T_A_11 = 6*self.n_to_i_dict[self.image_reference_name]

		# # X = X/X[T_A_11]

		# # print(X)

		# image_corners_dict = {}
		# absolute_transformations_dict = {}

		# for img_name in self.n_to_i_dict:

		#	 T_A_11 = 4*self.n_to_i_dict[img_name]
		#	 T_A_12 = 4*self.n_to_i_dict[img_name]+1
		#	 T_A_13 = 4*self.n_to_i_dict[img_name]+2
		#	 T_A_21 = T_A_12
		#	 T_A_22 = T_A_11
		#	 T_A_23 = 4*self.n_to_i_dict[img_name]+3

		#	 T = np.eye(3)
		#	 T[0,0] = X[T_A_11]
		#	 T[0,1] = X[T_A_12]
		#	 T[0,2] = X[T_A_13]
		#	 T[1,0] = -X[T_A_21]
		#	 T[1,1] = X[T_A_22]
		#	 T[1,2] = X[T_A_23]

		#	 UL_ref = [0,0,1]
		#	 UR_ref = [self.width,0,1]
		#	 LR_ref = [self.width,self.height,1]
		#	 LL_ref = [0,self.height,1]

		#	 UL = np.matmul(T,UL_ref)
		#	 UL = UL/UL[2]
		#	 UL = UL[:2]

		#	 UR = np.matmul(T,UR_ref)
		#	 UR = UR/UR[2]
		#	 UR = UR[:2]

		#	 LR = np.matmul(T,LR_ref)
		#	 LR = LR/LR[2]
		#	 LR = LR[:2]

		#	 LL = np.matmul(T,LL_ref)
		#	 LL = LL/LL[2]
		#	 LL = LL[:2]

		#	 image_corners_dict[img_name] = {'UL':UL,'UR':UR,'LR':LR,'LL':LL}
		#	 absolute_transformations_dict[img_name] = T

		# return image_corners_dict,absolute_transformations_dict

	def get_jacobians(self,X):

		jacobians = []

		for img1_name in self.pairwise_transformations:

			if img1_name == self.image_reference_name:
				i = self.n_to_i_dict[img1_name]

				jacobians.append(0*np.eye(9*self.num_images)[i*9+1])
				jacobians.append(0*np.eye(9*self.num_images)[i*9+2])
				jacobians.append(0*np.eye(9*self.num_images)[i*9+3])
				jacobians.append(0*np.eye(9*self.num_images)[i*9+5])
				jacobians.append(0*np.eye(9*self.num_images)[i*9+6])
				jacobians.append(0*np.eye(9*self.num_images)[i*9+7])
				jacobians.append(0*np.eye(9*self.num_images)[i*9+0])
				jacobians.append(0*np.eye(9*self.num_images)[i*9+4])
				jacobians.append(0*np.eye(9*self.num_images)[i*9+8])

			for img2_name in self.pairwise_transformations[img1_name]:
				
				matches = self.pairwise_transformations[img1_name][img2_name][1]

				inliers = self.pairwise_transformations[img1_name][img2_name][3]

				i = self.n_to_i_dict[img1_name]
				j = self.n_to_i_dict[img2_name]

				H1 = np.eye(3)
				H2 = np.eye(3)

				if img1_name != self.image_reference_name:
					
					H1_tmp = X[i*9:i*9+9]
					H1_tmp = H1_tmp.reshape(3,3)
					H1[0,:] = H1_tmp[0,:]
					H1[1,:] = H1_tmp[1,:]
					H1[2,:2] = H1_tmp[2,:2]

					jacobians.append(1000*np.eye(9*self.num_images)[i*9+8])

				if img2_name != self.image_reference_name:
					
					H2_tmp = X[j*9:j*9+9]
					H2_tmp = H2_tmp.reshape(3,3)				
					H2[0,:] = H2_tmp[0,:]
					H2[1,:] = H2_tmp[1,:]
					H2[2,:2] = H2_tmp[2,:2]
					
					jacobians.append(1000*np.eye(9*self.num_images)[j*9+8])

				inliers_iterator = 0

				for i,m in enumerate(matches):

					if inliers[i,0] == 0:
						continue

					kp1 = self.images_dict[img1_name].kp[m.trainIdx]
					kp2 = self.images_dict[img2_name].kp[m.queryIdx]

					p1 = [kp1.pt[0],kp1.pt[1],1]
					p2 = [kp2.pt[0],kp2.pt[1],1]

					p1_r_no_div = np.matmul(H1,p1) 
					p1_r = p1_r_no_div/p1_r_no_div[2]

					p2_r_no_div = np.matmul(H2,p2)
					p2_r = p2_r_no_div/p2_r_no_div[2]

					
					diff_x = p1_r[0]-p2_r[0]
					diff_y = p1_r[1]-p2_r[1]
					residual = (diff_x)**2+(diff_y)**2

					# H11
					rond_x = p1[0]/(p1_r_no_div[2])
					rond_y = 0
					jac_H1_11 = 0.5*residual**(-0.5)*(2*diff_x*rond_x+2*diff_y*rond_y)

					# H12
					rond_x = p1[1]/(p1_r_no_div[2])
					rond_y = 0
					jac_H1_12 = 0.5*residual**(-0.5)*(2*diff_x*rond_x+2*diff_y*rond_y)

					# H13
					rond_x = 1/(p1_r_no_div[2])
					rond_y = 0
					jac_H1_13 = 0.5*residual**(-0.5)*(2*diff_x*rond_x+2*diff_y*rond_y)

					# H21
					rond_x = 0
					rond_y = p1[0]/(p1_r_no_div[2])
					jac_H1_21 = 0.5*residual**(-0.5)*(2*diff_x*rond_x+2*diff_y*rond_y)

					# H22
					rond_x = 0
					rond_y = p1[1]/(p1_r_no_div[2])
					jac_H1_22 = 0.5*residual**(-0.5)*(2*diff_x*rond_x+2*diff_y*rond_y)

					# H23
					rond_x = 0
					rond_y = 1/(p1_r_no_div[2])
					jac_H1_23 = 0.5*residual**(-0.5)*(2*diff_x*rond_x+2*diff_y*rond_y)

					# H31
					rond_x = - p1[0]*p1_r_no_div[0]/((p1_r_no_div[2])**2) 
					rond_y = - p1[0]*p1_r_no_div[1]/((p1_r_no_div[2])**2)
					jac_H1_31 = 0.5*residual**(-0.5)*(2*diff_x*rond_x+2*diff_y*rond_y)

					# H32
					rond_x = - p1[1]*p1_r_no_div[0]/((p1_r_no_div[2])**2) 
					rond_y = - p1[1]*p1_r_no_div[1]/((p1_r_no_div[2])**2)
					jac_H1_32 = 0.5*residual**(-0.5)*(2*diff_x*rond_x+2*diff_y*rond_y)

					# H33
					rond_x = - 1*p1_r_no_div[0]/((p1_r_no_div[2])**2) 
					rond_y = - 1*p1_r_no_div[1]/((p1_r_no_div[2])**2)
					jac_H1_33 = 0.5*residual**(-0.5)*(2*diff_x*rond_x+2*diff_y*rond_y)

					# -----------------

					# H11
					rond_x = -p2[0]/(p2_r_no_div[2])
					rond_y = 0
					jac_H2_11 = 0.5*residual**(-0.5)*(2*diff_x*rond_x+2*diff_y*rond_y)

					# H12
					rond_x = -p2[1]/(p2_r_no_div[2])
					rond_y = 0
					jac_H2_12 = 0.5*residual**(-0.5)*(2*diff_x*rond_x+2*diff_y*rond_y)

					# H13
					rond_x = -1/(p2_r_no_div[2])
					rond_y = 0
					jac_H2_13 = 0.5*residual**(-0.5)*(2*diff_x*rond_x+2*diff_y*rond_y)

					# H21
					rond_x = 0
					rond_y = -p2[0]/(p2_r_no_div[2])
					jac_H2_21 = 0.5*residual**(-0.5)*(2*diff_x*rond_x+2*diff_y*rond_y)

					# H22
					rond_x = 0
					rond_y = -p2[1]/(p2_r_no_div[2])
					jac_H2_22 = 0.5*residual**(-0.5)*(2*diff_x*rond_x+2*diff_y*rond_y)

					# H23
					rond_x = 0
					rond_y = -1/(p2_r_no_div[2])
					jac_H2_23 = 0.5*residual**(-0.5)*(2*diff_x*rond_x+2*diff_y*rond_y)

					# H31
					rond_x = p2[0]*p2_r_no_div[0]/((p2_r_no_div[2])**2) 
					rond_y = p2[0]*p2_r_no_div[1]/((p2_r_no_div[2])**2)
					jac_H2_31 = 0.5*residual**(-0.5)*(2*diff_x*rond_x+2*diff_y*rond_y)

					# H32
					rond_x = p2[1]*p2_r_no_div[0]/((p2_r_no_div[2])**2) 
					rond_y = p2[1]*p2_r_no_div[1]/((p2_r_no_div[2])**2)
					jac_H2_32 = 0.5*residual**(-0.5)*(2*diff_x*rond_x+2*diff_y*rond_y)

					# H33
					rond_x = 1*p2_r_no_div[0]/((p2_r_no_div[2])**2) 
					rond_y = 1*p2_r_no_div[1]/((p2_r_no_div[2])**2)
					jac_H2_33 = 0.5*residual**(-0.5)*(2*diff_x*rond_x+2*diff_y*rond_y)

					ii = self.n_to_i_dict[img1_name]
					jj = self.n_to_i_dict[img2_name]
					jac = np.zeros(9*self.num_images)
					jac[ii*9+0] = jac_H1_11
					jac[ii*9+1] = jac_H1_12
					jac[ii*9+2] = jac_H1_13
					jac[ii*9+3] = jac_H1_21
					jac[ii*9+4] = jac_H1_22
					jac[ii*9+5] = jac_H1_23
					jac[ii*9+6] = jac_H1_31
					jac[ii*9+7] = jac_H1_32
					jac[ii*9+8] = jac_H1_33

					jac[jj*9+0] = jac_H2_11
					jac[jj*9+1] = jac_H2_12
					jac[jj*9+2] = jac_H2_13
					jac[jj*9+3] = jac_H2_21
					jac[jj*9+4] = jac_H2_22
					jac[jj*9+5] = jac_H2_23
					jac[jj*9+6] = jac_H2_31
					jac[jj*9+7] = jac_H2_32
					jac[jj*9+8] = jac_H2_33
					
					jacobians.append(jac)

					inliers_iterator+=1

					if inliers_iterator>=self.max_matches_to_use:
						break

		return np.array(jacobians)
					
	def get_residuals(self,X):

		residuals = []

		for img1_name in self.pairwise_transformations:

			if img1_name == self.image_reference_name:
				i = self.n_to_i_dict[img1_name]
				H = X[i*9:i*9+9]
				H = H.reshape(3,3)

				residuals.append(1000*H[0,1])
				residuals.append(1000*H[0,2])
				residuals.append(1000*H[1,0])
				residuals.append(1000*H[1,2])
				residuals.append(1000*H[2,0])
				residuals.append(1000*H[2,1])
				residuals.append(1000*H[0,0]-1000*1)
				residuals.append(1000*H[1,1]-1000*1)
				residuals.append(1000*H[2,2]-1000*1)

			for img2_name in self.pairwise_transformations[img1_name]:
				
				matches = self.pairwise_transformations[img1_name][img2_name][1]

				inliers = self.pairwise_transformations[img1_name][img2_name][3]

				i = self.n_to_i_dict[img1_name]
				j = self.n_to_i_dict[img2_name]

				H1 = np.eye(3)
				H2 = np.eye(3)

				if img1_name != self.image_reference_name:
					
					H1_tmp = X[i*9:i*9+9]
					H1_tmp = H1_tmp.reshape(3,3)
					H1[0,:] = H1_tmp[0,:]
					H1[1,:] = H1_tmp[1,:]
					H1[2,:2] = H1_tmp[2,:2]

					residuals.append(1000*H1_tmp[2,2]-1000*1)

				if img2_name != self.image_reference_name:
					
					H2_tmp = X[j*9:j*9+9]
					H2_tmp = H2_tmp.reshape(3,3)				
					H2[0,:] = H2_tmp[0,:]
					H2[1,:] = H2_tmp[1,:]
					H2[2,:2] = H2_tmp[2,:2]
					
					residuals.append(1000*H2_tmp[2,2]-1000*1)

				inliers_iterator = 0

				for i,m in enumerate(matches):

					if inliers[i,0] == 0:
						continue

					kp1 = self.images_dict[img1_name].kp[m.trainIdx]
					kp2 = self.images_dict[img2_name].kp[m.queryIdx]

					p1 = [kp1.pt[0],kp1.pt[1],1]
					p2 = [kp2.pt[0],kp2.pt[1],1]

					p1_r = np.matmul(H1,p1)
					p1_r = p1_r/p1_r[2]

					p2_r = np.matmul(H2,p2)
					p2_r = p2_r/p2_r[2]

					residuals.append(np.sqrt((p1_r[0]-p2_r[0])**2+(p1_r[1]-p2_r[1])**2))

					inliers_iterator+=1

					if inliers_iterator>=self.max_matches_to_use:
						break
					

		return residuals

	def solve_homography(self):

		H_0 = np.zeros(9*self.num_images)


		for img_name in self.n_to_i_dict:
			i = self.n_to_i_dict[img_name]
			
			if self.absolute_transformations is not None:

				H = self.absolute_transformations[img_name]

			else:

				H = np.eye(3)

			H_0[9*i:9*i+9] = H.reshape(9)

		
		# X, flag = leastsq(self.get_residuals, H_0)
		# X, flag = leastsq(self.get_residuals, H_0,Dfun=self.get_jacobians)
		# res = least_squares(self.get_residuals, H_0,jac='2-point')
		# res = least_squares(self.get_residuals, H_0,jac='3-point')
		res = least_squares(self.get_residuals, H_0,jac=self.get_jacobians)

		X = res.x

		print(np.sqrt(np.mean(self.get_residuals(X))))

		image_corners_dict = {}
		absolute_transformations_dict = {}

		for img_name in self.n_to_i_dict:
			
			i = self.n_to_i_dict[img_name]
			H = X[9*i:9*i+9]
			H = H.reshape(3,3)

			UL_ref = [0,0,1]
			UR_ref = [self.width,0,1]
			LR_ref = [self.width,self.height,1]
			LL_ref = [0,self.height,1]

			UL = np.matmul(H,UL_ref)
			UL = UL/UL[2]
			UL = UL[:2]

			UR = np.matmul(H,UR_ref)
			UR = UR/UR[2]
			UR = UR[:2]

			LR = np.matmul(H,LR_ref)
			LR = LR/LR[2]
			LR = LR[:2]

			LL = np.matmul(H,LL_ref)
			LL = LL/LL[2]
			LL = LL[:2]

			image_corners_dict[img_name] = {'UL':UL,'UR':UR,'LR':LR,'LL':LL}
			absolute_transformations_dict[img_name] = H

		return image_corners_dict,absolute_transformations_dict

class GPS_Corner_Based_Bundle_Adjustment:

	def __init__(self,images,pairwise_trans,coords,tr_type,w,h,s_gps,s_rpj,anchors_list,scale,max_mat,perc_drop):
		
		self.images = images
		self.images_dict = {}
		self.n_to_i_dict = {}

		for i,img in enumerate(self.images):
			self.images_dict[img.name] = img
			self.n_to_i_dict[img.name] = i

		self.num_images = len(self.images)

		self.pairwise_transformations = pairwise_trans
		self.transformation_type = tr_type

		self.initial_GPS_coords = coords
		
		self.width = w
		self.height = h

		self.sigma_GPS = s_gps
		self.sigma_RPJ = s_rpj

		self.anchors_dict = {}

		for d in anchors_list:
			self.anchors_dict[d['img_name']] = d

		self.scale_images = scale

		self.max_matches_to_use = max_mat

		self.perc_inliers_discard = perc_drop
		
	def solve(self,use_homogenouse=False):
		
		# print('Started optimization')
		res = None

		if self.transformation_type == cv_util.Transformation.translation:
			res = self.solve_translation() 
		elif self.transformation_type == cv_util.Transformation.similarity:
			res = self.solve_similarity_affine() 
		elif self.transformation_type == cv_util.Transformation.affine:
			res = self.solve_similarity_affine()
		elif self.transformation_type == cv_util.Transformation.homography:
			res = self.solve_homography()

		# print('Finished optimization')

		return res

	def phi(self,k,x,y):

		if k == 0:
			# UL
			return (1-(y/self.height))*(1-(x/self.width))
		elif k == 1:
			# UR
			return (1-(y/self.height))*(x/self.width)
		elif k == 2:
			# LL
			return (y/self.height)*(1-(x/self.width))
		elif k == 3:
			# LR
			return (y/self.height)*(x/self.width)


	def get_num_rows(self):

		n = 0

		for img_A_name in self.pairwise_transformations:
			for img_B_name in self.pairwise_transformations[img_A_name]:
				inliers = self.pairwise_transformations[img_A_name][img_B_name][3]
				num_inliers = int(np.sum(inliers))
				n+= 2*min(num_inliers,self.max_matches_to_use)

		n+= 12 * self.num_images

		n+= 2 * len([n for n in self.n_to_i_dict if n in self.anchors_dict])

		return n

	def solve_translation(self):

		C = np.eye(4*self.num_images)

		n = self.get_num_rows()

		A = np.zeros((n,4*self.num_images))
		b = np.zeros(n)

		lower_bounds = [-np.inf]*(4*self.num_images)
		upper_bounds = [np.inf]*(4*self.num_images)
		margin = 1e-5
		GPS_c = 1/self.sigma_GPS

		off = GPS_c*self.num_images
		
		row_it = 0

		for img_A_name in self.pairwise_transformations:

			Corner_A = {\
			'UL':[4*self.n_to_i_dict[img_A_name]+0,4*self.n_to_i_dict[img_A_name]+1],\
			'UR':[4*self.n_to_i_dict[img_A_name]+2,4*self.n_to_i_dict[img_A_name]+1],\
			'LL':[4*self.n_to_i_dict[img_A_name]+0,4*self.n_to_i_dict[img_A_name]+3],\
			'LR':[4*self.n_to_i_dict[img_A_name]+2,4*self.n_to_i_dict[img_A_name]+3]}

			for img_B_name in self.pairwise_transformations[img_A_name]:
				
				Corner_B = {\
				'UL':[4*self.n_to_i_dict[img_B_name]+0,4*self.n_to_i_dict[img_B_name]+1],\
				'UR':[4*self.n_to_i_dict[img_B_name]+2,4*self.n_to_i_dict[img_B_name]+1],\
				'LL':[4*self.n_to_i_dict[img_B_name]+0,4*self.n_to_i_dict[img_B_name]+3],\
				'LR':[4*self.n_to_i_dict[img_B_name]+2,4*self.n_to_i_dict[img_B_name]+3]}

				matches = self.pairwise_transformations[img_A_name][img_B_name][1]
				inliers = self.pairwise_transformations[img_A_name][img_B_name][3]
				perc_inliers = self.pairwise_transformations[img_A_name][img_B_name][2]

				if perc_inliers<self.perc_inliers_discard:
					continue

				num_inliers = min(np.sum(inliers),self.max_matches_to_use)

				EQ_c = 1/((self.sigma_RPJ)*np.sqrt(num_inliers))
				# print('Number and Percentage of inliers: ',int(np.sum(inliers)),perc_inliers)
				inlier_counter = 0

				for i,m in enumerate(matches):

					if inliers[i,0] == 0:
						continue

					kp_A = self.images_dict[img_A_name].kp[m.trainIdx]
					kp_B = self.images_dict[img_B_name].kp[m.queryIdx]

					p_A = (kp_A.pt[0],kp_A.pt[1])
					p_B = (kp_B.pt[0],kp_B.pt[1])

					row_x = np.zeros(4*self.num_images)
					row_y = np.zeros(4*self.num_images)

					for i,k in enumerate(['UL','UR','LL','LR']):

						row_x += (self.phi(i,p_A[0],p_A[1])*C[Corner_A[k][0]] - self.phi(i,p_B[0],p_B[1])*C[Corner_B[k][0]])
						row_y += (self.phi(i,p_A[0],p_A[1])*C[Corner_A[k][1]] - self.phi(i,p_B[0],p_B[1])*C[Corner_B[k][1]])

					A[row_it,:] = EQ_c*row_x
					row_it+=1

					A[row_it,:] = EQ_c*row_y
					row_it+=1

					inlier_counter+=1

					if inlier_counter >= self.max_matches_to_use:
						break


		
		for img_name in self.n_to_i_dict:

			Corner = {\
			'UL':[4*self.n_to_i_dict[img_name]+0,4*self.n_to_i_dict[img_name]+1],\
			'UR':[4*self.n_to_i_dict[img_name]+2,4*self.n_to_i_dict[img_name]+1],\
			'LL':[4*self.n_to_i_dict[img_name]+0,4*self.n_to_i_dict[img_name]+3],\
			'LR':[4*self.n_to_i_dict[img_name]+2,4*self.n_to_i_dict[img_name]+3]}

			for i,k in enumerate(['UL','UR','LL','LR']):

				A[row_it,:] = GPS_c*C[Corner[k][0]]
				b[row_it] = GPS_c*self.initial_GPS_coords[img_name][k]['lon']
				row_it+=1

				A[row_it,:] = GPS_c*C[Corner[k][1]]
				b[row_it] = -1*GPS_c*self.initial_GPS_coords[img_name][k]['lat']
				row_it+=1

			A[row_it,:] = off*(C[Corner['UR'][0]]-C[Corner['UL'][0]])
			b[row_it] = off*(self.initial_GPS_coords[img_name]['UR']['lon']-self.initial_GPS_coords[img_name]['UL']['lon'])
			row_it+=1

			A[row_it,:] = off*(C[Corner['LL'][1]]-C[Corner['UL'][1]])
			b[row_it] = off*(self.initial_GPS_coords[img_name]['UL']['lat']-self.initial_GPS_coords[img_name]['LL']['lat'])
			row_it+=1

			if img_name in self.anchors_dict:

				A_w = self.anchors_dict[img_name]['img_x']*self.scale_images
				A_h = self.anchors_dict[img_name]['img_y']*self.scale_images
				A_lon = self.anchors_dict[img_name]['gps_lon']
				A_lat = -1*self.anchors_dict[img_name]['gps_lat']

				row_x = np.zeros(4*self.num_images)
				row_y = np.zeros(4*self.num_images)

				for i,k in enumerate(['UL','UR','LL','LR']):

					row_x += (self.phi(i,A_w,A_h)*C[Corner[k][0]])
					row_y += (self.phi(i,A_w,A_h)*C[Corner[k][1]])

				A[row_it,:] = off*row_x
				b[row_it] = off*A_lon
				row_it+=1

				A[row_it,:] = off*row_y
				b[row_it] = off*A_lat
				row_it+=1

		A = A[:row_it,:]
		b = b[:row_it]

		print('>>> Optimization with {0} number of equations is beginning.'.format(row_it))

		res = lsq_linear(A, b, max_iter=4*self.num_images,verbose=0)
		# res = lsq_linear(A, b, bounds=(lower_bounds,upper_bounds), max_iter=4*self.num_images,verbose=0)

		X = res.x

		residuals = res.fun
		
		# print('>>>\t Final RMSE of residuals at this step: {0}'.format(np.sqrt(np.mean(residuals**2))))
		print('>>>\t {0}'.format(np.sqrt(np.mean(residuals**2))))

		image_corners_dict = {}

		for img_name in self.n_to_i_dict:

			Corner = {\
			'UL':[4*self.n_to_i_dict[img_name]+0,4*self.n_to_i_dict[img_name]+1],\
			'UR':[4*self.n_to_i_dict[img_name]+2,4*self.n_to_i_dict[img_name]+1],\
			'LL':[4*self.n_to_i_dict[img_name]+0,4*self.n_to_i_dict[img_name]+3],\
			'LR':[4*self.n_to_i_dict[img_name]+2,4*self.n_to_i_dict[img_name]+3]}

			UL = [X[Corner['UL'][0]],X[Corner['UL'][1]]]

			UR = [X[Corner['UR'][0]],X[Corner['UR'][1]]]

			LR = [X[Corner['LR'][0]],X[Corner['LR'][1]]]

			LL = [X[Corner['LL'][0]],X[Corner['LL'][1]]]

			image_corners_dict[img_name] = {'UL':UL,'UR':UR,'LR':LR,'LL':LL}
		
			print(img_name,':')
			print('\tUL: ',UL)
			print('\tUR: ',UR)
			print('\tLL: ',LL)
			print('\tLR: ',LR)
			print('-------------------------------------------------')

		return image_corners_dict
	

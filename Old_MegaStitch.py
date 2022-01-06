import numpy as np
import math
import computer_vision_utils as cv_util
import multiprocessing
import random
import sys
import Bundle_Adjustment

from scipy.sparse.linalg import lsmr
from scipy.optimize import lsq_linear

class Group:

	def __init__(self,imgs_dict,pairwise_tr,ref_img_n,img_ref_coord,coefs,tr_type):
		
		self.images_dict = imgs_dict
		self.images = [self.images_dict[k] for k in self.images_dict]

		for img in self.images:
			img.kp = None
			img.desc = None

		self.pairwise_trasformations = pairwise_tr
		self.reference_image = self.images_dict[ref_img_n]
		self.reference_image_coord = img_ref_coord
		
		self.corrected_images = [self.reference_image]
		self.corrected_coordinates = {self.reference_image.name:self.reference_image_coord}
		self.corrected_absolute_transformations = {self.reference_image.name:np.eye(3)}

		self.coefs = coefs
		self.transformation_type = tr_type

	def intersect_corrected_images(self,group):

		self_names = {ci.name for ci in self.corrected_images}
		group_names = {ci.name for ci in group.corrected_images}

		intersect_names = list(self_names.intersection(group_names))

		return [img for img in self.corrected_images if img.name in intersect_names]

	def overlap_corrected_images(self,group):

		overlap_pairs = []

		for img1 in self.corrected_images:
			for img2 in group.corrected_images:
				if img1.name == img2.name:
					continue

				if img1.name in self.pairwise_trasformations and img2.name in self.pairwise_trasformations[img1.name]:
					overlap_pairs.append((img1.name,img2.name))

		return overlap_pairs

	def is_correction_finished(self):

		if len(self.corrected_images) == len(self.images):
			return True
		else:
			return False

	def get_intersections_overlaps_and_transformations_with_other_groups(self,current_groups):

		intersections_with_other_groups = {}
		overlaps_with_other_groups = {}

		for G_j_name in current_groups:
			
			if G_j_name == self.reference_image.name:
				continue

			G_j = current_groups[G_j_name]

			I_Gi_Gj = self.intersect_corrected_images(G_j)
			overlaps_with_other_groups[G_j_name] = self.overlap_corrected_images(G_j)

			intersections_with_other_groups[G_j_name] = {}

			if len(I_Gi_Gj) == 0:
				continue
			
			for img in I_Gi_Gj:

				img1_coord = self.corrected_coordinates[img.name]
				img2_coord = G_j.corrected_coordinates[img.name]

				pts1 = np.float32([[img1_coord['UL'][0],img1_coord['UL'][1]],\
				[img1_coord['UR'][0],img1_coord['UR'][1]],\
				[img1_coord['LR'][0],img1_coord['LR'][1]],\
				[img1_coord['LL'][0],img1_coord['LL'][1]]])

				pts2 = np.float32([[img2_coord['UL'][0],img2_coord['UL'][1]],\
				[img2_coord['UR'][0],img2_coord['UR'][1]],\
				[img2_coord['LR'][0],img2_coord['LR'][1]],\
				[img2_coord['LL'][0],img2_coord['LL'][1]]])

				T = cv_util.estimate_base_transformations(pts1,pts2,self.transformation_type)

				intersections_with_other_groups[G_j_name][img.name] = T

		return intersections_with_other_groups,overlaps_with_other_groups

	def pairwise_transformation_exists(self,img1,img2):

		if img1.name in self.pairwise_trasformations and img2.name in self.pairwise_trasformations[img1.name]:
			return True
		else:
			return False

	def get_corner_coordinates(self,border_images,adjacent_to_border_images,absolute_transformations,border_images_coords):

		eq_tuples = []

		for brd_img in border_images:

			for adj_img1 in adjacent_to_border_images:

				if self.pairwise_transformation_exists(brd_img,adj_img1):

					Bj_name = adj_img1.name
					Bi_name = brd_img.name
					H_BiBj = self.pairwise_trasformations[Bi_name][Bj_name]
					H_ABi = absolute_transformations[Bi_name]
					
					eq_tuples.append((Bj_name, Bi_name, H_BiBj, H_ABi))

				for adj_img2 in adjacent_to_border_images:

					if self.pairwise_transformation_exists(adj_img2,adj_img1) and \
					self.pairwise_transformation_exists(adj_img2,brd_img):

						B1_name = adj_img1.name
						B2_name = adj_img2.name
						Br_name = brd_img.name

						#Br is X in our notations
						H_ABr = absolute_transformations[Br_name]
						H_BrB2 = self.pairwise_trasformations[B2_name][Br_name]

						H_B2B1 = self.pairwise_trasformations[B2_name][B1_name]
						H_AB2 = np.matmul(H_BrB2,H_ABr)

						eq_tuples.append((B1_name, B2_name, H_B2B1, H_AB2))


		if len(eq_tuples) == 0:
			return {}

		lsq = LinearLeastSquares_Solver(eq_tuples,border_images_coords,[self.coefs[0],self.coefs[1]],self.transformation_type)
		
		new_coords = lsq.solve()
		
		return new_coords

	def update_corner_coords_and_absolute_transformations(self,new_coords):

		for img_name in new_coords:
			self.corrected_coordinates[img_name] = new_coords[img_name]

		for img_name in self.corrected_coordinates:
			
			if self.corrected_coordinates[img_name] is None:
				continue

			pts1 = np.float32([[self.corrected_coordinates[img_name]['UL'][0],self.corrected_coordinates[img_name]['UL'][1]],\
			[self.corrected_coordinates[img_name]['UR'][0],self.corrected_coordinates[img_name]['UR'][1]],\
			[self.corrected_coordinates[img_name]['LR'][0],self.corrected_coordinates[img_name]['LR'][1]],\
			[self.corrected_coordinates[img_name]['LL'][0],self.corrected_coordinates[img_name]['LL'][1]]])

			pts2 = np.float32([[self.reference_image_coord['UL'][0],self.reference_image_coord['UL'][1]],\
			[self.reference_image_coord['UR'][0],self.reference_image_coord['UR'][1]],\
			[self.reference_image_coord['LR'][0],self.reference_image_coord['LR'][1]],\
			[self.reference_image_coord['LL'][0],self.reference_image_coord['LL'][1]]])

			T = cv_util.estimate_base_transformations(pts1,pts2,self.transformation_type)

			if T is None:
				
				print(' ___ Error in estimating absolute transformation for image {0}.'.format(img_name))
				continue


			self.corrected_absolute_transformations[img_name] = T

			if img_name not in [img.name for img in self.corrected_images]:
				self.corrected_images.append(self.images_dict[img_name]) 

	def single_step_three_loop_least_squares_correction(self):

		border_images = [img for img in self.corrected_images]
		adjacent_to_border_images = []

		for img_border in border_images:

			for img in self.images:

				if img == img_border:
					continue

				if img_border.name in self.pairwise_trasformations and img.name in self.pairwise_trasformations[img_border.name]\
				and img not in adjacent_to_border_images and img not in border_images:

					adjacent_to_border_images.append(img)
					

		new_coords = self.get_corner_coordinates(border_images,adjacent_to_border_images,self.corrected_absolute_transformations,self.corrected_coordinates)
		self.update_corner_coords_and_absolute_transformations(new_coords)


	def expand_neighbors_3_loop_method(self):
		
		megatstitch = MegaStitch_3L1P(self.images,self.pairwise_trasformations,self.coefs,self.reference_image.name,self.reference_image_coord,self.transformation_type)
		new_coords = megatstitch.single_step_three_loop_least_squares_correction()

		for img_name in new_coords:
			if new_coords[img_name] is not None:
				img = self.images_dict[img_name]

				pts1 = np.float32([[new_coords[img_name]['UL'][0],new_coords[img_name]['UL'][1]],\
				[new_coords[img_name]['UR'][0],new_coords[img_name]['UR'][1]],\
				[new_coords[img_name]['LR'][0],new_coords[img_name]['LR'][1]],\
				[new_coords[img_name]['LL'][0],new_coords[img_name]['LL'][1]]])

				pts2 = np.float32([[self.reference_image_coord['UL'][0],self.reference_image_coord['UL'][1]],\
				[self.reference_image_coord['UR'][0],self.reference_image_coord['UR'][1]],\
				[self.reference_image_coord['LR'][0],self.reference_image_coord['LR'][1]],\
				[self.reference_image_coord['LL'][0],self.reference_image_coord['LL'][1]]])

				T = cv_util.estimate_base_transformations(pts1,pts2,self.transformation_type)

				self.corrected_images.append(img)
				self.corrected_coordinates[img_name] = new_coords[img_name]
				self.corrected_absolute_transformations[img_name] = T


class LinearLeastSquares_Solver:

	def __init__(self,eq_tuple,init_corners,coefs,tr_type):
		
		self.all_image_names = list(set([n for n,_,_,_ in eq_tuple] + [n for _,n,_,_ in eq_tuple if n is not None]+[n for n in init_corners]))
		self.n_to_i_dict = {}

		for i,img_name in enumerate(self.all_image_names):
			self.n_to_i_dict[img_name] = i

		self.num_images = len(self.all_image_names)
		self.equations_tuple = eq_tuple
		self.initial_corners = init_corners
		self.coefs = coefs
		self.transformation_type = tr_type

	def solve(self,use_homogenouse=False):
		
		# print('Started optimization')
		res = None

		if self.transformation_type == cv_util.Transformation.translation:
			res = self.solve_similarity_affine() 
		elif self.transformation_type == cv_util.Transformation.similarity:
			res = self.solve_similarity_affine() 
		elif self.transformation_type == cv_util.Transformation.affine:
			res = self.solve_similarity_affine()
		elif self.transformation_type == cv_util.Transformation.homography:
			res = self.solve_homography(use_homogenouse)

		# print('Finished optimization')

		return res

	def solve_similarity_affine(self):

		V = np.eye(8*self.num_images)

		A = []
		b = []

		tr_c = self.coefs[0]
		co_c = self.coefs[1]

		for Bj_name, Bi_name, H_BiBj, H_ABi in self.equations_tuple:

			if Bi_name is not None:

				# two point equation (step 3 and 7 and i)

				T = np.matmul(np.linalg.inv(H_ABi),np.matmul(H_BiBj,H_ABi))

				for k,key in enumerate(['UL','UR','LR','LL']):

					index_x_Bj = 4 * self.n_to_i_dict[Bj_name] + k
					index_y_Bj = 4 * self.n_to_i_dict[Bj_name] + 4 * self.num_images + k

					index_x_Bi = 4 * self.n_to_i_dict[Bi_name] + k
					index_y_Bi = 4 * self.n_to_i_dict[Bi_name] + 4 * self.num_images + k

					row = tr_c*(T[0,0]*(V[index_x_Bi,:]) + T[0,1]*(V[index_y_Bi,:])  - V[index_x_Bj,:] )
					b_val = -tr_c*T[0,2]

					A.append(row)
					b.append(b_val)

					row = tr_c*(T[1,0]*(V[index_x_Bi,:]) + T[1,1]*(V[index_y_Bi,:])  - V[index_y_Bj,:] )
					b_val = -tr_c*T[1,2]

					A.append(row)
					b.append(b_val)

			else:

				# one point equation (step 4 and 6)
				# here H_BiBj is just a point

				new_B_j = H_BiBj

				for k,key in enumerate(['UL','UR','LR','LL']):

					index_x_Bj = 4 * self.n_to_i_dict[Bj_name] + k
					index_y_Bj = 4 * self.n_to_i_dict[Bj_name] + 4 * self.num_images + k

					A.append(0.1*tr_c*V[index_x_Bj,:])
					b.append(0.1*tr_c*new_B_j[key][0])

					A.append(0.1*tr_c*V[index_y_Bj,:])
					b.append(0.1*tr_c*new_B_j[key][1])


		for img_name in self.initial_corners:

			if img_name not in self.n_to_i_dict:
				continue

			for k,key in enumerate(['UL','UR','LR','LL']):
				
				x = 4 * self.n_to_i_dict[img_name] + k
				y = 4 * self.n_to_i_dict[img_name] + 4 * self.num_images + k

				A.append(co_c*V[x,:])
				b.append(co_c*self.initial_corners[img_name][key][0])

				A.append(co_c*V[y,:])
				b.append(co_c*self.initial_corners[img_name][key][1])

		A = np.array(A)
		b = np.array(b)

		res = lsq_linear(A, b, max_iter=4*self.num_images,verbose=0)
		X = res.x

		residuals = res.fun
		# print('>>>\t Final RMSE of residuals at this step: {0}'.format(np.sqrt(np.mean(residuals**2))))
		print('>>>\t {0}'.format(np.sqrt(np.mean(residuals**2))))


		image_corners_dict = {}

		for img_name in self.n_to_i_dict:

			x_i = 4 * self.n_to_i_dict[img_name]
			y_i = 4 * self.n_to_i_dict[img_name] + 4 * self.num_images

			UL = (X[x_i],X[y_i])
			UR = (X[x_i+1],X[y_i+1])
			LR = (X[x_i+2],X[y_i+2])
			LL = (X[x_i+3],X[y_i+3])

			image_corners_dict[img_name] = {'UL':UL,'UR':UR,'LR':LR,'LL':LL}

		return image_corners_dict

	def solve_homography(self,use_homogenouse):

		V = np.eye(12*self.num_images)

		A = []
		b = []

		lower_bounds = [-np.inf]*(12*self.num_images)
		upper_bounds = [np.inf]*(12*self.num_images)

		tr_c = self.coefs[0]
		co_c = self.coefs[1]

		for Bj_name, Bi_name, H_BiBj, H_ABi in self.equations_tuple:
			
			# if Bi_name in self.initial_corners:
			#	 tr_c=self.coefs[0]*10000
			# else:
			#	 tr_c=self.coefs[0]

			if Bi_name is not None:

				# two point equation (step 3 and 7 and i)
				tmp = np.linalg.inv(H_ABi)
				# tmp = tmp/tmp[2,2]
				T = np.matmul(tmp,np.matmul(H_BiBj,H_ABi))
				# T = T/T[2,2]

				for k,key in enumerate(['UL','UR','LR','LL']):

					index_x_Bj = 4 * self.n_to_i_dict[Bj_name] + k
					index_y_Bj = 4 * self.n_to_i_dict[Bj_name] + 4 * self.num_images + k
					index_w_Bj = 4 * self.n_to_i_dict[Bj_name] + 8 * self.num_images + k

					index_x_Bi = 4 * self.n_to_i_dict[Bi_name] + k
					index_y_Bi = 4 * self.n_to_i_dict[Bi_name] + 4 * self.num_images + k
					index_w_Bi = 4 * self.n_to_i_dict[Bi_name] + 8 * self.num_images + k

					row = tr_c*(T[0,0]*(V[index_x_Bi,:]) + T[0,1]*(V[index_y_Bi,:]) + T[0,2]*(V[index_w_Bi,:]) - V[index_x_Bj,:] )
					b_val = 0

					A.append(row)
					b.append(b_val)

					row = tr_c*(T[1,0]*(V[index_x_Bi,:]) + T[1,1]*(V[index_y_Bi,:]) + T[1,2]*(V[index_w_Bi,:])  - V[index_y_Bj,:] )
					b_val = 0

					A.append(row)
					b.append(b_val)

					row = tr_c*(T[2,0]*(V[index_x_Bi,:]) + T[2,1]*(V[index_y_Bi,:]) + T[2,2]*(V[index_w_Bi,:])  - V[index_w_Bj,:] )
					b_val = 0

					A.append(row)
					b.append(b_val)

			else:

				# one point equation (step 4 and 6)
				# here H_BiBj is just a point

				new_B_j = H_BiBj

				for k,key in enumerate(['UL','UR','LR','LL']):

					index_x_Bj = 4 * self.n_to_i_dict[Bj_name] + k
					index_y_Bj = 4 * self.n_to_i_dict[Bj_name] + 4 * self.num_images + k
					index_w_Bj = 4 * self.n_to_i_dict[Bj_name] + 8 * self.num_images + k

					A.append(0.1*tr_c*V[index_x_Bj,:])
					b.append(0.1*tr_c*new_B_j[key][0])

					A.append(0.1*tr_c*V[index_y_Bj,:])
					b.append(0.1*tr_c*new_B_j[key][1])

					A.append(0.1*tr_c*V[index_w_Bj,:])
					b.append(0.1*tr_c*new_B_j[key][2])

				# if Bj_name in self.initial_corners or Bi_name in self.initial_corners:

				#	 row = tr_c*(T[0,0]*(V[index_x_Bi,:]) + T[0,1]*(V[index_y_Bi,:]) + T[0,2]*(V[index_w_Bi,:]) - V[index_x_Bj,:] )
				#	 b_val = 0

				#	 A.append(row)
				#	 b.append(b_val)

				#	 row = tr_c*(T[1,0]*(V[index_x_Bi,:]) + T[1,1]*(V[index_y_Bi,:]) + T[1,2]*(V[index_w_Bi,:])  - V[index_y_Bj,:] )
				#	 b_val = 0

				#	 A.append(row)
				#	 b.append(b_val)

				#	 row = tr_c*(T[2,0]*(V[index_x_Bi,:]) + T[2,1]*(V[index_y_Bi,:]) + T[2,2]*(V[index_w_Bi,:])  - V[index_w_Bj,:] )
				#	 b_val = 0

				#	 A.append(row)
				#	 b.append(b_val)

				# else:

				#	 row = tr_c*(T[0,0]*(V[index_x_Bi,:]) + T[0,1]*(V[index_y_Bi,:]) - V[index_x_Bj,:] )
				#	 b_val = -tr_c*T[0,2]

				#	 A.append(row)
				#	 b.append(b_val)

				#	 row = tr_c*(T[1,0]*(V[index_x_Bi,:]) + T[1,1]*(V[index_y_Bi,:])  - V[index_y_Bj,:] )
				#	 b_val = -tr_c*T[1,2]

				#	 A.append(row)
				#	 b.append(b_val)

				#	 row = tr_c*(T[2,0]*(V[index_x_Bi,:]) + T[2,1]*(V[index_y_Bi,:])  - V[index_w_Bj,:] )
				#	 b_val = -tr_c*T[2,2]

				#	 A.append(row)
				#	 b.append(b_val)


		for img_name in self.initial_corners:

			if img_name not in self.n_to_i_dict:
				continue

			for k,key in enumerate(['UL','UR','LR','LL']):
				
				x = 4 * self.n_to_i_dict[img_name] + k
				y = 4 * self.n_to_i_dict[img_name] + 4 * self.num_images + k
				w = 4 * self.n_to_i_dict[img_name] + 8 * self.num_images + k

				if use_homogenouse:
					w_value = self.initial_corners[img_name][key][2]
				else:
					w_value = 1

				A.append(co_c*V[x,:])
				b.append(co_c*self.initial_corners[img_name][key][0])

				A.append(co_c*V[y,:])
				b.append(co_c*self.initial_corners[img_name][key][1])

				A.append(co_c*V[w,:])
				b.append(co_c*w_value)

				lower_bounds[x] = self.initial_corners[img_name][key][0]-1e-10
				upper_bounds[x] = self.initial_corners[img_name][key][0]+1e-10

				lower_bounds[y] = self.initial_corners[img_name][key][1]-1e-10
				upper_bounds[y] = self.initial_corners[img_name][key][1]+1e-10

				lower_bounds[w] = w_value-1e-10
				upper_bounds[w] = w_value+1e-10
				

		A = np.array(A)
		b = np.array(b)

		if self.transformation_type == cv_util.Transformation.homography:
			res = lsq_linear(A, b, bounds = (lower_bounds,upper_bounds), max_iter=4*self.num_images,verbose=0)
			X = res.x
		else:
			res = lsq_linear(A, b, max_iter=4*self.num_images,verbose=0)
			X = res.x



		residuals = res.fun
		# print('>>>\t Final RMSE of residuals at this step: {0}'.format(np.sqrt(np.mean(residuals**2))))
		print('>>>\t {0}'.format(np.sqrt(np.mean(residuals**2))))

		# print(residuals)

		image_corners_dict = {}

		for img_name in self.n_to_i_dict:

			x_i = 4 * self.n_to_i_dict[img_name]
			y_i = 4 * self.n_to_i_dict[img_name] + 4 * self.num_images
			w_i = 4 * self.n_to_i_dict[img_name] + 8 * self.num_images

			if use_homogenouse:
				UL = (X[x_i],X[y_i],X[w_i])
				UR = (X[x_i+1],X[y_i+1],X[w_i+1])
				LR = (X[x_i+2],X[y_i+2],X[w_i+2])
				LL = (X[x_i+3],X[y_i+3],X[w_i+3])
			else:
				UL = (X[x_i]/X[w_i],X[y_i]/X[w_i],X[w_i]/X[w_i])
				UR = (X[x_i+1]/X[w_i+1],X[y_i+1]/X[w_i+1],X[w_i+1]/X[w_i+1])
				LR = (X[x_i+2]/X[w_i+2],X[y_i+2]/X[w_i+2],X[w_i+2]/X[w_i+2])
				LL = (X[x_i+3]/X[w_i+3],X[y_i+3]/X[w_i+3],X[w_i+3]/X[w_i+3])

			image_corners_dict[img_name] = {'UL':UL,'UR':UR,'LR':LR,'LL':LL}

		return image_corners_dict


class MegaStitch_3L2P:

	def __init__(self,imgs,pairwise_tr,coefs,img_R_n,img_R_coords,transf_type):

		self.images = imgs
		self.images_dict = {}

		for img in self.images:
			self.images_dict[img.name] = img

		self.pairwise_trasformations = {}

		for img1_name in pairwise_tr:
			self.pairwise_trasformations[img1_name] = {}
			for img2_name in pairwise_tr[img1_name]:
				self.pairwise_trasformations[img1_name][img2_name] = pairwise_tr[img1_name][img2_name][0]

		self.coefs = coefs
		self.R_image = self.images_dict[img_R_n]
		self.R_coords = img_R_coords
		self.transformation_type = transf_type

	def pairwise_transformation_exists(self,img1,img2):

		if img1.name in self.pairwise_trasformations and img2.name in self.pairwise_trasformations[img1.name]:
			return True
		else:
			return False

	def get_corner_coordinates_phase_1(self,adjacent_to_border_images,border_images,absolute_transformations,border_images_coords):

		eq_tuples = []

		for adj_img in adjacent_to_border_images:
			for brd_img in border_images:

				if self.pairwise_transformation_exists(brd_img,adj_img):

					Bj_name = adj_img.name
					Bi_name = brd_img.name
					H_BiBj = self.pairwise_trasformations[Bi_name][Bj_name]
					H_ABi = absolute_transformations[Bi_name]

					eq_tuples.append((Bj_name, Bi_name, H_BiBj, H_ABi))

		lsq = LinearLeastSquares_Solver(eq_tuples,border_images_coords,self.coefs,self.transformation_type)

		new_coords = lsq.solve()

		return new_coords

	def get_corner_coordinates_phase_2(self,adjacent_to_border_images,absolute_transformations,adjacent_to_border_images_coords):

		eq_tuples = []

		for adj_img1 in adjacent_to_border_images:
			for adj_img2 in adjacent_to_border_images:

				if self.pairwise_transformation_exists(adj_img2,adj_img1):

					B1_name = adj_img1.name
					B2_name = adj_img2.name
					H_B2B1 = self.pairwise_trasformations[adj_img2.name][adj_img1.name]
					H_AB2 = absolute_transformations[B2_name]

					eq_tuples.append((B1_name, B2_name, H_B2B1, H_AB2))

		if len(eq_tuples) == 0:
			return {}

		lsq = LinearLeastSquares_Solver(eq_tuples,adjacent_to_border_images_coords,[self.coefs[0],self.coefs[1]/5],self.transformation_type)

		new_coords = lsq.solve()

		return new_coords
					 
	def update_corner_coords_and_absolute_transformations(self,final_corners,new_coords,absolute_transformations):

		for img_name in new_coords:
			final_corners[img_name] = new_coords[img_name]

		for img_name in final_corners:
			
			if final_corners[img_name] is None:
				continue

			pts1 = np.float32([[final_corners[img_name]['UL'][0],final_corners[img_name]['UL'][1]],\
			[final_corners[img_name]['UR'][0],final_corners[img_name]['UR'][1]],\
			[final_corners[img_name]['LR'][0],final_corners[img_name]['LR'][1]],\
			[final_corners[img_name]['LL'][0],final_corners[img_name]['LL'][1]]])

			pts2 = np.float32([[self.R_coords['UL'][0],self.R_coords['UL'][1]],\
			[self.R_coords['UR'][0],self.R_coords['UR'][1]],\
			[self.R_coords['LR'][0],self.R_coords['LR'][1]],\
			[self.R_coords['LL'][0],self.R_coords['LL'][1]]])

			T = cv_util.estimate_base_transformations(pts1,pts2,self.transformation_type)

			absolute_transformations[img_name] = T

		return final_corners,absolute_transformations

	def update_border_and_adjacent_to_border_image_lists(self,border_images,adjacent_to_border_images,finalized_image_names):

		new_border_images = adjacent_to_border_images
		new_adjacent_to_border_images = []
		new_finalized_image_names = finalized_image_names.copy()

		for img in border_images:
			new_finalized_image_names.append(img.name)

		for img in new_border_images:

			if img.name not in self.pairwise_trasformations:
				continue

			for img2_name in self.pairwise_trasformations[img.name]:
				if img2_name not in [a.name for a in new_adjacent_to_border_images] and \
				img2_name not in [a.name for a in new_border_images] and \
				img2_name not in [a for a in new_finalized_image_names]:

					new_adjacent_to_border_images.append(self.images_dict[img2_name])

		return new_border_images,new_adjacent_to_border_images,new_finalized_image_names

	def three_loop_least_squares_correction(self):

		final_corners = {}
		finalized_image_names = []

		for img in self.images:
			final_corners[img.name] = None

		final_corners[self.R_image.name] = self.R_coords

		absolute_transformations = {self.R_image.name:np.eye(3)}
		
		border_images = [self.R_image]
		adjacent_to_border_images = [self.images_dict[img_name] for img_name in self.pairwise_trasformations[self.R_image.name]]

		while True:

			# get corner coordinates and absolute transformations of the images adjacent to border images w.r.t. border images (first phase)
			print('>>> Phase 1')

			border_images_coords = {img.name:final_corners[img.name] for img in border_images if final_corners[img.name] is not None}
			new_coords = self.get_corner_coordinates_phase_1(adjacent_to_border_images,border_images,absolute_transformations,border_images_coords)
			final_corners, absolute_transformations = self.update_corner_coords_and_absolute_transformations(final_corners,new_coords,absolute_transformations)

			# get corner coordinates and absolute transformations of the images adjacent to border images w.r.t. themselves (second phase)
			print('>>> Phase 2')

			adjacent_to_border_images_coords = {img.name:final_corners[img.name] for img in adjacent_to_border_images if final_corners[img.name] is not None}
			new_coords = self.get_corner_coordinates_phase_2(adjacent_to_border_images,absolute_transformations,adjacent_to_border_images_coords)
			final_corners, absolute_transformations = self.update_corner_coords_and_absolute_transformations(final_corners,new_coords,absolute_transformations)

			# update border and adjacent images

			border_images , adjacent_to_border_images, finalized_image_names = self.update_border_and_adjacent_to_border_image_lists(border_images,adjacent_to_border_images,finalized_image_names)


			if len(adjacent_to_border_images) == 0:
				break

		print('>>> Total number of finalized images: {0}'.format(len(finalized_image_names)+len(border_images)))

		return final_corners


class MegaStitch_3L1P:

	def __init__(self,imgs,pairwise_tr,coefs,img_R_n,img_R_coords,transf_type):

		self.images = imgs
		self.images_dict = {}

		for img in self.images:
			self.images_dict[img.name] = img

		self.pairwise_trasformations = {}

		for img1_name in pairwise_tr:
			self.pairwise_trasformations[img1_name] = {}
			for img2_name in pairwise_tr[img1_name]:

				if type(pairwise_tr[img1_name][img2_name]) == np.ndarray:
					self.pairwise_trasformations[img1_name][img2_name] = pairwise_tr[img1_name][img2_name]
				else:
					self.pairwise_trasformations[img1_name][img2_name] = pairwise_tr[img1_name][img2_name][0]

		self.coefs = coefs
		self.R_image = self.images_dict[img_R_n]
		self.R_coords = img_R_coords
		self.transformation_type = transf_type

	def pairwise_transformation_exists(self,img1,img2):

		if img1.name in self.pairwise_trasformations and img2.name in self.pairwise_trasformations[img1.name]:
			return True
		else:
			return False


	def get_corner_coordinates(self,border_images,adjacent_to_border_images,absolute_transformations,border_images_coords):

		# if T multiplied by the corners of pairwise[0] (in pairwise[0] system) gives the corners of pairwise[1] (in pairwise[0] system)

		eq_tuples = []

		for brd_img in border_images:

			for adj_img1 in adjacent_to_border_images:

				if self.pairwise_transformation_exists(brd_img,adj_img1):

					Bj_name = adj_img1.name
					Bi_name = brd_img.name
					H_BiBj = self.pairwise_trasformations[Bi_name][Bj_name]
					H_ABi = absolute_transformations[Bi_name]
					
					eq_tuples.append((Bj_name, Bi_name, H_BiBj, H_ABi))
					# print(Bj_name,Bi_name)

				for adj_img2 in adjacent_to_border_images:

					if self.pairwise_transformation_exists(adj_img2,adj_img1) and \
					self.pairwise_transformation_exists(adj_img2,brd_img):

						B1_name = adj_img1.name
						B2_name = adj_img2.name
						Br_name = brd_img.name

						#Br is X in our notations
						H_ABr = absolute_transformations[Br_name]
						H_BrB2 = self.pairwise_trasformations[B2_name][Br_name]

						H_B2B1 = self.pairwise_trasformations[B2_name][B1_name]
						H_AB2 = np.matmul(H_BrB2,H_ABr)

						eq_tuples.append((B1_name, B2_name, H_B2B1, H_AB2))

						# H_B2Br = self.pairwise_trasformations[Br_name][B2_name]
						# H_B2A = np.matmul(np.linalg.inv(H_ABr),H_B2Br)

						# new_H = np.matmul(H_B2A,np.matmul(H_B2B1,H_AB2))
						# eq_tuples.append((B1_name, B2_name, new_H , np.eye(3)))

						# print("*",B1_name,B2_name)
						# print(H_AB2)
						# p = {'3.jpg':[386,-202,1],'4.jpg':[296,184,1]}
						# p = {'3.jpg':[443,-249,1],'4.jpg':[286,200,1]}

						# tmp = np.matmul(np.matmul(H_B2A,H_B2B1),H_AB2)
						# newA = np.matmul(tmp,p[B2_name])
						# newA = newA/newA[2]
						# print(newA)
						# print(tmp)

		if len(eq_tuples) == 0:
			return {}

		lsq = LinearLeastSquares_Solver(eq_tuples,border_images_coords,[self.coefs[0],self.coefs[1]],self.transformation_type)
		
		new_coords = lsq.solve()
		
		return new_coords
					 
	def update_corner_coords_and_absolute_transformations(self,final_corners,new_coords,absolute_transformations):

		for img_name in new_coords:
			final_corners[img_name] = new_coords[img_name]

		for img_name in final_corners:
			
			if final_corners[img_name] is None:
				continue

			pts1 = np.float32([[final_corners[img_name]['UL'][0],final_corners[img_name]['UL'][1]],\
			[final_corners[img_name]['UR'][0],final_corners[img_name]['UR'][1]],\
			[final_corners[img_name]['LR'][0],final_corners[img_name]['LR'][1]],\
			[final_corners[img_name]['LL'][0],final_corners[img_name]['LL'][1]]])

			pts2 = np.float32([[self.R_coords['UL'][0],self.R_coords['UL'][1]],\
			[self.R_coords['UR'][0],self.R_coords['UR'][1]],\
			[self.R_coords['LR'][0],self.R_coords['LR'][1]],\
			[self.R_coords['LL'][0],self.R_coords['LL'][1]]])

			T = cv_util.estimate_base_transformations(pts1,pts2,self.transformation_type)

			# new_pts1 = np.matmul(np.linalg.inv(T),[pts2[2][0],pts2[2][1],1])
			# print(new_pts1)
			# new_pts1 = new_pts1/new_pts1[2]
			# print(new_pts1)
			# print(pts1[2])

			if T is None:
				# print(pts1)
				# print(pts2)
				# print(final_corners[img_name])
				final_corners[img_name] = None
				print(' ___ Error in estimating absolute transformation for image {0}.'.format(img_name))
				continue


			absolute_transformations[img_name] = T

		return final_corners,absolute_transformations

	def update_border_and_adjacent_to_border_image_lists(self,border_images,adjacent_to_border_images,finalized_image_names,absolute_transformations):

		new_border_images = [img for img in adjacent_to_border_images if img.name in absolute_transformations]
		new_adjacent_to_border_images = [img for img in adjacent_to_border_images if img.name not in absolute_transformations]
		new_finalized_image_names = finalized_image_names.copy()

		for img in border_images:
			new_finalized_image_names.append(img.name)


		for img in new_border_images:

			if img.name not in self.pairwise_trasformations:
				continue

			for img2_name in self.pairwise_trasformations[img.name]:
				if img2_name not in [a.name for a in new_adjacent_to_border_images] and \
				img2_name not in [a.name for a in new_border_images] and \
				img2_name not in [a for a in new_finalized_image_names]:

					new_adjacent_to_border_images.append(self.images_dict[img2_name])

		

		return new_border_images,new_adjacent_to_border_images,new_finalized_image_names

	def three_loop_least_squares_correction(self):

		final_corners = {}
		finalized_image_names = []

		for img in self.images:
			final_corners[img.name] = None

		final_corners[self.R_image.name] = self.R_coords

		absolute_transformations = {self.R_image.name:np.eye(3)}
		
		border_images = [self.R_image]
		adjacent_to_border_images = [self.images_dict[img_name] for img_name in self.pairwise_trasformations[self.R_image.name]]

		while True:

			
			# get corner coordinates and absolute transformations of the images adjacent to border images

			border_images_coords = {img.name:final_corners[img.name] for img in border_images if final_corners[img.name] is not None}
			new_coords = self.get_corner_coordinates(border_images,adjacent_to_border_images,absolute_transformations,border_images_coords)
			final_corners, absolute_transformations = self.update_corner_coords_and_absolute_transformations(final_corners,new_coords,absolute_transformations)

			# update border and adjacent images

			border_images , adjacent_to_border_images,finalized_image_names = self.update_border_and_adjacent_to_border_image_lists(border_images,adjacent_to_border_images,finalized_image_names,absolute_transformations)


			if len(adjacent_to_border_images) == 0:
				break

		print('>>> Total number of finalized images: {0}'.format(len(finalized_image_names)+len(border_images)))
		# np.set_printoptions(suppress=True)
		# print(absolute_transformations)
		return final_corners


	def single_step_three_loop_least_squares_correction(self):

		final_corners = {}
		finalized_image_names = []

		for img in self.images:
			final_corners[img.name] = None

		final_corners[self.R_image.name] = self.R_coords

		absolute_transformations = {self.R_image.name:np.eye(3)}
		
		border_images = [self.R_image]
		adjacent_to_border_images = [self.images_dict[img_name] for img_name in self.pairwise_trasformations[self.R_image.name]]

		# get corner coordinates and absolute transformations of the images adjacent to border images

		border_images_coords = {img.name:final_corners[img.name] for img in border_images if final_corners[img.name] is not None}
		new_coords = self.get_corner_coordinates(border_images,adjacent_to_border_images,absolute_transformations,border_images_coords)
		final_corners, absolute_transformations = self.update_corner_coords_and_absolute_transformations(final_corners,new_coords,absolute_transformations)

		return final_corners


class MegaStitch_Multi_Group_Drone:

	def __init__(self,imgs,pairwise_tr,coefs,transf_type,x,y,is_normalized_keypoints,use_homogenouse,parallel,parallel_cores,eq_to_pick):

		self.images = imgs
		self.images_dict = {}

		for img in self.images:
			self.images_dict[img.name] = img

		self.pairwise_trasformations = {}

		for img1_name in pairwise_tr:
			self.pairwise_trasformations[img1_name] = {}
			for img2_name in pairwise_tr[img1_name]:
				if type(pairwise_tr[img1_name][img2_name]) == np.ndarray:
					self.pairwise_trasformations[img1_name][img2_name] = pairwise_tr[img1_name][img2_name]
				else:
					self.pairwise_trasformations[img1_name][img2_name] = pairwise_tr[img1_name][img2_name][0]

		self.coefs = coefs
		self.transformation_type = transf_type

		self.is_normalized_keypoints = is_normalized_keypoints
		
		self.use_homogenouse = use_homogenouse

		if use_homogenouse:

			if is_normalized_keypoints:
				self.ref_coords = {'UL':[-0.5,-0.5,1],'UR':[0.5,-0.5,1],'LR':[0.5,0.5,1],'LL':[-0.5,0.5,1]}
			else:
				self.ref_coords = {'UL':[0,0,1],'UR':[x,0,1],'LR':[x,y,1],'LL':[0,y,1]}
		else:

			if is_normalized_keypoints:
				self.ref_coords = {'UL':[-0.5,-0.5],'UR':[0.5,-0.5],'LR':[0.5,0.5],'LL':[-0.5,0.5]}
			else:
				self.ref_coords = {'UL':[0,0],'UR':[x,0],'LR':[x,y],'LL':[0,y]}

		self.use_parallel_process = parallel
		self.cores = parallel_cores
		self.number_equation_to_pick_from_unique_tuples = eq_to_pick


	def pairwise_transformation_exists(self,img1,img2):

		if img1.name in self.pairwise_trasformations and img2.name in self.pairwise_trasformations[img1.name]:
			return True
		else:
			return False


	def initialize_groups(self,groups):
		
		for r_key in groups:

			groups[r_key].expand_neighbors_3_loop_method()

	def get_equations_for_overlapping_images_step3(self,current_groups,G_i_name,G_j_name,overlaps_dict):

		eq_tuples = []

		G_i = current_groups[G_i_name]
		G_j = current_groups[G_j_name]

		overlaps = overlaps_dict[G_j_name]

		for ovr in overlaps:

			B_m1_name = ovr[0]
			B_m2_name = ovr[1]

			H_Gi_m1 = G_i.corrected_absolute_transformations[B_m1_name]
			H_m1_m2 = self.pairwise_trasformations[B_m1_name][B_m2_name]

			eq_tuples.append((B_m2_name, B_m1_name, H_m1_m2, H_Gi_m1))

		return eq_tuples

	def get_equations_for_all_images_based_on_overlappings_step4(self,current_groups,G_i_name,G_j_name,overlaps_dict):

		eq_tuples = []

		G_i = current_groups[G_i_name]
		G_j = current_groups[G_j_name]

		overlaps = overlaps_dict[G_j_name]

		for Bm_prim in G_j.corrected_images:
			for ovr in overlaps:

				B_m1_name = ovr[0]
				B_m2_name = ovr[1]

				if Bm_prim.name == B_m2_name:
					continue

				Bm_prim_new = {}

				H_Gj_m2 = G_j.corrected_absolute_transformations[B_m2_name]
				H_m1_m2 = self.pairwise_trasformations[B_m1_name][B_m2_name]
				H_m1_Gi = np.linalg.inv(G_i.corrected_absolute_transformations[B_m1_name])

				for k,key in enumerate(['UL','UR','LR','LL']):

					coord_Bm_prime = G_j.corrected_coordinates[Bm_prim.name]
				
					Bm_prim_Gj = [coord_Bm_prime[key][0],coord_Bm_prime[key][1],1]
					tmp = np.matmul(H_m1_Gi,np.matmul(H_m1_m2,np.matmul(H_Gj_m2,Bm_prim_Gj)))
					tmp = tmp/tmp[2]

					Bm_prim_new[key] = [tmp[0],tmp[1],tmp[2]]
				
				eq_tuples.append((Bm_prim.name, None, Bm_prim_new, None))

		return eq_tuples

	def get_equations_for_all_images_based_on_intersections_step6(self,current_groups,G_i_name,G_j_name,intersections_dict):

		eq_tuples = []

		G_i = current_groups[G_i_name]
		G_j = current_groups[G_j_name]

		intersections = intersections_dict[G_j_name]
		
		for Bm_prim in G_j.corrected_images:
			for Bm in intersections:

				Bm_prim_new = {}

				H_Gj_Gi_m = np.linalg.inv(intersections[Bm])

				for k,key in enumerate(['UL','UR','LR','LL']):

					coord_Bm_prime = G_j.corrected_coordinates[Bm_prim.name]
				
					Bm_prim_Gj = [coord_Bm_prime[key][0],coord_Bm_prime[key][1],1]
					tmp = np.matmul(H_Gj_Gi_m,Bm_prim_Gj)
					tmp = tmp/tmp[2]

					Bm_prim_new[key] = [tmp[0],tmp[1],tmp[2]]
				
				eq_tuples.append((Bm_prim.name, None, Bm_prim_new, None))

		return eq_tuples

	def get_equations_for_overlapping_images_based_on_intersections_step7(self,current_groups,G_i_name,G_j_name,overlaps_dict,intersections_dict):

		eq_tuples = []

		G_i = current_groups[G_i_name]
		G_j = current_groups[G_j_name]

		overlaps = overlaps_dict[G_j_name]
		intersections = intersections_dict[G_j_name]

		for Bm in intersections:

			for ovr in overlaps:

				B_m1_name = ovr[0]
				B_m2_name = ovr[1]

				H_Gi_Gj_m = intersections[Bm]
				H_Gj_m2 = G_j.corrected_absolute_transformations[B_m2_name]
				H_m2_m1 = np.linalg.inv(self.pairwise_trasformations[B_m1_name][B_m2_name])

				H_Gi_m2 = np.matmul(H_Gj_m2,H_Gi_Gj_m)

				eq_tuples.append((B_m1_name,B_m2_name,H_m2_m1,H_Gi_m2))

		return eq_tuples

	def get_equations_for_overlapping_groups_based_on_intersections_step_i(self,current_groups,G_i_name,intersections_dict_G_i):

		eq_tuples = []
		G_i = current_groups[G_i_name]

		for G_j_1_name in current_groups:

			if G_j_1_name == G_i_name:
				continue

			G_j_1 = current_groups[G_j_1_name]
			_ , overlaps_dict = G_j_1.get_intersections_overlaps_and_transformations_with_other_groups(current_groups)
			intersections = intersections_dict_G_i[G_j_1_name]

			for G_j_2_name in current_groups:

				if G_j_2_name == G_j_1_name or G_j_2_name == G_i_name:
					continue

				G_j_2 = current_groups[G_j_2_name]

				overlaps = overlaps_dict[G_j_2_name]

				for B_mj1 in intersections:

					for ovr in overlaps:

						B_m1_name = ovr[0]
						B_m2_name = ovr[1]

						H_Gi_Gj1_mj1 = intersections[B_mj1]
						H_Gj1_m1 = G_j_1.corrected_absolute_transformations[B_m1_name]
						H_m1_m2 = self.pairwise_trasformations[B_m1_name][B_m2_name]

						H_Gi_m1 = np.matmul(H_Gj1_m1,H_Gi_Gj1_mj1)

						eq_tuples.append((B_m2_name,B_m1_name,H_m1_m2,H_Gi_m1))

		return eq_tuples


	def get_new_G_i_using_new_coords(self,new_coords,G_i_name,):

		new_ref_coords = new_coords[G_i_name]
		G_i_new = Group(self.images_dict,self.pairwise_trasformations,G_i_name,new_ref_coords,self.coefs,self.transformation_type)
		
		for img_name in new_coords:
			if img_name == G_i_name:
				continue

			coords = new_coords[img_name]

			G_i_new.corrected_images.append(self.images_dict[img_name])
			G_i_new.corrected_coordinates[img_name] = coords

			pts1 = np.float32([[coords['UL'][0],coords['UL'][1]],\
			[coords['UR'][0],coords['UR'][1]],\
			[coords['LR'][0],coords['LR'][1]],\
			[coords['LL'][0],coords['LL'][1]]])

			pts2 = np.float32([[new_ref_coords['UL'][0],new_ref_coords['UL'][1]],\
			[new_ref_coords['UR'][0],new_ref_coords['UR'][1]],\
			[new_ref_coords['LR'][0],new_ref_coords['LR'][1]],\
			[new_ref_coords['LL'][0],new_ref_coords['LL'][1]]])

			T = cv_util.estimate_base_transformations(pts1,pts2,self.transformation_type)

			G_i_new.corrected_absolute_transformations[img_name] = T

		return G_i_new

	def get_next_group(self,G_i_name,current_groups):

		eq_tuples = []
		G_i = current_groups[G_i_name]

		intersections_dict , overlaps_dict = G_i.get_intersections_overlaps_and_transformations_with_other_groups(current_groups)
		

		for G_j_name in current_groups:

			G_j = current_groups[G_j_name]

			if G_j_name == G_i_name:
				continue

			# step 3
			eq_tuples += self.get_equations_for_overlapping_images_step3(current_groups,G_i_name,G_j_name,overlaps_dict)

			# step 4
			eq_tuples += self.get_equations_for_all_images_based_on_overlappings_step4(current_groups,G_i_name,G_j_name,overlaps_dict)

			# step 6
			eq_tuples += self.get_equations_for_all_images_based_on_intersections_step6(current_groups,G_i_name,G_j_name,intersections_dict)

			# step 7
			eq_tuples += self.get_equations_for_overlapping_images_based_on_intersections_step7(current_groups,G_i_name,G_j_name,overlaps_dict,intersections_dict)

		# step (i)
		eq_tuples += self.get_equations_for_overlapping_groups_based_on_intersections_step_i(current_groups,G_i_name,intersections_dict)

		# print('Number of corrected vs. number of equations: {0},{1}'.format(len(G_i.corrected_images),len(eq_tuples)))

		# print(len(list(set([(a,b) for a,b,c,d in eq_tuples]))),len(eq_tuples))

		eq_tuples = self.drop_equations(eq_tuples,self.number_equation_to_pick_from_unique_tuples)

		print('   \tNumber of corrected vs. number of equations: {0} --> {1}'.format(len(G_i.corrected_images),len(eq_tuples)))
		sys.stdout.flush()

		lsq = LinearLeastSquares_Solver(eq_tuples,G_i.corrected_coordinates,G_i.coefs,G_i.transformation_type)

		new_coords = lsq.solve(self.use_homogenouse)

		G_i_new = self.get_new_G_i_using_new_coords(new_coords,G_i_name)

		return G_i_new

	def drop_equations(self,eq_tuples,number_eq_to_pick):
		
		unique_names = list(set([(a,b) for a,b,c,d in eq_tuples]))
		
		unique_names_dict = {}

		for a,b in unique_names:
			unique_names_dict[(a,b)] = []

		for a,b,c,d in eq_tuples:
			unique_names_dict[(a,b)].append((a,b,c,d))

		new_tuples = []

		for a,b in unique_names:

			random.shuffle(unique_names_dict[(a,b)])

			for a1,b1,c1,d1 in unique_names_dict[(a,b)][:number_eq_to_pick]:

				new_tuples.append((a1,b1,c1,d1))

		return new_tuples

	def get_next_group_parallel(args):

		G_i_name = args[0]
		current_groups = args[1]
		self_obj = args[2]

		eq_tuples = []
		G_i = current_groups[G_i_name]

		intersections_dict , overlaps_dict = G_i.get_intersections_overlaps_and_transformations_with_other_groups(current_groups)
		

		for G_j_name in current_groups:

			G_j = current_groups[G_j_name]

			if G_j_name == G_i_name:
				continue

			# step 3
			eq_tuples += self_obj.get_equations_for_overlapping_images_step3(current_groups,G_i_name,G_j_name,overlaps_dict)

			# step 4
			eq_tuples += self_obj.get_equations_for_all_images_based_on_overlappings_step4(current_groups,G_i_name,G_j_name,overlaps_dict)

			# step 6
			eq_tuples += self_obj.get_equations_for_all_images_based_on_intersections_step6(current_groups,G_i_name,G_j_name,intersections_dict)

			# step 7
			eq_tuples += self_obj.get_equations_for_overlapping_images_based_on_intersections_step7(current_groups,G_i_name,G_j_name,overlaps_dict,intersections_dict)

		# step (i)
		eq_tuples += self_obj.get_equations_for_overlapping_groups_based_on_intersections_step_i(current_groups,G_i_name,intersections_dict)

		# print('Number of corrected vs. number of equations: {0},{1}'.format(len(G_i.corrected_images),len(eq_tuples)))

		# print(len(list(set([(a,b) for a,b,c,d in eq_tuples]))),len(eq_tuples))

		eq_tuples = self_obj.drop_equations(eq_tuples,self_obj.number_equation_to_pick_from_unique_tuples)

		print('   \tNumber of corrected vs. number of equations: {0} --> {1}'.format(len(G_i.corrected_images),len(eq_tuples)))
		sys.stdout.flush()

		lsq = LinearLeastSquares_Solver(eq_tuples,G_i.corrected_coordinates,G_i.coefs,G_i.transformation_type)

		new_coords = lsq.solve(self_obj.use_homogenouse)

		G_i_new = self_obj.get_new_G_i_using_new_coords(new_coords,G_i_name)

		return G_i_name,G_i_new

	def all_groups_correction_finished(self,groups):

		flag = True

		for r_name in groups:
			if not groups[r_name].is_correction_finished():
				flag = False

		return flag

	def get_hierarchy_of_grids(self,grid_size_level_0):
		
		min_lat = sys.maxsize
		min_lon = sys.maxsize
		max_lat = -sys.maxsize
		max_lon = -sys.maxsize

		for img in self.images:
			
			if img.lat>max_lat:
				max_lat = img.lat
			if img.lat<min_lat:
				min_lat = img.lat

			if img.lon>max_lon:
				max_lon = img.lon
			if img.lon<min_lon:
				min_lon = img.lon

		distance_lon = (max_lon - min_lon)/grid_size_level_0[0]
		distance_lat = (max_lat - min_lat)/grid_size_level_0[1]

		X = np.arange(min_lon+distance_lon/2,max_lon,distance_lon)
		Y = np.arange(min_lat+distance_lat/2,max_lat,distance_lat)

		# print(min_lon,max_lon,min_lat,max_lat)
		# print(distance_lon,distance_lat)

		associated_img = {}
		used_images_names = []

		if len(X)*len(Y)>len(self.images):
			print('Grid size is large for this number of images. Choose smaller grid size.')
			return None

		for x in X:
			for y in Y:
				min_distance = sys.maxsize
				min_image = None

				for img in self.images:
					d = img.get_distance_point([x,y])
					if d<min_distance:
						if len(used_images_names) == 0 or img.name not in used_images_names:
							min_distance = d
							min_image = img

				associated_img[(x,y)] = min_image.name
				used_images_names.append(min_image.name)

		current_level = 0
		hierarchy_of_grids = {}

		while True:

			hierarchy_of_grids[current_level] = {}

			for i,x in enumerate(X):
				for j,y in enumerate(Y):
					hierarchy_of_grids[current_level][associated_img[(x,y)]] = []

					if i>0:
						if j>0:
							hierarchy_of_grids[current_level][associated_img[(x,y)]].append(associated_img[(X[i-1],Y[j])])
							hierarchy_of_grids[current_level][associated_img[(x,y)]].append(associated_img[(X[i],Y[j-1])])
						else:
							hierarchy_of_grids[current_level][associated_img[(x,y)]].append(associated_img[(X[i-1],Y[j])])
					else:
						if j>0:
							hierarchy_of_grids[current_level][associated_img[(x,y)]].append(associated_img[(X[i],Y[j-1])])
						
					if i<len(X)-1:
						if j<len(Y)-1:
							hierarchy_of_grids[current_level][associated_img[(x,y)]].append(associated_img[(X[i+1],Y[j])])
							hierarchy_of_grids[current_level][associated_img[(x,y)]].append(associated_img[(X[i],Y[j+1])])
						else:
							hierarchy_of_grids[current_level][associated_img[(x,y)]].append(associated_img[(X[i+1],Y[j])])
					else:
						if j<len(Y)-1:
							hierarchy_of_grids[current_level][associated_img[(x,y)]].append(associated_img[(X[i],Y[j+1])])
						

			current_level+=1

			if len(X) == 1 and len(Y) == 1:
				break

			if len(X) >=2:
				X = X[1::2]
			else:
				X = X[len(X)-1:len(X)]

			if len(Y) >=2:
				Y = Y[1::2]
			else:
				Y = Y[len(Y)-1:len(Y)]

		
		# import matplotlib.pyplot as plt


		# for i in hierarchy_of_grids:

		#	 img_names = hierarchy_of_grids[i]

		#	 for img_n in img_names:
				
		#		 plt.clf()

		#		 plt.scatter([min_lon,max_lon,max_lon,min_lon],[min_lat,min_lat,max_lat,max_lat],color='r')
				
		#		 neighbor_names = img_names[img_n]

		#		 plt.scatter(self.images_dict[img_n].lon,self.images_dict[img_n].lat,color='b',marker='*')

		#		 for ne in neighbor_names:
		#			 plt.scatter(self.images_dict[ne].lon,self.images_dict[ne].lat,color='g',marker='.')

		# plt.savefig('~/tmp.png')
				# plt.show()

		return hierarchy_of_grids

	def enough_itersections_in_next_hierarchy_level(self,hierarchy_of_grids,next_hierarchy_level,current_groups,min_intersect):
		
		if next_hierarchy_level not in hierarchy_of_grids:

			next_hierarchy_level -= 1

		grid_image_neighbor_dict_in_current_level = hierarchy_of_grids[next_hierarchy_level]

		flag = True

		for img_name in grid_image_neighbor_dict_in_current_level:

			neighbor_names = grid_image_neighbor_dict_in_current_level[img_name]

			for neighbor_name in neighbor_names:

				G_i = current_groups[img_name]
				G_j = current_groups[neighbor_name]

				inter = len(G_i.intersect_corrected_images(G_j))+len(G_i.overlap_corrected_images(G_j))

				if inter < min_intersect:
					# print(inter)
					flag = False

		return flag

	def drop_groups(self,hierarchy_of_grids,current_hierarchy_level,next_iteration_groups):
		
		new_next_iteration_groups = {}

		grid_image_neighbor_dict_in_current_level = hierarchy_of_grids[current_hierarchy_level]

		for grid_image in grid_image_neighbor_dict_in_current_level:

			new_next_iteration_groups[grid_image] = next_iteration_groups[grid_image]

		return new_next_iteration_groups

	def group_propagation_correction(self):

		current_groups = {}
		for img in self.images:
			current_groups[img.name] = Group(self.images_dict,self.pairwise_trasformations,img.name,self.ref_coords,self.coefs,self.transformation_type)

		current_hierarchy_level = 0

		# hierarchy_of_grids = self.get_hierarchy_of_grids([2,6])
		hierarchy_of_grids = self.get_hierarchy_of_grids([4,110])

		# self.initialize_groups(current_groups)

		next_iteration_groups = {}

		while True:

			if current_hierarchy_level in hierarchy_of_grids:
				print('>>>>>> Hierarchy Level {0} with {1} number of grid images and {2} number of total groups.'.format(current_hierarchy_level,\
				len(hierarchy_of_grids[current_hierarchy_level]),len(current_groups)))
			else:
				print('>>>>>> Final round of corrected with {0} number of grid images and {1} number of total groups.'.format(\
					len(hierarchy_of_grids[current_hierarchy_level-1]),len(current_groups)))

			if self.use_parallel_process:
				
				args_list = []

				for group_img_name in current_groups:
					args_list.append((group_img_name,current_groups,self))

				processes = multiprocessing.Pool(self.cores)
				results = processes.map(MegaStitch_Multi_Group.get_next_group_parallel,args_list)
				processes.close()

				for gp_name,gp in results:
					next_iteration_groups[gp_name] = gp

			else:

				for group_img_name in current_groups:

					next_iteration_groups[group_img_name] = self.get_next_group(group_img_name,current_groups)

			if  len(next_iteration_groups) == 1 and self.all_groups_correction_finished(next_iteration_groups):
				break

			if self.enough_itersections_in_next_hierarchy_level(hierarchy_of_grids,current_hierarchy_level+1,current_groups,1):

				next_iteration_groups = self.drop_groups(hierarchy_of_grids,current_hierarchy_level,next_iteration_groups)
				current_hierarchy_level += 1

			current_groups = next_iteration_groups
			
			next_iteration_groups = {}


		final_group_corrected_images = [next_iteration_groups[img_name] for img_name in next_iteration_groups][0].corrected_coordinates
		return final_group_corrected_images


class MegaStitch_Hybrid_Multi_Group_Drone:

	def __init__(self,imgs,pairwise_tr,coefs,transf_type,x,y,is_normalized_keypoints,use_homogenouse,parallel,parallel_cores,eq_to_pick):

		self.images = imgs
		self.images_dict = {}

		for img in self.images:
			self.images_dict[img.name] = img

		self.pairwise_trasformations = {}

		for img1_name in pairwise_tr:
			self.pairwise_trasformations[img1_name] = {}
			for img2_name in pairwise_tr[img1_name]:
				if type(pairwise_tr[img1_name][img2_name]) == np.ndarray:
					self.pairwise_trasformations[img1_name][img2_name] = pairwise_tr[img1_name][img2_name]
				else:
					self.pairwise_trasformations[img1_name][img2_name] = pairwise_tr[img1_name][img2_name][0]

		self.coefs = coefs
		self.transformation_type = transf_type

		self.is_normalized_keypoints = is_normalized_keypoints
		
		self.use_homogenouse = use_homogenouse

		if use_homogenouse:

			if is_normalized_keypoints:
				self.ref_coords = {'UL':[-0.5,-0.5,1],'UR':[0.5,-0.5,1],'LR':[0.5,0.5,1],'LL':[-0.5,0.5,1]}
			else:
				self.ref_coords = {'UL':[0,0,1],'UR':[x,0,1],'LR':[x,y,1],'LL':[0,y,1]}
		else:

			if is_normalized_keypoints:
				self.ref_coords = {'UL':[-0.5,-0.5],'UR':[0.5,-0.5],'LR':[0.5,0.5],'LL':[-0.5,0.5]}
			else:
				self.ref_coords = {'UL':[0,0],'UR':[x,0],'LR':[x,y],'LL':[0,y]}

		self.use_parallel_process = parallel
		self.cores = parallel_cores
		self.number_equation_to_pick_from_unique_tuples = eq_to_pick

	def pairwise_transformation_exists(self,img1,img2):

		if img1.name in self.pairwise_trasformations and img2.name in self.pairwise_trasformations[img1.name]:
			return True
		else:
			return False

	def initialize_groups(self,groups):
		
		for r_key in groups:

			groups[r_key].expand_neighbors_3_loop_method()

	def get_equations_for_overlapping_images_step3(self,current_groups,G_i_name,G_j_name,overlaps_dict):

		eq_tuples = []

		G_i = current_groups[G_i_name]
		G_j = current_groups[G_j_name]

		overlaps = overlaps_dict[G_j_name]

		for ovr in overlaps:

			B_m1_name = ovr[0]
			B_m2_name = ovr[1]

			H_Gi_m1 = G_i.corrected_absolute_transformations[B_m1_name]
			H_m1_m2 = self.pairwise_trasformations[B_m1_name][B_m2_name]

			eq_tuples.append((B_m2_name, B_m1_name, H_m1_m2, H_Gi_m1))

		return eq_tuples

	def get_equations_for_all_images_based_on_overlappings_step4(self,current_groups,G_i_name,G_j_name,overlaps_dict):

		eq_tuples = []

		G_i = current_groups[G_i_name]
		G_j = current_groups[G_j_name]

		overlaps = overlaps_dict[G_j_name]

		for Bm_prim in G_j.corrected_images:
			for ovr in overlaps:

				B_m1_name = ovr[0]
				B_m2_name = ovr[1]

				if Bm_prim.name == B_m2_name:
					continue

				Bm_prim_new = {}

				H_Gj_m2 = G_j.corrected_absolute_transformations[B_m2_name]
				H_m1_m2 = self.pairwise_trasformations[B_m1_name][B_m2_name]
				H_m1_Gi = np.linalg.inv(G_i.corrected_absolute_transformations[B_m1_name])

				for k,key in enumerate(['UL','UR','LR','LL']):

					coord_Bm_prime = G_j.corrected_coordinates[Bm_prim.name]
				
					Bm_prim_Gj = [coord_Bm_prime[key][0],coord_Bm_prime[key][1],1]
					tmp = np.matmul(H_m1_Gi,np.matmul(H_m1_m2,np.matmul(H_Gj_m2,Bm_prim_Gj)))
					tmp = tmp/tmp[2]

					Bm_prim_new[key] = [tmp[0],tmp[1],tmp[2]]
				
				eq_tuples.append((Bm_prim.name, None, Bm_prim_new, None))

		return eq_tuples

	def get_equations_for_all_images_based_on_intersections_step6(self,current_groups,G_i_name,G_j_name,intersections_dict):

		eq_tuples = []

		G_i = current_groups[G_i_name]
		G_j = current_groups[G_j_name]

		intersections = intersections_dict[G_j_name]
		
		for Bm_prim in G_j.corrected_images:
			for Bm in intersections:

				Bm_prim_new = {}

				H_Gj_Gi_m = np.linalg.inv(intersections[Bm])

				for k,key in enumerate(['UL','UR','LR','LL']):

					coord_Bm_prime = G_j.corrected_coordinates[Bm_prim.name]
				
					Bm_prim_Gj = [coord_Bm_prime[key][0],coord_Bm_prime[key][1],1]
					tmp = np.matmul(H_Gj_Gi_m,Bm_prim_Gj)
					tmp = tmp/tmp[2]

					Bm_prim_new[key] = [tmp[0],tmp[1],tmp[2]]
				
				eq_tuples.append((Bm_prim.name, None, Bm_prim_new, None))

		return eq_tuples

	def get_equations_for_overlapping_images_based_on_intersections_step7(self,current_groups,G_i_name,G_j_name,overlaps_dict,intersections_dict):

		eq_tuples = []

		G_i = current_groups[G_i_name]
		G_j = current_groups[G_j_name]

		overlaps = overlaps_dict[G_j_name]
		intersections = intersections_dict[G_j_name]

		for Bm in intersections:

			for ovr in overlaps:

				B_m1_name = ovr[0]
				B_m2_name = ovr[1]

				H_Gi_Gj_m = intersections[Bm]
				H_Gj_m2 = G_j.corrected_absolute_transformations[B_m2_name]
				H_m2_m1 = np.linalg.inv(self.pairwise_trasformations[B_m1_name][B_m2_name])

				H_Gi_m2 = np.matmul(H_Gj_m2,H_Gi_Gj_m)

				eq_tuples.append((B_m1_name,B_m2_name,H_m2_m1,H_Gi_m2))

		return eq_tuples

	def get_equations_for_overlapping_groups_based_on_intersections_step_i(self,current_groups,G_i_name,intersections_dict_G_i):

		eq_tuples = []
		G_i = current_groups[G_i_name]

		for G_j_1_name in current_groups:

			if G_j_1_name == G_i_name:
				continue

			G_j_1 = current_groups[G_j_1_name]
			_ , overlaps_dict = G_j_1.get_intersections_overlaps_and_transformations_with_other_groups(current_groups)
			intersections = intersections_dict_G_i[G_j_1_name]

			for G_j_2_name in current_groups:

				if G_j_2_name == G_j_1_name or G_j_2_name == G_i_name:
					continue

				G_j_2 = current_groups[G_j_2_name]

				overlaps = overlaps_dict[G_j_2_name]

				for B_mj1 in intersections:

					for ovr in overlaps:

						B_m1_name = ovr[0]
						B_m2_name = ovr[1]

						H_Gi_Gj1_mj1 = intersections[B_mj1]
						H_Gj1_m1 = G_j_1.corrected_absolute_transformations[B_m1_name]
						H_m1_m2 = self.pairwise_trasformations[B_m1_name][B_m2_name]

						H_Gi_m1 = np.matmul(H_Gj1_m1,H_Gi_Gj1_mj1)

						eq_tuples.append((B_m2_name,B_m1_name,H_m1_m2,H_Gi_m1))

		return eq_tuples

	def get_new_G_i_using_new_coords(self,new_coords,G_i_name,):

		new_ref_coords = new_coords[G_i_name]
		G_i_new = Group(self.images_dict,self.pairwise_trasformations,G_i_name,new_ref_coords,self.coefs,self.transformation_type)
		
		for img_name in new_coords:
			if img_name == G_i_name:
				continue

			coords = new_coords[img_name]

			G_i_new.corrected_images.append(self.images_dict[img_name])
			G_i_new.corrected_coordinates[img_name] = coords

			pts1 = np.float32([[coords['UL'][0],coords['UL'][1]],\
			[coords['UR'][0],coords['UR'][1]],\
			[coords['LR'][0],coords['LR'][1]],\
			[coords['LL'][0],coords['LL'][1]]])

			pts2 = np.float32([[new_ref_coords['UL'][0],new_ref_coords['UL'][1]],\
			[new_ref_coords['UR'][0],new_ref_coords['UR'][1]],\
			[new_ref_coords['LR'][0],new_ref_coords['LR'][1]],\
			[new_ref_coords['LL'][0],new_ref_coords['LL'][1]]])

			T = cv_util.estimate_base_transformations(pts1,pts2,self.transformation_type)

			G_i_new.corrected_absolute_transformations[img_name] = T

		return G_i_new

	def get_next_group(self,G_i_name,current_groups):	

		eq_tuples = []
		G_i = current_groups[G_i_name]

		if len(current_groups) == 1:
			while not G_i.is_correction_finished():
				G_i.single_step_three_loop_least_squares_correction()

			return G_i

		intersections_dict , overlaps_dict = G_i.get_intersections_overlaps_and_transformations_with_other_groups(current_groups)
		

		for G_j_name in current_groups:

			G_j = current_groups[G_j_name]

			if G_j_name == G_i_name:
				continue

			# step 3
			eq_tuples += self.get_equations_for_overlapping_images_step3(current_groups,G_i_name,G_j_name,overlaps_dict)

			# step 4
			eq_tuples += self.get_equations_for_all_images_based_on_overlappings_step4(current_groups,G_i_name,G_j_name,overlaps_dict)

			# step 6
			eq_tuples += self.get_equations_for_all_images_based_on_intersections_step6(current_groups,G_i_name,G_j_name,intersections_dict)

			# step 7
			eq_tuples += self.get_equations_for_overlapping_images_based_on_intersections_step7(current_groups,G_i_name,G_j_name,overlaps_dict,intersections_dict)

		# step (i)
		eq_tuples += self.get_equations_for_overlapping_groups_based_on_intersections_step_i(current_groups,G_i_name,intersections_dict)

		# print('Number of corrected vs. number of equations: {0},{1}'.format(len(G_i.corrected_images),len(eq_tuples)))

		# print(len(list(set([(a,b) for a,b,c,d in eq_tuples]))),len(eq_tuples))

		eq_tuples = self.drop_equations(eq_tuples,self.number_equation_to_pick_from_unique_tuples)

		print('   \tNumber of corrected vs. number of equations: {0} --> {1}'.format(len(G_i.corrected_images),len(eq_tuples)))
		sys.stdout.flush()

		lsq = LinearLeastSquares_Solver(eq_tuples,G_i.corrected_coordinates,G_i.coefs,G_i.transformation_type)

		new_coords = lsq.solve(self.use_homogenouse)

		G_i_new = self.get_new_G_i_using_new_coords(new_coords,G_i_name)

		return G_i_new

	def drop_equations(self,eq_tuples,number_eq_to_pick):
		
		unique_names = list(set([(a,b) for a,b,c,d in eq_tuples]))
		
		unique_names_dict = {}

		for a,b in unique_names:
			unique_names_dict[(a,b)] = []

		for a,b,c,d in eq_tuples:
			unique_names_dict[(a,b)].append((a,b,c,d))

		new_tuples = []

		for a,b in unique_names:

			random.shuffle(unique_names_dict[(a,b)])

			for a1,b1,c1,d1 in unique_names_dict[(a,b)][:number_eq_to_pick]:

				new_tuples.append((a1,b1,c1,d1))

		return new_tuples

	def get_next_group_parallel(args):

		G_i_name = args[0]
		current_groups = args[1]
		self_obj = args[2]

		eq_tuples = []
		G_i = current_groups[G_i_name]

		intersections_dict , overlaps_dict = G_i.get_intersections_overlaps_and_transformations_with_other_groups(current_groups)
		

		for G_j_name in current_groups:

			G_j = current_groups[G_j_name]

			if G_j_name == G_i_name:
				continue

			# step 3
			eq_tuples += self_obj.get_equations_for_overlapping_images_step3(current_groups,G_i_name,G_j_name,overlaps_dict)

			# step 4
			eq_tuples += self_obj.get_equations_for_all_images_based_on_overlappings_step4(current_groups,G_i_name,G_j_name,overlaps_dict)

			# step 6
			eq_tuples += self_obj.get_equations_for_all_images_based_on_intersections_step6(current_groups,G_i_name,G_j_name,intersections_dict)

			# step 7
			eq_tuples += self_obj.get_equations_for_overlapping_images_based_on_intersections_step7(current_groups,G_i_name,G_j_name,overlaps_dict,intersections_dict)

		# step (i)
		eq_tuples += self_obj.get_equations_for_overlapping_groups_based_on_intersections_step_i(current_groups,G_i_name,intersections_dict)

		# print('Number of corrected vs. number of equations: {0},{1}'.format(len(G_i.corrected_images),len(eq_tuples)))

		# print(len(list(set([(a,b) for a,b,c,d in eq_tuples]))),len(eq_tuples))

		eq_tuples = self_obj.drop_equations(eq_tuples,self_obj.number_equation_to_pick_from_unique_tuples)

		print('   \tNumber of corrected vs. number of equations: {0} --> {1}'.format(len(G_i.corrected_images),len(eq_tuples)))
		sys.stdout.flush()

		lsq = LinearLeastSquares_Solver(eq_tuples,G_i.corrected_coordinates,G_i.coefs,G_i.transformation_type)

		new_coords = lsq.solve(self_obj.use_homogenouse)

		G_i_new = self_obj.get_new_G_i_using_new_coords(new_coords,G_i_name)

		return G_i_name,G_i_new

	def all_groups_correction_finished(self,groups):

		flag = True

		for r_name in groups:
			if not groups[r_name].is_correction_finished():
				flag = False

		return flag

	def get_hierarchy_of_grids(self,grid_size_level_0):
		
		min_lat = sys.maxsize
		min_lon = sys.maxsize
		max_lat = -sys.maxsize
		max_lon = -sys.maxsize

		for img in self.images:
			
			if img.lat>max_lat:
				max_lat = img.lat
			if img.lat<min_lat:
				min_lat = img.lat

			if img.lon>max_lon:
				max_lon = img.lon
			if img.lon<min_lon:
				min_lon = img.lon

		distance_lon = (max_lon - min_lon)/grid_size_level_0[0]
		distance_lat = (max_lat - min_lat)/grid_size_level_0[1]

		X = np.arange(min_lon+distance_lon/2,max_lon,distance_lon)
		Y = np.arange(min_lat+distance_lat/2,max_lat,distance_lat)

		# print(min_lon,max_lon,min_lat,max_lat)
		# print(distance_lon,distance_lat)

		associated_img = {}
		used_images_names = []

		if len(X)*len(Y)>len(self.images):
			print('Grid size is large for this number of images. Choose smaller grid size.')
			return None

		for x in X:
			for y in Y:
				min_distance = sys.maxsize
				min_image = None

				for img in self.images:
					d = img.get_distance_point([x,y])
					if d<min_distance:
						if len(used_images_names) == 0 or img.name not in used_images_names:
							min_distance = d
							min_image = img

				associated_img[(x,y)] = min_image.name
				used_images_names.append(min_image.name)

		current_level = 0
		hierarchy_of_grids = {}

		while True:

			hierarchy_of_grids[current_level] = {}

			for i,x in enumerate(X):
				for j,y in enumerate(Y):
					hierarchy_of_grids[current_level][associated_img[(x,y)]] = []

					if i>0:
						if j>0:
							hierarchy_of_grids[current_level][associated_img[(x,y)]].append(associated_img[(X[i-1],Y[j])])
							hierarchy_of_grids[current_level][associated_img[(x,y)]].append(associated_img[(X[i],Y[j-1])])
						else:
							hierarchy_of_grids[current_level][associated_img[(x,y)]].append(associated_img[(X[i-1],Y[j])])
					else:
						if j>0:
							hierarchy_of_grids[current_level][associated_img[(x,y)]].append(associated_img[(X[i],Y[j-1])])
						
					if i<len(X)-1:
						if j<len(Y)-1:
							hierarchy_of_grids[current_level][associated_img[(x,y)]].append(associated_img[(X[i+1],Y[j])])
							hierarchy_of_grids[current_level][associated_img[(x,y)]].append(associated_img[(X[i],Y[j+1])])
						else:
							hierarchy_of_grids[current_level][associated_img[(x,y)]].append(associated_img[(X[i+1],Y[j])])
					else:
						if j<len(Y)-1:
							hierarchy_of_grids[current_level][associated_img[(x,y)]].append(associated_img[(X[i],Y[j+1])])
						

			current_level+=1

			if len(X) == 1 and len(Y) == 1:
				break

			if len(X) >=2:
				X = X[1::2]
			else:
				X = X[len(X)-1:len(X)]

			if len(Y) >=2:
				Y = Y[1::2]
			else:
				Y = Y[len(Y)-1:len(Y)]

		
		# import matplotlib.pyplot as plt


		# for i in hierarchy_of_grids:

		#	 img_names = hierarchy_of_grids[i]

		#	 for img_n in img_names:
				
		#		 plt.clf()

		#		 plt.scatter([min_lon,max_lon,max_lon,min_lon],[min_lat,min_lat,max_lat,max_lat],color='r')
				
		#		 neighbor_names = img_names[img_n]

		#		 plt.scatter(self.images_dict[img_n].lon,self.images_dict[img_n].lat,color='b',marker='*')

		#		 for ne in neighbor_names:
		#			 plt.scatter(self.images_dict[ne].lon,self.images_dict[ne].lat,color='g',marker='.')

		# plt.savefig('~/tmp.png')
				# plt.show()

		return hierarchy_of_grids

	def enough_itersections_neighboring_group(self,hierarchy_of_grids,hierarchy_level,current_groups,min_intersect):
		
		if hierarchy_level not in hierarchy_of_grids:

			hierarchy_level -= 1

		grid_image_neighbor_dict_in_current_level = hierarchy_of_grids[hierarchy_level]

		flag = True

		for img_name in grid_image_neighbor_dict_in_current_level:

			neighbor_names = grid_image_neighbor_dict_in_current_level[img_name]

			for neighbor_name in neighbor_names:

				G_i = current_groups[img_name]
				G_j = current_groups[neighbor_name]

				inter = len(G_i.intersect_corrected_images(G_j))

				if inter < min_intersect:
					# print(inter)
					flag = False

		return flag

	def drop_groups(self,hierarchy_of_grids,current_hierarchy_level,next_iteration_groups):
		
		if current_hierarchy_level+1 in hierarchy_of_grids:
			new_next_iteration_groups = {}

			grid_image_neighbor_dict_in_current_level = hierarchy_of_grids[current_hierarchy_level+1]

			for grid_image in grid_image_neighbor_dict_in_current_level:

				new_next_iteration_groups[grid_image] = next_iteration_groups[grid_image]

			return new_next_iteration_groups

		else:

			return next_iteration_groups

	def group_propagation_correction(self,grid_w,grid_h,min_intersect):

		current_hierarchy_level = 0

		hierarchy_of_grids = self.get_hierarchy_of_grids([grid_w,grid_h])

		current_groups = {}
		for img_name in hierarchy_of_grids[0]:
			current_groups[img_name] = Group(self.images_dict,self.pairwise_trasformations,img_name,self.ref_coords,self.coefs,self.transformation_type)

		while not self.enough_itersections_neighboring_group(hierarchy_of_grids,current_hierarchy_level,current_groups,min_intersect):
			
			print('>>> Expanding initial groups with 3 loop algorithm.')

			for gp_name in current_groups:
				current_groups[gp_name].single_step_three_loop_least_squares_correction()

		print('>>> Enough overlap between initial groups found.')

		next_iteration_groups = {}

		while True:

			if current_hierarchy_level in hierarchy_of_grids:
				print('>>>>>> Hierarchy Level {0} with {1} number of grid images and {2} number of total groups.'.format(current_hierarchy_level,\
				len(hierarchy_of_grids[current_hierarchy_level]),len(current_groups)))
			elif current_hierarchy_level - 1 in hierarchy_of_grids:
				print('>>>>>> Final round of corrected with {0} number of grid images and {1} number of total groups.'.format(\
					len(hierarchy_of_grids[current_hierarchy_level-1]),len(current_groups)))
			else:
				print('>>>>>> Final round. Loop?')

			if self.use_parallel_process:
				
				args_list = []

				for group_img_name in current_groups:
					args_list.append((group_img_name,current_groups,self))

				processes = multiprocessing.Pool(self.cores)
				results = processes.map(MegaStitch_Multi_Group.get_next_group_parallel,args_list)
				processes.close()

				for gp_name,gp in results:
					next_iteration_groups[gp_name] = gp

			else:

				for group_img_name in current_groups:

					next_iteration_groups[group_img_name] = self.get_next_group(group_img_name,current_groups)

			if  len(next_iteration_groups) == 1 and self.all_groups_correction_finished(next_iteration_groups):
				break

			if self.enough_itersections_neighboring_group(hierarchy_of_grids,current_hierarchy_level,current_groups,1):

				next_iteration_groups = self.drop_groups(hierarchy_of_grids,current_hierarchy_level,next_iteration_groups)
				current_hierarchy_level += 1

			current_groups = next_iteration_groups
			
			next_iteration_groups = {}


		final_group_corrected_images = [next_iteration_groups[img_name] for img_name in next_iteration_groups][0].corrected_coordinates

		return final_group_corrected_images


class MegaStitch_Multi_Group:

	def __init__(self,imgs,pairwise_tr,coefs,transf_type,x,y,is_normalized_keypoints,use_homogenouse,parallel,parallel_cores,eq_to_pick):

		self.images = imgs
		self.images_dict = {}

		for img in self.images:
			self.images_dict[img.name] = img

		self.pairwise_trasformations = {}

		for img1_name in pairwise_tr:
			self.pairwise_trasformations[img1_name] = {}
			for img2_name in pairwise_tr[img1_name]:
				if type(pairwise_tr[img1_name][img2_name]) == np.ndarray:
					self.pairwise_trasformations[img1_name][img2_name] = pairwise_tr[img1_name][img2_name]
				else:
					self.pairwise_trasformations[img1_name][img2_name] = pairwise_tr[img1_name][img2_name][0]

		self.coefs = coefs
		self.transformation_type = transf_type

		self.is_normalized_keypoints = is_normalized_keypoints
		
		self.use_homogenouse = use_homogenouse

		if use_homogenouse:

			if is_normalized_keypoints:
				self.ref_coords = {'UL':[-0.5,-0.5,1],'UR':[0.5,-0.5,1],'LR':[0.5,0.5,1],'LL':[-0.5,0.5,1]}
			else:
				self.ref_coords = {'UL':[0,0,1],'UR':[x,0,1],'LR':[x,y,1],'LL':[0,y,1]}
		else:

			if is_normalized_keypoints:
				self.ref_coords = {'UL':[-0.5,-0.5],'UR':[0.5,-0.5],'LR':[0.5,0.5],'LL':[-0.5,0.5]}
			else:
				self.ref_coords = {'UL':[0,0],'UR':[x,0],'LR':[x,y],'LL':[0,y]}

		self.use_parallel_process = parallel
		self.cores = parallel_cores
		self.number_equation_to_pick_from_unique_tuples = eq_to_pick


	def pairwise_transformation_exists(self,img1,img2):

		if img1.name in self.pairwise_trasformations and img2.name in self.pairwise_trasformations[img1.name]:
			return True
		else:
			return False


	def initialize_groups(self,groups):
		
		for r_key in groups:

			groups[r_key].expand_neighbors_3_loop_method()

	def get_equations_for_overlapping_images_step3(self,current_groups,G_i_name,G_j_name,overlaps_dict):

		eq_tuples = []

		G_i = current_groups[G_i_name]
		G_j = current_groups[G_j_name]

		overlaps = overlaps_dict[G_j_name]

		for ovr in overlaps:

			B_m1_name = ovr[0]
			B_m2_name = ovr[1]

			H_Gi_m1 = G_i.corrected_absolute_transformations[B_m1_name]
			H_m1_m2 = self.pairwise_trasformations[B_m1_name][B_m2_name]

			eq_tuples.append((B_m2_name, B_m1_name, H_m1_m2, H_Gi_m1))

		return eq_tuples

	def get_equations_for_all_images_based_on_overlappings_step4(self,current_groups,G_i_name,G_j_name,overlaps_dict):

		eq_tuples = []

		G_i = current_groups[G_i_name]
		G_j = current_groups[G_j_name]

		overlaps = overlaps_dict[G_j_name]

		for Bm_prim in G_j.corrected_images:
			for ovr in overlaps:

				B_m1_name = ovr[0]
				B_m2_name = ovr[1]

				if Bm_prim.name == B_m2_name:
					continue

				Bm_prim_new = {}

				H_Gj_m2 = G_j.corrected_absolute_transformations[B_m2_name]
				H_m1_m2 = self.pairwise_trasformations[B_m1_name][B_m2_name]
				H_m1_Gi = np.linalg.inv(G_i.corrected_absolute_transformations[B_m1_name])

				for k,key in enumerate(['UL','UR','LR','LL']):

					coord_Bm_prime = G_j.corrected_coordinates[Bm_prim.name]
				
					Bm_prim_Gj = [coord_Bm_prime[key][0],coord_Bm_prime[key][1],1]
					tmp = np.matmul(H_m1_Gi,np.matmul(H_m1_m2,np.matmul(H_Gj_m2,Bm_prim_Gj)))
					tmp = tmp/tmp[2]

					Bm_prim_new[key] = [tmp[0],tmp[1],tmp[2]]
				
				eq_tuples.append((Bm_prim.name, None, Bm_prim_new, None))

		return eq_tuples

	def get_equations_for_all_images_based_on_intersections_step6(self,current_groups,G_i_name,G_j_name,intersections_dict):

		eq_tuples = []

		G_i = current_groups[G_i_name]
		G_j = current_groups[G_j_name]

		intersections = intersections_dict[G_j_name]
		
		for Bm_prim in G_j.corrected_images:
			for Bm in intersections:

				Bm_prim_new = {}

				H_Gj_Gi_m = np.linalg.inv(intersections[Bm])

				for k,key in enumerate(['UL','UR','LR','LL']):

					coord_Bm_prime = G_j.corrected_coordinates[Bm_prim.name]
				
					Bm_prim_Gj = [coord_Bm_prime[key][0],coord_Bm_prime[key][1],1]
					tmp = np.matmul(H_Gj_Gi_m,Bm_prim_Gj)
					tmp = tmp/tmp[2]

					Bm_prim_new[key] = [tmp[0],tmp[1],tmp[2]]
				
				eq_tuples.append((Bm_prim.name, None, Bm_prim_new, None))

		return eq_tuples

	def get_equations_for_overlapping_images_based_on_intersections_step7(self,current_groups,G_i_name,G_j_name,overlaps_dict,intersections_dict):

		eq_tuples = []

		G_i = current_groups[G_i_name]
		G_j = current_groups[G_j_name]

		overlaps = overlaps_dict[G_j_name]
		intersections = intersections_dict[G_j_name]

		for Bm in intersections:

			for ovr in overlaps:

				B_m1_name = ovr[0]
				B_m2_name = ovr[1]

				H_Gi_Gj_m = intersections[Bm]
				H_Gj_m2 = G_j.corrected_absolute_transformations[B_m2_name]
				H_m2_m1 = np.linalg.inv(self.pairwise_trasformations[B_m1_name][B_m2_name])

				H_Gi_m2 = np.matmul(H_Gj_m2,H_Gi_Gj_m)

				eq_tuples.append((B_m1_name,B_m2_name,H_m2_m1,H_Gi_m2))

		return eq_tuples

	def get_equations_for_overlapping_groups_based_on_intersections_step_i(self,current_groups,G_i_name,intersections_dict_G_i):

		eq_tuples = []
		G_i = current_groups[G_i_name]

		for G_j_1_name in current_groups:

			if G_j_1_name == G_i_name:
				continue

			G_j_1 = current_groups[G_j_1_name]
			_ , overlaps_dict = G_j_1.get_intersections_overlaps_and_transformations_with_other_groups(current_groups)
			intersections = intersections_dict_G_i[G_j_1_name]

			for G_j_2_name in current_groups:

				if G_j_2_name == G_j_1_name or G_j_2_name == G_i_name:
					continue

				G_j_2 = current_groups[G_j_2_name]

				overlaps = overlaps_dict[G_j_2_name]

				for B_mj1 in intersections:

					for ovr in overlaps:

						B_m1_name = ovr[0]
						B_m2_name = ovr[1]

						H_Gi_Gj1_mj1 = intersections[B_mj1]
						H_Gj1_m1 = G_j_1.corrected_absolute_transformations[B_m1_name]
						H_m1_m2 = self.pairwise_trasformations[B_m1_name][B_m2_name]

						H_Gi_m1 = np.matmul(H_Gj1_m1,H_Gi_Gj1_mj1)

						eq_tuples.append((B_m2_name,B_m1_name,H_m1_m2,H_Gi_m1))

		return eq_tuples


	def get_new_G_i_using_new_coords(self,new_coords,G_i_name,):

		new_ref_coords = new_coords[G_i_name]
		G_i_new = Group(self.images_dict,self.pairwise_trasformations,G_i_name,new_ref_coords,self.coefs,self.transformation_type)
		
		for img_name in new_coords:
			if img_name == G_i_name:
				continue

			coords = new_coords[img_name]

			G_i_new.corrected_images.append(self.images_dict[img_name])
			G_i_new.corrected_coordinates[img_name] = coords

			pts1 = np.float32([[coords['UL'][0],coords['UL'][1]],\
			[coords['UR'][0],coords['UR'][1]],\
			[coords['LR'][0],coords['LR'][1]],\
			[coords['LL'][0],coords['LL'][1]]])

			pts2 = np.float32([[new_ref_coords['UL'][0],new_ref_coords['UL'][1]],\
			[new_ref_coords['UR'][0],new_ref_coords['UR'][1]],\
			[new_ref_coords['LR'][0],new_ref_coords['LR'][1]],\
			[new_ref_coords['LL'][0],new_ref_coords['LL'][1]]])

			T = cv_util.estimate_base_transformations(pts1,pts2,self.transformation_type)

			G_i_new.corrected_absolute_transformations[img_name] = T

		return G_i_new

	def get_next_group(self,G_i_name,current_groups):

		eq_tuples = []
		G_i = current_groups[G_i_name]

		intersections_dict , overlaps_dict = G_i.get_intersections_overlaps_and_transformations_with_other_groups(current_groups)
		

		for G_j_name in current_groups:

			G_j = current_groups[G_j_name]

			if G_j_name == G_i_name:
				continue

			# step 3
			eq_tuples += self.get_equations_for_overlapping_images_step3(current_groups,G_i_name,G_j_name,overlaps_dict)

			# step 4
			eq_tuples += self.get_equations_for_all_images_based_on_overlappings_step4(current_groups,G_i_name,G_j_name,overlaps_dict)

			# step 6
			eq_tuples += self.get_equations_for_all_images_based_on_intersections_step6(current_groups,G_i_name,G_j_name,intersections_dict)

			# step 7
			eq_tuples += self.get_equations_for_overlapping_images_based_on_intersections_step7(current_groups,G_i_name,G_j_name,overlaps_dict,intersections_dict)

		# step (i)
		eq_tuples += self.get_equations_for_overlapping_groups_based_on_intersections_step_i(current_groups,G_i_name,intersections_dict)

		# print('Number of corrected vs. number of equations: {0},{1}'.format(len(G_i.corrected_images),len(eq_tuples)))

		# print(len(list(set([(a,b) for a,b,c,d in eq_tuples]))),len(eq_tuples))

		eq_tuples = self.drop_equations(eq_tuples,self.number_equation_to_pick_from_unique_tuples)

		print('   \tNumber of corrected vs. number of equations: {0} --> {1}'.format(len(G_i.corrected_images),len(eq_tuples)))

		lsq = LinearLeastSquares_Solver(eq_tuples,G_i.corrected_coordinates,G_i.coefs,G_i.transformation_type)

		new_coords = lsq.solve(self.use_homogenouse)

		G_i_new = self.get_new_G_i_using_new_coords(new_coords,G_i_name)

		return G_i_new

	def drop_equations(self,eq_tuples,number_eq_to_pick):
		
		unique_names = list(set([(a,b) for a,b,c,d in eq_tuples]))
		
		unique_names_dict = {}

		for a,b in unique_names:
			unique_names_dict[(a,b)] = []

		for a,b,c,d in eq_tuples:
			unique_names_dict[(a,b)].append((a,b,c,d))

		new_tuples = []

		for a,b in unique_names:

			random.shuffle(unique_names_dict[(a,b)])

			for a1,b1,c1,d1 in unique_names_dict[(a,b)][:number_eq_to_pick]:

				new_tuples.append((a1,b1,c1,d1))

		return new_tuples

	def get_next_group_parallel(args):

		G_i_name = args[0]
		current_groups = args[1]
		self_obj = args[2]

		eq_tuples = []
		G_i = current_groups[G_i_name]

		intersections_dict , overlaps_dict = G_i.get_intersections_overlaps_and_transformations_with_other_groups(current_groups)
		

		for G_j_name in current_groups:

			G_j = current_groups[G_j_name]

			if G_j_name == G_i_name:
				continue

			# step 3
			eq_tuples += self_obj.get_equations_for_overlapping_images_step3(current_groups,G_i_name,G_j_name,overlaps_dict)

			# step 4
			eq_tuples += self_obj.get_equations_for_all_images_based_on_overlappings_step4(current_groups,G_i_name,G_j_name,overlaps_dict)

			# step 6
			eq_tuples += self_obj.get_equations_for_all_images_based_on_intersections_step6(current_groups,G_i_name,G_j_name,intersections_dict)

			# step 7
			eq_tuples += self_obj.get_equations_for_overlapping_images_based_on_intersections_step7(current_groups,G_i_name,G_j_name,overlaps_dict,intersections_dict)

		# step (i)
		eq_tuples += self_obj.get_equations_for_overlapping_groups_based_on_intersections_step_i(current_groups,G_i_name,intersections_dict)

		# print('Number of corrected vs. number of equations: {0},{1}'.format(len(G_i.corrected_images),len(eq_tuples)))

		# print(len(list(set([(a,b) for a,b,c,d in eq_tuples]))),len(eq_tuples))

		eq_tuples = self_obj.drop_equations(eq_tuples,self_obj.number_equation_to_pick_from_unique_tuples)

		print('   \tNumber of corrected vs. number of equations: {0} --> {1}'.format(len(G_i.corrected_images),len(eq_tuples)))

		lsq = LinearLeastSquares_Solver(eq_tuples,G_i.corrected_coordinates,G_i.coefs,G_i.transformation_type)

		new_coords = lsq.solve(self_obj.use_homogenouse)

		G_i_new = self_obj.get_new_G_i_using_new_coords(new_coords,G_i_name)

		return G_i_name,G_i_new

	def all_groups_correction_finished(self,groups):

		flag = True

		for r_name in groups:
			if not groups[r_name].is_correction_finished():
				flag = False

		return flag

	def group_propagation_correction(self):

		current_groups = {}
		for img in self.images:
			current_groups[img.name] = Group(self.images_dict,self.pairwise_trasformations,img.name,self.ref_coords,self.coefs,self.transformation_type)


		next_iteration_groups = {}

		while True:

			if self.use_parallel_process:
				
				args_list = []

				for group_img_name in current_groups:
					args_list.append((group_img_name,current_groups,self))

				processes = multiprocessing.Pool(self.cores)
				results = processes.map(MegaStitch_Multi_Group.get_next_group_parallel,args_list)
				processes.close()

				for gp_name,gp in results:
					next_iteration_groups[gp_name] = gp

			else:

				for group_img_name in current_groups:

					next_iteration_groups[group_img_name] = self.get_next_group(group_img_name,current_groups)

			if self.all_groups_correction_finished(next_iteration_groups):
				break

			current_groups = next_iteration_groups
			
			next_iteration_groups = {}


		final_group_corrected_images = [next_iteration_groups[img_name] for img_name in next_iteration_groups][0].corrected_coordinates
		return final_group_corrected_images


class MegaStitch_3L1P_Homogenouse:

	def __init__(self,imgs,pairwise_tr,coefs,img_R_n,img_R_coords,transf_type):

		self.images = imgs
		self.images_dict = {}

		for img in self.images:
			self.images_dict[img.name] = img

		self.pairwise_trasformations = {}

		for img1_name in pairwise_tr:
			self.pairwise_trasformations[img1_name] = {}
			for img2_name in pairwise_tr[img1_name]:

				if type(pairwise_tr[img1_name][img2_name]) == np.ndarray:
					self.pairwise_trasformations[img1_name][img2_name] = pairwise_tr[img1_name][img2_name]
				else:
					self.pairwise_trasformations[img1_name][img2_name] = pairwise_tr[img1_name][img2_name][0]

		self.coefs = coefs
		self.R_image = self.images_dict[img_R_n]
		self.R_coords = img_R_coords
		self.transformation_type = transf_type

	def pairwise_transformation_exists(self,img1,img2):

		if img1.name in self.pairwise_trasformations and img2.name in self.pairwise_trasformations[img1.name]:
			return True
		else:
			return False


	def get_corner_coordinates(self,border_images,adjacent_to_border_images,absolute_transformations,border_images_coords):

		# if T multiplied by the corners of pairwise[0] (in pairwise[0] system) gives the corners of pairwise[1] (in pairwise[0] system)

		eq_tuples = []

		for brd_img in border_images:

			for adj_img1 in adjacent_to_border_images:

				if self.pairwise_transformation_exists(brd_img,adj_img1):

					Bj_name = adj_img1.name
					Bi_name = brd_img.name
					H_BiBj = self.pairwise_trasformations[Bi_name][Bj_name]
					H_ABi = absolute_transformations[Bi_name]

					eq_tuples.append((Bj_name, Bi_name, H_BiBj, H_ABi))

				for adj_img2 in adjacent_to_border_images:

					if self.pairwise_transformation_exists(adj_img2,adj_img1) and \
					self.pairwise_transformation_exists(adj_img2,brd_img):

						B1_name = adj_img1.name
						B2_name = adj_img2.name
						Br_name = brd_img.name

						#Br is X in our notations
						H_ABr = absolute_transformations[Br_name]
						H_BrB2 = self.pairwise_trasformations[B2_name][Br_name]
						
						H_B2B1 = self.pairwise_trasformations[B2_name][B1_name]
						H_AB2 = np.matmul(H_BrB2,H_ABr)

						eq_tuples.append((B1_name, B2_name, H_B2B1, H_AB2))

		if len(eq_tuples) == 0:
			return {}

		lsq = LinearLeastSquares_Solver(eq_tuples,border_images_coords,[self.coefs[0],self.coefs[1]],self.transformation_type)

		new_coords = lsq.solve(True)

		return new_coords
					 
	def update_corner_coords_and_absolute_transformations(self,final_corners,new_coords,absolute_transformations):

		for img_name in new_coords:
			final_corners[img_name] = new_coords[img_name]

		for img_name in final_corners:
			
			if final_corners[img_name] is None:
				continue

			pts1 = np.float32([[final_corners[img_name]['UL'][0],final_corners[img_name]['UL'][1],final_corners[img_name]['UL'][2]],\
			[final_corners[img_name]['UR'][0],final_corners[img_name]['UR'][1],final_corners[img_name]['UR'][2]],\
			[final_corners[img_name]['LR'][0],final_corners[img_name]['LR'][1],final_corners[img_name]['LR'][2]],\
			[final_corners[img_name]['LL'][0],final_corners[img_name]['LL'][1],final_corners[img_name]['LL'][2]]])

			pts2 = np.float32([[self.R_coords['UL'][0],self.R_coords['UL'][1],self.R_coords['UL'][2]],\
			[self.R_coords['UR'][0],self.R_coords['UR'][1],self.R_coords['UR'][2]],\
			[self.R_coords['LR'][0],self.R_coords['LR'][1],self.R_coords['LR'][2]],\
			[self.R_coords['LL'][0],self.R_coords['LL'][1],self.R_coords['LL'][2]]])

			T = cv_util.estimate_base_transformations(pts1,pts2,self.transformation_type)

			absolute_transformations[img_name] = T

		return final_corners,absolute_transformations

	def update_border_and_adjacent_to_border_image_lists(self,border_images,adjacent_to_border_images,finalized_image_names):

		new_border_images = adjacent_to_border_images
		new_adjacent_to_border_images = []
		new_finalized_image_names = finalized_image_names.copy()

		for img in border_images:
			new_finalized_image_names.append(img.name)

		for img in new_border_images:

			if img.name not in self.pairwise_trasformations:
				continue

			for img2_name in self.pairwise_trasformations[img.name]:
				if img2_name not in [a.name for a in new_adjacent_to_border_images] and \
				img2_name not in [a.name for a in new_border_images] and \
				img2_name not in [a for a in new_finalized_image_names]:

					new_adjacent_to_border_images.append(self.images_dict[img2_name])

		

		return new_border_images,new_adjacent_to_border_images,new_finalized_image_names

	def three_loop_least_squares_correction(self):

		final_corners = {}
		finalized_image_names = []

		for img in self.images:
			final_corners[img.name] = None

		final_corners[self.R_image.name] = self.R_coords

		absolute_transformations = {self.R_image.name:np.eye(3)}
		
		border_images = [self.R_image]
		adjacent_to_border_images = [self.images_dict[img_name] for img_name in self.pairwise_trasformations[self.R_image.name]]

		while True:

			
			# get corner coordinates and absolute transformations of the images adjacent to border images

			border_images_coords = {img.name:final_corners[img.name] for img in border_images if final_corners[img.name] is not None}
			new_coords = self.get_corner_coordinates(border_images,adjacent_to_border_images,absolute_transformations,border_images_coords)
			final_corners, absolute_transformations = self.update_corner_coords_and_absolute_transformations(final_corners,new_coords,absolute_transformations)

			# update border and adjacent images

			border_images , adjacent_to_border_images,finalized_image_names = self.update_border_and_adjacent_to_border_image_lists(border_images,adjacent_to_border_images,finalized_image_names)


			if len(adjacent_to_border_images) == 0:
				break

		print('>>> Total number of finalized images: {0}'.format(len(finalized_image_names)+len(border_images)))

		return final_corners


class MegaStitch_Homography:

	def __init__(self,imgs,pairwise_tr,coefs,img_R_n,img_R_coords,w,h,norm,maxmatch):

		self.images = imgs
		self.images_dict = {}

		for img in self.images:
			self.images_dict[img.name] = img

		self.pairwise_trasformations = pairwise_tr

		self.coefs = coefs
		self.R_image = self.images_dict[img_R_n]
		self.R_coords = img_R_coords
		self.transformation_type = cv_util.Transformation.homography

		self.width = w
		self.height = h
		self.is_normalized = norm
		self.max_inliers_to_use_bndl = maxmatch
		# self.max_inliers_to_use_bndl = sys.maxsize

	def pairwise_transformation_exists(self,img1,img2):

		if img1 in self.pairwise_trasformations and img2 in self.pairwise_trasformations[img1]:
			return True
		else:
			return False

	def single_hop_correction(self,corrected_images_corner_and_transformations):

		eq_tuples = []
		corrected_coordinates = {}

		for img_name in corrected_images_corner_and_transformations:
			corrected_coordinates[img_name] = corrected_images_corner_and_transformations[img_name][0]

		for adj_img in self.images_dict:

			if adj_img in corrected_images_corner_and_transformations:
				continue

			for brd_img in corrected_images_corner_and_transformations:

				if self.pairwise_transformation_exists(brd_img,adj_img):

					Bj_name = adj_img
					Bi_name = brd_img
					H_BiBj = self.pairwise_trasformations[Bi_name][Bj_name][0]
					H_ABi = corrected_images_corner_and_transformations[Bi_name][1]

					eq_tuples.append((Bj_name, Bi_name, H_BiBj, H_ABi))

				if self.pairwise_transformation_exists(adj_img,brd_img):

					Bj_name = adj_img
					Bi_name = brd_img
					H_BiBj = np.linalg.inv(self.pairwise_trasformations[Bj_name][Bi_name][0])
					H_ABi = corrected_images_corner_and_transformations[Bi_name][1]

					eq_tuples.append((Bj_name, Bi_name, H_BiBj, H_ABi))

		lsq = LinearLeastSquares_Solver(eq_tuples,corrected_coordinates,self.coefs,self.transformation_type)

		new_coords = lsq.solve()

		newly_corrected_images_corner_and_transformations = {}

		for img_name in new_coords:

			T = self.get_absolute_transformation(new_coords[img_name])
			newly_corrected_images_corner_and_transformations[img_name] = (new_coords[img_name],T)

		return newly_corrected_images_corner_and_transformations
	 
	def get_absolute_transformation(self,coords):

		pts1 = np.float32([[coords['UL'][0],coords['UL'][1]],\
		[coords['UR'][0],coords['UR'][1]],\
		[coords['LR'][0],coords['LR'][1]],\
		[coords['LL'][0],coords['LL'][1]]])

		pts2 = np.float32([[self.R_coords['UL'][0],self.R_coords['UL'][1]],\
		[self.R_coords['UR'][0],self.R_coords['UR'][1]],\
		[self.R_coords['LR'][0],self.R_coords['LR'][1]],\
		[self.R_coords['LL'][0],self.R_coords['LL'][1]]])

		T = cv_util.estimate_base_transformations(pts1,pts2,self.transformation_type)

		return T

	def bundle_adjustment_step(self,corrected_images_corner_and_transformations):

		corrected_coordinates = {}

		for img_name in corrected_images_corner_and_transformations:
			corrected_coordinates[img_name] = corrected_images_corner_and_transformations[img_name][0]

		new_images = [img for img in self.images if img.name in corrected_images_corner_and_transformations]
		new_pairwise = {}

		for img1 in self.pairwise_trasformations:
			new_pairwise[img1] = {}
			for img2 in self.pairwise_trasformations[img1]:
				if img1 in corrected_images_corner_and_transformations and img2 in corrected_images_corner_and_transformations:
					new_pairwise[img1][img2] = self.pairwise_trasformations[img1][img2]

		Bndl = Bundle_Adjustment.Bundle_Adjustment(new_images,new_pairwise,corrected_coordinates,self.coefs,
			self.transformation_type,self.width,self.height,self.is_normalized,self.R_image.name,self.max_inliers_to_use_bndl)

		new_coords,_ = Bndl.solve()

		newly_corrected_images_corner_and_transformations = {}

		for img_name in new_coords:

			T = self.get_absolute_transformation(new_coords[img_name])
			newly_corrected_images_corner_and_transformations[img_name] = (new_coords[img_name],T)

		return newly_corrected_images_corner_and_transformations

	def solve(self):

		corrected_images_corner_and_transformations = {}

		corrected_images_corner_and_transformations[self.R_image.name] = (self.R_coords,np.eye(3))

		while True:

			# get corner coordinates and absolute transformations of the images adjacent to border images w.r.t. border images
			
			corrected_images_corner_and_transformations = self.single_hop_correction(corrected_images_corner_and_transformations)

			# do small bundle adjustment

			corrected_images_corner_and_transformations = self.bundle_adjustment_step(corrected_images_corner_and_transformations)

			if len(corrected_images_corner_and_transformations) == len(self.images):
				break


		corrected_corners = {}

		for img_name in self.images_dict:

			corrected_corners[img_name] = corrected_images_corner_and_transformations[img_name][0]

		return corrected_corners
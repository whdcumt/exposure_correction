
import cv2
import numpy as np
import scipy.sparse as spsparse

		# 1. find all boundaries between two components using adj_mat
		# 2. find the minimum weight between them
		# scratch that, instead need to form and maintain a new adjacency matrix
		# which retains the minimum weight between each component

		# Adjacency matrix is going to shrink by 1 for each component merged 
		# at stage 0 the adjacency matrix and components hash are consistent
		# updates need to handle both components hash AND adjacency matrix

		# NEWEST ISSUE:
		# Need to deal with max internal weights in adjacency

		# TODO: solve out ground truth for test image

class ExposureCorrector():
	img = []
	adj_mat = []
	# ^ this guy represents adjacency of COMPONENTS, not individual pixels
	# the weights indicate the minimum weight between neighboring components
	# diagonal = max internal weight

	components = {}
	# self.internal_weights = {}
	w = 0
	h = 0
	
	k = 300

	def get_adjacency_matrix(self, img):
		self.img = img
		rows = np.shape(img)[0]
		cols = np.shape(img)[1]

		self.w = cols
		self.h = rows

		h_connections = np.ones((cols, 1))
		h_connections[-1] = 0;

		diag_h = np.tile(h_connections, (rows, 1))
		diag_h = diag_h[0:-1]
		print diag_h.shape
		assert(diag_h.shape[0] == rows*cols - 1)
		
		diag_v = np.ones((cols*(rows-1), 1))
		print diag_v.shape
		assert(diag_v.shape[0] == rows*cols - cols)

		diagonals = [diag_h.T, diag_v.T]

		self.adj_mat = spsparse.diags(diagonals, [1, cols], shape=(rows*cols, rows*cols), format='lil', dtype=np.uint8)
		
		return self.adj_mat

	def init_components(self):
		# make a hash of the initial components with each pixel in it's own 
		# component
		num_pixels = self.w * self.h

		for i in range(num_pixels):
			pixel_row = int(i / self.w);
			pixel_col = i % self.w;
			self.components[i] = [(pixel_row, pixel_col)]

		return self.components


	def init_adj_weights(self):
		# at this point each pixel is in it's own component

		# for each pixel, initial weight = abs(pix[i] - pix[j])
		nonzero = self.adj_mat.nonzero()
		print nonzero[0]
		print nonzero[1]		

		for i in range(len(nonzero[0])):
			home_pix = nonzero[0][i]
			neighbor = nonzero[1][i]

			home_i = self.pix_num_to_index(home_pix)
			home_val = self.img[home_i[0], home_i[1]]
			
			neighbor_i = self.pix_num_to_index(neighbor)
			neighbor_val = self.img[neighbor_i[0], neighbor_i[1]]

			self.adj_mat[home_pix, neighbor] = abs(home_val - neighbor_val)

		# for i in range(len(self.components.keys())):
		# 	self.adj_mat[i, i] = 0

	def segment_img(self):
		num_changes = 1
		while (num_changes != 0):
			keys_to_delete = []
			num_changes = 0

			components = self.components.keys()

			for c in components:
				if (c in self.components):
					# keys = self.update_component(c)
					num_changes += self.update_component(c)
					
					# for j in keys:
					# 	keys_to_delete.append(j)
			
			# Delete Keys from merges:
			# for k in keys_to_delete:
			# 	self.components.pop(k, None)

			print "Number of merges: " + str(num_changes)
			print "Total components: " + str(len(self.components.keys()))
			print ""

		return self.components


	def update_component(self, c1):
		keys_to_delete = []
		num_changes = 0
		adj_components = self.adj_mat.getrow(c1).nonzero()
		print adj_components[1]
		for c2 in adj_components[1]:
			# skip if we deleted component already
			if (c2 != c1) and (c2 in self.components):
				if (self.check_boundary(c1, c2, self.k) == False):
					self.merge_components(c1, c2)
					print "merged " + str(c1) + " and " + str(c2)
					num_changes += 1
					# keys_to_delete.append(c2)


		return num_changes


	def check_boundary(self, c1, c2, k):
		boundary = True
		m_int = min([self.adj_mat[c1, c1] + k/len(self.components[c1]), self.adj_mat[c2, c2] + k/len(self.components[c2])]) 
		
		if self.adj_mat[c1, c2] > m_int:
			boundary = True
		else:
			boundary = False

		return boundary

	def merge_components(self, c1, c2):
		# copy all elements of c2 into c1 
		to_copy = self.components[c2]
		for pixel in to_copy:
			self.components[c1].append(pixel)

		# copy adjacency matrix info
		self.merge_adj_mat(c1, c2)

		# delete component in component hash
		self.components.pop(c2, None)

		print self.adj_mat.toarray()
		return 0

	def merge_adj_mat(self, c1, c2):
		c1_adj = self.adj_mat.getrow(c1)
		c2_adj = self.adj_mat.getrow(c2)

		c2_nonzero = c2_adj.nonzero()

		for component in c2_nonzero[1]:
			if component == c1:
				print "update max"
				# update max internal weight
				c1_adj[0, component] = max(c1_adj[0, component], c2_adj[0, component])
			else:
				# update minimum connection to neighbors
				if (c1_adj[0, component] == 0):
					c1_adj[0, component] = c2_adj[0, component]
				elif (c2_adj[0, component] == 0): 
					c1_adj[0, component] = c1_adj[0, component]
				else:
					c1_adj[0, component] = min(c1_adj[0, component], c2_adj[0, component])
					


		self.adj_mat[c1, :] = c1_adj

		# delete c2's row and col in adj_mat?
		# for now, zero out
		self.adj_mat[c2, :] = 0
		self.adj_mat[:, c2] = 0

		return 0

	
	def pix_num_to_index(self, pix_num):
		row = int(pix_num / self.w)
		col = pix_num % self.w
		return (row, col)


	def min_x_weight(self, c1, c2):
		# get the minimum weight between two components

		pass


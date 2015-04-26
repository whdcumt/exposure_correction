
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

		# ----------------------------------------------------------
		# TODO: when merging, grow a single component as much as possible before 
		# moving on 
		# use a queue of pixels to check!
		# ---------------------------------------------------------

class ExposureCorrector():
	img = []
	adj_mat = []
	# ^ this guy represents adjacency of COMPONENTS, not individual pixels
	# 0 == not connected
	# 1 == connected

	# Hash of components (ie the segmentation)
	components = {}
	
	# pairwise weights of connected components
	weights = {}
	
	w = 0
	h = 0
	
	k = 25

	def show_segmentation(self):
		out_img = np.zeros((self.h, self.w)).astype(np.uint8)
		for c in self.components:
			pixels = self.components[c]
			pix_list = []
			for p in pixels:
				pix_list.append(self.img[p[0], p[1]])

			mean_val = int(np.mean(pix_list))

			for p in pixels:
				out_img[p[0], p[1]] = mean_val

		return out_img


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
		# print diag_h.shape
		assert(diag_h.shape[0] == rows*cols - 1)
		
		diag_v = np.ones((cols*(rows-1), 1))
		# print diag_v.shape
		assert(diag_v.shape[0] == rows*cols - cols)

		diagonals = [diag_h.T, diag_v.T]

		self.adj_mat = spsparse.diags(diagonals, [1, cols], shape=(rows*cols, rows*cols), format='lil', dtype=np.uint8)
		
		self.adj_mat = (self.adj_mat + self.adj_mat.transpose()).tolil()
		# print spsparse.isspmatrix_lil(self.adj_mat)
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


	def init_weights(self):
		# at this point each pixel is in it's own component

		# for each pixel, initial weight = abs(pix[i] - pix[j])
		nonzero = self.adj_mat.nonzero()
		# print nonzero[0]
		# print nonzero[1]		

		for i in range(len(nonzero[0])):
			home_pix = nonzero[0][i]
			neighbor = nonzero[1][i]

			home_i = self.pix_num_to_index(home_pix)
			home_val = self.img[home_i[0], home_i[1]]
			
			neighbor_i = self.pix_num_to_index(neighbor)
			neighbor_val = self.img[neighbor_i[0], neighbor_i[1]]

			self.weights[(home_pix, neighbor)] = abs(int(home_val) - int(neighbor_val))
			self.weights[(neighbor, home_pix)] = self.weights[(home_pix, neighbor)]

		for i in range(len(self.components.keys())):
			self.weights[(i, i)] = 0

	def segment_img(self):
		num_changes = 1
	
		components = self.components.keys()

		for c in components:
			while(num_changes > 0):
				if (c in self.components):
					num_changes = self.update_component(c)
				else:
					num_changes = 0
					
			print "Component: " + str(c)

		# print "Number of merges: " + str(num_changes)
		# print "Total components: " + str(len(self.components.keys()))
		# print ""

		return self.components


	def update_component(self, c1):
		keys_to_delete = []
		num_changes = 0
		adj_components = self.adj_mat.getrow(c1).nonzero()
		# print adj_components[1]
		components_to_check = list(adj_components[1])

		while components_to_check != []:
			# skip if we deleted component already
			c2 = components_to_check.pop(0)
			if (c2 != c1) and (c2 in self.components):
				if (self.check_boundary(c1, c2, self.k) == False):
					# print "merged " + str(c1) + " and " + str(c2)
					
					new_components = (self.adj_mat.getrow(c2).nonzero())[1]

					self.merge_components(c1, c2)
					
					# print new_components
					components_to_check = components_to_check + list(new_components)
					# print "merged " + str(c1) + " and " + str(c2)
					num_changes += 1
					


		return num_changes


	def check_boundary(self, c1, c2, k):
		boundary = True
		m_int = min([self.weights[(c1, c1)] + k/len(self.components[c1]), self.weights[(c2, c2)] + k/len(self.components[c2])]) 
		
		if self.weights[(c1, c2)] > m_int:
			boundary = True
		else:
			boundary = False

		return boundary

	def merge_components(self, c1, c2):

		# copy all elements of c2 into c1 
		# print "copying components"
		self.components[c1] = self.components[c1] + self.components[c2]
		# print "done"

		# copy adjacency matrix info
		# print "merging adj mat"
		self.merge_adj_mat(c1, c2)
		# print "done"

		# delete component in component hash
		self.components.pop(c2, None)

		# print self.adj_mat.toarray()
		return 0

	def merge_adj_mat(self, c1, c2):
		# c1_adj = self.adj_mat.getrow(c1)
		c2_adj = self.adj_mat.getrow(c2)

		c2_nonzero = c2_adj.nonzero()[1]


		# print "updating max internal weight"
		# update max internal weight:
		self.weights[(c1, c1)] = max(self.weights[(c1, c1)], self.weights[(c1, c2)])


		# print "updating weights and neighbors"
		for component in c2_nonzero:
			# update connection to neighbors
			if (self.adj_mat[c1, component] == 1) or (c2_adj[0, component] == 1):
				self.adj_mat[c1, component] = 1
			
			# update min neighbor weights
			if ((c1, component) in self.weights) and ((c2, component) in self.weights):
				self.weights[(c1, component)] = min([self.weights[(c1, component)], self.weights[(c2, component)]])
				self.weights[(component, c1)] = self.weights[(c1, component)]
				
				# delete old component weights
				self.weights.pop((c2, component), None)
				self.weights.pop((component, c2), None)
			
			elif ((c2, component) in self.weights):
				self.weights[(c1, component)] = self.weights[(c2, component)]
				self.weights[(component, c1)] = self.weights[(c1, component)]

				# delete old component weights
				self.weights.pop((c2, component), None)
				self.weights.pop((component, c2), None)
			else:
				# retain current min weight
				pass

			# clear old neighbor info
			self.adj_mat[c2, component] = 0
			self.adj_mat[component, c2] = 0	
			
		return 0

	
	def pix_num_to_index(self, pix_num):
		row = int(pix_num / self.w)
		col = pix_num % self.w
		return (row, col)


	def min_x_weight(self, c1, c2):
		# get the minimum weight between two components

		pass


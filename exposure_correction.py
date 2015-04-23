
import cv2
import numpy as np
import scipy.sparse as spsparse


def get_unweighted_adjacency(img):
	# create an unweighted adjacency matrix (4 neighbors)
	rows = np.shape(img)[0]
	cols = np.shape(img)[1]

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

	adj = spsparse.diags(diagonals, [1, cols], shape=(rows*cols, rows*cols), format='lil', dtype=np.uint8)
	
	return adj



def main():
	# img = cv2.imread("test_img.JPG", 0)
	img = np.ones((600, 600))
	
	adj_matrix = get_unweighted_adjacency(img)

	# print adj_matrix[0, 1]

	# cv2.namedWindow("output")
	# dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
	# ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)




if __name__ == "__main__":
	main()
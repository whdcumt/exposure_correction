# test exposure corrector


import numpy as np
import ExposureCorrector as EC



img = np.array([[0, 10, 255], [5, 200, 255]])

corrector = EC.ExposureCorrector()



def main():
	adj = corrector.get_adjacency_matrix(img)
	print adj.toarray()
	corrector.init_components()
	print corrector.components
	
	corrector.init_adj_weights()

	print corrector.adj_mat.toarray()

	corrector.segment_img()

	return 0

if __name__ == "__main__":
	main()
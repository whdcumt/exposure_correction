# test exposure corrector


import numpy as np
import ExposureCorrector as EC
import cv2
import numpy.random as nprand

# img = np.array([[0, 10, 255], [5, 200, 255]])

# img = np.zeros((200, 200)).astype(np.uint8)
# img[:, 0:100] = 200
# img[0:100, 0:100] = 100
# img[100:200, 0:99] = 0
# img = img + (nprand.random((200,200))*55).astype(np.uint8)

# img = cv2.imread("test_img_200x150.png", cv2.CV_LOAD_IMAGE_GRAYSCALE)
img = cv2.imread("./amir.jpg", cv2.CV_LOAD_IMAGE_GRAYSCALE)

corrector = EC.ExposureCorrector()



def main():
	adj = corrector.get_adjacency_matrix(img)
	

	# print adj.toarray()
	

	corrector.init_components()
	# print corrector.components
	
	corrector.init_weights()
	# print corrector.weights

	# print corrector.adj_mat.toarray()

	components = corrector.segment_img()

	# print components

	cv2.namedWindow("OG")
	cv2.namedWindow("Segmentation")

	seg_img = corrector.show_segmentation()

	# cv2.imshow("OG", img)
	# cv2.imshow("Segmentation", seg_img)

	cv2.imwrite("seg_img.jpg", seg_img)
	cv2.imwrite("og.jpg", img)




	# cv2.waitKey(0)
	return 0

if __name__ == "__main__":
	main()
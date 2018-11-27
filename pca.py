import numpy as np
import csv
import math
import random
reader = csv.reader(open("pima-indians-diabetes.csv", "rb"), delimiter=",")
raw_list = list(reader)

accuracy_list = []

float_list = map(lambda x: map(lambda y: float(y),x),raw_list[9:])
for k in range(20):	
	correct = 0
	total = 0
	random.shuffle(float_list)
	training_data_float = float_list[:len(float_list)//2  ]
	test_data_float = float_list[len(float_list)//2:]
	X_training = np.array(map(lambda x: x[0:8] ,training_data_float))
	X_training_mean = X_training.mean(0)
	X_training = X_training- X_training_mean
	X_training_class = map(lambda x: x[8],training_data_float)
	X_test = map(lambda x: x[0:8] ,test_data_float)
	X_test_class = map(lambda x: x[8],test_data_float)
	X_cov_matrix_training = np.cov(X_training.T) 
	eig_val_cov, eig_vec_cov = np.linalg.eig(X_cov_matrix_training)

	eigenvector_index = sorted(range(len(eig_val_cov)), key=lambda k: eig_val_cov[k])[::-1]

	eig_vec_matrix = np.array([eig_vec_cov.T[eigenvector_index[0]],eig_vec_cov.T[eigenvector_index[1]],eig_vec_cov.T[eigenvector_index[2]]]).T
	if k ==1:
		print("attributes: "+raw_list[eigenvector_index[0]][0]+ " "+raw_list[eigenvector_index[1]][0]+raw_list[eigenvector_index[2]][0])

	projected_training_data = np.dot(X_training,eig_vec_matrix)
	projected_data_class_zero = []
	projected_data_class_one = []


	for i in range(len(X_training_class)):
		if X_training_class[i] ==0.0 :
			projected_data_class_zero.append(projected_training_data[i])
		elif X_training_class[i] ==1.0:
			projected_data_class_one.append(projected_training_data[i])
	prior_class_zero = float(len(projected_data_class_zero))/(float(len(projected_data_class_zero))+float(len(projected_data_class_one)))
	prior_class_one = float(len(projected_data_class_one))/(float(len(projected_data_class_zero))+float(len(projected_data_class_one)))
	projected_data_class_zero = np.array(projected_data_class_zero)
	projected_data_class_one = np.array(projected_data_class_one)

	class_zero_mean= projected_data_class_zero.mean(0)
	class_one_mean = projected_data_class_one.mean(0)
	class_zero_cov = np.cov(projected_data_class_zero.T)
	class_one_cov = np.cov(projected_data_class_one.T)


	def distribution_class_zero(vect):
			# print(np.dot(np.dot(np.subtract(vect,class_zero_mean),np.linalg.inv(class_zero_cov)),np.subtract(vect,class_zero_mean).T))
		return (1.0/math.pow((2*math.pi),1.5))*math.pow(np.linalg.det(class_zero_cov),-0.5)*math.exp(-0.5*np.dot(np.dot(np.subtract(vect,class_zero_mean),np.linalg.inv(class_zero_cov)),np.subtract(vect,class_zero_mean).T))

	def distribution_class_one(vect):
			# print(np.dot(np.dot(np.subtract(vect,class_zero_mean),np.linalg.inv(class_zero_cov)),np.subtract(vect,class_zero_mean).T))
		return (1.0/math.pow((2*math.pi),1.5))*math.pow(np.linalg.det(class_one_cov),-0.5)*math.exp(-0.5*np.dot(np.dot(np.subtract(vect,class_one_mean),np.linalg.inv(class_one_cov)),np.subtract(vect,class_one_mean).T))
	for i in X_test:
		test_vect = np.array(i)-X_training_mean
		test_vect = np.dot(test_vect,eig_vec_matrix)
		d0 = distribution_class_zero(test_vect)*prior_class_zero
		d1 = distribution_class_one(test_vect)*prior_class_one
		if(d0>d1 and X_test_class[total]==0.0):
			correct = correct+1
		elif(d0<d1 and X_test_class[total]==1.0):
			correct = correct+1
		total = total+1
	accuracy_list.append(float(correct)/float(total))

print("mean accuracy: "+ str(np.array(accuracy_list).mean(0)))

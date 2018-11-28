import numpy as np
import csv
import math
import random
reader = csv.reader(open("pima-indians-diabetes.csv", "rb"), delimiter=",")
raw_list = list(reader)

accuracy_list = []

float_list = map(lambda x: map(lambda y: float(y),x),raw_list[9:])


for j in range(20):
	correct = 0
	total = 0
	random.shuffle(float_list)

	training_data_float = float_list[:len(float_list)//2]
	test_data_float = float_list[len(float_list)//2:]

	class0 = filter(lambda x: x[8]==0, training_data_float)
	class1 = filter(lambda x: x[8]==1, training_data_float)
	prior_class_zero = float(len(class0))/(float(len(class0))+float(len(class1)))
	prior_class_one = float(len(class1))/(float(len(class0))+float(len(class1)))
	class0 = np.array(map(lambda x: x[:8],class0))
	class1 = np.array(map(lambda x: x[:8],class1))
	class0_mean = class0.mean(0)
	class1_mean = class1.mean(0)
	scatter_matrix_class0 = np.cov(class0.T)*(len(class0)-1)
	scatter_matrix_class1 = np.cov(class1.T)*(len(class1)-1)
	scatter_within = scatter_matrix_class0+scatter_matrix_class1

	V_optimal_line = np.dot(np.linalg.inv(scatter_within),(class0_mean-class1_mean)) 

	projected_class0 = np.dot(V_optimal_line,class0.T)
	projected_class1 = np.dot(V_optimal_line,class1.T)

	class_zero_mean= projected_class0.mean(0)
	class_one_mean = projected_class1.mean(0)
	class_zero_cov = np.var(projected_class0.T)
	class_one_cov = np.var(projected_class1.T)

	def distribution_class_zero(vect):
				# print(np.dot(np.dot(np.subtract(vect,class_zero_mean),np.linalg.inv(class_zero_cov)),np.subtract(vect,class_zero_mean).T))
		return (1/math.sqrt(2*math.pi*class_zero_cov))*math.exp(-1*math.pow(vect-class_zero_mean,2.0)/(2*class_zero_cov))

	def distribution_class_one(vect):
				# print(np.dot(np.dot(np.subtract(vect,class_zero_mean),np.linalg.inv(class_zero_cov)),np.subtract(vect,class_zero_mean).T))
		return (1/math.sqrt(2*math.pi*class_one_cov))*math.exp(-1*math.pow(vect-class_one_mean,2.0)/(2*class_one_cov))


	for i in test_data_float:
		vect = np.dot(np.array(i[:8]),V_optimal_line)
		
		d0 = distribution_class_zero(vect)*prior_class_zero
		d1 = distribution_class_one(vect)*prior_class_one
		if(d0>d1 and i[8]==0.0):
			correct = correct+1
		elif(d0<d1 and i[8]==1.0):
			correct = correct+1
		total = total+1
	accuracy_list.append(float(correct)/float(total))

print("accuracy: "+str(np.array(accuracy_list).mean(0)))


	
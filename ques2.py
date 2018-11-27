import numpy as np
import math
def proj(a,b):
	return (np.dot(np.array(a),b)/np.dot(b,b))*b

class1 = np.array([[-2,1],[-5,-4],[-3,1],[0,-3],[-8,-1]])
class2 = np.array([[2,5],[1,0],[5,-1],[-1,-3],[6,1]])



class1_mean = class1.mean(0)
class2_mean = class2.mean(0)
print("class 1 means: ")
print(class1_mean)
print("class 2 means: ")
print(class2_mean)

scatter_matrix_class1 = np.cov(class1.T)*(len(class1)-1)
scatter_matrix_class2 = np.cov(class2.T)*(len(class2)-1)

print("class 1 scatter matrix: ")
print(scatter_matrix_class1)
print("class 2 scatter matrix: ")
print(scatter_matrix_class2)

scatter_within = scatter_matrix_class1+scatter_matrix_class2 

print("scatter matrix within ")
print(scatter_within)

V_optimal_line = np.dot(np.linalg.inv(scatter_within),(class1_mean-class2_mean))
print("optimal line vector: ")
print(V_optimal_line)
print("---------------")


class1_projection = np.array(np.dot(V_optimal_line,class1.T))

class2_projection = np.array(np.dot(V_optimal_line,class2.T))

for i in range(len(class1_projection)):
	if(class1_projection[i]<0):
		
		print("point "+ str(class1[i])+" was incorrectly projected")
for i in range(len(class2_projection)):
	if(class2_projection[i]>0):
		
		print("point "+ str(class2[i])+" was incorrectly projected")





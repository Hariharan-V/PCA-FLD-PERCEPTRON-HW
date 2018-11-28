import numpy as np
def dot(a,b):
	sum = 0
	for x in range(len(b)):
		sum = sum + a[x]*b[x]
	return sum
def normalize(a):
	for i in range(len(a)):
		if a[i][0]==2:
			for j in range(1,len(a[i])):
				a[i][j] = -1*a[i][j]

	return a

data = [[2,1,1,1,-1,0,2],[1,1,0,0, 1, 2, 0],[2,1,-1, -1, 1, 1, 0],[1,1,4,0,1,2,1],[1,1,-1,1,1,1,0],[1,1,-1,-1,-1,1,0],[2,1,-1,1,1,2,1]]
# data = [[1,1,1,1,-1,-1],[2,1,1,1,1,1],[2,1,-1,-1,-1,1],[1,1,1,-1,-1,1]]

weight = np.array([3,1,1,-1,2,-7])
# weight = np.array([0.25, 0.25, 0.25, 0.25, 0.25])
data = normalize(data)
for x in data:
	d= dot(weight,x[1:])
	print("calculated value = "+ str(d))
	if(d>0 ):
		print("data: " + str(x)+" classified correctly by weight: "+ str(weight))
	else:
		print("data: " + str(x)+" classified incorrectly by weight: "+ str(weight))
		weight = np.array(weight)+np.array(x[1:])
		print("new weight: "+ str(weight))









import numpy as np
flag=False
row=0
column=0
row_str=""
col_str=""
f=open("/home/rana/mypython/file.txt")
for W in f:
    if flag==True:
        row=len(W)
        row_str=W
    else:
        column=len(W)
        col_str=W
    flag=True
f.close()
matrix=np.zeros((row,column),np.int32)

#scoring
sub_cost=input("enter substitution cost")
insrt_cost=input("enter insertion cost")
del_cost=input("enter deletion cost")


def _compute_array():
    for i in range(1, row):
        for j in range(1, column):
            matrix[i,j] = min(  matrix[i-1, j-1] + _get_score(i, j),
                                    matrix[i-1, j] + insrt_cost,
                                    matrix[i, j-1] + del_cost)
def _get_score( i, j):
    if row_str[i-1]==col_str[j-1]:
        return 0
    else:
        return sub_cost 

flag=True
matrix[0,0]=0
matrix[0,1]=1 
matrix[1,0]=1
for i in range(0,1):
    for j in range(2,column):
        matrix[i][j]=matrix[i][j-1]+1
for i in range(0,1):
    for j in range(2,row):
        matrix[j][i]=matrix[j-1][i]+1
_compute_array()
#_traceback()
print (matrix)


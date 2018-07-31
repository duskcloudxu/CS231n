# 草稿暂存

```python
>>> import numpy
>>> import numpy as np
>>> a=np.array([[1,2,3],[4,5,6],[7,8,9]])
>>> b=np.array([[1,1,1],[2,2,2]])
>>> a
array([[1, 2, 3],
       [4, 5, 6],
       [7, 8, 9]])
>>> b
array([[1, 1, 1],
       [2, 2, 2]])
>>> c=b.dot(a.T)
>>> c
array([[ 6, 15, 24],
       [12, 30, 48]])
>>> c=-2*c
>>> c
array([[-12, -30, -48],
       [-24, -60, -96]])
>>> np.square(b)
array([[1, 1, 1],
       [4, 4, 4]], dtype=int32)
>>> bSum=np.square(b)
>>> bSum
array([[1, 1, 1],
       [4, 4, 4]], dtype=int32)
>>> bSum.sum(axis=1)
array([ 3, 12], dtype=int32)
>>> bSum=bSum.sum(axis=1)
>>> bSum.reshape((1,2))
array([[ 3, 12]], dtype=int32)
>>> b+c
array([[-11, -29, -47],
       [-22, -58, -94]])
>>> bSum
array([ 3, 12], dtype=int32)
>>> np.reshape(bSum,(1,2))
array([[ 3, 12]], dtype=int32)
>>> bSum
array([ 3, 12], dtype=int32)
>>> bSum=bSum.reshape((1,2))
>>> bSum
array([[ 3, 12]], dtype=int32)
>>> bSum=bSum.T
>>> bSum
array([[ 3],
       [12]], dtype=int32)
>>> c=bSum+c
>>> c
array([[ -9, -27, -45],
       [-12, -48, -84]])
>>> np.square(a).sum(axis=1)
array([ 14,  77, 194], dtype=int32)
>>> a
array([[1, 2, 3],
       [4, 5, 6],
       [7, 8, 9]])
>>> np.square(a).sum(axis=1)+c
array([[  5,  50, 149],
       [  2,  29, 110]])
>>> a
array([[1, 2, 3],
       [4, 5, 6],
       [7, 8, 9]])
>>> b
array([[1, 1, 1],
       [2, 2, 2]])
>>>
```


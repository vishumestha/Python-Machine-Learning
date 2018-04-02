import pandas as pd

from scipy import misc
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt
from scipy import misc
from sklearn.manifold import Isomap

# Look pretty...
# matplotlib.style.use('ggplot')
plt.style.use('ggplot')


#
# TODO: Start by creating a regular old, plain, "vanilla"
# python list. You can call it 'samples'.
#
# .. your code here .. 
samples=[]
#
# TODO: Write a for-loop that iterates over the images in the
# Module4/Datasets/ALOI/32/ folder, appending each of them to
# your list. Each .PNG image should first be loaded into a
# temporary NDArray, just as shown in the Feature
# Representation reading.
path='D:/Data Science with Python/6.Programming with Python for Data Science/DAT210x-master/DAT210x-master/Module4/Datasets/ALOI/32/'
#32_r0.png

for i in  range(0,355,5):
    imagename="32_r{0}.png".format(i)
    imagepath=path+imagename
    img=misc.imread(imagepath)
    
    #img=img[::2,::2]
    
    X=(img/250.0).reshape(-1)
    samples.append(X)
sam_cont=len(samples)
colors=[]
for i in range(1,sam_cont):
    colors.append('b')
    
import os
samples2=[]
#for file in os.listdir('D:/Data Science with Python/6.Programming with Python for Data Science/DAT210x-master/DAT210x-master/Module4/Datasets/ALOI/32i'):
#    imagepath=os.path.join('D:/Data Science with Python/6.Programming with Python for Data Science/DAT210x-master/DAT210x-master/Module4/Datasets/ALOI/32i',file)
#    img=misc.imread(imagepath).reshape(-1)
#    samples.append(img)

    
#
# Optional: Resample the image down by a factor of two if you
# have a slower computer. You can also convert the image from
# 0-255  to  0.0-1.0  if you'd like, but that will have no
# effect on the algorithm's results.
#
# .. your code here .. 


#
# TODO: Once you're done answering the first three questions,
# right before you converted your list to a dataframe, add in
# additional code which also appends to your list the images
# in the Module4/Datasets/ALOI/32_i directory. Re-run your
# assignment and answer the final question below.
#
# .. your code here .. 
for file1 in os.listdir('D:/Data Science with Python/6.Programming with Python for Data Science/DAT210x-master/DAT210x-master/Module4/Datasets/ALOI/32i'):	# Also append the 32i images to the list/dataframe
     b = os.path.join('D:/Data Science with Python/6.Programming with Python for Data Science/DAT210x-master/DAT210x-master/Module4/Datasets/ALOI/32i', file1)
     img1 = misc.imread(b)
     X=(img1/250.0).reshape(-1)
     samples.append(X)

sam_cont=len(samples)-sam_cont    
for i in range(1,sam_cont):
    colors.append('r')

#
# TODO: Convert the list to a dataframe
#
# .. your code here .. 

df=pd.DataFrame(samples)
#
# TODO: Implement Isomap here. Reduce the dataframe df down
# to three components, using K=6 for your neighborhood size
#
# .. your code here .. 

isomap=Isomap(n_components=3,n_neighbors=6)
T=isomap.fit_transform(df)

#from sklearn.decomposition import PCA
#pca=PCA(n_components=3)
#T=pca.fit_transform(df)


#
# TODO: Create a 2D Scatter plot to graph your manifold. You
# can use either 'o' or '.' as your marker. Graph the first two
# isomap components

#
# .. your code here ..
plt.scatter(x=T[:,0],y=T[:,1],marker='*',c=colors)




#
# TODO: Create a 3D Scatter plot to graph your manifold. You
# can use either 'o' or '.' as your marker:
#
# .. your code here .. 

fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')
ax.set_xlabel("component1")
ax.set_ylabel("component2")
ax.set_zlabel("component2")
ax.scatter(T[:,0],T[:,1],T[:,2],marker='*',c=colors)




plt.show()


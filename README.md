# Face-Recogntion
Problem statement
Facial recognition is an important and more preferred bio metric when compared to other bio metrics. One of the primary reason is that it requires no physical interaction on behalf of the user. For face recognition there are two types of comparisons:

1. Verification:
In this the system compares the given individual to that the individual says they are and responds with a yes or a no.

2. Identification:
This is where the system compares the given individual to all the other individuals in the database and recognizes that person.

In this project we implement the Identification system using Machine Learning concepts such as Principal Component Analysis (PCA) and Support Vector Machine (SVM).

DESCRIPTION OF DATA AND SOURCE

For this project we use the Yale Face Database. It contains 165 images in total. The data is of 15 persons with 11 images per person. The title of each image is in the format “Subject99.expression” where the expression can be sad, happy, surprised etc.

It also contains images with different configurations such as center light, right light, with glasses etc. We train our classifier using all the images except those which have a “sad” expression. Those with “sad” expression would be used for testing our classifier. Each subject has a number which acts as a label for that subject. Although this label can also be a name but for easier data representation we take it as number.

The Dataset can be obtained for the link: 
http://vision.ucsd.edu/content/yale-face-database

DETAILS OF PREPROCESSING DONE

A. GRAYSCALING AND FACE EXTRACTION
The Input dataset has a lot of area apart from the face that is not required for facial recognition. If we don’t extract the face only it will lead to more attributes in a single image which would further increase the time taken by the system to recognize the person. So to decrease the time as well as the number of attributes in the image we extract only the face from the image using Haar Cascade provided by OpenCV.
Also, we store the image data in the form of a 3 dimensional array of grayscale values. To convert coloured images into grayscale, we use Python Image Library.
Once we get the details of the face we store the gray scale values into a 3 dimensional array.

B. PRINCIPAL COMPONENT ANALYSIS
The objective of PCA is to perform dimensionality reduction while preserving as much of the randomness in the high-dimensional space as possible. It is mainly concerned with identifying correlations in the data. We use PCA because many algorithms that work fine in low dimensions become intractable when the input is high-dimensional. In our project we have restricted the Number of components for PCA to 10. We first convert our 3 dimensional array to a 2 dimensional array by flattening the array and then fit it to a PCA with n_components = 10. We then transform our dataset using the given PCA to train our Classifier.

ClASSIFICATION USING SVM

Once the preprocessing of the dataset is done we can now use a classifier to train it for the given dataset. We chose Support Vector Machine as our classifier for this project. SVM’s are supervised learning models with associated learning algorithms that analyze data and recognize patterns. We use a nonlinear support vector classification model with the kernel as radial basis function (rbf). We first fit the classifier with the data that was transformed using PCA and the labels which were extracted while reading the images in the directory. Once the classifier is trained we then take our testing dataset that is the images with “sad” expression and check whether the label predicted by our classifier is same as that given in the file name. 
To check the predictions on the testing dataset we first take the gray scale value of the face in the image, store in a 2 D array, flatten the array and then transform it using our PCA which we fitted earlier. Once we get the transformed array we predict the label for the test image and print it out.

The above classification model worked very well. The testing dataset consisted of 15 images, and it recognized correctly in all the cases. 

https://geekinsideyou.wordpress.com/2015/10/14/face-recognition-using-machine-learning-concepts-such-as-svm-and-pca/

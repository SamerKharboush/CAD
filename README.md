# CAD 

What is coronary artery disease?
Coronary artery disease (CAD) is the most common type of heart disease in the United States. It is sometimes called coronary heart disease or ischemic heart disease.

For some people, the first sign of CAD is a heart attack. You and your health care team may be able to help reduce your risk for CAD.
What causes coronary artery disease?
CAD is caused by plaque buildup in the walls of the arteries that supply blood to the heart (called coronary arteries) and other parts of the body.

Plaque is made up of deposits of cholesterol and other substances in the artery. Plaque buildup causes the inside of the arteries to narrow over time, which can partially or totally block the blood flow. This process is called atherosclerosis.
Coronary artery disease (CAD) is known as a common cardiovascular disease. 
A standard clinical tool for diagnosing CAD is angiography. 

The main challenges are dangerous side effects and high angiography costs. Today, the development of artificial intelligence-based methods is a valuable achievement for diagnosing disease., so I tried different Machine Learning methods such as (CNN),Pytorch, Tensorflow and Pre-trained model ResNet18  to detect CAD patient MRI also Autoencoder which will try to understand the underlying distribution of Data, and try to create CAD images using Models.



# First pytorch notebook
Custom Data Generator A lot of effort in solving any machine learning problem goes into preparing the data. PyTorch provides many tools to make data loading easy and preprocess/augment data from a non-trivial dataset.
torch.utils.data.Dataset is an abstract class representing a dataset. Your custom dataset should inherit Dataset and override the following methods:
__len__ so that len(dataset) returns the size of the dataset.
__getitem__ to support the indexing such that dataset[i] can be used to get ith sample.
Learn More: Pytorch Data Loading Documentation
Custom Neural Network Class
This class will inherit the properties from torch.nn.Module and should have two functions __init__ for defining neural network and forward for forward propogation. In General, forward function, will be dealing with output from each layer. I can apply MaxPool, Dropouts and other activation functions.

Convolutional Neural Network - Classification
# Model Training
1.	Loss Function: torch.nn Module having predefined loss functions and you can create your own Loss functions as defined above. Using CrossEntropyLoss for 2 Outputs, for examining the individual probabilities
2.	Optimizer: torch.nn having pre-defined optimizers, with many parameters, using Adam here
3.	After defining epochs, need to loop over some steps
•	Fitting Data with batch of dataset
•	Compute the Loss
•	Make Gradient Zero, for previous computations
•	Compute the backward propogation (Derivate calculation)
•	Optimizing the weight and biase

# fast.ai notebook
fastai is a free deep learning API built on PyTorch V1. The fast.ai team incorporates their reseach breakthroughs into the software, enabling users to achieve more accurate results faster and with fewer lines of code.
I will be deploying standard techniques taught in the fast.ai course to see how well these techniques can perform without needing expert knowledge. The techniques are:
1.	Learning rate finder
2.	1-cycle learning
3.	Differential learning rates for model finetuning
4.	Data augmentation
5.	Test time augmentation
6.	Transfer learning via low-resolution images

# Exploratory data analysis 
then, I will only check for the number of classes and the number of items per class. Imbalanced datasets may require resampling of the data to ensure proper training.
Data loading and preparation using ImageDataBunch 

# Model creation
Submissions into the competition are evaluated on the area under the ROC curve between the predicted probability and the observed target. Since we have a limited number of submissions per day, implementing a metric for the ROC AUC (which is non-standard in the fast.ai v1 library) allows us to run as many experiments we want.
At this point, I am not sure if changing the metric changes the loss function in the Learner to optimize the metric. I will be doing more reading up in that area.

# Model training
The most important hyperparameter in training neural networks in general is the learning rate. Unfortunately, as of now, there is no way of finding a good learning rate without trial-and-error.
The library has made it convenient to test different learning rates. We find a good learning rate using the method lr_find, then plotting the graph of learning rates against losses. As a rule of thumb, the learning rate is chosen from a part of the graph where it is steepest and most consistent.
Transfer learning
We will now train the model on the same dataset; except we are using images of higher resolution. Intuitively, the 'concepts' learnt by the neural network will continue to be applied in training with the new set of images.
Conclusion
At this stage, we would like to check the effectiveness of the learn model against our validation set (which is automatically generated by the ImageDataBunch object). We will use the following methods to evaluate the effectiveness.
1.	Confusion matrix.
2.	Accuracy.
3.	ROC-AUC, as dictated in the competition evaluation.

# Tensorflow notebook
in this notebook, I've used CNN to perform Image Classification on the Brain Tumor dataset.
Since this dataset is small, if we train a neural network to it, it won't really give us a good result.
Therefore, I'm going to use the concept of Transfer Learning to train the model to get really accurate results.
Deep convolutional neural network models may take days or even weeks to train on very large datasets.
The include_top parameter is set to False so that the network doesn't include the top layer/ output layer from the pre-built model which allows us to add our own output layer depending upon our use case!
GlobalAveragePooling2D -> This layer acts similar to the Max Pooling layer in CNNs, the only difference being is that it uses the Average values instead of the Max value while pooling. This really helps in decreasing the computational load on the machine while training.

Dropout -> This layer omits some of the neurons at each step from the layer making the neurons more independent from the neibouring neurons. It helps in avoiding overfitting. Neurons to be ommitted are selected at random. The rate parameter is the liklihood of a neuron activation being set to 0, thus dropping out the neuron
Dense -> This is the output layer which classifies the image into 1 of the 2 possible classes.
A way to short-cut this process is to re-use the model weights from pre-trained models that were developed for standard computer vision benchmark datasets, such as the ResNet18  and ImageNet image recognition tasks. Top performing models can be downloaded and used directly or integrated into a new model for your own computer vision problems.
Callbacks -> Callbacks can help you fix bugs more quickly, and can help you build better models. They can help you visualize how your model’s training is going, and can even help prevent overfitting by implementing early stopping or customizing the learning rate on each iteration.

By definition, "A callback is a set of functions to be applied at given stages of the training procedure. You can use callbacks to get a view on internal states and statistics of the model during training.

# Flask app
a very simple example application that has just few lines of code. Instead of repeating that trivial example,
in this app (just a web app), we load our pretrained model and allow the app to upload the data and pass it through our model to be classified based on our model’s weights and fittings.
The application will exist in a package. In Python, a sub-directory that includes a __init__.py file is considered a package and can be imported. When you import a package, the __init__.py executes and defines what symbols the package exposes to the outside world.
For more access Flask website.
In sum, it’s a very simple deployment method.

# Autoencoder notebook
Autoencoder is a type neural network where the output layer has the same dimensionality as the input layer. In simpler words, the number of output units in the output layer is equal to the number of input units in the input layer. An autoencoder replicates the data from the input to the output in an unsupervised manner and is therefore sometimes referred to as a replicator neural network.
Architecture of autoencoders
An autoencoder consists of three components:
•	Encoder: An encoder is a feedforward, fully connected neural network that compresses the input into a latent space representation and encodes the input image as a compressed representation in a reduced dimension. The compressed image is the distorted version of the original image.
•	Code: This part of the network contains the reduced representation of the input that is fed into the decoder.
•	Decoder: Decoder is also a feedforward network like the encoder and has a similar structure to the encoder. This network is responsible for reconstructing the input back to the original dimensions from the code.

an Autoencoder Implementation, which will try to understand the underlying distribution of Data, and try to create CAD images using Models, (noise denoise) concept
Dataloading and preparation, then build a linear (NN), loss MSELoss and Adam optimizer,
Show the result of the autoencoder using clear_output from  IPython.display.


# CAD Prediction | Web App Demo (Flask) 
CAD prediction using Web App (Flask) that can classify if the subjects is a CAD patient or not based on uploaded MRI image.

The image data that was used for this project is CAD MRI images.(https://www.kaggle.com/danialsharifrazi/cad-cardiac-mri-dataset)

## For more projects visit my website
Click on image to play :point_down:

[![Coronary artery disease prediction | Web App Demo (Flask)](https://www.hospitaltimes.co.uk/wp-content/uploads/2019/04/heart-image-2L.jpg)](https://www.samerkharboush.tk/)



## Want to run this project in your computer
- **Follow these Steps**
 1. Clone or Download
  2. Open the terminal/CMD in project directory
  3. Then create virtual environment using this command: 
  
      ```py -m venv env```
  4. Activate virtual environment using: 
  
      ```env\Scripts\activate```
  5. Install all the requirements using: 
  
      ```pip install -r requirements.txt```
      
      It will take some time to download till that take a sip of coffee :coffee: 
      
  6. After successful download of all above requirements, run the app using:
      
      ``` flask run ```
      
      Wait for few seconds till it shows like : ```Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)```
   7. Then open this URL in browser : http://127.0.0.1:5000/
   8. Voila :thumbsup:

**Car Detection Project**

**Contents**

1.	Introduction
2.	Project Objectives
3.	Project Limitations
4.	Importing Datasets
5.	Data Cleaning and Manipulation
6.	Exploratory Data Analysis (EDA)
7.	Data Preprocessing
8.	Model Training and Testing
9.	Results and Conclusion
10.	Detectron2 Introduction
11.	Detectron2 Preprocessing
12.	Detectron2 Model Training
13.	Detectron2 Prediction and Evaluation
14.	Streamlit local host - Model deployment into GUI API

**1. Introduction**

**1.1	Data Description**

The Car Detection Dataset is using the Stanford’s Car Dataset which has 16,185 images of 196 classes of cars. The data is split into 8,144 train and 8,041 Annotations and test images, where each class has been split roughly in a 50-50 split. The Classes are typically at the level of Brand, Model, Year, Type.

**The Dataset is as follows:**

i) Annotations Train and Test Dataset: The annotations dataset was provided as a .ZIP file. After extraction, the annotations train and test dataset is found to be in .csv format, where the features are “Image Name”, “Bounding Box Coordinates” and “Image Class”. There are 8,114 train and 8041 test annotations.

ii) Car Names and Make Dataset: The dataset is stored in .csv format and it has only 1 feature “Car Name”. This feature can be further processed using data analytics and create newer features from the existing one.

The features that can be created from “Car Name” are “Brand”, “Model”, “Year” and “Type”.

iii) The Train and Test Images in a .ZIP file: The Images were stored in a .ZIP file with a folder directory of car names, the images are in .jpg format. There are 8,114 train and 8,041 test images.

**1.2	Project Description**

The Stanford Car Dataset will be utilized to build a Car Detection and Recognition predictive model. The main goal of the model is to classify a car’s year, make and model given an input image. 

This model could be further developed to be used in creating a mobile
application that assists users in identifying cars of interest. The users would simply take a picture of the vehicle of interest and the application would return information (year, make, type and model) regarding the recognized vehicle. 

The users could also input a picture of a car and this application would bring out details pertaining to it such as (year, Brand, type and model)

This model can be useful in the hands of car dealership websites as this model can recognize vehicle name and it can be improved further as it can be used in searching the partners’ database to retrieve information such as availability, price etc. 

This application could help people who are not familiar with cars or who simply want quick information without searching the Internet themselves. 
				
![image](https://user-images.githubusercontent.com/88423149/184920876-6b1f3e87-8167-4283-9cc9-5abc4bb0477a.png)
This image is taken from the internet for displaying the application of this project

**2. Project Objectives** 

	The Main Objective of this project is to develop a Car detection model that predicts the Car’s Model, Brand/Make, Year of Manufacturing, and Type of Car (Sedan, Hatchback, SUV etc.). 

•	The main objectives of this project is as follows:

1)	Develop a powerful car detection model that correctly detects the model, make, year and type of car, then displays a hyperlink of the car which redirects straight to the internet.
2)	Make the model detect the car in real time via camera.
3)	Create a mobile application to make it available to consumers all around the world. 
4)	To Include Car images and classes all the way till 2022 for wider application in real life.

•	The objective for the upcoming milestone 2 is to create a powerful car detection model that correctly detects the model, make, year and type of car. The finalization of the model would be completed by then.

•	The objective for milestone 3 is to create a model detect car in real time via camera and a mobile application with GUI for the real time car detection. This mobile application should be available to consumers by mid of June. 

**3. Project Limitations** 

	The Interim limitation of the project is as follows (using CNN) :

1)	The dataset only has limited number of classes for each feature. This results in poorly performance due to limited data.
2)	 The dataset due to its large size, it requires a large allocation of RAM of around 20+ GB.
3)	The Brand/Make of the car can only be determined using the Brand Logo from the Car. For the model to make the accurate detection, it needs to be trained with lots of high-quality images to give accurate prediction in production environment. However, the dataset has many car images with unclear brand Logo which makes it more difficult for the models to make the perfect prediction.
4)	The dataset is only limited to a 30-year timeline of cars, because It only has images of cars manufactured from 1991 to 2012. Hence, it can only detect cars from this timeline.

	The Final limitations of the project are as follows (using Detectron2):

1)	The detectron2 research tool only has examples with google colab notebook, the tool has never been run in Jupyter Notebook. Hence, time consuming to resolve issues.
2)	The dictionary for model annotations is case sensitive and unless it is in a specified format, the model wont train and would keep pulling an error.
3)	The detectron2 tool weights are not getting registered in local directory due to difference in directory from cloud to local, resulting in empty images without bounding boxes.
4)	The accuracy is not great, because the model is predicting with initial pre trained model weights and not trained custom model weights.

**4. Importing Datasets**  

The Car Detection project has 3 different types of datasets: “Annotations”, “Images”, “Car Names and Make” and they are extracted using the following:

a) The Car Names and Make and the Annotations datasets are stored as .CSV, Hence, Pandas function “pd.read_csv” is used for extraction.

b) The Images are stored as .jpg, they were also zipped, so they were extracted and read using “for” loop and appended into a list, converted into an array. The folder name and image name for the images were also extracted the same way.

Since they were all extracted at the same order, the index of the images and labels are the same. Hence, the accurate label for the image can be pulled out while displaying the image.

There aren’t any major roadblocks while importing the datasets, however, while importing the images, the images were not resized as it affected the bounding box. During the model training phase, the images need to be imported again resized. 

**5. Data Cleaning and Manipulation**        
 
 The Dataset is cleaned and manipulated to create 4 new features “Brand”, “Type”, “Year”, “Model”. There were no missing or duplicate values in the dataset. 

Using split and replace, the data was wrangled. The “Brand” column typically has name of the car manufacturer. The “Type” column has the type of car such as “Sedan”, “Hatchback”, “SUV”, “Convertible”, “Coupe”, “Wagon”, “Cab”, “Van”. The “Year” column has the year at which the Car was manufactured. 
	
The “Year” column has the year of manufacturing of the cars, the dataset typically ranges between 1991 to 2012. The “Model” column has the Car models extracted from the Car Name.

**6. Exploratory Data Analysis:**

In Exploratory Data Analysis (EDA), the data is observed holistically, and insights are gathered. The following are the insights gathered from the Train Data is as follows:

i.	The brand with the highest population in the dataset is “Chevrolet” with 905 instances, followed by “Dodge” with 630 instances. The Lowest populated brand is “Maybach” with 29 instances, followed by “Mazda” with 36 instances. 

ii.	The Year with the highest population in the dataset is “2012” with 4818 instances, followed by “2007” with 1059 instances. The Lowest populated Year is “1999” and “1997” with 44 instances. 

iii.	There are 49 different brands, and 189 unique car models in the dataset. 

iv.	There are 7 different types of Cars in the dataset that are as follows:  Sedan, Hatchback, SUV, Convertible, Coupe, Wagon, Cab, Van. The Car Type with the highest population in the dataset is “Sedan” which has about 2159 instances, followed by “SUV” with 1556 instances. The Lowest populated Car Type is “Wagon” with 253 instances, followed by “Van” with 541 instances.

v.	Statistical analysis is performed as part of EDA, and First, Five Summary Analysis was performed, where the max value was ‘2012’ for the Year feature, this signified that the latest car model entry is of a car from 2012. Hence, this dataset doesn’t have data over 2012. 

vi.	The Min value was ‘1991’ for the Year feature, this signified that the dataset has cars all the way from 1991 to 2012. This means that the dataset has roughly cars of 30 years’ timeline. 

vii.	The Second Statistical analysis was correlation matrix, where bbx3 and bbx4 have the highest correlation of 96%, which means that there’s multicollinearity.  The lighter the color, the more correlation it has with the feature. The following heatmap, helps us visualize the multicollinearity between the features. However, these features don’t impact the model as they are only for creating the bounding box for imposing into image. 

![image](https://user-images.githubusercontent.com/88423149/184921324-6dbd3a03-ab05-48d2-a9cd-903a15b9893c.png)
Heat map for Correlation

viii.	After the statistical analysis, the data visualization is performed, where the data is visualized in plots as Univariate, and Multivariate. 

In Univariate Analysis, The “Year” feature is displayed over a bar plot for understanding the pattern of data. It is understood that the year “2012” has the highest population in the dataset as most of the cars included in the dataset are manufactured in that year. The pie plot helps in understanding the distribution of data.

![image](https://user-images.githubusercontent.com/88423149/184921369-bd4934c5-3113-42ef-b7e6-7e03c0715aa3.png)
Bar plot for “Year” feature

![image](https://user-images.githubusercontent.com/88423149/184921390-3bee1017-0e81-4a4d-930a-48c9bbaed2c2.png)
Pie plot for “Year” feature

ix.	In Univariate analysis, the Bar plot of “Type” feature is visualized. From this, it is understood that “Sedan” car type has the highest population in the dataset, followed by “SUV” and “Coupe”. The lowest being “Wagon” followed by “Van”. The pie plot helps in understanding the distribution of data.

![image](https://user-images.githubusercontent.com/88423149/184921434-3bf81b03-2e04-48c5-b55f-a274bd629c41.png)
Bar plot for “Type” feature

![image](https://user-images.githubusercontent.com/88423149/184921467-a2466505-8929-48fc-949d-866a11e209e5.png)
Pie plot for “Type” feature

x.	In Multivariate Analysis, the Bar plot of “Brand” with Hue of “Model”. From this plot it is understood that, “Dodge” has the highest populated cars in the dataset, followed by “Ford” and “Volkswagon”. The Lowest populated cars being “Maybach”, followed by “FIAT” and “Infiniti.

![image](https://user-images.githubusercontent.com/88423149/184921504-16f53890-8699-46a3-9ca2-91ceda6c35e8.png)
Bar plot for “Brand” with hue of “Model” feature

xi.	In Multivariate Analysis, the Bar plot of “Type” with Hue of “Brand”. From this plot it is understood that “Cab” Car Type has the highest populated count of cars in the dataset, followed by “Sedan” and “SUV”. The Lowest populated cars being “Van”, followed by “Hatchback” and “Convertible.

![image](https://user-images.githubusercontent.com/88423149/184921555-e73c35cd-f731-422f-afb0-8a11eb26f026.png)
Bar plot for “Type” with hue of “Brand” feature

xii.	In Multivariate Analysis, the Tree Map shows a hierarchical distribution of clusters. This plot is created using plotly, so it is interactable. The various insights drawn will be from each of the cluster in the Tree map.

![image](https://user-images.githubusercontent.com/88423149/184921604-91112287-6962-464e-9a82-08bb2ff45876.png)
GUI interactive Tree Map in hierarchical order of Car Type > Year > Model

xiii.	In the Tree map, the “Sedan” Car type has Cars from 1993, 1994, 2001, 2007, 2008, 2009, 2010, 2011, 2012.

 ![image](https://user-images.githubusercontent.com/88423149/185035442-e48e9729-a492-4019-9568-d95719a4c784.png)
Tree Map of Sedan with Year and Brand

xiv.	In the Tree map, the “SUV” Car type has Cars from 2000, 2007, 2008, 2009, 2010, 2011, 2012.

 ![image](https://user-images.githubusercontent.com/88423149/185035500-801b9d31-ac37-4dd6-912d-171c30502de7.png)
Tree Map of SUV with Year and Brand

xv.	In the Tree map, the “Convertible” Car type has Cars from 1993, 2007, 2008, 2009, 2010, 2012.
  ![image](https://user-images.githubusercontent.com/88423149/185035520-aff0ac90-546f-4d87-afe2-6139fcb79309.png)
Tree Map of Convertible with Year and Brand

xvi.	In the Tree map, the “Cab” Car type has Cars from 2007, 2009, 2010, 2011, 2012.
        ![image](https://user-images.githubusercontent.com/88423149/185035541-d57e1531-158e-4207-afa5-a16864cd594d.png)
Tree Map of Cab with Year and Brand

xvii.	In the Tree map, the “Coupe” Car type has Cars from 1998, 1999, 2001, 2006, ,2007, 2008 , 2009 , 2012.
![image](https://user-images.githubusercontent.com/88423149/185035564-1eb8f641-06ac-4b82-b25d-bc211b117efd.png)
Tree Map of Coupe with Year and Brand

xviii.	In the Tree map, the “Hatchback” Car type has Cars from 1991, 1998, 2011, 2012.
![image](https://user-images.githubusercontent.com/88423149/185035579-abeaf587-4cc2-4890-838a-eb2d647d65d9.png)
Tree Map of Hatchback with Year and Brand

xix.	In the Tree map, the “Van” Car type has Cars from 1997, 2007, 2009, 2012.
![image](https://user-images.githubusercontent.com/88423149/185035604-746f0a10-c2ae-4b80-ab9a-1e75fc3a92c7.png)
Tree Map of Van with Year and Brand

xx.	In the Tree map, the “Wagon” Car type has Cars from 1994, 2002, 2007, 2008, 2012.
![image](https://user-images.githubusercontent.com/88423149/185035619-4c1b8fe4-90fe-47ad-a3ae-c67ac0926146.png)

**7. Data Preprocessing**
In Data Preprocessing, the two major activities performed are: 

i)	Mapping Train and Test Images to Classes,
ii)	Mapping Train and Test Images to Annotations
iii)	Displaying Images with Bounding box from the annotations. 

However, as part of preprocessing, the Train and Test images and labels were converted into arrays. Columns “Type”, “Year”, “Brand”, “Car name” are used as the labels/classes for images.

 For Better understanding of the dataset, the Train and Test dataset’s duplicates were removed and a dataset with unique values was created. The following image is from the unique car dataset, it has the list of all 189 unique cars in the dataset.
![image](https://user-images.githubusercontent.com/88423149/185035643-dfe86c79-049a-4b9d-b6ca-a6a23da07db6.png)

**Mapping Train and Test Images to Classes:**

The Image and Label are mapped to each other via Index since the images and labels were extracted from file directory. It is already mapped. The image and label are converted into an array before displaying the images.

![image](https://user-images.githubusercontent.com/88423149/185035713-3bb6bb36-3046-4fa8-9f47-c34e76b9f06b.png)
Classes/Labels mapped to Image

**Mapping Train and Test Images to Annotations**

The Image and Annotations are mapped to each other via Index since the images and labels were extracted from file directory. It is already mapped. The annotations have bounding box coordinates X1, Y1, X2,Y2. These coordinates are mapped into a rectangular box by calculating the width and height. This is the formula for Width = X2 – X1 and Height = Y2 – Y1. 

**Displaying Images with Bounding box from the annotations. **

The image mapping to bounding box is displayed, the bounding box is retrieved using “patches.Rectangle” function. The following are the images with Bounding box:
![image](https://user-images.githubusercontent.com/88423149/185035776-a748f2a0-834f-4254-9e4b-4552c05b561a.png)
![image](https://user-images.githubusercontent.com/88423149/185035783-c59ea9fd-d890-4368-b628-9a256aacff93.png)
Classes/Labels and Bounding box mapped to Image

![image](https://user-images.githubusercontent.com/88423149/185035790-64646903-5493-4ba5-9875-dfc117f6292f.png)
189 Unique Cars from the Dataset with Bounding box

**8. CNN Model Training and Testing**
In Model building, training and testing, the CNN Model architecture was built, the CNN model was trained with images and labels/classes (‘carname’,’type’,’year’,’brand’). 

The CNN model designed has 22 layers in total: 4 Conv2D layers, 4 max pooling layers, 6 batch normalization layers, 4 dropout layers, 3 dense layers and 1 flatten layer. The final dense layer for each CNN differs as the following feature have different number of classes: Year (16 classes), Type (8 classes), Brand (49 classes), Car names (189 classes).

![image](https://user-images.githubusercontent.com/88423149/185035851-ac823c7b-9bbd-4025-9964-a0f2ddbfbd84.png)
CNN architecture for “car names” class

Data Augmentation is performed using Image data generator to reduce overfitting of images. After which, the models were fit with data for model training.

For the first CNN model, which had “Car names” as the class and images as feature. The following are the results from the training.	
![image](https://user-images.githubusercontent.com/88423149/185035890-c9a180a8-270d-4485-bd11-8df9d9ee71c7.png)
![image](https://user-images.githubusercontent.com/88423149/185035899-2c3c69b2-a72b-4ae3-981e-5196983b40fe.png)

i.	This is the accuracy and loss for batch size of 500 over 30 epochs

#	Train	Validation
Accuracy 6.8%	3.8%
Loss	45.7%	51.3%

ii.	For the Second CNN model, which had “Brand” as the class and images as feature. The following are the results from the training
![image](https://user-images.githubusercontent.com/88423149/185035933-b1b9318c-88bd-4ce9-93ec-60fc1f02d46a.png)
![image](https://user-images.githubusercontent.com/88423149/185035938-cc152f3a-cfec-40ea-a47b-91da5de75cdc.png)

This is the accuracy and loss for batch size of 500 over 30 epochs

#	Train	Validation
Accuracy 13.7%	11.0%
Loss	31.7%	33.3%

iii.	For the Third CNN model, which had “Type” as the class and images as feature. The following are the results from the training. 

![image](https://user-images.githubusercontent.com/88423149/185035983-91711038-f7b8-4c75-bebb-721b026abaac.png)
![image](https://user-images.githubusercontent.com/88423149/185035991-5155fb15-6fc8-47ff-9092-105f30f1209d.png)
This is the accuracy and loss for batch size of 500 over 30 epochs

#	Train	Validation
Accuracy 31.04%	30.6%
Loss	18.6%	18.8%

iv.	For the Fourth CNN model, which had “Year” as the class and images as feature. The following are the results from the training.

![image](https://user-images.githubusercontent.com/88423149/185036034-d36b55bb-b1bf-41db-8712-c39602a674d1.png)
![image](https://user-images.githubusercontent.com/88423149/185036041-690614b7-9248-42af-83cd-4b13c929406d.png)

This is the accuracy and loss for batch size of 500 over 30 epochs

#	Train	Validation
Accuracy 59.16%	59.10%
Loss	15.07%	15.3%

The model was tested, where Label/classes were predicted for an image. The following were the images for which the prediction was carried out, the resolution being 60x60 is why the image is unclear.

![image](https://user-images.githubusercontent.com/88423149/185036089-5168ef9a-e748-4298-be49-a055d0a3cfdb.png)
![image](https://user-images.githubusercontent.com/88423149/185036101-4935eb43-c7ce-4a4e-89f4-204e904b3584.png)
![image](https://user-images.githubusercontent.com/88423149/185036106-f00d154a-4249-435d-9608-bd3df0a7a1ca.png)
![image](https://user-images.githubusercontent.com/88423149/185036109-d5892fed-e94e-4a7e-8152-5a00a46792ef.png)

•	The Labels were incorrectly predicted; hence the model is not viable for production.

**9. Results and Conclusion (CNN Model)**

•	From the above model related observations, the reason for low accuracy on ‘Car names’ dataset is due to insufficient class values in the dataset. As Car names dataset has 189 unique classes and had limited number of duplicate instances, resulting in data being underfit for modelling purpose.

•	The main reason why the model is performing poorly for all the classes is because the input resolution is only 60x60 which gives a very unclear image, that affects the model prediction capability.

•	The models were pickled and then “Brand”, “Year” and “Type” CNN models were retrained with weights loaded on batch size of 500 over 40 epochs. However, the results were no different. 

•	To improve the model performance, the final step that was taken was to increase the resolution of the model to 500x500. However, the model needed to allocate 22.8 gb of memory for this model, but due to the unavailability, it displayed an error message. 

![image](https://user-images.githubusercontent.com/88423149/185036218-b0743060-e2f9-46e0-b296-beedd679f7e4.png)

•	The only viable solution is to use pre trained models for training the dataset, which will make it easier in prediction process and also taking images as a batch for modelling purpose.

**10. Detectron2 Introduction:**

•	 Detectron2 is a tool that uses the Pytorch framework that was created by Facebook AI research group and has state of the art detection and efficiency in predicting images. It uses different object related algorithms (eg: Mask RCNN, RetinaNet, Fast RCNN etc.) which are embedded to accurately classify objects. It can be combined with the Car Detection dataset for much better prediction capabilities.

•	On that note, we had used a custom dataset for which it was trained using Faster RCNN and RetinaNet, the model had detected, labelled, and put bounding boxes over the objects in the images. From the Detectron2 tool some of the pre trained model weights such as “COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml” for Faster RCNN and "COCO-Detection/retinanet_R_101_FPN_3x.yaml” for RetinaNet were used.

•	The following are the images with which it had performed object detection from model weights alone. 

![image](https://user-images.githubusercontent.com/88423149/185036260-873568e6-355b-4a0c-acf1-b21f2da8169c.png)

![image](https://user-images.githubusercontent.com/88423149/185036268-efab69c1-5408-4f95-bfae-2a7fbb0b54fc.png)
Detectron2 – RetinaNet object detection using pretrained weights

![image](https://user-images.githubusercontent.com/88423149/185036316-1d0a42d6-d30b-473d-96b5-d1576ad99a14.png)

![image](https://user-images.githubusercontent.com/88423149/185036326-23973909-f334-4d7e-9b95-9baf914cc540.png)
Detectron2 – Faster RCNN object detection using pretrained weights

•	From the above images, we can understand that the Faset RCNN gave results which are more useful to Car Detection dataset as it only gave concise predictions, although RetinaNet gave lots of predictions, it gave unnecessary predictions. 

•	Hence, from the Detectron2 tool, for Car Detection project, for the next milestone, the Faster RCNN can be utilized and tested for Car Detection dataset. Keras tuner can be used for Hyperparameter tuning of the dataset.

**11. Detectron2 Preprocessing**
•	Detectron2 model zoo was tested for its model weights and pre trained bounding box and performance for FasterRCNN.

•	From the testing, a dataset with “Car” and “Truck” classes were considered, the IOU accuracy was averaging around 96%, which is high, this means that, the model has accurate detection in production environment as well using test data.

•	The Detectron2 models require a class ID column which was created separately for the datasets.

•	The main class used for the dataset is the “Type” feature, which is the only feature which can be accurate predicted, and it is due to it only having 8 classes. Hence, better learning for the model.

•	The annotations for the Detectron2 model should be created in COCO format, which is like JSON that is in a dictionary format. The template for the Detectron2 dataset is given in this guideline https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html.

![image](https://user-images.githubusercontent.com/88423149/185036360-1dfd9382-69c2-4c22-8756-d0f8b6e9f947.png)
Detectron2 dataset In COCO format (in dictionary)

•	After the annotations, filename, imageid, height, width, bbox_mode, category_id, and segmentation are defined for the dataset in COCO format, the detectron2 classes should be mapped to the annotations.

•	Using, DatasetCatalog, the model classes are mapped to the annotations as objects within the Detectron2 tool.

![image](https://user-images.githubusercontent.com/88423149/185036405-0b0b76c3-5cf8-4926-b3e2-6d3316d545fe.png)
Mapping Classes to Annotations using Detectron2 function

•	Images are bound with boxes and class via Detectron2 tool.

![image](https://user-images.githubusercontent.com/88423149/185036449-ca145c91-5e28-45c7-a18c-aa7945ad4203.png)
Image with bounding boxes and classes using Detectron2 tool

**12. Detectron2 Model Training**
•	 The Detectron2 model was trained using Faster RCNN .yaml file from COCO-Detection. The pre trained weights reduce the model training time and memory requirement. Hence, this is ideal for Car Detection dataset, as it has over 8000+ images each for train and test.

![image](https://user-images.githubusercontent.com/88423149/185036475-d25c8ff7-f2ea-484e-a219-706dc1d30788.png)
Detectron2 Custom Model Training with pre trained weights

![image](https://user-images.githubusercontent.com/88423149/185036486-5e1701d3-b6bc-4517-a18a-5e25859b2841.png)
Detectron2 Custom Model Training classes

**13. Detectron2 Prediction/Evaluation**
•	The Faster RCNN pre trained model from Detectron2 tool was predicted using model zoo weights with custom classes. The results are as follows:

![image](https://user-images.githubusercontent.com/88423149/185036507-50c7c5aa-efcd-4f30-8e13-af5980ea8444.png)
Detectron2 Custom FasterRCNN Model Prediction

•	The model was then evaluated using COCOEvaluator which is an inbuilt function from Detectron2. 

![image](https://user-images.githubusercontent.com/88423149/185036546-27dd4331-59b1-499a-9de0-b7bc5da78cdc.png)
Detectron2 Custom FasterRCNN Model Evaluation

•	From the evaluation metrics, we can tell that the Average precision and Recall is very low. This is due to the usage of the model zoo weights on the model

•	There was a limitation with the model prediction, where the OUTPUT directory with “model_final.pth” file, which has the weights of the trained model was not displaying weights. Although, this was working in Colabs. As this model was run in Jupyter notebook – GPU environment.  There seems to be some unknown issue.

•	However, the model weights from model zoo were slightly accurate. Hence, we’re considering this and proceeding with this, as the final model used in production.

•	The model was also checked with RetinaNet and Masked RCNN pre trained model from model zoo of Detectron2 tool. The predicted results are as follows:

![image](https://user-images.githubusercontent.com/88423149/185036559-5a032aba-0081-434e-8fa2-a7ebe87292a9.png)
Detectron2 Custom MaskedRCNN Prediction

![image](https://user-images.githubusercontent.com/88423149/185036571-b8d1ff70-4189-4034-9408-19219c5c77ec.png)
Detectron2 Custom RetinaNet Prediction

•	Out of RetinaNet, Masked RCNN and Faster RCNN. The Faster RCNN had the best result, it however didn’t perform object masking, but it had better IOU value overall. Hence, the Faster RCNN model will be deployed into Streamlit API in local environment.

14. Streamlit Model Deployment
•	The FasterRCNN model is deployed into Streamlit GUI API for an interactive model prediction. 

•	The custom model is stored as objectDetection.py in the local directory.

•	The streamlit app is stored as detectron2_app.py in the local directory.

The files are attached for reference, the code is also Hashed in the notebook.

![image](https://user-images.githubusercontent.com/88423149/185036663-c4be91b6-99b5-44ba-ab9d-fdc487823ce0.png)
Detectron2 Model Prediction in Streamlit

The Faster RCNN model on rendering a Video 

https://user-images.githubusercontent.com/88423149/185037267-c0bca9e2-f93c-4391-b75f-69ba54173a24.mp4

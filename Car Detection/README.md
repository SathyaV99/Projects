**Car Detection Project**

![image](https://user-images.githubusercontent.com/88423149/184920574-8690d296-de57-43fa-af12-cadce0a4f0bc.png)


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


        

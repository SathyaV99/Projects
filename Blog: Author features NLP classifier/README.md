**SUMMARY**

The main objective of the project is to use over 600,000 posts from from more than 19 thousand bloggers and label the blog such that they can be categorized according to different interests.

In this project, using regular expressions: corpus was cleaned removing unwanted characters, spaces, stop words, and converted to lowercase. Labels were made from the categories.

Data was split into train and test, using vectorizer to maps words, count the number of words, and multilabel binarizer for encoding the multi variable classes.

Using logistic regression, the classification of the corpus in test instance is performed.

<img width="889" alt="image" src="https://user-images.githubusercontent.com/88423149/181878266-6820b3a2-4093-4d53-b132-ed86caee411e.png">

Precision : 0.710154	
Recall : 0.359492
F1 Score: 0.477344

Here accuracy is not actually of any concern, the classification metrics are valued more.

<img width="899" alt="image" src="https://user-images.githubusercontent.com/88423149/181878364-28c7c607-f0e2-4e7a-9705-407515767dbf.png">

# BrainWaves - Machine Learning
## Societe Generale Global Solution Centre
### Asset Clustering
World is changing fast. We are experiencing a technological revolution as big as industrial revolution. One of the major drivers of this revolution is Machine learning. Banking and financial services are at forefront of digital revolution and are using machine learning in their businesses.

The problem you will be working on is related to trading in financial markets and it is quite challenging. You will be clustering financial assets and building forecasting model which are the initial steps needed to build a machine learning based quantitative trading strategy. The common problem in trading is to forecast a variable based on other variables. Since financial data is time related one has to use recent past information to capture all the dynamics. We have divided the problem into two parts. First problem is of clustering which is needed to understand the data and the second problem is of prediction/classification.

Following is the first part of the problem. **Given the time series data of 100 financial assets which are correlated, you have to cluster them into similar groups. These groups may relate to market segments such as Metals & Minerals, Banking/Finance etc.**
#### Data
The dataset is common for both the problems. The training file(train.csv) contains the time-series data. Below is the relevant explanation of the different columns.

|Column   |Explanation|
|:-------:|-----------|
|Time     |Integer representing the time factor|
|X1..X100 |Stock prices belonging to various industries|
|Y        |Y is forecast value of the financial variable which we are interested in. Y has +1 if the underlying asset goes up in the future and it has -1 if it goes down.|
**Note**: The actual number of clusters are between 5 to 15
#### Submission
You have to submit a file containing your predictions in the following format.
```
Asset,Cluster
X1,6
X2,6
X3,7
X4,2
X5,9
```
### Predict the growth
Following is the second part of the problem. **Given the time series data of 100 financial assets which are correlated, you have to build a model to forecast return of the financial asset from which Y is derived. You can also use the clustering info from the first problem as needed.**
**Note**: The problem is to forecast return of financial asset from which Y is derived. The accuracy of the models will be relatively low (closer to 50%).
#### Submission
You have to submit a file containing your predictions in the following format.
```
Time,Y
3001,1
3002,-1
3003,1
3004,-1
3005,1
```

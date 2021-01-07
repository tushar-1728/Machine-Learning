# UE17CS303 - MACHINE LEARNING ASSIGNMENT
##### Applying Naive Bayes classifier to predict voting pattern in a two party electoral system.
![Machine learning](./images/machine-learning.jpg)

## INTRODUCTION
The objective of this project is to apply Naive Bayes classifier on 1984 American election dataset to predict whether a voter is republican or democrat. The dataset has 435 entries each consisting of 16 binary attributes and one target attribute which is also binary in nature. So basically, the dataset has 16⨯435 data points out of which 392 are missing. Since all the attributes are binary, we have converted all ‘yes’ and ‘no’ to ones and zeros. We have also assumed democrat as positive classification and republican as negative.

## DATA CLEANING AND VISUALIZATION
As we have discussed before, our house-votes dataset has 16⨯435 data points out of which 392 are missing and these missing values are distributed over 203 rows and last attribute has about 25% data missing, hence we can’t perform list wise deletion.
![Missing data-points in the dataset](./images/missing-datapoints-in-the-dataset.png)
Yellow lines in the above figure show the distribution of missing values in all attributes.

So, what we have done is as follows:

We have plotted the distribution of ‘yes’ and ‘no’ for all the 16 attributes across the target attributes: democrat and republican. And now we have used this distribution to impute the missing values.

Suppose the `duty free import` attribute for the second row is missing and the target attribute for that row is democrat. Then we see the distribution of ‘yes’ and ‘no’ in the `duty free import` attribute for all the voters who are democratic. Now whichever attribute value has higher count, we replace the missing value with that attribute value.

The graphs below show the above mentioned distributions:
![crime](./images/crime.png){:height="50%" width="50%"}
![immigration](./images/immigration.png){:height="50%" width="50%"}




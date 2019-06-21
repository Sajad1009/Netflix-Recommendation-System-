# How Netflix Recommendation System Work (Collaborative filtering) 

Netflix offers large number of of TV shows available for streaming. It recommends titles for the users.
If you use Netflix you may have noticed they create amazing precises genres:Romantic Dramas Where The Main Character is Left Handed.
How do they come up with those genres? 
How to they deal with giving great recommendations to their 90 million-plus subscribers who are already used to getting recommendations
from pretty much every platform they use? Deep learning, algorithms and creativity.


Collaborative filtering is commonly used for recommender systems. These techniques aim to fill in the missing entries of a user-item association matrix. spark.ml currently supports model-based collaborative filtering, in which users and products are described by a small set of latent factors that can be used to predict missing entries. spark.ml uses the alternating least squares (ALS) algorithm to learn these latent factors. The implementation in spark.ml has the following parameters:

PySpark is the collaboration of Apache Spark and Python. Apache Spark is an open-source cluster-computing framework, built around speed, ease of use, and streaming analytics whereas Python is a general-purpose, high-level programming language. 


first we have to install pyspark and get it ready to work....I  provied some links to help you with this task 

https://medium.com/@GalarnykMichael/install-spark-on-windows-pyspark-4498a5d8d66c
https://aws.amazon.com/premiumsupport/knowledge-center/emr-pyspark-python-3x/


I  installed pyspark in google could (however spark installed by default in google could but I wanted to run on my virtual env)

```
In your bash cmd  use pip install findspark  and then set SparkContext
```

```
import findspark
findspark.init('/home/sak/spark-2.4.3-bin-hadoop2.7')  #here you sould set the path to spark folder in your machine/server
import pyspark
from pyspark import SparkContext, SparkConf
conf = SparkConf().setAppName("MyApp).setMaster("local") # you could replace local with year url if you want
sc = SparkContext(conf=conf)
```

Here we have to start spark session in order to start with spark sql 

from pyspark.sql import SparkSession

```
spark = SparkSession \
    .builder \
    .appName("Python Spark SQL basic example") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()
 ```   

Now we import the required lib

```
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
``` 
Now we reading the data into RDD spark format and give it titles names and then convert it to sql.Data frame 

``` 
lines = spark.read.text("sample_movielens_ratings.txt").rdd
parts = lines.map(lambda row: row.value.split("::"))
ratingsRDD = parts.map(lambda p: Row(userId=int(p[0]), movieId=int(p[1]),rating=float(p[2]), timestamp=int(p[3])))
ratings = spark.createDataFrame(ratingsRDD)
(training, test) = ratings.randomSplit([0.8, 0.2])
```

###Now we build the recommendation model using ALS on the training data
###Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics

```
als = ALS(maxIter=5, regParam=0.01, userCol="userId", itemCol="movieId", ratingCol="rating",coldStartStrategy="drop")
model = als.fit(training)
```

#Now we can evaluate the model by computing the RMSE on the test data

```
predictions = model.transform(test)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))
```

```
```
### Generate top 10 movie recommendations for each user
```
userRecs = model.recommendForAllUsers(10)
```
### Generate top 10 user recommendations for each movie
```
movieRecs = model.recommendForAllItems(10)
```
### Generate top 10 movie recommendations for a specified set of users
```
users = ratings.select(als.getUserCol()).distinct().limit(3)
userSubsetRecs = model.recommendForUserSubset(users, 10)
```
### Generate top 10 user recommendations for a specified set of movies
```
movies = ratings.select(als.getItemCol()).distinct().limit(3)
movieSubSetRecs = model.recommendForItemSubset(movies, 10)
```
### Show the results 
```
userRecs.show()
movieRecs.show()
userSubsetRecs.show()
movieSubSetRecs.show()
```


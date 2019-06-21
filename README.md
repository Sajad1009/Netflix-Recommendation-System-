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
```

Root-mean-square error = 1.6638709815338888
+------+--------------------+
|userId|     recommendations|
+------+--------------------+
|    28|[[34, 6.2745895],...|
|    26|[[94, 5.3556576],...|
|    27|[[32, 4.5518517],...|
|    12|[[19, 5.7531867],...|
|    22|[[44, 5.652228], ...|
|     1|[[52, 4.6947684],...|
|    13|[[93, 3.8497524],...|
|     6|[[25, 4.4996967],...|
|    16|[[51, 4.749304], ...|
|     3|[[62, 5.261743], ...|
|    20|[[90, 6.108018], ...|
|     5|[[55, 4.80991], [...|
|    19|[[32, 3.8459191],...|
|    15|[[46, 4.9723687],...|
|    17|[[46, 5.135776], ...|
|     9|[[49, 5.1126733],...|
|     4|[[52, 3.8264365],...|
|     8|[[52, 5.0733185],...|
|    23|[[90, 6.6885557],...|
|     7|[[25, 5.151322], ...|
+------+--------------------+
only showing top 20 rows

+-------+--------------------+
|movieId|     recommendations|
+-------+--------------------+
|     31|[[12, 3.7749054],...|
|     85|[[8, 4.8402247], ...|
|     65|[[23, 4.866437], ...|
|     53|[[8, 4.883639], [...|
|     78|[[23, 1.4780338],...|
|     34|[[28, 6.2745895],...|
|     81|[[28, 4.8189664],...|
|     28|[[18, 5.008406], ...|
|     76|[[14, 4.9252243],...|
|     26|[[26, 4.680863], ...|
|     27|[[11, 5.1526017],...|
|     44|[[22, 5.652228], ...|
|     12|[[28, 4.7525043],...|
|     91|[[28, 5.519838], ...|
|     22|[[22, 4.2934175],...|
|     93|[[2, 5.158542], [...|
|     47|[[10, 4.1678576],...|
|      1|[[11, 4.5958223],...|
|     52|[[8, 5.0733185], ...|
|     13|[[23, 3.8065495],...|
+-------+--------------------+
only showing top 20 rows

+------+--------------------+
|userId|     recommendations|
+------+--------------------+
|    26|[[94, 5.3556576],...|
|    19|[[32, 3.8459191],...|
|    29|[[46, 4.9788456],...|
+------+--------------------+

+-------+--------------------+
|movieId|     recommendations|
+-------+--------------------+
|     65|[[23, 4.866437], ...|
|     26|[[26, 4.680863], ...|
|     29|[[14, 5.337469], ...|
+-------+--------------------+

```


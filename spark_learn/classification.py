#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql import DataFrameNaFunctions
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import Binarizer
from pyspark.ml.feature import VectorAssembler, StringIndexer, VectorIndexer

#%%
sc = SparkContext()
sql = SQLContext(sc)

df = sql.read.csv('big-data-4/daily_weather.csv',
                   header='true',inferSchema='true')
df.columns

featureColumns = ['air_pressure_9am',
 'air_temp_9am',
 'avg_wind_direction_9am',
 'avg_wind_speed_9am',
 'max_wind_direction_9am',
 'max_wind_speed_9am',
 'rain_accumulation_9am',
 'rain_duration_9am']

df.drop('number')

df = df.drop('number')
df = df.na.drop()
df.count()

binarizer = Binarizer(threshold=24.99999,
                      inputCol='relative_humidity_3pm',
                      outputCol='label')
binarizedDF = binarizer.transform(df)

binarizedDF.select('relative_humidity_3pm','label').show(4)

assembler = VectorAssembler(inputCols=featureColumns,
                            outputCol='features')
assembled = assembler.transform(binarizedDF)
assembled.show(2)

(train, test) = assembled.randomSplit([0.8,0.2],seed=13234)
train.count()
test.count()

dt = DecisionTreeClassifier(featuresCol='features',
                            labelCol='label',
                            maxDepth=5,
                            minInstancesPerNode=20,
                            impurity='gini')

pipeline = Pipeline(stages = [dt])
model = pipeline.fit(train)
predictions = model.transform(test)
predictions.select('prediction','label').show(10)
#predictions.select('prediction','label').write.csv('big-data-4/predictions.csv',
#                 header='true')

#%% evaluation
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics

predictions = predictions.select('prediction','label')
evaluator = MulticlassClassificationEvaluator(labelCol='label',
                                              predictionCol='prediction',
                                              metricName='accuracy')
accuracy = evaluator.evaluate(predictions)
print('Acc = %g' % accuracy)

predictions.rdd.take(2)
predictions.rdd.map(tuple).take(2)
metrics = MulticlassMetrics(predictions.rdd.map(tuple))
metrics.confusionMatrix().toArray().T

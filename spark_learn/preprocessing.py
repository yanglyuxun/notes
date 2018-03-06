#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

#%% read data
from pyspark import SparkContext
sc =SparkContext()

from pyspark.sql import SQLContext

sqlContext = SQLContext(sc)

df = sqlContext.read.load('big-data-4/daily_weather.csv',
                     format='com.databricks.spark.csv',
                     header='true',inferSchema='true')

#%% exploration

df.columns
df.printSchema()
df.describe().show()
df.describe().toPandas().T
df.describe('air_pressure_9am').show()
df.count()

df2 = df.na.drop(subset=['air_pressure_9am'])
df2.count()

df2.stat.corr('rain_accumulation_9am','rain_duration_9am')

#%% preparation

removeAllDF = df.na.drop()
df.describe('air_pressure_9am').show()
removeAllDF.describe('air_pressure_9am').show()

#from pyspark.sql.functions import avg
imputeDF = df
for x in imputeDF.columns:
    meanValue = removeAllDF.agg({x:'avg'}).first()[0]
    print(x, meanValue)
    imputeDF = imputeDF.na.fill(meanValue,[x])

df.describe('air_pressure_9am').show()
imputeDF.describe('air_pressure_9am').show()

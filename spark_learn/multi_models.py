#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
multiple models
"""
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler

import gzip,os

import utils

sc = SparkContext()
sql = SQLContext(sc)

with gzip.open('big-data-4/minute_weather.csv.gz') as zf:
    with open('/tmp/a.csv','wb') as f:
        f.write(zf.read())
df = sql.read.csv('/tmp/a.csv',
                  header = 'true',
                  inferSchema='true')
df.count()
filtered = df.filter((df.rowID %10)==0)
filtered.count()

filtered.describe().toPandas().transpose()
filtered.filter(filtered.rain_accumulation==0.0).count()
filtered.filter(filtered.rain_duration==0.0).count()
working = filtered.drop('rain_accumulation','rain_duration','hpwren_timestamp')
working2 = working.na.drop()
working.count()
working2.count()

#%% scale
working = working2
working.columns
features = ['air_pressure',
 'air_temp',
 'avg_wind_direction',
 'avg_wind_speed',
 'max_wind_direction',
 'max_wind_speed',
 'relative_humidity']
assembler = VectorAssembler(inputCols = features,
                            outputCol='features_unscaled')
assembled = assembler.transform(working)
scaler = StandardScaler(inputCol='features_unscaled',
                        outputCol='features',
                        withStd=True,
                        withMean=True)
scalerModel = scaler.fit(assembled)
scaled = scalerModel.transform(assembled)

#%% elbow plot
scaled = scaled.select('features','rowID')
elbowset = scaled.filter((scaled.rowID % 3)==0).select('features')
elbowset.persist()

clusters = range(2,31)
wsseList = utils.elbow(elbowset, clusters)

utils.elbow_plot(wsseList, clusters)

#%% clustering

scaledF = scaled.select('features')
scaledF.persist()

kmeans = KMeans(k=12, seed=1)
model = kmeans.fit(scaledF)
transformed = model.transform(scaledF)
clusters = model.clusterCenters()

#%% parallel plots
P = utils.pd_centers(features, clusters)
# dry days
utils.parallel_plot(P[P['relative_humidity']<-0.5], P)
# warm days
utils.parallel_plot(P[P['air_temp']>0.5], P)



#%%
try:
    os.remove('/tmp/a.csv')
except:
    pass

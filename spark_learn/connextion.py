#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

from pyspark import SparkContext, SparkConf

appName = 'test_app'
master = 'spark://192.168.1.55:7077'
conf = SparkConf().setAppName(appName).setMaster(master)
sc = SparkContext(conf=conf)

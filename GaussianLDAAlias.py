#!/usr/bin/python2.7  
# -*- coding: utf-8 -*-
import numpy as np
from Data import Data
import util
import random
import math
import sys
import NormalInverseWishart
from VoseAlias import VoseAlias

dataVectors = []
corpus = []
numIterations = 0
K = 0
N = 0
tableCounts = dict()
tableCountsPerDoc = []
tableAssignments = []
tableMeans = []
tableCholeskyLTriangularMat = []
logDeterminants = []
currentIteration = 0
prior = NormalInverseWishart()
CholSigma0 = None
dirName = ""
alpha = 0.0
q = []
done = False
MH_STEPS = 2

def updateTableParams(tableId,custId,isRemoved):
	count = tableCounts[tableId]
	k_n = prior.k_0 + count
	nu_n = prior.nu_0 + count
	scaleTdistrn = (k_n + 1) / (k_n * (nu_n - Data.D + 1))
	oldLTriangularDecomp = tableCholeskyLTriangularMat[tableId]

	if isRemoved:
		x = dataVectors[custId] - tableMeans[tableId]
		coeff = math.sqrt((k_n + 1) / k_n)
		x = x * coeff
		oldLTriangularDecomp,x = Util.cholRank1Downdate(oldLTriangularDecomp,x)
		tableCholeskyLTriangularMat[tableId] = oldLTriangularDecomp
		newMean = (k_n + 1) * tableMeans[tableId]
		newMean = newMean - dataVectors[custId]
		newMean = newMean / k_n
		tableMeans[tableId] = newMean
	else:
		newMean = tableMeans[tableId] * (k_n - 1)
		newMean = newMean + dataVectors[custId]
		newMean = newMean / k_n
		tableMeans[tableId] = newMean
		x = dataVectors[custId] - tableMeans[tableId]
		x = coeff * x
		oldLTriangularDecomp,x = util.cholRank1Update(oldLTriangularDecomp,x)
		tableCholeskyLTriangularMat[tableId] = oldLTriangularDecomp
	
	logDet = 0.0
	for l in range(Data.D):
		logDet += math.log(oldLTriangularDecomp[l][l])
	logDet += Data.D * math.log(scaleTdistrn) / 2.0

	if tableId < len(logDeterminants):
		logDeterminants[tableId] = logDet
	else:
		logDeterminants.append(logDet)


def initialize():
	currentIteration = 0
	if prior.nu_0 < Data.D:
		print "The initial degrees of freedom of the prior is less than the dimension!. Setting it to the number of dimension: %d" % Data.D
		prior.nu_0 = Data.D

	scaleTdistrn = (prior.k_0 + 1) / (prior.k_0 * (prior.nu_0 - Data.D + 1))
	for i in range(K):
		priorMean = prior.mu_0
		initialCholesky = CholSigma0
		logDet = 0.0
		for l in range(Data.D):
			logDet += math.log(CholSigma0[l][l])
		logDet += (Data.D * math.log(scaleTdistrn) / 2.0)
		logDeterminants.append(logDet)
		tableMeans.append(priorMean)
		tableCholeskyLTriangularMat.append(initialCholesky)

	for d in range(N):
		doc = corpus[d]
		wordCounter = 0
		assignment = []
		for i in doc:
			tableId = int(K * random.random())
			assignment.append(tableId)
			if tableCounts.get(tableId,-1) >= 0:
				tableCounts[tableId] += 1
			else:
				tableCounts[tableId] = 1
			tableCountsPerDoc[tableId][d] += 1
			updateTableParams(tableId,i,false)
			wordCounter += 1
		tableAssignments.append(assignment)

	for i in range(K):
		if tableCounts.get(i,-1) < 0:
			print "Still some tables are empty....exiting!"
			sys,exit(1)
	print "Initialization complete"
	avgLL = util.calculateAvgLL(corpus, tableAssignments, dataVectors, tableMeans, tableCholeskyLTriangularMat, K,N,prior,tableCountsPerDoc)
	print "Average ll at the begining %lf" % avgLL
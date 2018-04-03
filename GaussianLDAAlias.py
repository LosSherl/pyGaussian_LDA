#!/usr/bin/python2.7  
# -*- coding: utf-8 -*-
import numpy as np
from Data import Data
import time
import util
import random
import math
import sys
import threading
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

def logMultivariateTDensity(x,tableId):
	dlogprob = 0.0
	count = tableCounts[tableId]
	k_n = prior.k_0 + count
	nu_n = prior.nu_0 + count	
	scaleTdistrn = math.sqrt((k_n + 1) / (k_n * (nu_n - Data.D + 1)))
	nu = prior.nu_0 + count - Data.D + 1
	x_minus_mu = x - tableMeans[tableId]
	lTriangularChol = tableCholeskyLTriangularMat[tableId] * scaleTdistrn
	x_minus_mu = np.linalg(lTriangularChol,x_minus_mu)
	x_minus_mu_T = x_minus_mu.T
	mul = x_minus_mu_T * x_minus_mu
	val = mul[0][0]
	logprob = math.lgamma((nu + Data.D) / 2) - (math.lgamma(nu / 2) + Data.d / 2 * (math.log(nu) + math.log(math.pi)) + logDeterminants[tableId] + (nu + Data.D) / 2 * math.log(1 + val / nu))

	return logprob

def sample():
	initRun()
	t = threading.thread(target = run,name = "runThread")
	t.start()
	for currentIteration in range(numIterations):
		startTime = time.time()
		print "Starting iteration %d" % currentIteration
		for d in range(len(corpus)):
			document = corpus[d]
			wordCounter = 0
			for custId in document:
				oldTableId = tableAssignments[d][wordCounter]
				tableAssignments[d][wordCounter] -= 1
				oldCount = tableCounts[oldTableId]
				tableCounts[oldTableId] -= 1
				tableCountsPerDoc[oldTableId][d] -= 1
				updateTableParams(oldTableId,custId,True)
				posterior = []
				nonZeroTopicIndex = []
				Max = float("-inf")
				pSum = 0.0
				for k in range(K):
					if tableCountsPerDoc[k][d] > 0:
						logLikelihood = logMultivariateTDensity(dataVectors[custId],k)
						logPosterior = math.log(tableCountsPerDoc[k][d]) + logLikelihood
						nonZeroTopicIndex.append[k]
						posterior.append(logPosterior)
						if logPosterior > Max:
							Max = logPosterior
				for k in range(K):
					p = posterior[k]
					p -= p - Max
					expP = math.exp(p)
					pSum += expP
					posterior[k] = pSum
				select_pr = pSum / (pSum + alpha * q[custId].wsum)
				newTableId = -1
				for r in range(MH_STEPS):
					if random.random() < select_pr:
						u = random.random() * pSum
						temp = util.binSearchArrayList(posterior,u,0,len(posterior) - 1)
						newTableId = nonZeroTopicIndex[temp]
					else:
						newTableId = q[custId].sampleVose()

					if oldTableId != newTableId:
						temp_old = logMultivariateTDensity(dataVectors[custId],oldTableId)
						temp_new = logMultivariateTDensity(dataVectors[custId],newTableId)
						acceptance = (tableCountsPerDoc[newTableId][d] + alpha) / (tableCountsPerDoc[oldTableId][d] + alpha) \
						* math.exp(temp_new - temp_old) \
						* (tableCountsPerDoc[oldTableId][d] * temp_old + alpha * q[custId].w[oldTableId]) \
						/ (tableCountsPerDoc[newTableId][d] * temp_new + alpha * q[custId].w[newTableId])
						u = random.random()
						if u < acceptance:
							oldTableId = newTableId
				tableAssignments[d][wordCounter] = newTableId
				tableCounts[newTableId] += 1
				tableCountsPerDoc[newTableId][d] += 1
				updateTableParams(newTableId,custId,false)
				wordCounter += 1
			if d % 10 == 0:
				print "Done for document %d" % d
				print "Time for document %d  %lf" % (d,time.time() - startTime)
		stopTime = time.time()
		elapsedTime = (stopTime - startTime) / 1000
		print "Time taken for this iteration %lf" % elapsedTime
		avgLL = util.calculateAvgLL(corpus,tableAssignments,dataVectors,tableMeans,tableCholeskyLTriangularMat,K,N,prior,tableCountsPerDoc)
		print "Avg log-likelihood at the end of iteration %d is %lf" % (currentIteration,avgLL)
	done = true
	t.join()

def main():
	startTime = time.time()
	args = sys.argv
	inputFile = args[0]
	Data.inputFileName = inputFile
	D = int(args[1])
	numIterations = int(ars[2])
	Data.D = D
	K = int(args[3])
	dirName = args[4]
	data = Data.readData()
	dataVectors = []
	for i in range(data.shape[0]):
		dataVectors.append(data[i].T)
	print "Total number of vectors are %d" % data.shape[0]
	inputCorpusFile = args[5]
	corpus = Data.readCorpus(inputCorpusFile)
	print "Corpus file read"
	N = len(corpus)
	print "Total number of documents are %d" % N
	prior.mu_0 = util.getSampleMean(dataVectors)
	prior.nu_0 = Data.D
	prior.sigma_0 = np.eye(Data.D)
	prior.sigma_0 = 3 * Data.D * prior.sigma_0
	prior.k_0 = 0.1
	CholSigma0 = np.zeros((Data.D,Data.D))
	CholSigma0 = CholSigma0 + prior.sigma_0
	alpha = 1 / K

	CholSigma0 = np.linalg.cholesky(CholSigma0)
	tableAssignments = []
	tableCountsPerDoc = np.zeros((K,N))
	q = [0] * Data.numVectors
	for w in range(Data.numVectors):
		q[w] = VoseAlias()
		q[w].init(K)

	print "Starting to initialize"
	initialize()
	print "Gibbs sampler will run for %d iterations" % numIterations
	sample()
	stopTime = time.time()
	elapsedTime = (stopTime - startTime) / 1000
	print "Time taken %lf" % elapsedTime
	util.printGaussians(tableMeans,tableCholeskyLTriangularMat,K,dirName)
	util.printDocumentTopicDistribution(tableCountsPerDoc,N,K,dirName,alpha)
	util.printTableAssignments(tableAssignments,dirName)
	util.printNumCustomersPerTopic(tableCountsPerDoc,dirName,K,N)
	print "Done"

def run():
	temp = VoseAlias()
	temp.init(K)

	while not done:
		for w in range(Data.numVectors):
			Max = float("-inf")
			for k in range(K):
				logLikelihood = logMultivariateTDensity(dataVectors[w],k)
				temp.w[k] = logLikelihood
				if logLikelihood > Max:
					Max = logLikelihood
			temp.wsum = 0.0
			for k in range(K):
				p = temp.w[k]
				p -= Max
				expP = math.exp(p)
				temp.wsum += expP
				temp.w[k] = expP
			temp.generateTable()
			q[w].copy(temp)

def initRun():
	temp = VoseAlias()
	temp.init(K)
	for w in range(Data.numVectors):
		Max = float("-inf")
		for k in range(K):
			logLikelihood = logMultivariateTDensity(dataVectors[w],k)
			temp.w[k] = logLikelihood
			if logLikelihood > Max:
				Max = logLikelihood
		temp.wsum = 0.0
		for k in range(K):
			p = temp.w[k]
			p -= Max
			expP = math.exp(p)
			temp.wsum += expP
			temp.w[k] = expP
		temp.generateTable()
		q[w].copy(temp)
		
			
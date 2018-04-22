#!/usr/bin/python2.7  
# -*- coding: utf-8 -*-
import numpy as np
from Data import Data
import time
import Util
import random
import math
import sys
import threading
from NormalInverseWishart import NormalInverseWishart
from VoseAlias import VoseAlias

class Gaussian_LDA(object):
	def __init__(self):
		self.dataVectors = []
		self.corpus = []
		self.numIterations = 0
		self.K = 0
		self.N = 0
		self.tableCounts = dict()
		self.tableCountsPerDoc = []
		self.tableAssignments = []
		self.tableMeans = []
		self.tableCholeskyLTriangularMat = []
		self.logDeterminants = []
		self.currentIteration = 0
		self.prior = NormalInverseWishart()
		self.CholSigma0 = None
		self.dirName = ""
		self.alpha = 0.0
		self.q = []
		self.done = False
		self.MH_STEPS = 2

	def updateTableParams(self,tableId,custId,isRemoved):
		s = time.time()
		count = self.tableCounts[tableId]
		k_n = self.prior.k_0 + count
		nu_n = self.prior.nu_0 + count
		scaleTdistrn = (k_n + 1) / (k_n * (nu_n - Data.D + 1))
		oldLTriangularDecomp = self.tableCholeskyLTriangularMat[tableId]

		if isRemoved:
			x = self.dataVectors[custId] - self.tableMeans[tableId]
			coeff = math.sqrt((k_n + 1) / k_n)
			x *= coeff
			Util.cholRank1Downdate(oldLTriangularDecomp,x)
			self.tableCholeskyLTriangularMat[tableId] = oldLTriangularDecomp
			newMean = (k_n + 1) * self.tableMeans[tableId]
			newMean -= self.dataVectors[custId]
			newMean /= k_n
			self.tableMeans[tableId] = newMean
		else:
			newMean = self.tableMeans[tableId] * (k_n - 1)
			newMean += self.dataVectors[custId]
			newMean /= k_n
			self.tableMeans[tableId] = newMean
			x = self.dataVectors[custId] - self.tableMeans[tableId]
			coeff = math.sqrt(k_n / (k_n - 1))
			x *= coeff
			Util.cholRank1Update(oldLTriangularDecomp,x)
			self.tableCholeskyLTriangularMat[tableId] = oldLTriangularDecomp
		
		logDet = 0.0
		for l in range(Data.D):
			logDet += math.log(oldLTriangularDecomp[l][l])
		logDet += Data.D * math.log(scaleTdistrn) / 2.0

		if tableId < len(self.logDeterminants):
			self.logDeterminants[tableId] = logDet
		else:
			self.logDeterminants.append(logDet)

	def initialize(self):
		self.currentIteration = 0
		if self.prior.nu_0 < Data.D:
			print "The initial degrees of freedom of the prior is less than the dimension!. Setting it to the number of dimension: %d" % Data.D
			self.prior.nu_0 = Data.D

		scaleTdistrn = (self.prior.k_0 + 1) / (self.prior.k_0 * (self.prior.nu_0 - Data.D + 1))
		for i in range(self.K):
			priorMean = self.prior.mu_0
			initialCholesky = self.CholSigma0
			logDet = 0.0
			for l in range(Data.D):
				logDet += math.log(self.CholSigma0[l][l])
			logDet += (Data.D * math.log(scaleTdistrn) / 2.0)
			self.logDeterminants.append(logDet)
			self.tableMeans.append(priorMean)
			self.tableCholeskyLTriangularMat.append(initialCholesky)
		for d in range(self.N):
			print d
			doc = self.corpus[d]
			wordCounter = 0
			assignment = []
			for i in doc:
				tableId = int(self.K * random.random())
				assignment.append(tableId)
				if self.tableCounts.get(tableId,-1) >= 0:
					self.tableCounts[tableId] += 1
				else:
					self.tableCounts[tableId] = 1
				self.tableCountsPerDoc[tableId][d] += 1
				self.updateTableParams(tableId,i,False)
				wordCounter += 1
			self.tableAssignments.append(assignment)
		for i in range(self.K):
			if self.tableCounts.get(i,-1) < 0:
				print "Still some tables are empty....exiting!"
				sys,exit(1)
		print "Initialization complete"
		avgLL = Util.calculateAvgLL(self.corpus, self.tableAssignments, self.dataVectors, self.tableMeans, self.tableCholeskyLTriangularMat, self.K,self.N,self.prior,self.tableCountsPerDoc)
		print "Average ll at the begining %lf" % avgLL

	def logMultivariateTDensity(self,x,tableId):
		dlogprob = 0.0
		count = self.tableCounts[tableId]
		k_n = self.prior.k_0 + count
		nu_n = self.prior.nu_0 + count	
		scaleTdistrn = math.sqrt((k_n + 1) / (k_n * (nu_n - Data.D + 1)))
		nu = self.prior.nu_0 + count - Data.D + 1
		x_minus_mu = x - self.tableMeans[tableId]
		lTriangularChol = self.tableCholeskyLTriangularMat[tableId] * scaleTdistrn
		x_minus_mu = np.linalg(lTriangularChol,x_minus_mu)
		x_minus_mu_T = x_minus_mu.T
		mul = x_minus_mu_T * x_minus_mu
		val = mul[0][0]
		logprob = math.lgamma((nu + Data.D) / 2) - (math.lgamma(nu / 2) + Data.D / 2 * (math.log(nu) + math.log(math.pi)) + self.logDeterminants[tableId] + (nu + Data.D) / 2 * math.log(1 + val / nu))

		return logprob

	def sample(self):
		initRun()
		t = threading.thread(target = run,name = "runThread")
		t.start()
		for self.currentIteration in range(self.numIterations):
			startTime = time.time()
			print "Starting iteration %d" % self.currentIteration
			for d in range(len(self.corpus)):
				document = self.corpus[d]
				wordCounter = 0
				for custId in document:
					oldTableId = self.tableAssignments[d][wordCounter]
					self.tableAssignments[d][wordCounter] -= 1
					oldCount = self.tableCounts[oldTableId]
					self.tableCounts[oldTableId] -= 1
					self.tableCountsPerDoc[oldTableId][d] -= 1
					updateTableParams(oldTableId,custId,True)
					posterior = []
					nonZeroTopicIndex = []
					Max = float("-inf")
					pSum = 0.0
					for k in range(K):
						if self.tableCountsPerDoc[k][d] > 0:
							logLikelihood = logMultivariateTDensity(self.dataVectors[custId],k)
							logPosterior = math.log(self.tableCountsPerDoc[k][d]) + logLikelihood
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
							temp = Util.binSearchArrayList(posterior,u,0,len(posterior) - 1)
							newTableId = nonZeroTopicIndex[temp]
						else:
							newTableId = self.q[custId].sampleVose()

						if oldTableId != newTableId:
							temp_old = logMultivariateTDensity(self.dataVectors[custId],oldTableId)
							temp_new = logMultivariateTDensity(self.dataVectors[custId],newTableId)
							acceptance = (self.tableCountsPerDoc[newTableId][d] + alpha) / (self.tableCountsPerDoc[oldTableId][d] + alpha) \
							* math.exp(temp_new - temp_old) \
							* (self.tableCountsPerDoc[oldTableId][d] * temp_old + alpha * self.q[custId].w[oldTableId]) \
							/ (self.tableCountsPerDoc[newTableId][d] * temp_new + alpha * self.q[custId].w[newTableId])
							u = random.random()
							if u < acceptance:
								oldTableId = newTableId
					self.tableAssignments[d][wordCounter] = newTableId
					self.tableCounts[newTableId] += 1
					self.tableCountsPerDoc[newTableId][d] += 1
					updateTableParams(newTableId,custId,false)
					wordCounter += 1
				if d % 10 == 0:
					print "Done for document %d" % d
					print "Time for document %d  %lf" % (d,time.time() - startTime)
			stopTime = time.time()
			elapsedTime = (stopTime - startTime) / 1000
			print "Time taken for this iteration %lf" % elapsedTime
			avgLL = Util.calculateAvgLL(self.corpus,self.tableAssignments,self.dataVectors,self.tableMeans,self.tableCholeskyLTriangularMat,self.K,self.N,self.prior,self.tableCountsPerDoc)
			print "Avg log-likelihood at the end of iteration %d is %lf" % (self.currentIteration,avgLL)
		done = true
		t.join()

	def main(self):
		startTime = time.time()
		args = sys.argv
		inputFile = args[1]
		Data.inputFileName = inputFile
		D = int(args[2])
		self.numIterations = int(args[3])
		Data.D = D
		self.K = int(args[4])
		dirName = args[5]
		data = Data.readData()
		self.dataVectors = []
		for i in range(data.shape[0]):
			self.dataVectors.append(data[i].T)
		print "Total number of vectors are %d" % data.shape[0]
		inputcorpusFile = args[6]
		self.corpus = Data.readCorpus(inputcorpusFile)
		print "corpus file read"
		self.N = len(self.corpus)
		print "Total number of documents are %d" % self.N
		self.prior.mu_0 = Util.getSampleMean(self.dataVectors)
		self.prior.nu_0 = Data.D
		self.prior.sigma_0 = np.eye(Data.D)
		self.prior.sigma_0 = 3 * Data.D * self.prior.sigma_0
		self.prior.k_0 = 0.1
		self.CholSigma0 = np.zeros((Data.D,Data.D))
		self.CholSigma0 = self.CholSigma0 + self.prior.sigma_0
		self.alpha = 1 / self.K
		self.CholSigma0 = np.linalg.cholesky(self.CholSigma0)
		self.tableAssignments = []
		self.tableCountsPerDoc = np.zeros((self.K,self.N))
		self.q = [0] * Data.numVectors
		for w in range(Data.numVectors):
			self.q[w] = VoseAlias()
			self.q[w].init(self.K)

		print "Starting to initialize"
		self.initialize()
		print "Gibbs sampler will run for %d iterations" % self.numIterations
		self.sample()
		stopTime = time.time()
		elapsedTime = (stopTime - startTime) / 1000
		print "Time taken %lf" % elapsedTime
		Util.printGaussians(self.tableMeans,self.tableCholeskyLTriangularMat,self.K,dirName)
		Util.printDocumentTopicDistribution(self.tableCountsPerDoc,self.N,self.K,dirName,self.alpha)
		Util.printTableAssignments(self.tableAssignments,dirName)
		Util.printNumCustomersPerTopic(self.tableCountsPerDoc,dirName,self.K,self.N)
		print "Done"

	def run(self):
		temp = VoseAlias()
		temp.init(K)

		while not done:
			for w in range(Data.numVectors):
				Max = float("-inf")
				for k in range(K):
					logLikelihood = self.logMultivariateTDensity(self.dataVectors[w],k)
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
				self.q[w].copy(temp)

	def initRun():
		temp = VoseAlias()
		temp.init(K)
		for w in range(Data.numVectors):
			Max = float("-inf")
			for k in range(K):
				logLikelihood = self.logMultivariateTDensity(self.dataVectors[w],k)
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
			self.q[w].copy(temp)

model = Gaussian_LDA()
model.main()
#!/usr/bin/python2.7  
# -*- coding: utf-8 -*-
from Data import Data
import numpy as np
import math
import random
import sys

def cholRank1Update(L,X):
	for k in range(Data.D):
		r = math.sqrt(math.pow(L[k][k],2) + math.pow(X[k][0],2))
		c = r / L[k][k]
		s = X[k][0] / L[k][k]
		L[k][k] = r
		for l in range(k + 1,Data.D):
			val = (L[l][k] + s * X[l][0]) / c
			L[l][k] = val
			val = c * X[l][0] - s * val
			X[l][0] = val
	# return L,X

def cholRank1Downdate(L,X):
	for k in range(Data.D):
		r = math.sqrt(L[k][k] * L[k][k]) - X[k][0] * X[k][0]
		c = r / L[k][k]
		s = X[k][0] / L[k][k]
		L[k][k] = r
		for l in range(k + 1,Data.D):
			val = (L[l][k] - s * X[l][0]) / c
			L[l][k] = val
			val = c * X[l][0] - s * L[l][k]
			X[l][0] = val

	# return L,X

def sample(probs):
	cumulative_probs = [0.0] * len(probs)
	sum_probs = 0.0
	counter = 0
	for prob in probs:
		sum_probs += prob
		cumulative_probs[counter] = sum_probs
		counter += 1
	if sum_probs != 1:
		for i in range(len(probs)):
			probs[i] = probs[i] / sum_probs
			cumulative_probs[i] = cumulative_probs[i] / sum_probs
	r = random.random()
	res = binSearch(cumulative_probs,r,0,len(cumulative_probs) - 1)
	return res

def binSearch(cumProb,key,start,end):
	if start > end:
		return start
	mid = (start + end) / 2
	if key == cumProb[mid]:
		return mid + 1
	elif key < cumProb[mid]:
		return binSearch(cumProb,key,start,mid - 1)
	else:
		return binSearch(cumProb,key,mid + 1,end)
	return -1

def getSampleMean(data):
	mean = np.zeros((Data.D,1))
	for vec in data:
		mean = mean + vec
	mean /= len(data)

	return mean;

def getSampleCovariance(data,mean):
	sampleCovariance = np.zeors((Data.D,Data.D))
	for i in range(Data.numVectors):
		x_minus_x_bar = np.zeros((Data.D,1))
		x_minus_x_bar = data[i] + x_minus_x_bar
		x_minus_x_bar = x_minus_x_bar - mean
		x_minus_x_bar_T = x_minus_x_bar.T
		mul = np.dot(x_minus_x_bar,x_minus_x_bar_T)
		sampleCovariance += mul
		sampleCovariance /= Data.numVectors - 1
	return sampleCovariance


def printGaussians(tableMeans,tableCholeskyLTriangularMat,K,dirName):
	for i in range(K):
		fout = open(dirName + str(i) + ".txt","w")
		for l in range(tableMeans[i].shape[0]):
			fout.write(str(tableMeans[i][l][0]) + " ")
		fout.write("\n")
		chol = tableCholeskyLTriangularMat[i]
		for r in range(chol.shape[0]):
			for c in range(chol.shape[1]):
				fout.write(str(chol[r][c]) + " ")
			fout.write("\n")
		fout.close()

def printNumCustomersPerTopic(tableCountsPerDoc,dirName,K,N):
	fout = open(dirName + "topic_counts.txt","w")
	for k in range(K):
		n_k = 0
		for n in range(N):
			n_k += tableCountsPerDoc[k][n]
		fout.write(str(n_k) + "\n")
	fout.close()

def printDocumentTopicDistribution(tableCountsPerDoc,numDocs,K,dirName,alpha):
	fout = open(dirName + "document_topic.txt","w")
	for i in range(numDocs):
		Sum = 0.0
		for k in range(K):
			Sum += tableCountsPerDoc[k][i]
		for k in range(K):
			temp = (tableCountsPerDoc[k][i] + alpha) / (Sum + K * alpha)
			fout.write(str(temp) + " ")
		fout.write("\n")
	fout.close()

def printTableAssignments(tableAssignments,dirName):
	fout = open(dirName + "tableAssignments.txt","w")
	for i in range(len(tableAssignments)):
		eachDoc = tableAssignments[i]
		for assignment in eachDoc:
			fout.write(str(assignment) + " ")
		fout.write("\n")
	fout.close()

def calculateAvgLL(corpus,tableAssignments,dataVectors,tableMeans,tableCholeskyLTriangularMat,K,N,prior,tableCountsPerDoc):
	N_k = [0] * K
	for k in range(K):
		n_k = 0
		for n in range(N):
			n_k += tableCountsPerDoc[k][n]
		N_k[k] = n_k
		if n_k == 0:
			print "table %d is empty.....exiting"
			sys.exit(1)
	scalar = [0.0] * K
	for k in range(K):
		scalar[k] = prior.nu_0 + N_k[k] - Data.D

	scaledCholeskies = []
	for k in range(K):
		scaledCholesky = tableCholeskyLTriangularMat[k] / math.sqrt(scalar[k])
		scaledCholeskies.append(scaledCholesky)
	
	logDeterminants = []
	for i in range(K):
		logDet = 0.0
		for l  in range(Data.D):
			logDet += math.log(scaledCholeskies[i][l][l])
		logDeterminants.append(logDet)

	docCounter = 0
	totalWordCounter = 0
	totalLogLL = 0.0
	for eachDoc in corpus:
		wordCounter = 0
		for word in eachDoc:
			x = dataVectors[word]
			tableId = tableAssignments[docCounter][wordCounter]
			x_minus_mu = x - tableMeans[tableId]
			lTriangularChol = scaledCholeskies[tableId]
			x_minus_mu = np.linalg.solve(lTriangularChol,x_minus_mu) #questioned
			x_minus_mu_T = x_minus_mu.T
			mul = np.dot(x_minus_mu_T,x_minus_mu)
			val = mul[0][0]
			logDensity = 0.5 * (val + Data.D * math.log(2 * math.pi)) + logDeterminants[tableId]
			totalLogLL -= logDensity
			wordCounter += 1
			totalWordCounter += 1
		docCounter += 1
	
	avgDensity = totalLogLL / totalWordCounter

	return avgDensity

def getCustomerIdWordMappings(vocabFile):
	Hmap = dict()
	fin = open(vocabFile,"r")
	counter = 0
	for line in fin:
		word = line.strip()
		Hmap[counter] = word
		counter += 1
	fin.close() 
	return Hmap
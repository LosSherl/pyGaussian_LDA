#!/usr/bin/python2.7  
# -*- coding: utf-8 -*-
import random
import Queue

class VoseAlias(object):
	def __init__(self):
		self.wsum = 0.0

	def init(self,num):
		self.n = num
		self.Alias = [0] * self.n
		self.Prob = [0.0] * self.n
		self.w = [0.0] * self.n
		self.p = [0.0] * self.n

	def copy(self,other):
		self.n = other.n;
		self.wsum = other.wsum;
		self.w = other.w
		self.Prob = other.Prob
		self.Alias = other.Alias

	def generateTable(self):

		Small = Queue.Queue()
		Large = Queue.Queue()

		for i in range(self.n):
			self.p[i] = (self.w[i] * n) / self.wsum

 		for i in range(self.n):
 			if p[i] < 1:
 				Small.put(i)
 			else:
 				Large.put(i)
		
		while not (Small.empty() or Large.empty()):
			l = Small.get()
			g = Large.get()
			self.Prob[l] = self.p[l]
			self.Alias[l] = g
			self.p[g] = (self.p[g] + self.p[l]) - 1
			if self.p[g] < 1:
				Small.put(g)
			else:
				Large.put(g)

		while not Large.empty():
			g = Large.get()
			self.Prob[g] = 1

		while not Small.empty():
			l = Small.get()
			self.Prob[l] = 1

	def sampleVose():
		fair_die = int(self.n * random.random())
		m = random.random()
		res = fair_die
		if m > self.Prob[fair_die]:
			res = self.Alias[fair_die]
		return res
	
#!/usr/bin/python2.7  
# -*- coding: utf-8 -*-

import numpy as np

class Data(object):
	inputFileName = ""
	D = 1
	nVectors = 0
	def __init__(self):
		pass
	
	def readData(self):
		fin = open(self.inputFileName,"r")
		lines = []
		for line in fin:
			lines.append(line)
		fin.close()
		self.nVectors = len(lines)
		data = np.zeros((self.nVectors,self.D))
		for i in range(nVectors):
			vals = lines[i].split()
			for j in range(self.D):
				data[i][j] = float(vals[j])

		return data

	def readCorpus(self,inputCorpusName):
		corpus = []
		fin = open(inputCorpusName,"r")
		for line in fin:
			doc = []
			words = line.split()
			for word in words:
				doc.append(int(word))
			corpus.append(doc)
		fin.close()
		return corpus

	# def readClusterPrintAsHtml(self):
	# 	wordLangMap = dict()
	# 	wordLangFile = "data/test/all/all_word_langid_map_max.txt"
	# 	clusterFileName = "./last_iteration_table_members.txt"
	# 	outputFileName = "./last_iteration_table_members.html"
	# 	fout = open(outputFileName,"w")
	# 	fout.write("<html>\n")
	# 	fout.write("<meta http-equiv=\"Content-Type\" content=\"text/html; charset=UTF-8\" />\n")
	# 	fout.write("<head>\n")
	# 	fout.write("<link rel=\"stylesheet\" href=\"https://code.jquery.com/ui/1.11.1/themes/smoothness/jquery-ui.css\">\n")
	# 	fout.write("<script src=\"https://code.jquery.com/jquery-1.10.2.js\"></script>\n")
	# 	fout.write("<script src=\"https://code.jquery.com/ui/1.11.1/jquery-ui.js\"></script>\n")			
	# 	fout.write("<script>\n")
	# 	fout.write("$(function() {\n")
	# 	fout.write("$( \"#accordion\" ).accordion({collapsible: true,heightStyle:\"content\",active:false});\n")
	# 	fout.write("});</script>\n")
	# 	fout.write("<style type=\"text/css\"></style>\n")
	# 	fout.write("</head>\n")
	# 	fout.write("<body>\n")
	# 	fout.write("<div id=\"accordion\" class=\"ui-accordion ui-widget ui-helper-reset\",role=\"tablist\">\n")
	# 	fin = open(wordLangFile,"r")
	# 	for line in fin:
	# 		split = line.split()
	# 		word = split[0]
	# 		langsCounts = split[1]
	# 		split = langsCounts.split(":")
	# 		splits = langsCounts.split(":")
	# 		lang = splits[0]
	# 		count = int(splits[1])
	# 		wordLangMap[word] = lang
	# 	fin.close()
	# 	fin = open(clusterFileName,"r")
	# 	NumClusterGtThan10 = 0
	# 	languages = set()
	# 	for line in fin:
	# 		langCountMap = dict()
	# 		split = line.split(":")
	# 		clusterNum = int(split[0])
	# 		clusterWords = split[1].split()
	# 		count = 0
	# 		for m in range(1,len(clusterWords)):
	# 			word = clusterWords[m]
	# 			if wordLangMap.get(word):
	# 				lang = wordLangMap[word]
	# 				languages.add(lang)
	# 				if not langCountMap.get(lang):
	# 					langCountMap[lang] = 1.0
	# 				else:
	# 					langCountMap[lang] += 1.0
	# 				count += 1

	# 		List = sorted(langCountMap.items(),lambda x,y:cmp(x[1],y[1]),reverse = True)
	# 		Sum = 0
	# 		for item in List:
	# 			Sum += item[1]

	# 		for i in range(len(List)):
	# 			List[i] = (List[i][0],List[i][1] / Sum)

	# 		if count >= 10:
	# 			NumClusterGtThan10 += 1
	# 			fout.write("<h3>\n")
	# 			fout.write(str(NumClusterGtThan10) + ": Number of Words " + str(count) + "<br/>\n")
	# 			print str(NumClusterGtThan10) + ": Number of Words " + str(count)
	# 			sumProb = 0.0
	# 			for i in range(len(List)):
	# 				if sumProb < 0.95:
	# 					fout.write(str(List[i][0]) + " : %.2f% " % (List[i][1] * 100))
	# 					print str(List[i][0]) + " : " + str(List[i][1] * 100),
	# 				else:
	# 					left = 1 - sumProb
	# 					fout.write("Others : %.2f%\n" % (left * 100))
	# 					print " Others : %f" % left
	# 					break					
	# 				sumProb = sumProb + List[i][1]
	# 			fout.write("\n")
	# 			print ""
	# 			fout.write("</h3>\n")
	# 			fout.write("<div> <p>\n")
	# 			for m in range(1,len(clusterWords)):
	# 				word = clusterWords[m]
	# 				fout.write(word + " ")
	# 			fout.write("</p></div>\n")
	# 	fout.write("</div>\n")
	# 	fout.write("</div>\n")
	# 	fout.write("</body>\n")
	# 	fout.write("</html>\m")
	# 	fout.close()
	# 	fin.close()
	# 	print "Total number of languages are %d" % len(languages)
	# 	for language in languages:
	# 		print languages + " ",

	# def create20NewsCorpus(self,fileName,blackListFile):
	# 	blackList = set()
	# 	fin = open(blackListFile,"r")
	# 	for line in fin:
	# 		blackList.add(line)
	# 	fin.close()
	# 	fin = open(fileName,"r")
	# 	prevDocNum = "1"
	# 	corpus = []
	# 	doc = []
	# 	for line in fin:
	# 		split = line.split()
	# 		docNum = split[0]
	# 		if not docNum == prevDocNum:
	# 			corpus.append(doc)
	# 			doc = []
	# 			print "Finished document "+ prevDocNum
	# 		wordId = split[1]
	# 		count = int(split[2])
	# 		if not wordId in blackList:
	# 			for i in range(count):
	# 				doc.append(wordId)	
	# 		prevDocNum = docNum

	# 	for(ArrayList<String> eachDoc:corpus)
 #  			Collections.shuffle(eachDoc);
	# 	PrintStream out = new PrintStream("20_news/corpus.test","UTF-8");
	# 	for(ArrayList<String> eachDoc:corpus)
	# 	{
	# 		for(String word:eachDoc)
	# 			out.print(word+" ");
	# 		out.println();
	# 	}
	# 	out.close();
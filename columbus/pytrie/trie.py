import sys
import pdb
from trieNode import Node
from tagrepo import Tagrepo

class Trie():
	def __init__(self):
		self.root = Node()
		self.tagrepo = Tagrepo()
		
	def insert(self, word):
		parent = self.root
		tagfound = False

		childs = self.root.__get_childs__()
		for idx in xrange(len(word)):	
			currChar = word[idx]
			if currChar in childs:
				childs[currChar].__inc_frequency__()
				if childs[currChar].__is_tag__():
					tag = word[:idx+1]
					self.tagrepo.__inc_tag_frequency__(tag)

				parent = childs[currChar]
			else:
				tnode = Node()
				tnode.__create_node__(currChar, parent)
				parent.__add_child__(currChar, tnode)
				if parent.__get_frequency__() > tnode.__get_frequency__():
					tag = word[:idx]
					if len(tag) > 3:
						#pdb.set_trace()
						self.tagrepo.__insert_tag__(tag, parent.__get_frequency__())
						tagfound = True
						parent.__set_tag__()

				parent = tnode
			childs = parent.__get_childs__()
			if(idx == len(word)-1):
				parent.__set_leaf__()
				if len(word) >3 and parent.__get_frequency__() >= 1 and not tagfound:
					self.tagrepo.__insert_tag__(word, parent.__get_frequency__())
					parent.__set_tag__()


	def direct_tag_insert(self, word):
		self.tagrepo.__insert_tag__(word, 1)
				
	def print_frequency(self):	
		self.tagrepo.__print_all_tags__()		

	def get_topK(self, k, fthreshold):
		return self.tagrepo.__get_topK_tags__(k, fthreshold)

	def get_all_tags(self):
		return self.tagrepo.__get_all_tags__()


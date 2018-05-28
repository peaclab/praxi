import pdb
import sys

class Node():
	def __init__(self):
		self.char = None
		self.childs = {}
		self.isLeaf = False
		self.frequency = 0
		self.isTag = False;
	
	def __create_node__(self, char, parent):
		self.char = char
		self.parent = parent
		self.frequency = 1
		
	def __get_char__(self):
		return self.char

	def __add_child__(self, char, node):
		self.childs[char] = node

	def __get_childs__(self):
		return self.childs

	def __inc_frequency__(self):
		self.frequency+=1

	def __set_leaf__(self):
		self.isLeaf = True

	def __get_frequency__(self):
		return self.frequency

	def __set_tag__(self):
		self.isTag = True

	def __is_tag__(self):
		return self.isTag


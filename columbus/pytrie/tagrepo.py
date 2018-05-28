import sys
from collections import OrderedDict
import pdb

class Tagrepo():
	def __init__(self):
		self.tags = {}
		self.filterTags = ['lib', 'CA', 'update', 'font', 'ucf', 'var', 'man', 'ssl', 'info', 'cert', 'dpkg', 'locale','Locale', 'encoding', 'fixes',\
		'local', 'dist', 'Unicode', 'Collate', 'debconf', 'zone', 'zoneinfo', 'system', 'posix', 'encod', 'vendor', 'bundle', 'logs', 'unicore', 'gconv',\
		'copyright','auto', 'test', 'man1', 'man8', 'systemd','LC_M','right', 'yum', 'mirror' ]
		self.filterPrefixTags = ['local', 'dist','x86_64', 'CP', 'ISO', 'IBM', 'yum', 'encod', 'cert', 'ca-', 'changelog', 'LC_M', 'America', 'Asia', 'system', 'man', 'Europe', 'Africa', 'apt', 'dpkg']

		self.removedTags = []

	def __insert_tag__(self, tag, frequency):
		validTag = False
		#if tag.startswith('redis'):
		#	pdb.set_trace()
		for char in tag:
			if char.isalpha():
				validTag = True
				break
		
		if not validTag:
			return
		
		if tag in self.filterTags:
			return
		for filterPrefix in self.filterPrefixTags:
			if tag.startswith(filterPrefix):
				return

		#pdb.set_trace()

		for existing_tag in self.tags:
			if tag == existing_tag and self.tags[existing_tag] < frequency:
				self.tags[existing_tag] = frequency
				return

			#if tag.startswith(existing_tag):
			#	self.tags[existing_tag] = frequency
				return
			if len(existing_tag) > len(tag) and existing_tag.startswith(tag):
				self.tags.pop(existing_tag)
				self.tags[tag] = frequency
				return
			elif len(existing_tag) < len(tag) and tag.startswith(existing_tag):
				return

			'''
				#self.tags[existing_tag] = 0
				if existing_tag == 'cassandra':
					pdb.set_trace()
				self.tags.pop(existing_tag)
				self.removedTags.append(existing_tag)
				self.tags[tag] = frequency
				#print "removed %s"%(existing_tag)
				return
			elif tag == existing_tag and self.tags[existing_tag] < frequency:
				self.tags[existing_tag] = frequency
				return 
			'''
		self.tags[tag] = frequency
				
		'''
		if tag in self.tags:
			if self.tags[tag] < frequency:
				self.tags[tag] = frequency
		else:
			self.tags[tag] = frequency
		'''
	def __inc_tag_frequency__(self, tag):
		if tag in self.tags:
			self.tags[tag]+=1		

	def __print_all_tags__(self):
		sorted_dict =  OrderedDict(reversed(sorted(self.tags.items(), key=lambda x: x[1])))
		print sorted_dict

	def __get_all_tags__(self):
		sorted_dict =  OrderedDict(reversed(sorted(self.tags.items(), key=lambda x: x[1]))) 

		return sorted_dict

	def __get_topK_tags__(self, k, fthreshold):
		sorted_dict =  OrderedDict(reversed(sorted(self.tags.items(), key=lambda x: x[1])))	
		res = dict()
		for key in sorted_dict:
			if sorted_dict[key] < fthreshold:
				break
			res[key] = sorted_dict[key]
			k-=1
			if k == 0:
				break	
		return OrderedDict(reversed(sorted(res.items(), key=lambda x: x[1])))	
		

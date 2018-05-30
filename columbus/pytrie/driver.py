from .trie import *
import pdb


def main():
	t = Trie()
	fp = open('bnames/mysql-binames')
	bins = fp.readlines()
	for biname in bins:
		t.insert(biname.rstrip())

	print(t.get_topK(3))


if __name__=="__main__":
	main()

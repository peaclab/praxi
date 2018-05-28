from pytrie.tagrepo import Tagrepo
import pdb
from pytrie.trie import Trie

def main():
	trie = Trie()
	trie.insert('python2.7')
	print trie.get_topK(10, 2)
	trie.insert('python2.7')
	print trie.get_topK(10, 2)
	trie.insert('python2.7')
	print trie.get_topK(10, 2)

	trie.insert('python')
	print trie.get_topK(10, 2)
	trie.insert('python')
	print trie.get_topK(10, 2)
	trie.insert('python')


	print trie.get_topK(10, 2)

if __name__=="__main__":
	main()


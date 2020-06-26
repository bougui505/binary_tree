# Binary tree class in python
The tree as an array data structure as described here: https://en.wikipedia.org/w/index.php?title=Binary_tree&oldid=964115444#Arrays
```
$ ./bintree.py

Adding left child a to parent 0
Adding right child b to parent 0
Adding left child c to parent a
Adding left child d to parent b
Adding right child e to parent b
Adding left child f to parent e
Adding right child g to parent e
       0        
     a    b     
   c  -  d  e   
- - - - - - f g 
Tree array: [0, 'a', 'b', 'c', None, 'd', 'e', None, None, None, None, None, None, 'f', 'g']
Tree depth: 3
Tree leaves: ['c', 'd', 'f', 'g']
Get nodes at depth 2: ['c', 'd', 'e']
Parent of nodes 'd' and 'e': ('b', 'b')
Leaves from node 'b': ['d', 'f', 'g']
Offspring from node 'b': ['d', 'e', 'f', 'g']
Offspring indices from node 'b': [5, 6, 13, 14]
================================================================================
Swapping branches a and b
       0        
     b    a     
   d  e  c  -   
- - f g - - - - 
```

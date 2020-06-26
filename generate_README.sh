#!/usr/bin/env zsh
# -*- coding: UTF8 -*-

# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr
# https://research.pasteur.fr/en/member/guillaume-bouvier/
# 2020-05-29 23:23:44 (UTC+0200)

func runcmd () {
    OUTPUT=$(eval $1)
    echo "\`\`\`"
    echo "$ $1\n"
    echo "$OUTPUT"
    echo "\`\`\`"
}

cat << EOF
# Binary tree class in python
The tree as an array data structure as described here: https://en.wikipedia.org/w/index.php?title=Binary_tree&oldid=964115444#Arrays
EOF

runcmd "./bintree.py"

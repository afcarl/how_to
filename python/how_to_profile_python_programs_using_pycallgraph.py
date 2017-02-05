# How to profile python programs using pycallgraph
# http://pycallgraph.readthedocs.io/en/latest/guide/filtering.html
# 
# Installation:
#   $ pip install pycallgraph
#
# An example:

import time


class Banana:

    def __init__(self):
        pass

    def eat(self):
        self.secret_function()
        self.chew()
        self.swallow()

    def secret_function(self):
        time.sleep(0.2)

    def chew(self):
        pass

    def swallow(self):
        pass



from pycallgraph import PyCallGraph
from pycallgraph.output import GraphvizOutput

# after executing the script, check 'python/profile.png'
graphviz = GraphvizOutput(output_file='python/profile.png')

with PyCallGraph(output=graphviz):
    banana = Banana()
    banana.eat()

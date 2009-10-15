import os
import pyximport; pyximport.install()
import callback

# from report import report_cheese

def report_cheese(name):
     print("Found cheese: " + name)

callback.find(report_cheese)


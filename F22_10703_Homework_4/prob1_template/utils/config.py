####################
###   Logging    ###
####################
import logging
import os
if not os.path.exists("./out"): os.mkdir("./out")
log_path = "out"
log_file = "log.txt"

class MyHandler(logging.StreamHandler):

	def __init__(self):

		logging.StreamHandler.__init__(self)

		# Formatting
		formatter = logging.Formatter(
			fmt="%(asctime)s %(filename)-10s %(levelname)-8s: %(message)s", 
			datefmt="%Y-%m-%d %T")
		self.setFormatter(formatter)

		# Create another handler to log.txt, in addtion to stdout
		fileHandler = logging.FileHandler("{0}/{1}".format(log_path, log_file))
		fileHandler.setFormatter(formatter)

		# Add this handler
		rootLogger = logging.getLogger()
		rootLogger.addHandler(fileHandler)


####################
###     Timer    ###
####################
import datetime
class Now():
    def __init__(self):
        pass
    def get(self):
        return datetime.datetime.now().strftime("%H:%M:%S")


####################
###   Plotting   ###
####################
import matplotlib.pyplot as plt
import matplotlib
font = {"family" : "normal",
        "weight" : "bold",
        "size"	 : 14
        }
savefig = { "dpi" : 300}
matplotlib.rc("font", **font)
matplotlib.rc("savefig", **savefig)


####################
###     MISC     ###
####################
# Mute tensorflow deprecation warnings
import tensorflow as tf
# if type(tf.contrib) != type(tf): tf.contrib._warning = None
import matplotlib
matplotlib.use('TkAgg') # Manually define a different backend to use
import matplotlib.pyplot as plt
import numpy as np

# create 1000 populations of greyhounds and labradors
# with ration 50:50
greyhounds = 500
labradors = 500

grey_height = 28 + 4 * np.random.randn(greyhounds)
lab_height = 24 + 4 * np.random.randn(labradors)


# Greyounds - red, labradors - blue
plt.hist([grey_height, lab_height], stacked=True, color=['r', 'b'])
plt.show()

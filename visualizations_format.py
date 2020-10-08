import matplotlib.pyplot as plt
import numpy as np

"""
Ways to define colors
"""
samples = range(1,4)

for i, col in zip(samples, [(0.0, 0.0, 1.0), 'blue', '#0000FF']):
    plt.plot([0, 10], [0, i], lw=3, color=col)

plt.legend(['RGB values: (0.0, 0.0, 1.0)',
            "matplotlib names: 'blue'",
            "HTML hex values: '#0000FF'"],
           loc='upper left')
plt.title('3 alternatives to define the color blue')

plt.show()

"""
MY Way
"""

fig,ax = plt.subplots(figsize=(12,9))
fig.patch.set_facecolor('xkcd:gray')
ax.set_facecolor((0.,0.,0.))

plt.plot([0,10], [0,1], lw=3, color=(1.0,0.0,0.0))
plt.plot([0,10], [0,2], lw=3, color='red')
plt.plot([0,10], [0,3], lw=3, color='#f54842')
plt.legend(['RGB values: (1.0,0.0,0.0)',
            "matplotlib names: 'red'",
            "HTML hex values: '#f54842'"],
           loc='upper left')
plt.title('3 ways to define colors', fontsize=13)
plt.savefig("nine_plot.png")
plt.show()

"""
color names
"""
cols = ['blue', 'green', 'red', 'cyan',  'magenta', 'yellow', 'black', 'white']

samples = range(1, len(cols)+1)
plt.figure(figsize=(10,8))
fig.patch.set_facecolor('xkcd:gray')

for i, col in zip(samples, cols):
    plt.plot([0, 10], [0, i], label=col, lw=3, color=col)

plt.legend(loc='upper left')
plt.title('Matplotlib Sample Colors', fontsize=13)
plt.savefig('ten_plot.png')
plt.show()

"""
gray levels
"""
plt.figure(figsize=(10,8))

samples = np.arange(0, 1.1, 0.1)

for i in samples:
    plt.plot([0, 10], [0, i], label='gray-level %s'%i, lw=3,
             color=str(i)) # ! gray level has to be parsed as string

plt.legend(loc='upper left')
plt.title('Gray Gradients with RGBA Formatting', fontsize=13)
plt.savefig('eleven_plot.png')
plt.show()

"""
Color gradients
"""
import matplotlib.pyplot as plt
import matplotlib.colors as col
import matplotlib.cm as cm
import numpy as np

# input data
mean_values = np.random.randint(1, 101, 100)
x_pos = range(len(mean_values))

fig = plt.figure(figsize=(20,5))

# create colormap
cmap = cm.ScalarMappable(col.Normalize(min(mean_values),
                                       max(mean_values),
                                       cm.hot))

# plot bars
plt.subplot(131)
plt.bar(x_pos, mean_values, align='center', alpha=0.5,
        color=cmap.to_rgba(mean_values))
plt.ylim(0, max(mean_values) * 1.1)

plt.subplot(132)
plt.bar(x_pos, np.sort(mean_values), align='center', alpha=0.5,
        color=cmap.to_rgba(mean_values))
plt.ylim(0, max(mean_values) * 1.1)

plt.subplot(133)
plt.bar(x_pos, np.sort(mean_values), align='center', alpha=0.5,
        color=cmap.to_rgba(np.sort(mean_values)))
plt.ylim(0, max(mean_values) * 1.1)

plt.show()

"""
Marker styles
"""
marker_name = ['point', 'pixel', 'circle', 'triangle down', 'triangle up', 'triangle_left', 'triangle_right',
               'tri_down', 'tri_up', 'tri_left', 'tri_right', 'octagon', 'square', 'pentagon', 'star', 'hexagon1',
               'hexagon2', 'plus', 'x', 'diamond', 'thin_diamond', 'vline']

markers = [

'.', # point
',', # pixel
'o', # circle
'v', # triangle down
'^', # triangle up
'<', # triangle_left
'>', # triangle_right
'1', # tri_down
'2', # tri_up
'3', # tri_left
'4', # tri_right
'8', # octagon
's', # square
'p', # pentagon
'*', # star
'h', # hexagon1
'H', # hexagon2
'+', # plus
'x', # x
'D', # diamond
'd', # thin_diamond
'|', # vline

]
samples = range(len(markers))

plt.figure(figsize=(13, 10))
for i in samples:
    plt.plot([i-1, i, i+1], [i, i, i], label=marker_name[i], marker=markers[i], markersize=11)

# Annotation
plt.title('Matplotlib Marker styles', fontsize=20)
plt.ylim([-1, len(markers)+1])
plt.legend(loc='lower right')
plt.savefig('twelve_plot.png')
plt.show()

"""
line style
"""
linestyles = ['-.', '--', 'None', '-', ':']

plt.figure(figsize=(8, 5))
samples = range(len(linestyles))


for i in samples:
    plt.plot([i-1, i, i+1], [i, i, i],
             label='"%s"' %linestyles[i],
             linestyle=linestyles[i],
             lw=4
             )

# Annotation

plt.title('Matplotlib line styles', fontsize=20)
plt.ylim([-1, len(linestyles)+1])
plt.legend(loc='lower right')


plt.show()

"""
removing frames
"""
x = range(10)
y = range(10)

fig = plt.gca()

plt.plot(x, y)

# removing frame
fig.spines["top"].set_visible(False)
fig.spines["bottom"].set_visible(False)
fig.spines["right"].set_visible(False)
fig.spines["left"].set_visible(False)

# removing ticks
plt.tick_params(axis="both", which="both", bottom="off", top="off",
                labelbottom="on", left="off", right="off", labelleft="on")

plt.show()

"""
custom tick labels
"""
x = range(10)
y = range(10)
labels = ['super long axis label' for i in range(10)]

fig, ax = plt.subplots()

plt.plot(x, y)

# set custom tick labels
ax.set_xticklabels(labels, rotation=45, horizontalalignment='right')

plt.show()

"""
grid styling
"""
import numpy as np
import random, math
from matplotlib import pyplot as plt

data = np.random.normal(0, 20, 1000)
bins = np.arange(-100, 100, 5) # fixed bin size
# horizontal grid
plt.hist(data, bins=bins, alpha=0.5)
ax = plt.gca()
ax.yaxis.grid(True)

plt.show()

# vertical grid
plt.hist(data, bins=bins, alpha=0.5)
ax = plt.gca()
ax.xaxis.grid(True)

plt.show()

# linestyle
from matplotlib import rcParams

rcParams['grid.linestyle'] = '-'
rcParams['grid.color'] = 'blue'
rcParams['grid.linewidth'] = 0.2

plt.grid()
plt.hist(data, bins=bins, alpha=0.5)

plt.show()

"""
outside of the box labels
"""
# above
fig = plt.figure()
ax = plt.subplot(111)

x = np.arange(10)

for i in range(1, 4):
    ax.plot(x, i * x**2, label='Group %d' % i)

ax.legend(loc='upper center',
          bbox_to_anchor=(0.5,  # horizontal
                          1.15),# vertical
          ncol=3, fancybox=True)
plt.show()

# right
fig = plt.figure()
ax = plt.subplot(111)

x = np.arange(10)

for i in range(1, 4):
    ax.plot(x, i * x**2, label='Group %d' % i)

ax.legend(loc='upper center',
          bbox_to_anchor=(1.15, 1.02),
          ncol=1, fancybox=True)
plt.show()

# transparent
x = np.arange(10)

for i in range(1, 4):
    plt.plot(x, i * x**2, label='Group %d' % i)

plt.legend(loc='upper right', framealpha=0.1)
plt.show()

"""
style sheets
"""
print(plt.style.available)

# 1st way: set the style for our coding environment globally via the plt.style.use function
plt.style.use('ggplot')

x = np.arange(10)
plt.figure(figsize=(10,8))
for i in range(1, 4):
    plt.plot(x, i * x**2, label='Group %d' % i)
plt.legend(loc='best')
plt.title('Style sheet formatting', fontsize=13)
plt.savefig('13_plot.png')
plt.show()

# 2nd way: via the with context manager, which applies the styling to a specific code block only
with plt.style.context('fivethirtyeight'):
    plt.figure(figsize=(10,8))
    for i in range(1, 4):
        plt.plot(x, i * x**2, label='Group %d' % i)
    plt.legend(loc='best')
    plt.title('Style sheet formatting', fontsize=13)
    plt.savefig('14_plot.png')
    plt.show()

# All styles
import math

n = len(plt.style.available)
num_rows = math.ceil(n/4)

fig = plt.figure(figsize=(15, 15))

for i, s in enumerate(plt.style.available):
    with plt.style.context(s):
        ax = fig.add_subplot(num_rows, 4, i+1)
        for i in range(1, 4):
            ax.plot(x, i * x**2, label='Group %d' % i)
            ax.set_xlabel(s, color='black')
            ax.legend(loc='best')

fig.tight_layout()
plt.show()
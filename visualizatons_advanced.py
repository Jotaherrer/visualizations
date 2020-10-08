from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import pyplot as plt

"""
3D PLOT WITH SAMPLE DATA
"""
mu_vec1 = np.array([0,0,0]) # mean vector
cov_mat1 = np.array([[1,0,0],[0,1,0],[0,0,1]]) # covariance matrix

class1_sample = np.random.multivariate_normal(mu_vec1, cov_mat1, 20)
class2_sample = np.random.multivariate_normal(mu_vec1 + 1, cov_mat1, 20)
class3_sample = np.random.multivariate_normal(mu_vec1 + 2, cov_mat1, 20)

# class1_sample.shape -> (20, 3), 20 rows, 3 columns

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(class1_sample[:,0], class1_sample[:,1], class1_sample[:,2],
           marker='x', color='blue', s=40, label='class 1')
ax.scatter(class2_sample[:,0], class2_sample[:,1], class2_sample[:,2],
           marker='o', color='green', s=40, label='class 2')
ax.scatter(class3_sample[:,0], class3_sample[:,1], class3_sample[:,2],
           marker='^', color='red', s=40, label='class 3')

ax.set_xlabel('variable X')
ax.set_ylabel('variable Y')
ax.set_zlabel('variable Z')

plt.title('3D Scatter Plot')

plt.show()

"""
BAR PLOTS
1. Horizontal bar plot with error bars
"""
# input data
mean_values = [1, 2, 3]
std_dev = [0.2, 0.3, 0.4]
bar_labels = ['Bar 1', 'Bar 2', 'Bar 3']

fig = plt.figure(figsize=(12,8))

# plot bars
y_pos = np.arange(len(mean_values))
y_pos = [x for x in y_pos]
plt.yticks(y_pos, bar_labels, fontsize=13)
plt.barh(y_pos, mean_values, xerr=std_dev,
         align='center', alpha=0.5, color='red')

# annotation and labels
plt.title('Horizontal Bar plot with error', fontsize=13)
#plt.ylim([-1,len(mean_values)+0.5])
plt.xlim([0, 3.5])
plt.grid()
plt.savefig('first_plot.png')
plt.show()

"""
2. Back-to-back bar plot
"""
# input data
X1 = np.array([1, 2, 3])
X2 = np.array([3, 2, 1])

bar_labels = ['Bar 1', 'Bar 2', 'Bar 3']

fig = plt.figure(figsize=(12,8))

# plot bars
y_pos = np.arange(len(X1))
y_pos = [x for x in y_pos]
plt.yticks(y_pos, bar_labels, fontsize=13)

plt.barh(y_pos, X1,
         align='center', alpha=0.5, color='blue')

# we simply negate the values of the numpy array for
# the second bar:
plt.barh(y_pos, -X2,
         align='center', alpha=0.5, color='purple')

# annotation and labels
plt.title('Back-to-back Bar Plot', fontsize=13)
plt.ylim([-1,len(X1)+0.1])
#plt.xlim([-max(X2)-1, max(X1)+1])
plt.grid()
plt.savefig('second_plot.png')
plt.show()

"""
3. Grouped bar plot
"""
# Input data
green_data = [1, 2, 3]
blue_data = [3, 2, 1]
red_data = [2, 3, 3]
labels = ['group 1', 'group 2', 'group 3']

# Setting the positions and width for the bars
pos = list(range(len(green_data)))
width = 0.2

# Plotting the bars
fig, ax = plt.subplots(figsize=(8,6))

plt.bar(pos, green_data, width,
                 alpha=0.5,
                 color='g',
                 label=labels[0])

plt.bar([p + width for p in pos], blue_data, width,
                 alpha=0.5,
                 color='b',
                 label=labels[1])

plt.bar([p + width*2 for p in pos], red_data, width,
                 alpha=0.5,
                 color='r',
                 label=labels[2])

# Setting axis labels and ticks
ax.set_ylabel('y-value')
ax.set_title('Grouped bar plot')
ax.set_xticks([p + 1.5 * width for p in pos])
ax.set_xticklabels(labels)

# Setting the x-axis and y-axis limits
plt.xlim(min(pos)-width, max(pos)+width*4)
plt.ylim([0, max(green_data + blue_data + red_data) * 1.5])

# Adding the legend and showing the plot
plt.legend(['green', 'blue', 'red'], loc='upper left')
plt.grid()
plt.show()

"""
4. Bar plot with plot labels/text 1
"""
data = range(200, 225, 5)

bar_labels = ['Bar 1', 'Bar 2', 'Bar 3', 'Bar 4', 'Bar 5']

fig = plt.figure(figsize=(12,8))
# plot bars
y_pos = np.arange(len(data))
plt.yticks(y_pos, bar_labels, fontsize=15)
bars = plt.barh(y_pos, data,
         align='center', alpha=0.5, color='orange',edgecolor='red')
# annotation and labels
for b,d in zip(bars, data):
    plt.text(b.get_width() + b.get_width()*0.08, b.get_y() + b.get_height()/2,
        '{0:.2%}'.format(d/min(data)),
        ha='center', va='bottom', fontsize=12)

plt.title('Horizontal bar plot with labels', fontsize=15)
plt.ylim([-1,len(data)+0.5])
plt.xlim((100,240))
plt.vlines(min(data), -1, len(data)+0.5, linestyles='dashed')
plt.savefig('three_plot.png')
plt.show()

"""
5. Bar plot with plot labels/text 2
"""
idx = range(4)
values = [3000, 5000, 12000, 20000]
labels = ['Group 1', 'Group 2',
          'Group 3', 'Group 4']

fig, ax = plt.subplots(figsize=(12,8))
ax.set_facecolor('xkcd:gray')
fig.patch.set_facecolor('xkcd:gray')

# Automatically align and rotate tick labels:
fig.autofmt_xdate()

bars = plt.bar(idx, values, align='center', color='peru', edgecolor='blue')
plt.xticks(idx, labels, fontsize=13)

# Add text labels to the top of the bars
def autolabel(bars):
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., 1.05 * height,
                '%d' % int(height),
                ha='center', va='bottom', fontsize=13)

autolabel(bars)
plt.ylim([0, 25000])
plt.title('Bar plot with Height Labels', fontsize=14)
plt.tight_layout()
plt.savefig('four_plot.png')
plt.show()

"""
6. Bar plot with color gradients
"""
import matplotlib.pyplot as plt
import matplotlib.colors as col
import matplotlib.cm as cm

# input data
mean_values = range(10,18)
x_pos = range(len(mean_values))


# create colormap
cmap1 = cm.ScalarMappable(col.Normalize(min(mean_values), max(mean_values), cm.spring))
cmap2 = cm.ScalarMappable(col.Normalize(0, 20, cm.spring))

# plot bars
fig, ax = plt.subplots(figsize=(12,8))
plt.subplot(121)
plt.bar(x_pos, mean_values, align='center', alpha=0.5, color=cmap1.to_rgba(mean_values))
plt.ylim(0, max(mean_values) * 1.1)

plt.subplot(122)
plt.bar(x_pos, mean_values, align='center', alpha=0.5, color=cmap2.to_rgba(mean_values))
plt.ylim(0, max(mean_values) * 1.1)
plt.savefig('five_plot.png')
plt.show()

"""
7. Bar plot pattern fill
"""
patterns = ('-', '+', 'x', '\\', '*', 'o', 'O', '.')

fig = plt.gca()

# input data
mean_values = range(1, len(patterns)+1)

# plot bars
x_pos = list(range(len(mean_values)))
bars = plt.bar(x_pos,
               mean_values,
               align='center',
               color='steelblue',
               )

# set patterns
for bar, pattern in zip(bars, patterns):
     bar.set_hatch(pattern)


# set axes labels and formatting
fig.axes.get_yaxis().set_visible(False)
plt.ylim([0, max(mean_values) * 1.1])
plt.xticks(x_pos, patterns)
plt.show()

"""
MY WAY
"""
fig, ax = plt.subplots(figsize=(12,8))
#ax.set_facecolor('xkcd:gray')
#fig.patch.set_facecolor('xkcd:gray')

patterns = ('-', '+', 'x', '\\', '*', 'o', 'O', '.')
mean_values = range(1, len(patterns)+1)
x_pos = list(range(len(mean_values)))
bars = plt.bar(x_pos,
               mean_values,
               align='center',
               color='salmon'
               )
for bar, pattern in zip(bars, patterns):
     bar.set_hatch(pattern)
plt.xticks(x_pos, patterns, fontsize=13)
plt.title('Bar plot with patterns')
plt.savefig('six_plot.png')
plt.show()


"""
HEATMAPS
1. SIMPLE HEATMAP
"""
mean = [0,0]
cov = [[0,1],[1,0]]
x, y = np.random.multivariate_normal(mean, cov, 10000).T

fig = plt.figure(figsize=(10,8))
plt.hist2d(x, y, bins=10)
plt.colorbar()
plt.grid(False)
plt.title("Heatmap representation", fontsize=13)
plt.show()

"""
"""
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(0)

# normal distribution center at x=0 and y=5
x = np.random.randn(100000)
y = np.random.randn(100000) + 5
print(y.mean())
print(x.mean())

plt.figure(figsize=(10,8))
plt.hist2d(x, y, bins=40)
plt.xlabel('X values - Centered at 0', fontsize=13)
plt.ylabel('Y values - Centered at 5', fontsize=13)
cbar = plt.colorbar()
cbar.ax.set_ylabel('Number of observations', fontsize=13)
plt.savefig('seven_plot.png')
plt.show()

"""
2. Color maps
"""
from math import ceil
import numpy as np

# Sample from a bivariate Gaussian distribution
mean = [0,0]
cov = [[0,1],[1,0]]
x, y = np.random.multivariate_normal(mean, cov, 10000).T


size = len(plt.cm.datad.keys())
all_maps = list(plt.cm.datad.keys())

fig, ax = plt.subplots(ceil(size/4), 4, figsize=(12,100))

counter = 0
for row in ax:
    for col in row:
        try:
            col.imshow(hist, cmap=all_maps[counter])
            col.set_title(all_maps[counter])
        except IndexError:
           break
        counter += 1

plt.tight_layout()
plt.show()

"""
PIE CHART
1. modified pie chart
"""
def piechart_modified():

plt.figure(figsize=(10,8))
plt.pie(
        (10,5),
        labels=('Blue','Orange'),
        shadow=True,
        colors=('steelblue', 'orange'),
        explode=(0,0.15), # space between slices
        startangle=90,    # rotate conter-clockwise by 90 degrees
        autopct='%1.1f%%',# display fraction as percentage
        )
plt.legend(fancybox=True, fontsize=13)
plt.axis('equal')     # plot pyplot as circle
plt.title('Shadowed Pie Chart',fontsize=15)
plt.savefig('eight_plot.png')
plt.tight_layout()
plt.show()

piechart_modified()


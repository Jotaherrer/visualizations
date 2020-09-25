import matplotlib.pyplot as plt
import seaborn as sns

"""
Initial examples
"""
# Plotting two types of lines in one graph

"""
Site traffic graph
"""
# Plotting months (x) and site visits (y)
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
x_values = list(range(len(months)))
visits_per_month = [9695, 7909, 10831, 12942, 12495, 16794, 14161, 12762, 12777, 12439, 10309, 8724]

# Plot
fig = plt.figure(figsize=(20,8))
ax1 = fig.add_subplot(1,2,1)
plt.plot(x_values, visits_per_month, color='purple',marker='o')
plt.title('Visits per month')
plt.ylabel("Visits")
ax1.set_xticks(x_values)
ax1.set_xticklabels(months)
ax1.set_yticks(list(range(7000,18000,1000)))
plt.show()

"""
Limes graph
"""
# Plotting months (x) and lime sales per month(y)
key_limes_per_month = [92.0, 109.0, 124.0, 70.0, 101.0, 79.0, 106.0, 101.0, 103.0, 90.0, 102.0, 106.0]
persian_limes_per_month = [67.0, 51.0, 57.0, 54.0, 83.0, 90.0, 52.0, 63.0, 51.0, 44.0, 64.0, 78.0]
blood_limes_per_month = [75.0, 75.0, 76.0, 71.0, 74.0, 77.0, 69.0, 80.0, 63.0, 69.0, 73.0, 82.0]

# Plot
ax2 = fig.add_subplot(1,2,2)
plt.plot(x_values,key_limes_per_month, color='green',label='Key Limes',marker='o')
plt.plot(x_values,persian_limes_per_month, color='purple', label='Persian Limes',marker='*')
plt.plot(x_values,blood_limes_per_month, color='gray', label='Blood Limes',marker='s')
plt.title('Lime Sales per Month')
plt.ylabel("Sales in Thousands USD")
plt.legend(loc='best')
ax2.set_xticks(x_values)
ax2.set_xticklabels(months)
plt.subplots_adjust(wspace=0.17)
plt.show()

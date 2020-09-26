import matplotlib.pyplot as plt
import seaborn as sns

"""
Initial examples
"""
# Plotting tea sales
drinks = ["Cappuccino", "Latte", "Chai", "Americano", "Mocha", "Espresso"]
sales =  [91, 76, 56, 66, 52, 27]
tea_df = pd.DataFrame(sales, index=drinks, columns=['Total Sales'])

fig = plt.figure(figsize=(14,8))
plt.bar(range(len(drinks)), sales, color='peru',edgecolor='blue')
ax = plt.subplot()
ax.set_xticks(range(6))
ax.set_xticklabels(drinks)
plt.title('Tea Sales Summary')
plt.xlabel('Tea Type')
plt.ylabel('Gross Sales')
plt.grid(axis='y')
plt.savefig('plot_first.png')
plt.show()

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

"""
Side-by-side chart / coffee sales
"""
drinks = ["cappuccino", "latte", "chai", "americano", "mocha", "espresso"]
months_sales = ['Jan','Mar','May','Jun','Aug','Oct', 'Dec']
sales1 = [95, 72, 53, 62, 51, 25]
sales2 = [62, 81, 34, 62, 35, 42]

fig = plt.figure(figsize=(12,8))

ax3 = plt.subplot()
ax3.set_xticks(range(1,12,2))
ax3.set_xticklabels(months_sales)

n = 1
t = 2
d = 6
w = 0.8
store1_x = [t*element + w*n for element
             in range(d)]
plt.bar(store1_x, sales1, color='gray')

n = 2
t = 2
d = 6
w = 0.8
store2_x = [t*element + w*n for element
             in range(d)]
plt.bar(store2_x, sales2, color='purple')

plt.title("Coffee Sales Comparison")
plt.xlabel("Types of coffees")
plt.ylabel("Pounds sold")
plt.legend(labels=drinks, loc='upper right')
plt.savefig('plot_six.png')
plt.show()

"""
STACKED BARS EXAMPLE / Sales distribution
"""
import numpy as np

product = ['Computer', 'Keyboard', 'Headset', 'Mouse', 'Monitor']
sales_c = np.random.randint(1000,3000,5)
sales_k = np.random.randint(1000,3000,5)
sales_h = np.random.randint(1000,3000,5)
sales_m = np.random.randint(1000,3000,5)
sales_o = np.random.randint(1000,3000,5)

k_bottom = np.add(sales_c, sales_k)
h_bottom = np.add(k_bottom, sales_h)
m_bottom = np.add(h_bottom, sales_m)

fig = plt.figure(figsize=(10,8))
ax5 = plt.subplot()

plt.bar(range(len(sales_c)),sales_c, color='#D50071', label=product[0])
plt.bar(range(len(sales_k)),sales_k, bottom=sales_c, color='#0040FF',label=product[1])
plt.bar(range(len(sales_h)),sales_h, bottom=k_bottom, color='#00CA70',label=product[2])
plt.bar(range(len(sales_m)),sales_m, bottom=h_bottom, color='#C14200',label=product[3])
plt.bar(range(len(sales_o)),sales_o, bottom=m_bottom, color='#F0C300',label=product[4])

ax5.set_xticks(range(5))
ax5.set_xticklabels(['Monday','Tuesday', 'Wednesday', 'Thursday', 'Friday'])
plt.legend(loc='best')
plt.title('Sales Distribution by Product')
plt.ylabel("Products Sold")
plt.savefig('plot_seven')
plt.show()

"""
PIE CHART / Regional sales
"""
region = ['LATAM', 'North America','Europe','Asia','Africa']
sales = [3500,5500,4800,4500,2500]
explode_values = [0.1,0.1,0.1,0.1,0.1]
colors = ['cyan', 'steelblue','orange','blue','yellowgreen']

fig = plt.figure(figsize=(10,8))
plt.pie(sales, labels=region,autopct='%d%%', colors=colors,explode=explode_values)
plt.axis('equal')
plt.title('Global Sales Distribution', fontsize='20')
plt.savefig('plot_eight.png')
plt.show()

"""
HISTOGRAM
"""
clients_scoring1 = [62.58, 67.63, 81.37, 52.53, 62.98, 72.15, 59.05, 73.85, 97.24, 76.81, 89.34, 74.44, 68.52, 85.13, 90.75, 70.29, 75.62, 85.38, 77.82, 98.31, 79.08, 61.72, 71.33, 80.77, 80.31, 78.16, 61.15, 64.99, 72.67, 78.94]
clients_scoring2 = [72.38, 71.28, 79.24, 83.86, 84.42, 79.38, 75.51, 76.63, 81.48,78.81,79.23,74.38,79.27,81.07,75.42,90.35,82.93,86.74,81.33,95.1,86.57,83.66,85.58,81.87,92.14,72.15,91.64,74.21,89.04,76.54,81.9,96.5,80.05,74.77,72.26,73.23,92.6,66.22,70.09,77.2]

fig = plt.figure(figsize=(10,8))
plt.hist(clients_scoring1,bins=12,linewidth=2, alpha=0.5, color='yellowgreen')
plt.hist(clients_scoring2,bins=12,linewidth=2, alpha=0.5, color='steelblue')
plt.legend(['Clients - Group 1', 'Clients - Group 2'])
plt.title('Credit Scoring by Group of Client', fontsize='15')
plt.xlabel('Percentage')
plt.ylabel('Frequency')
plt.savefig('plot_nine.png')
plt.show()
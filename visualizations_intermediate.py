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
STACKED BARS EXAMPLE / grades distribution
"""
unit_topics = ['Limits', 'Derivatives', 'Integrals', 'Diff Eq', 'Applications']

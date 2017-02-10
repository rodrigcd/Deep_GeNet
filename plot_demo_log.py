import numpy as np
import matplotlib.pyplot as plt

data = np.load('../deep-genet-log-single-point.pkl')
son_props = data['son_props']
generations = np.array(range(len(son_props)))+1
fitness = data['fitness']
best_indiv = fitness.max(axis=1)
mean_indiv = fitness.mean(axis=1)

fig, ax1 = plt.subplots()
ax1.plot(generations, son_props, 'b-')
ax1.set_xlabel('Generation')

ax1.set_ylabel('Offspring proportion', color='b')
for tl in ax1.get_yticklabels():
    tl.set_color('b')

ax2 = ax1.twinx()
ax2.plot(generations, best_indiv, 'r--')
ax2.set_ylabel('Fitness', color='r')
for tl in ax2.get_yticklabels():
    tl.set_color('r')


ax2.plot(generations, mean_indiv, 'k--')

plt.savefig('demo_no_autocopy.png')
#plt.show()

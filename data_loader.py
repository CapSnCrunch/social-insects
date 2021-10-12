import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

save_file = 'square_data.dat' # Specify which file we want to load

data = [] # Data saved as SHDs, Rw, Rb, I
with open(os.path.dirname(__file__) + '/data/' + save_file, 'rb') as f:
    while True:
        try:
            data.append(pickle.load(f))
        except EOFError:
            break

print(len(data))
SHDs, R_w, R_b, I = data

plot1 = plt.figure(2)
legend = []
for i in range(len(SHDs)):
    plt.plot(np.arange(len(SHDs[i]))[30:], SHDs[i][30:])
    #legend.append('SF:' + str(colonies[i].f))
plt.title('RID')
#plt.legend(legend)
plt.xlabel('t')
plt.ylabel('SHD')

plot2 = plt.figure(3)
for i in range(len(I)):
    plt.plot(I[i])
plt.title('Proportion of Informed Ants')
#plt.legend(legend)
plt.xlabel('t')
plt.ylabel('I(t)')

plot3 = plt.figure(4)
plt.fill_between(np.arange(len(I[0])), np.mean(np.vstack(I), axis = 0) - 2*np.std(np.vstack(I), axis = 0), np.mean(np.vstack(I), axis = 0) + 2*np.std(np.vstack(I), axis = 0), alpha = 0.2)
plt.plot(np.mean(np.vstack(I), axis = 0))
plt.title('Proportion of Informed Ants')
#plt.legend(legend)
plt.xlabel('t')
plt.ylabel('I(t)')

plot4 = plt.figure(6)
for i in range(len(R_w)):
    pass
    #plt.scatter(np.arange(len(R_w[i])), R_w[i], s = 2)
plt.plot(np.mean(np.vstack(R_w), axis = 0))
plt.title('Rw')
#plt.legend(legend)
plt.xlabel('t')
plt.ylabel('Contact Rate')

plot5 = plt.figure(7)
for i in range(len(R_b)):
    pass
    #plt.scatter(np.arange(len(R_b[i])), R_b[i], s = 2)
plt.plot(np.mean(np.vstack(R_b), axis = 0))
plt.title('Rb')
#plt.legend(legend)
plt.xlabel('t')
plt.ylabel('Contact Rate')

plt.show()
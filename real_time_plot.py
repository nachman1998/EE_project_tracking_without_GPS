import os
import msvcrt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from multiprocessing.connection import Listener
import threading


def collector():
    global conn,x,y,z,label
    cun = 0
    while True:
        msg = conn.recv()
        x.append(msg[0]), y.append(msg[1]), z.append(-1 * msg[2]),count.append(cun)
        label.append(msg[3])
        cun+=1
        print(msg)


def update(frame):
    global x, y, z, label
    colors = {'downstairs': 'red', 'upstairs': 'blue', 'no_movement': 'green', 'standing_still': 'orange',
              'forward ': 'purple', 'backward ': 'brown', 'right ': 'pink', 'left ': 'gray', 'siting': 'cyan'}
    markers = {'downstairs': 'o', 'upstairs': 's', 'no_movement': '^', 'standing_still': 'D', 'forward ': 'v',
               'backward ': 'p', 'right ': '*', 'left ': 'h', 'siting': 'X'}

    ax1.clear()  # Clear previous plot
    ax2.clear()  # Clear previous plot
    ax1.plot(y, x, color='black')
    ax2.plot(count, z, color='black')
    for i in range(len(y)):
        ax1.scatter(y[i], x[i], c=colors[label[i]], marker=markers[label[i]], s=100, alpha=0.7,edgecolors='black', linewidth=1.5, label=label[i])
        ax2.scatter(count[i], z[i], c=colors[label[i]], marker=markers[label[i]], s=100, alpha=0.7, edgecolors='black',linewidth=1.5, label=label[i])

    ax1.set_title("XY")
    ax1.set_xlabel("X(m)")
    ax1.set_ylabel("Y(m)")
    ax1.grid()
    ax1.axis("equal")
    ax2.set_title("Z")
    ax2.set_xlabel("step Count")
    ax2.set_ylabel("Z(m)")

    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    return ax1,ax2

x=[]
y=[]
z=[]
label=[]
count=[]
address = ('localhost', 6000)  # family is deduced to be 'AF_INET'
print("dd")
listener = Listener(address, authkey=b'secret password')
conn = listener.accept()
print('connection accepted from', listener.last_accepted)

t = threading.Thread(target=collector)
t.start()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(19.2, 10.8), gridspec_kw={'height_ratios': [3, 1]})

ax1.set_title("XY")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.axis("equal")
ax2.set_title("Z")
ax2.set_xlabel("step Count")
ax2.set_ylabel("Z")
ani = animation.FuncAnimation(fig, update, interval=1500)
plt.show()
t.join()
listener.close()
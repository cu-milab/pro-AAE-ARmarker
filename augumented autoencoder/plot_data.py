#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

font_path = '/usr/share/fonts/truetype/takao-gothic/TakaoPGothic.ttf'
font_prop = FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()



train = np.loadtxt("./aae_loss_val.txt")
#x = range(0, len(train[:,0]))
print(train.shape)

# draw data
t = len(train)
epoch = range(0,t)

plt.plot(epoch, train[:t], linewidth = 1.5)
#plt.plot(epoch, train[:t,1], linewidth = 1.5)
#plt.plot(epoch, train[:t,2], linewidth = 1.5)


plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0)
plt.grid()
plt.xlim(0,len(train[:t]))
#plt.ylim(0,1000)
plt.gca().yaxis.set_tick_params(which='both', direction='in',bottom=True, top=True, left=True, right=True)
plt.gca().xaxis.set_tick_params(which='both', direction='in',bottom=True, top=True, left=True, right=True)
plt.title("")
plt.ylabel("loss")
plt.xlabel("epoch")
#plt.show()
plt.savefig("loss_train.png")

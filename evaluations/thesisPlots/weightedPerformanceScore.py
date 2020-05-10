import re
import os
import numpy as np

os.environ['PATH'] += os.pathsep + "/usr/local/texlive/2019/bin/x86_64-linux/"

import matplotlib.pyplot as plt
plt.rc('text', usetex=True)

spacing = 0.1
w = 0.025
textheight = 0.018
textwidth = 0.005

bar_names = ["Split02", "Add01", "LinAdd01"]
bars = [
    [(0, 0.85, "CT"), (0.85, 1.0, "TT")],
    [(0, 0.606, "CS"), (0.606, 0.973, "ST"), (0.973, 1.0, "SS")],
    [(0, 0.260, "CC"), (0.260, 0.81, "CT"), (0.81, 0.857, "CS"), (0.857, 1.0, "TT")],
]

cs=["red", "green", "blue", "orange"]

plt.figure()
#plt.title(title)
for i, bar in enumerate(bars):
    b = [v[0] for v in bar]
    h = [v[1]-v[0] for v in bar]
    l = [v[2] for v in bar]
    print(b, h, l)
    #plt.bar([i*spacing]*len(b), bottom=b, height=h, width=spacing/2, color=cs[0:len(b)])

    c = i * spacing
    for bb, hh, ll in zip(b, h, l):
        plt.plot([c-w, c+w], [bb,bb], color="black", lw=0.5)
        plt.text(c - textwidth, bb+(hh/2) - textheight, ll)

    plt.plot([c - w, c + w], [1.0, 1.0], color="black", lw=0.5)

    plt.plot([c - w, c - w], [0.0, 1.0], color="black", lw=0.5)
    plt.plot([c + w, c + w], [0.0, 1.0], color="black", lw=0.5)

plt.xticks([spacing*i for i in range(len(bars))], bar_names)
plt.ylim(-0.1, 1.0)
plt.ylabel("$\\alpha$")
plt.tight_layout()
#plt.gcf().autofmt_xdate()

#ax.spines['top'].set_visible(False)
#ax.spines['right'].set_visible(False)
#ax.spines['bottom'].set_visible(False)
#ax.spines['left'].set_visible(False)
plt.box(False)
plt.show()

import matplotlib.pyplot as plt
import matplotlib.patches as patches





patterns = ['-', '+', 'x', 'o', 'O', '.', '*']  # more patterns
fig4 = plt.figure()
ax4 = fig4.add_subplot(111, aspect='equal')
for p in [
    patches.Rectangle(
        (0.05 + (i * 0.13), 0.1),
        0.1,
        0.6,
        hatch=patterns[i],
        fill=False
    ) for i in range(len(patterns))
]:
    ax4.add_patch(p)

plt.show()
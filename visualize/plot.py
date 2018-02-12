import matplotlib.pyplot as plt

file = open('/Users/eirikvikanes/Dropbox/Skolearbeid/Master/struc2vec/emb/bitcoin.emb', 'r')

nodes = []
x = []
y = []

for line in file:
    split = line.split()
    if len(split) > 2:
        nodes.append(split[0])
        x.append(float(split[1]))
        y.append(float(split[2]))

plt.scatter(x, y)

for label, x, y in zip(nodes, x, y):
    plt.annotate(label, xy=(x, y))
plt.show()


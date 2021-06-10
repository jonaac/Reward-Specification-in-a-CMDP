import matplotlib.pyplot as plt

f = open('results.txt')
lines = [l.rstrip() for l in f]
f.close()

avg_r = []

for line in lines:
	if 'Average Reward' in line:
		text, n = line.split(': ')
		n = float(n)
		avg_r.append(n)

plt.plot(avg_r)
plt.xlabel("Episode")
plt.ylabel("Avg. Epsiodic Reward")
plt.show()
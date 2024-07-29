import matplotlib.pyplot as plt



rdl = open("/home/fede/Desktop/Tirocinio/logs/horus.log").readlines()
m = 10
y = [[],[]]
k = []
p = []
i = 0
for line in rdl:
    if "Mean Batch Loss:" in line:
        ll = line.split(" ")
        n = float(ll[12])
        if n < 0.0017:
            y[i].append(n)
            if len(y[i]) % m == 0 and len(y[i]) != 0:
                k.append(sum(y[i][-m:])/m)
                p.append(min(y[i][-m:]))
    if "EPOCH: 2 of 12" in line:
        i += 1

x = [i for i in range(len(y[0]))]
x1 = [i*m for i in range(len(k))]
x2 = [len(y[0])+i for i in range(0,len(y[1]))]
#fig, axs = plt.subplots(1, 1, layout='constrained')


plt.plot(x,y[0],label="Loss 1 Epoch")
plt.plot(x1,k,label=f"Average {m} Loss")
plt.plot(x1,p,label=f"Min {m} Loss")
# for i in range(len(p)):
#     plt.annotate(f"{p[i]:.4}", (x1[i], p[i]))
plt.plot(x2,y[1],color="blue",label="Loss 2 Epoch")
# plt.axis("off")
plt.legend(loc="upper left")
plt.ylabel("Loss")
plt.show()
#Credit goes to https://github.com/Wanogon/COVID-19-Project
#Generates dilution effect by creating a function, Gamma, that takes in the batch size, the number of positves in the batch
#and returns the false negative rate.
#These results are saved in F_N_d.npy

d_cens = 35.6
pi = [0.32, 0.53, 0.15];
mu = [20.14, 29.35, 34.78];
sigma = [3.60, 2.96, 1.32];

num_iter = 100000
            
            
def FNR(x):
    return 1/(1+math.exp(-12.5*(x-35.8)))

def Generate_Vload(PI, LoD):
    rand = random.random()
    if rand < PI[0]/sum(PI):
        x = np.random.normal(mu[0], sigma[0]);
    elif rand >= PI[0]/sum(PI) and rand <= (PI[0] + PI[1])/sum(PI):
        x = np.random.normal(mu[1], sigma[1]);
    else:
        x = np.random.normal(mu[2], sigma[2]);
    return x

def Gamma_censored_new(n, d, d_cens=35.6):
    count1, count2 = 0, 0
    record1, record2 = [], []
    for i in range(num_iter):
        add1, add2 = 0, 0
        V = np.zeros(d)
        for j in range(d):
            V[j] = Generate_Vload(pi, d_cens)
        V_pool = -math.log(sum(2**(-V))/n,2)
        U1 = np.random.uniform();
        if U1 <= FNR(V_pool):
            add1 = 1
            count1 += add1
        else:
            U2 = np.random.uniform()
            if U2 <= FNR(V[0]):
                add2 = 1
                count2 += add2
        record1.append(add1)
        record2.append(add2)
    hw1 = 1.96*np.std(record1)/math.sqrt(num_iter)
    hw2 = 1.96*np.std(record2)/math.sqrt(num_iter)
    return count1/num_iter, count2/num_iter, hw1, hw2

       
G = np.zeros([100, 10])
G_CI = np.zeros([100,10,2])
E = np.zeros([100, 10])
E_CI = np.zeros([100,10,2])
for n in range(1,101):
    print('n = '+str(n))
    for d in range(1, min(n+1,11)):
        print('d =', d)
        g, e, h1, h2 = Gamma_censored_new(n,d)
        G[n-1][d-1], E[n-1][d-1] = g, e
        G_CI[n-1][d-1] = [g-h1, g+h1]
        E_CI[n-1][d-1] = [e-h2, e+h2]
   # plt.plot(range(d,101), g, linestyle = '-')
   # plt.xlabel('Pool size n')
   # plt.ylabel('False negative rate')
   # plt.title('False negative rate vs. Pool size n with '+str(d)+' positive sample')
   # plt.show()

np.save('F_N_d.npy', G)
np.save('Eta_N_d.npy', E)
np.save('F_N_d_CI.npy', G_CI)
np.save('Eta_N_d_CI.npy', E_CI)

l1 = plt.plot(range(1,101), G[0:][1], label = 'd=1', color='orange', linestyle='-')
l2 = plt.plot(range(2,101), G[1:][2], label = 'd=2', color='blue', linestyle='-')
l3 = plt.plot(range(3,101), G[2:][3], label = 'd=3', color='coral', linestyle='-')
l4 = plt.plot(range(4,101), G[3:][4], label = 'd=4', color='green', linestyle='-')
l5 = plt.plot(range(5,101), G[4:][5], label = 'd=5', color='gray', linestyle='-')
#l6 = plt.plot(range(6,101), G[5:][6], label = 'd=6', color='teal', linestyle='-')
#l7 = plt.plot(range(7,101), G[6:][7], label = 'd=7', color='red', linestyle='-')
#l8 = plt.plot(range(8,101), G[7:][8], label = 'd=8', color='black', linestyle='-')
#l9 = plt.plot(range(9,101), G[8:][9], label = 'd=9', color='brown', linestyle='-')
#l10 = plt.plot(range(10,101), G[9:100][10], label = 'd=10', color='khaki', linestyle='-')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
plt.xlabel('Pool size n')
plt.ylabel('False negative rate')
plt.title('False negative rate vs. Pool size for 1<=d<=5')
plt.show()

G_round, E_round = np.zeros([100,10]), np.zeros([100,10])

for i in range(100):
    for j in range(min(i+1, 10)):
        hw_G = 0.5*(G_CI[i][j][1]-G_CI[i][j][0])
        if hw_G != 0:
            G_round[i][j] = round(G[i][j], math.ceil(-math.log(hw_G,10)))
        hw_E = 0.5*(E_CI[i][j][1]-E_CI[i][j][0])
        if hw_E != 0:
            E_round[i][j] = round(E[i][j], math.ceil(-math.log(hw_E,10)))

np.save('F_N_d_round.npy', G_round)
np.save('Eta_N_d_round.npy', E_round)

plt.matshow(G, cmap = plt.get_cmap('OrRd'))
plt.xlabel('Pool size')
plt.ylabel('Number of positive samples')
plt.title('Visualization of Gamma matrix')
plt.show()

plt.matshow(E, cmap = plt.get_cmap('OrRd'))
plt.xlabel('Pool size')
plt.ylabel('Number of positive samples')
plt.title('Visualizaton of Eta matrix')
plt.show()

G_d30, G_d30_CI = np.zeros([1,30]), np.zeros([1,30,2])
for d in range(1,31):
    print('d=',d)
    g, e, h1, h2 = Gamma_censored_new(100,d)
    G_d30[0][d-1] = g
    G_d30_CI[0][d-1] = [g-h1, g+h1]

plt.plot(range(1,31), G_d30[0], linestyle='-')
plt.xlabel('d')
plt.ylabel('False negative rate')
plt.title('False negative rate when n = 100')

    
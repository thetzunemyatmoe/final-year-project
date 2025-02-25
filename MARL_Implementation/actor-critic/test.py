t_max = 100
batch_size = 20
t = 0

for _ in range(t_max):
    t_start = t

    while not t - t_start == batch_size:
        print(t)
        t += 1

    print('------------')
    break

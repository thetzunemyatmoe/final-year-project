import random

for i in range(100):
    random.seed(i)
    print(random.randint(0, 100))

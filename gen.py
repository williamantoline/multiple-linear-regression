import random
import sys

n = int(sys.argv[1])

def rand():
    return str(random.randint(10,20))

with open("data.csv", "w") as f:
    f.write("y,x1,x2,x3\n")
    for i in range(n):
        a,b,c,d = rand(),rand(),rand(),rand()
        line = a + "," + b + "," + c + "," + d + "\n"
        f.write(line)
    f.close()



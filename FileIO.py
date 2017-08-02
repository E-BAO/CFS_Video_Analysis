filename = "replacement.txt"

a = [0,1,2]

with open(filename, "w") as f:
	for i in a:
		f.write(str(i)+"\n")

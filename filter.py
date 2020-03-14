username = input()
li = list(username.split(" "))
a=open('bad-words.csv').read()
list1 = list(a.split("\n"))
for x in li:
    if x in list1:
        flag = 1
        print(x)
        break
    else:
        flag = 0
if flag:
    print("True")
else:
    print("False")
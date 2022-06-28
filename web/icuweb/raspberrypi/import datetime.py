import datetime

i = datetime.datetime.now()
j = i.replace(microsecond=0)

print(i)
print(j)

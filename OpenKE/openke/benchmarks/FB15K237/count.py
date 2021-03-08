subject = set()
predicate = set()
object = set()

for line in open('Hash_Triple2HashValues_100.txt', 'r'):
    subject.add(line.split(',')[0])
    predicate.add(line.split(',')[1])
    object.add(line.split(',')[2])

print(len(subject))
print(len(predicate))
print(len(object))

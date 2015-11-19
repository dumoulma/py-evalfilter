from collections import deque

from fuzzywuzzy import fuzz

from datasets.fuman_base import load_fuman_rant

dataset = load_fuman_rant('data/20151023/bad-rants-4189.csv')
duplicates = set()
deduped = list()
n_elements = len(dataset.data)
rant_indexes = deque([i for i in range(n_elements)])
while rant_indexes:
    i = rant_indexes.popleft()
    end = min(i + 4, n_elements)
    window = [j for j in range(i + 1, end) if j not in duplicates and j in rant_indexes]
    r1 = dataset.data[i]
    dups = [j for j in window if fuzz.ratio(r1, dataset.data[j]) > 90]
    for j in dups:
        rant_indexes.remove(j)
    duplicates.update(dups)
    deduped.append(i)

rants = dataset.data
print('Found', len(duplicates), 'duplicates')
print('Deduped list has', len(deduped), 'elements')
assert len(rants) == len(deduped) + len(duplicates), "Missing rants!"

long_deduped = [rants[i].replace('\n', ' ') for i in deduped if len(rants[i]) > 50]
with open('data/output/bad-rants-deduped.csv', "wt") as fp:
    for rant in long_deduped:
        fp.write(rant + '\n')

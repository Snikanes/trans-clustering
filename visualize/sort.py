filename = "/Users/eirikvikanes/Developer/trans-clustering/graph/bitcoin.edgelist"
out = "/Users/eirikvikanes/Developer/trans-clustering/graph/bitcoin-undirected.edgelist"

def tuple_cmp(first, second):
    diff = first[0] - second[0]
    if diff == 0:
        return  first[1] - second[1]
    return diff

def read_trans(in_file):
    trans = []
    for line in in_file.readlines():
        trans.append(tuple(map(lambda x: int(x), line.split(' '))))
    return trans

def sort_transactions(trans):
    return sorted(trans, tuple_cmp)

def write_trans(out_file, trans): 
    for t in trans:        
        out_f.write(str(t[0]) + ' ' + str(t[1]) + '\n')

def remove_directionality(trans):
    # (not so) Quick and dirty
    saved = set()
    
    undirected_trans = []
    for t in trans:
        #print(saved)
        # Make sure a list exists for each key in map
        if (t[0], t[1]) not in saved:
            undirected_trans.append((t[0], t[1]))
            saved.add((t[1], t[0]))
    return undirected_trans

with open(filename, 'r') as f:
    transactions = read_trans(f)
    sorted_trans = sort_transactions(transactions)
    #print(transactions)
    undirected = remove_directionality(sorted_trans)

with open(out, 'w') as out_f:
    write_trans(out_f, undirected)

# trans = [(1,2), (1,3), (1,5), (2,1), (2,3), (3,1), (3,2), (3,2), (5,1)]
# print remove_directionality(trans)










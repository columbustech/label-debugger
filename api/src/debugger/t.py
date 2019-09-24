from collections import defaultdict

def merge(intervals):
        intervals.sort(key=lambda x: x[0])

        merged = []
        for interval in intervals:
            #print interval
            if not merged or (merged[-1][1] < interval[0]):
                merged.append(interval)
            else:
                merged[-1][1] = max(merged[-1][1], interval[1])

        return merged

def subsequence(array):
    
    count = defaultdict(list)
    intervals = []
    res = []
    
    for i,w in enumerate(array):
        if len(count[w]) > 0:
            count[w].insert(1,i)
        else:
             count[w].insert(0,i)



    for key,val in count.items():
        if len(val) ==1:
            val.append(val[0])
        intervals.append(val)

    print intervals
    merged_inter  =  merge(intervals)
    print merged_inter

    for item in merged_inter:
        s , e = item[0], item[1]
        res.append(e-s+1)

    print res

subsequence(['a','b','e','f','a','e','g','h'])
#subsequence(['a','b','a','b','c','b','a','c','a','d','e','f','e','g','d','e','h','i','j','h','k','l','i','j'])
#subsequence(['a','b','c'])
#subsequence(['a','b','c','a'])


'''

Algorithm: 
 1. First Algorithm computes the first and last occurence of each charcater and store it in interval array.
 2. Iterate over the intervals and Merge the overalapping intervals so that each shopper appear exactly once in the subsequence
 3. Compute the length non overlapping intervals

Complexity O(nlogn)

'''
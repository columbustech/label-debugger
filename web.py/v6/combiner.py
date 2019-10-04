'''
Created on Mar 6, 2019

@author: hzhang0418
'''

from operator import itemgetter

'''
Combine given ranked lists into a single ranked list
'''

def combine_two_lists(first, second):
    all_items = set(first+second)
    first2score = {}
    second2score = {}
    
    # assign scores to items in first list
    score = len(all_items)
    for item in first:
        first2score[item] = score
        score -= 1
    for item in second:
        if item not in first2score:
            first2score[item] = score
            score -=1
            
    # assign scores to items in second list
    score = len(all_items)
    for item in second:
        second2score[item] = score
        score -= 1
    for item in first:
        if item not in second2score:
            second2score[item] = score
            score -=1
            
    item_scores = []
    for item in all_items:
        first_score = first2score[item]
        second_score = second2score[item]
        item_scores.append( (item, first_score, second_score, first_score+second_score) )
        
    # sort them in descending order of score, if same score, then compare their first score
    item_scores.sort(key= itemgetter(3,1), reverse=True)
    
    return [t[0] for t in item_scores]


def combine_all_lists(ranked_lists):
    if len(ranked_lists)==0:
        raise Exception("Zero ranked lists are given!")
    
    all_items = set(ranked_lists[0])
    for cur_list in ranked_lists[1:]:
        for item in cur_list:
            all_items.add(item)
            
    item2score = {item: 0 for item in all_items} 
    for cur_list in ranked_lists:
        # assign scores to items in this list
        score = len(all_items)
        for item in cur_list: # items in original list
            item2score[item] += score
            score -= 1
        tmp = set(cur_list)
        for item in all_items: # items not in original list
            if item not in tmp:
                item2score[item] += score
                
    item_scores = [(item, score) for item, score in item2score.items()]
    # Note: order between items with same scores will be random 
    item_scores.sort(key=itemgetter(1), reverse=True)
    
    return [t[0] for t in item_scores]

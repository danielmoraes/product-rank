# utils

def get_item_pos(list, s): 
    for idx, item in enumerate(list): 
        if item == s: 
            return idx 
    return -1 

import copy

lower = -1.
upper = 1.

def normalize_instances(instances, ranges = None) :
    normalized_instances = copy.deepcopy(instances)
    if ranges == None :
        ranges_dict = dict()
    for attribute in normalized_instances[0].keys() :  # we iterate on the attributes
        column = [instance[attribute] for instance in normalized_instances]
        if ranges != None :
            minimum = ranges[attribute][0]
            maximum = ranges[attribute][1]
        else :
            minimum = min(column)
            maximum = max(column)
            ranges_dict[attribute] = [minimum, maximum]
        for i in range(len(column)) :
            if column[i] == minimum :
                column[i] = lower
            elif column[i] == maximum :
                column[i] = upper
            else :
                column[i] = lower + (upper-lower) * (column[i] - minimum) / (maximum - minimum)
        
        # Copying normalized values in memory
        for elem, instance in zip(column, normalized_instances):
            instance[attribute] = elem
   
    if ranges == None :
        return normalized_instances, ranges_dict
    else :
        return normalized_instances

def normalize_data(data, ranges = None) :
    normalized_data = copy.deepcopy(data)
    
    if ranges == None :
        ranges_dict = dict()
    
    if type(normalized_data[0]) is list:
        attrs = len(normalized_data[0])
    else:
        attrs = 1

    for i in range(attrs):
        if attrs > 1:
            column = [item[i] for item in normalized_data]
        else:
            column = normalized_data

        if ranges != None:
            minimum = ranges[i][0]
            maximum = ranges[i][1]
        else:
            minimum = min(column)
            maximum = max(column)
            ranges_dict[i] = [minimum, maximum]
        for j in range(len(column)):
            if column[j] == minimum:
                column[j] = lower
            elif column[j] == maximum:
                column[j] = upper
            else :
                column[j] = lower + (upper-lower) * (column[j] - minimum) / (maximum - minimum)

        # Copying normalized values in memory
        for elem, item in zip(column, normalized_data):
            if attrs > 1:
                item[i] = elem
            else:
                item = elem

    if ranges == None :
        return normalized_data, ranges_dict
    else :
        return normalized_data

# utils

def get_item_pos(list, s): 
    for idx, item in enumerate(list): 
        if item == s: 
            return idx 
    return -1            

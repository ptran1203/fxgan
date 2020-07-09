from const import CATEGORIES_MAP, INVERT_CATEGORIES_MAP
import numpy as np

def _safe_get(idx):
    try:
        return INVERT_CATEGORIES_MAP[idx]
    except:
        return idx

def draw_md_table(scores):
    """
    sample input:
    {
        'VGG16': [0.739 ,0.786 ,0.721 ,0.755 ,0.776 ,0.774 ,0.683 ,0.884],
        'standard aug': [0.731 ,0.779 ,0.721 ,0.771 ,0.774 ,0.755 ,0.690 ,0.865],
        'GAN v1': [0.744 ,0.774 ,0.736 ,0.772 ,0.775 ,0.757 ,0.695 ,0.878],
    }
    """
    table = '|  |'
    for name in scores.keys():
        table += ' {} |'.format(name)

    table += '\n|'
    for i in range(len(scores) + 1):
        table += '--|'

    table += '\n'
    head = scores[list(scores.keys())[0]]
    len_head = len(head)
    avgs = [sum(v)/len_head for v in scores.values()]
    for i in range(len_head):
        # use i + 1 because we don't care No Finding case 
        table += '| ' + _safe_get(i) + ' |'
        # find the best score value
        best = 0
        row = ''
        for name in scores.keys():
            point = round(scores[name][i], 3)
            if point > best:
                best = point
            row += ' {} |'.format(point)
        row = row.replace(str(best), '**{}**'.format(best))
        table += row + '\n'
    row = '| **Average** |'
    best = 0
    for avg in avgs:
        avg = round(avg, 3)
        if avg > best:
            best = avg
        row += ' {} |'.format(avg)
    row = row.replace(str(best), '**{}**'.format(best))
    table += row + '\n'
    return table



t = draw_md_table({
    'BAGAN': [0.68700851, 0.66911371, 0.73974217, 0.7589632 , 0.73636512,
        0.74040938, 0.75206689, 0.75399698, 0.68468512, 0.86447541],
    'GAN v1': [0.6940082 , 0.6558297 , 0.73341737, 0.774533  , 0.72823122,
            0.74645802, 0.7528848 , 0.76804866, 0.6655222 , 0.86680296],
    'VGG16': [0.64423613, 0.66321496, 0.72975991, 0.76752939, 0.7130005 ,
        0.72108968, 0.73559245, 0.77180551, 0.65813898, 0.84691835],
    'VGG16 + standard augment': [0.68320208, 0.66263178, 0.72914494, 0.77314687, 0.71750277,
            0.74320282, 0.75906515, 0.75608415, 0.66901341, 0.8622686 ]
})

print(t)
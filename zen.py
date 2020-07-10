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



t = draw_md_table(
    {
        'VGG16': [0.72008197, 0.65094216, 0.70600801, 0.76723819, 0.71350819,
            0.77838193, 0.7679022 , 0.73409787, 0.68693671, 0.88577081],

        'VGG16 + decay': [0.65971942, 0.67350554, 0.74284161, 0.78021341, 0.71633025,
            0.75088064, 0.75108217, 0.76870327, 0.67295667, 0.86955549]
    }
)

print(t)
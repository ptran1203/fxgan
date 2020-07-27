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
    'VGG16 + standard augment': [0.6786879452004883,
   0.6752725029828712,
   0.7354756531768718,
   0.7941096148413221,
   0.7013054341871434,
   0.753290263188177,
   0.7668212981593264,
   0.6395236344092106,
   0.6843162670123136,
   0.8765715667311411,
   0.7416795865633075,
   0.6375227444192962,
   0.48576365663322185,
   0.45813920655089646,
   0.6845064468399916],
    'GAN v2': [0.6949736751015648,
   0.7212805112337964,
   0.7595366330675493,
   0.8075364911950278,
   0.724808594219746,
   0.7531406684091992,
   0.7663928012519563,
   0.6896778026081299,
   0.7242611795204148,
   0.8878102836879431,
   0.695562938353636,
   0.6672036568588293,
   0.6709030100334449,
   0.38101087109981646,
   0.7110970196575777
    ],
    'GAN Vnew': [0.70022044, 0.72060254, 0.75214884, 0.82482814, 0.74100426,
        0.74721277, 0.73915717, 0.66768691, 0.69026248, 0.88203981,
        0.73242894, 0.62814539, 0.68270346, 0.54099958, 0.5380892 ]
    # 'BAGAN': [0.72840812, 0.70454355, 0.78345195, 0.80308178, 0.66570514,
    #     0.77344237, 0.69780662, 0.69037331, 0.66884569, 0.87527006,
    #     0.75006879, 0.72865296, 0.75385162, 0.5310418 , 0.52986067]
    }
)
print(t)

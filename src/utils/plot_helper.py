# -*- coding: utf-8 -*-
"""
.. codeauthor:: Tim Wengefeld <tim.wengefeld@tu-ilmenau.de>
"""

import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import copy

def plot_sample(rgb_images, depth_images, predictions, save_plots=False, plot_savepath=''):
    if not save_plots:
        plt.ion()
    fig = plt.figure()
    fig.set_figheight(4)
    fig.set_figwidth(20)
    spec = gridspec.GridSpec(ncols=8, nrows=1,
                             width_ratios=[0.2, 0.2, 0.2, 1, 0.2, 0.65, 0.15, 1], wspace=0.1,
                             hspace=0.1, height_ratios=[1])
    
    for i in range(0,len(rgb_images),1):
        plt.clf()

        # draw rgb patch
        ax1 = fig.add_subplot(spec[0])
        ax1.imshow(rgb_images[i], interpolation='nearest', aspect='auto')
        ax1.axis('off')

        # draw depth patch
        ax2 = fig.add_subplot(spec[1])
        ax2.imshow(depth_images[i],vmin=1000,vmax=4000, interpolation='nearest', aspect='auto')
        ax2.axis('off')

        phantom_image = copy.deepcopy(rgb_images[i]) 
        phantom_image.fill(255)

        ax3 = fig.add_subplot(spec[2])
        ax3.imshow(phantom_image, aspect='auto')
        ax3.axis('off')

        ax5 = fig.add_subplot(spec[4])
        ax5.imshow(phantom_image, aspect='auto')
        ax5.axis('off')
        
        ax6 = fig.add_subplot(spec[6])
        ax6.imshow(phantom_image, aspect='auto')
        ax6.axis('off')

        # draw soft biometric attributes
        ax4 = fig.add_subplot(spec[3])
        ax4.axes.get_xaxis().set_visible(False)
        ax4.set_xlim(-1,1)
        ax4.set_title('Soft-biometric Attributes')

        for k in ['has_long_hair','has_jacket','has_long_sleeves','has_long_trousers','gender']:
            if k == 'gender':
                y_label = 'Male'
            if k == 'has_long_hair':
                y_label = 'Long\nHair'
            if k == 'has_long_sleeves':
                y_label = 'Long\nSleeves'
            if k == 'has_jacket':
                y_label = 'Jacket'
            if k == 'has_long_trousers':
                y_label = 'Long\nTrousers'

            if k == 'gender':
                color=['#ff69b4','#00bfff']
            else:
                color=['#ffa500','#90ee90']

            ax4.barh(y_label,predictions[k][i][0],color=color[0])
            ax4.barh(y_label,0-predictions[k][i][1],color=color[1])

        secaxy = ax4.secondary_yaxis('right')
        secaxy.set_yticks([0,1,2,3,4])
        secaxy.set_yticklabels(['Short\nHair','No\nJacket','Short\nSleeves','Short\nTrousers','Female'])

        # draw orientation
        ax6 = fig.add_subplot(spec[5], polar=True)
        ax6.set_theta_zero_location('S', offset=0)
        ax6.set_yticks([])
        orientation = np.deg2rad(predictions['orientation'][i])
        style_offset = np.deg2rad(10)
        ax6.plot([0,orientation-style_offset,orientation,orientation+style_offset,0],[0,0.75,1,0.75,0])
        ax6.set_title('Orientation')

        # draw posture
        ax8 = fig.add_subplot(spec[7])
        ax8.bar(['Standing','Squatting','Sitting'],predictions['posture'][i])
        ax8.set_ylim(0,1)
        ax8.set_title('Posture')
 
        if save_plots:
            plt.savefig(plot_savepath+'{:03d}'.format(i)+'.png')
        else:
            plt.pause(0.0005) 
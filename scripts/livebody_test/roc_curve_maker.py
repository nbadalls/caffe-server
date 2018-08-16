#use to draw Roc curve
#Author Zkx@__@
#Date 2018-01-10
#Update 2018-01-17

import os
import sys
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
plt.switch_backend('agg')  #solve invalid display variable error

def readScore(path2ScoreFile):
    with open(path2ScoreFile,'r') as f:
        data = f.read().splitlines()
    data = np.array([[float(datum.split()[0]),float(datum.split()[1])] for datum in data])
    return data[:,0], data[:,1]#return distance label

def draw_roc(fpr,tpr,thresholds, output_figure_path, title=None):
    roc_auc = auc(fpr, tpr)

    fig = plt.figure()
    plt.plot(fpr, tpr, lw=1, label='ROC fold (area = %0.4f)' % (roc_auc))
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

    #save the fpr acc threshold
    roc_info = []
    for tolerance in [10**-7,10**-6,5.0*10**-6,10**-5, 10**-4, 1.2*10**-4, 10**-3, 10**-2, 10**-1]:
        fpr = np.around(fpr, decimals = 7)
        index = np.argmin(abs(fpr - tolerance))
        #same threshold match the multi-accuracy
        index_all = np.where(fpr == fpr[index])
        #select max accuracy
        max_acc = np.max(tpr[index_all])
        threshold = np.max(abs(thresholds[index_all]))
        x, y = fpr[index], max_acc

        plt.plot(x, y, 'x')
        plt.text(x, y, "({:.7f}, {:.7f}) threshold={:.7f}".format(x, y, threshold))
    	temp_info = 'fpr\t{}\tacc\t{}\tthreshlod\t{}'.format(tolerance, round(max_acc,5), threshold)
	roc_info.append(temp_info)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC - {}'.format(title))
    #plt.legend(loc="lower right")
    fig.savefig(output_figure_path)
    plt.close(fig)
    return roc_info

def roc_maker(score_file):
    filename = os.path.basename(score_file).strip('.txt')
    output_figure_path = score_file.replace('.txt', '_ROC.png')
    distance, label = readScore(score_file)
    fpr, tpr, thresholds = roc_curve(label, distance, pos_label=1)
    return draw_roc(fpr, tpr, thresholds, output_figure_path, filename)


def travel_folder(folder_path):
	for rootpath, folderpath, filenames in os.walk(folder_path):
		for filename in filenames:
			if filename.endswith('.txt'):
				 print filename
				 roc_maker('{}/{}'.format(rootpath, filename))


if __name__ == '__main__':

    if len(sys.argv) < 2:
        print "Please input distance file.."
    else:

    #score_file = "/home/minivision/SoftWare/caffe_reid_mini/asset/sphere_fintune_m2_l3/result_v2_faceNet-20_m2_l3_zkx_iter_15000.txt"
        score_file = sys.argv[1]
        roc_maker(score_file)
	#folder_path = sys.argv[1]
	#travel_folder(folder_path)

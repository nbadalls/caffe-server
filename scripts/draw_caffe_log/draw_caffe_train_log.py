
#use to draw caffe train result

import pandas as pd
import matplotlib.pyplot as plt
import parse_log
import os
import argparse
plt.switch_backend('agg')

def draw_log_image(parse_log_file):

    log_file_name = parse_log_file.rstrip('.log.train').split('/')[-1]
    image_figure_path = parse_log_file.replace('.log.train', '.png')
    train_log = pd.read_csv(parse_log_file)

    _, ax1 = plt.subplots()
    ax1.set_title(log_file_name)
    ax1.plot(train_log["NumIters"], train_log["softmax_loss"], alpha=0.5)
    ax1.plot(train_log["NumIters"], train_log["LearningRate"]*150, 'g')
    ax1.set_xlabel('iteration')
    ax1.set_ylabel('train loss')
    plt.legend(loc='upper right')

    plt.savefig(image_figure_path)
    plt.close()
    # plt.show()
    print(image_figure_path)
    #print ('Done.')
	


def parse_and_draw_log_image(log_file):
	outdir_path = os.path.abspath(os.path.join(log_file, '..'))
	parse_log.main2(log_file, outdir_path)
	if os.path.exists(log_file + '.train'):
		draw_log_image(log_file + '.train')


def parse_args():
    description = ('Parse a Caffe training log into two CSV files '
                   'containing training and testing information and draw loss curve')
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('logfile_path',
                        help='Path to log file')


    args = parser.parse_args()
    return args



def parse_args2():
    description = ('Parse a Caffe training log into two CSV files '
                   'containing training and testing information and draw loss curve')
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('logfile_folder',
                        help='Path to log folder')


    args = parser.parse_args()
    return args


def main_single():
	args = parse_args()
	parse_and_draw_log_image(args.logfile_path)

def main_folder():
	args = parse_args2()
	if not os.path.isdir(args.logfile_folder):
		print args.logfile_folder , '\nis not a directory...'
		return 

	for root_path, folder_path, filename_path in os.walk(args.logfile_folder):
		for filename in filename_path:
			if filename.endswith('.log'):
				parse_and_draw_log_image('{}/{}'.format(root_path, filename))

if __name__ == '__main__':
    main_folder()

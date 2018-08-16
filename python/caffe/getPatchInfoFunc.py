#Using these functions to split patches info
#center_id  norm_ratio height width
import os



#transform center id string into patchID
def stringTopatchID(patch_str):
    id_pairs = {
     'fc':-1,
     'le': 0,
     're': 1,
      'n': 2,
     'lm': 3,
     'rm': 4
	}

    center_id = []
    for elem in patch_str:
	center_id.append(id_pairs[elem])
    return  center_id

#transform patchID from number into English letter
def patchIDMatching(center_ind):
    id_pairs = {
    '-1': 'fc',
    '0' : 'le',
    '1' : 're',
    '2' : 'n',
    '3' : 'lm',
    '4' : 'rm'
    }

    patch_center = ""
    for index, elem in enumerate(center_ind):
        if index != len(center_ind)-1:
            patch_center +="{}_".format(id_pairs[str(elem)])
        else:
            patch_center +="{}".format(id_pairs[str(elem)])
    return patch_center


#split each patch info    fc_0.5_60x60_le_0.99_60x60_n_0.99_60x60_lm_0.99_60x60 -> <fc_0.5_60x60> <le_0.99_60x60>
def getPatchInfoList(patch_line):
    patch_list = []
    if patch_line.find('x') >= 0:
        f_letter = 'x'
    elif patch_line.find('X') >= 0:
        f_letter = 'X'
    else:
        return patch_list
    while True:
        if patch_line.find(f_letter) >=0:
            idex = patch_line.find(f_letter)
            sub_line  = patch_line[idex+1:]
            if sub_line.find('_') >=0:
                index_info =sub_line.find('_')
                index_info += idex+1
                patch_list.append(patch_line[0:index_info])
                patch_line = patch_line[index_info +1 :]
            else:
                patch_list.append(patch_line[0:])
                break
    return patch_list

#split single patch info's information
def splitPatchInfo(single_patch_line):

    dict_info = {}

    if single_patch_line.find('x') >= 0:
        f_letter = 'x'
    elif single_patch_line.find('X') >= 0:
        f_letter = 'X'


    pre_part = single_patch_line.split(f_letter)[0].split('_')
    height = int(pre_part[-1])
    width = int(single_patch_line.split(f_letter)[1])
    norm_ratio = float(pre_part[-2])
    center_id_str = pre_part[0:-2]
    center_id = stringTopatchID(center_id_str)

    #save into dict
    dict_info['center_id'] = center_id
    dict_info['norm_ratio'] = norm_ratio
    dict_info['height'] = height
    dict_info['width'] = width
    return dict_info




# line = "fc_0.5_60x60_le_re_n_0.99_60x60_le_n_0.99_60x60_lm_rm_0.99_60x60"
# list_result = getPatchInfoList(line)
# for ll in list_result:
# 	dict_result = splitPatchInfo(ll)
# 	for key in dict_result.keys():
# 		print dict_result[key]

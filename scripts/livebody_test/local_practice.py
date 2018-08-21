# -*- coding: utf-8 -*-
"""
Created on Thu May 31 17:32:01 2018
use to test local face
@author: minivision
"""
import face_gather_test
import sys


def model_test(select_date, test_set, batch_num=-1):

    test = face_gather_test.gethorModelTest(select_date, test_set, batch_num)
    test.testRun()

if __name__ == '__main__':

    if len(sys.argv)  == 3:
        select_date = sys.argv[1]
        test_set = sys.argv[2]
        model_test(select_date, test_set)
    elif len(sys.argv) == 4:
        select_date = sys.argv[1]
        test_set = sys.argv[2]
        batch_num =int( sys.argv[3])
        model_test(select_date, test_set, batch_num)
    else:
         print("input: \n  --select_date\n  --test_set\n --batch_num[-1]")

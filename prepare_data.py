import numpy as np
import os
import argparse

parser = argparse.ArgumentParser(description='Diffusion Model at 32x32')
parser.add_argument('-type', '--type', type=str, metavar='', help='type of example')
args = parser.parse_args()

if __name__ == "__main__":
    if args.type:
        print('User defined problem')
        type = args.type
    else:
        print('Not define problem type, use default poentential vorticity')
        type = 'ns'

    cwd = os.getcwd()
    ### define problem type ##########
    if type == 'wave':
        from config.config_wave import *
    if type == 'euler':
        from config.config_euler import *
    if type == 'ns':
        from config.config_ns import *
    ########################## gen data ##########################
    with open(Ori_data_high, 'rb') as ss:
        mat_f = np.load(ss)

    with open(Ori_data_low, 'rb') as ss:
        mat_c = np.load(ss)


    mat_f= mat_f[:N_gen, ..., None]
    mat_f = make_image(mat_f)

    mat_c = mat_c[:N_gen, ..., None]
    mat_c = make_image(mat_c)

    cv1_0, cv1_1, cv1_2 = prepare_cv_data(kernel=1, blur=1, mat=mat_f)
    cv1_0 = make_image(cv1_0)

    with open(Gen_data, 'wb') as ss:
        np.save(ss, mat_c)
        np.save(ss, cv1_0)

    ################################ Sup data ################################
    mat1_u_0, mat1_u_1, mat1_u_2, mat1_u_d = prepare_up_cv(L, points_x_0, points_x_1, points_x_2, points_x, mat_f,
                                                           kernel=1, blur=1)


    with open(Super_data, 'wb') as ss:

        np.save(ss, mat1_u_0)
        np.save(ss, mat1_u_1)
        np.save(ss, mat1_u_2)

        np.save(ss, mat_f)

    ##### prepare for testing
    test_ls = []
    with open(Ori_data_test, 'rb') as ss:
        tmat_c = np.load(ss)
        tmat_f = np.load(ss)

    tmat_f = tmat_f[:N_gen, ..., None]
    tmat_f = make_image(tmat_f)

    tmat_c = tmat_c[:N_gen, ..., None]
    tmat_c = make_image(tmat_c)

    cv1_0, cv1_1, cv1_2 = prepare_cv_data(kernel=1, blur=1, mat=tmat_f)
    cv1_0 = make_image(cv1_0)

    mat1_u_0, mat1_u_1, mat1_u_2, mat1_u_d = prepare_up_cv(L, points_x_0, points_x_1, points_x_2, points_x, tmat_f,
                                                           kernel=1, blur=1)


    with open(Test_data, 'wb') as ss:
        ### save at 32x32
        np.save(ss, tmat_c)
        np.save(ss, cv1_0)

        np.save(ss, mat1_u_0)
        np.save(ss, mat1_u_1)
        np.save(ss, mat1_u_2)

        np.save(ss, tmat_f)


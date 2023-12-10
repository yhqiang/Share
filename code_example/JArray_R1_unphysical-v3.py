import sys
import concurrent.futures
import time
import datetime
import numpy as np
from collections import Counter
from scipy import linalg
import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
Jtest = 1.0, 0.98, 1.01
# Jtest = 1.0,1.0,1.0
Ltest = np.int16(sys.argv[3])


def M_init2(L, J):
    Jx, Jy, Jz = J
    """ Initialize M matrix to all u=+1 configuration """
    ncell = L ** 2  # L = lattice size; L by L unit cells
    M = np.zeros([ncell, ncell])

    # Inner and boundary z bonds, intra unit cell i, A->B, z bond
    for i in range(ncell):
        M[i, i] = Jz  # intra unit cell i, A->B, z bond

    # Inner x bonds
    for i in range(ncell):
        if np.mod(i, L) == 0:  # the boundary terms will be fixed by pbc
            continue
        M[i, i-1] = Jx  # unit cell i to (i-1), A->B, x bond

    # Inner y bonds
    for i in range(L, ncell):
        M[i, i-L] = Jy  # unit cell i to (i-L), A->B, y bond

    # Apply pbc to x and y bonds on the boundary
    for i in range(L):
        M[i*L, i*L + (L-1)] = Jx  # uc (iL)A -> uc ((i-1)L+1)B
        M[i, (L - 1) * L + i] = Jy  # uc (i)A -> uc ((L-1)*L+i)B

    return M


def get_M2(L, J, bonds_to_flip=[]):
    # print(f'lattice size: {L} by {L}\n')
    # Initialization
    M = M_init2(L, J)
    if len(bonds_to_flip) > 0:
        ncell = L**2
        # print('bonds to be flipped: \n', bonds_to_flip,'\n')

        for (cell, bond_type) in bonds_to_flip:
            cell = np.mod(cell, ncell)  # cell index starts from 0
            # perform bond flip; a double flip would be identity operation
            if bond_type == 'z':
                M[cell, cell] *= -1
            if bond_type == 'x':
                if np.mod(cell, L) == 0:
                    M[cell, cell+L-1] *= -1
                else:
                    M[cell, cell - 1] *= -1
            if bond_type == 'y':
                if cell < L:
                    M[cell, cell+L*(L-1)] *= -1
                else:
                    M[cell, cell-L] *= -1

    return M


def M_svd(L, J, bond_flip=[]):
    M = get_M2(L, J, bonds_to_flip=bond_flip)
    u, s, vh = linalg.svd(M, lapack_driver='gesvd')
    return u, s, vh


def detQ(a, b):
    # construct a bock diagonal matrix
    zero_block = np.zeros_like(a)
    return linalg.det(np.block([[a, zero_block], [zero_block, b]]))


def gauge_trans_config(flux_config, L, sites):
    gauged_flux_config = flux_config
    for site in sites:
        if site < L**2:  # A sites 0,1,2, ..., N-1
            unit_cell_x = site
            unit_cell_y = site
            unit_cell_z = site

        else:         # B sites N, N+1, ..., 2N-1
            unit_cell_z = site-L**2

            if np.mod(unit_cell_z+1, L) == 0:
                unit_cell_x = unit_cell_z-L+1
            else:
                unit_cell_x = unit_cell_z + 1

            if unit_cell_z >= L*(L-1):
                unit_cell_y = unit_cell_z-L*(L-1)
            else:
                unit_cell_y = unit_cell_z+L
        gauged_flux_config = gauged_flux_config + \
            [(unit_cell_x, 'x'), (unit_cell_y, 'y'), (unit_cell_z, 'z')]
    # return gauged_flux_config
    r = [k for k, v in Counter(
        gauged_flux_config).items() if np.mod(v, 2) == 1]
    return r


def info(L, J, flux_config, gauge_sites=[]):
    gauged_flux_config = gauge_trans_config(flux_config, L, gauge_sites)
    M = get_M2(L, J, bonds_to_flip=gauged_flux_config)
    u, s, vh = linalg.svd(M, lapack_driver='gesvd')
    det = detQ(u, vh)
    # print(f'detQ = {det:.2f}, # of bond fermions = {len(gauged_flux_config)}, detQ * (-1)^N_Chi = {det*(-1)**len(gauged_flux_config):.2f}')
    return u, s, vh, np.round(det), len(gauged_flux_config)


GS_flux = []  # Assume GS has no bond flip
P_flux = GS_flux + [(0, 'x')]  # with x-link eminating from site 0 flipped
Q_flux = GS_flux + [(0, 'y')]  # with y-link eminating from site 0 flipped

u0, s0, vh0, detQ_0, Nchi_0 = info(Ltest, Jtest, GS_flux)
E_0_0 = -np.sum(s0)

uP, sP, vhP, detQ_P, Nchi_P = info(Ltest, Jtest, P_flux)
E_P_0 = -np.sum(sP)

uQ, sQ, vhQ, detQ_Q, Nchi_Q = info(Ltest, Jtest, Q_flux)
E_Q_0 = -np.sum(sQ)


# Elemenet 1
X0Q = 0.5*((uQ.T) @ u0 + vhQ @ (vh0.T))
X0Q_inv = linalg.inv(X0Q)
SADX_0Q = np.sqrt(np.abs(linalg.det(X0Q)))
# <0|s^y_0A|Q>
# minus sign because of -i*(UX_inv)_{\nu\lambda}
GS_yA_Q = -SADX_0Q * (u0@X0Q_inv)
# <0|s^y_L(L-1)B)|Q>
GS_yB_Q = -SADX_0Q * (vh0.T@X0Q_inv)


# Element 2
XPQ = 0.5*((uQ.T) @ uP + vhQ @ (vhP.T))
YPQ = 0.5*((uQ.T) @ uP - vhQ @ (vhP.T))
FPQ = linalg.inv(XPQ) @ YPQ
# matter vacuum overlap
SADX_PQ = np.sqrt(np.abs(linalg.det(XPQ)))
Q_zA_P = SADX_PQ * (XPQ-(YPQ@FPQ))


def Q_zB_P(mu, lbd):
    r = -(XPQ@(uP.T))[mu, 0] * (vhP.T)[0, lbd] - (XPQ@vhP)[mu,
                                                           0] * uP[0, lbd] + XPQ[mu, lbd] * (uP@vhP)[0, 0]
    r += -(XPQ@(uP.T))[mu, 0] * ((vhP.T)@FPQ)[0, lbd] + \
        (XPQ@vhP)[mu, 0] * (uP@FPQ)[0, lbd]
    r += - XPQ[mu, lbd] * (uP@FPQ@vhP)[0, 0] - \
        (YPQ@FPQ)[mu, lbd] * (uP@vhP)[0, 0]
    r += + (YPQ@FPQ@vhP)[mu, 0] * uP[0, lbd] + \
        (YPQ@FPQ@(uP.T))[mu, 0] * (vhP.T)[0, lbd]
    r += (YPQ@FPQ)[mu, lbd] * (uP@FPQ@vhP)[0, 0] + (YPQ@(FPQ.T)@vhP)[mu, 0] * \
        (uP@FPQ)[0, lbd] + (YPQ@FPQ@(uP.T))[mu, 0] * ((vhP.T)@FPQ)[0, lbd]
    return SADX_PQ * r

# mu is row index, lbd is col index
Q_zB_P_mat = np.asarray([[Q_zB_P(mu, lbd)
                          for lbd in range(Ltest**2)] for mu in range(Ltest**2)])


# Element 3
X0P = 0.5*((uP.T) @ u0 + vhP @ (vh0.T))
X0P_inv = linalg.inv(X0P)
SADX_0P = np.sqrt(np.abs(linalg.det(X0P)))
P_xA_GS = SADX_0P * (u0@X0P_inv)
P_xB_GS = SADX_0P * (vh0.T@X0P_inv)


def g_func(x, eta=0.2):  # eta should be larger than the freq step size
    r = 1j/(x+1j*eta)
    return r


def R1_sum_lbd_mu_mat_mul(L, E_0_0, E_P_0, E_Q_0, sP, sQ, freq):
    omega1, omega2 = freq
    N = L**2
    D_r = np.diag([g_func(omega1-((E_P_0+2*sP[lbd]) - E_0_0))
                   for lbd in range(N)])
    D_s = np.diag([g_func(omega2-((E_Q_0+2*sQ[mu]) - E_0_0))
                   for mu in range(N)])
    D_r_TR = np.diag([g_func(omega1+((E_P_0+2*sP[lbd]) - E_0_0))
                      for lbd in range(N)])
    D_s_TR = np.diag([g_func(omega2+((E_Q_0+2*sQ[mu]) - E_0_0))
                      for mu in range(N)])

    r = (GS_yA_Q @ D_s @ Q_zA_P @ D_r @ (P_xA_GS.T))[0, 0]
    r += (GS_yA_Q @ D_s @ Q_zA_P @ D_r @ (P_xB_GS.T))[0, L-1]
    r += (GS_yA_Q @ D_s @ Q_zB_P_mat @ D_r @ (P_xA_GS.T))[0, 0]
    r += (GS_yA_Q @ D_s @ Q_zB_P_mat @ D_r @ (P_xB_GS.T))[0, L-1]

    r += (GS_yB_Q @ D_s @ Q_zA_P @ D_r @ (P_xA_GS.T))[L*(L-1), 0]
    r += (GS_yB_Q @ D_s @ Q_zA_P @ D_r @ (P_xB_GS.T))[L*(L-1), L-1]
    r += (GS_yB_Q @ D_s @ Q_zB_P_mat @ D_r @ (P_xA_GS.T))[L*(L-1), 0]
    r += (GS_yB_Q @ D_s @ Q_zB_P_mat @ D_r @ (P_xB_GS.T))[L*(L-1), L-1]

    r -= (GS_yA_Q @ D_s_TR @ Q_zA_P @ D_r_TR @ (P_xA_GS.T))[0, 0]
    r -= (GS_yA_Q @ D_s_TR @ Q_zA_P @ D_r_TR @ (P_xB_GS.T))[0, L-1]
    r -= (GS_yA_Q @ D_s_TR @ Q_zB_P_mat @ D_r_TR @ (P_xA_GS.T))[0, 0]
    r -= (GS_yA_Q @ D_s_TR @ Q_zB_P_mat @ D_r_TR @ (P_xB_GS.T))[0, L-1]

    r -= (GS_yB_Q @ D_s_TR @ Q_zA_P @ D_r_TR @ (P_xA_GS.T))[L*(L-1), 0]
    r -= (GS_yB_Q @ D_s_TR @ Q_zA_P @ D_r_TR @ (P_xB_GS.T))[L*(L-1), L-1]
    r -= (GS_yB_Q @ D_s_TR @ Q_zB_P_mat @ D_r_TR @ (P_xA_GS.T))[L*(L-1), 0]
    r -= (GS_yB_Q @ D_s_TR @ Q_zB_P_mat @ D_r_TR @ (P_xB_GS.T))[L*(L-1), L-1]

    r *= 2j
    return (omega1, omega2, np.real(r), np.imag(r))


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def block(L, block_freq, executor, f_name):
    runs = [executor.submit(R1_sum_lbd_mu_mat_mul, L, E_0_0, E_P_0, E_Q_0, sP, sQ, block_freq[i])
            for i in range(len(block_freq))]
    # results = list([r.result() for r in runs])
    with open(f_name, 'a+') as f:
        # for result in results:
        #     f.write(f'{result[0]},{result[1]},{result[2]},{result[3]}\n')
        for r in concurrent.futures.as_completed(runs):
            f.write(
                f'{r.result()[0]},{r.result()[1]},{r.result()[2]},{r.result()[3]}\n')


def main(L, J, my_freq, n_workers, job_id):
    # number of blocks to split the frequency points into
    num_block = np.int16(np.sqrt(len(my_freq)))
    # number of points in each block
    block_size = 1 + len(my_freq)//num_block
    freq_tot = list(chunks(my_freq, block_size))          # frequncy parts

    if not os.path.exists('data'):
        os.mkdir('data')
    f_name = f'./data/matmul_JA_ktv2nd_R1_L{L}_Js{np.round(J,2)}_{today}_job{job_id}.csv'
    with open(f_name, 'w+') as f:                         # header of the output file
        f.write('omega1,omega2,R1_real,R1_imag\n')
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        for itv in range(num_block):
            block(L, freq_tot[itv], executor, f_name)


if __name__ == '__main__':
    job_name = str(sys.argv[0])
    # total number of jobs
    n_jobs = np.int16(sys.argv[1])
    job_id = np.int16(sys.argv[2])                         # job id
    # number of workers for parallel computing
    # works on some Unix systems
    n_workers = len(os.sched_getaffinity(0))
    print(f'max_workers = {n_workers}')

    # total number of freq points
    freq_max = 2.6                                         # maximum frequency
    freq_min = -freq_max
    # frequency step; should be smaller than the broadening of the Lorenzian
    freq_step = 0.1
    freq = np.arange(freq_min, freq_max+freq_step, freq_step)
    my_freq = np.array([(o1, o2) for o1 in freq for o2 in freq])
    # split freq points in to n_jobs
    # number of chunks to split the frequency interval into
    num_chunks = n_jobs
    freq_len = len(my_freq)
    # number of points in each chunk (job)
    part_len = freq_len//num_chunks + 1
    # generate frequncy part list
    freq_tot = list(chunks(my_freq, part_len))
    # frequency part of this current job
    freq_job = freq_tot[job_id-1]

    today = datetime.date.today()
    start = time.perf_counter()
    print(f'{datetime.datetime.now()} starting {job_name}')
    print(f'number of freq points: {len(freq_job)}')
    print(f'freq computed in job{job_id}:')
    print(freq_job)

    # run the main function
    main(Ltest, Jtest, freq_job, n_workers, job_id)

    end = time.perf_counter()
    time_taken = end - start
    print(f'Finished, time taken: {time_taken}')
    with open('time_log_R1.txt', 'a+') as f:
        f.write(
            f'matmul_JArray_ktv2nd_R1_L{Ltest}_Js{np.round(Jtest,2)}_{datetime.date.today()}_job{job_id}: {time_taken}; points:{len(freq_job)}; time per point: {time_taken/(len(freq_job))}\n')

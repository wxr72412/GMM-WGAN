seed = 23
image = True # generate_image
# image = False

BATCH_SIZE = 16
# BATCH_SIZE = 32
# BATCH_SIZE = 64  # Batch size (GMM with a small value, and its BATCH_SIZE is hyperPara.GMM_num_samples)
# BATCH_SIZE_plot = BATCH_SIZE
BATCH_SIZE_plot = BATCH_SIZE * 2


DATASET = 'gaussian'
num_gaussians = 1
# num_gaussians = 2
# num_gaussians = 3
#################### situation 7 #####################
V_e_in = {}
V_e_not = {}
V_q_in = {}
V_q_not = {}

# DATASET = 'hepar2'
# V_q_not = {'age': '82'}
# V_q_not = {'age': '82', 'bilirubin': '4'}
# V_q_not = {'age': '82', 'bilirubin': '4', 'platelet': '125'}
# V_q_not = {'age': '82', 'bilirubin': '4', 'platelet': '125', 'alt': '67'}
###############################################
# DATASET = 'ecoli70'

###############################################
# DATASET = 'heart'
# V_q_not = {'age': '60'}
# V_q_not = {'age': '60', 'trestbps': '160'}
# V_q_not = {'age': '60', 'trestbps': '160', 'thalach': '160'}
# V_q_not = {'age': '60', 'trestbps': '160', 'thalach': '160', 'oldpeak': '1'}
###############################################
# MODE = 'WM'
# MODE = 'SVM'

# MODE = 'KDE'
# MODE = 'GMM'
#
# MODE = 'GAN'
# MODE = 'WGAN'

# MODE = 'GMM_GAN'
# MODE = 'GMM_WGAN'

# MODE = 'GMM_in_GAN'
MODE = 'GMM_in_WGAN'


### GAN ###############################################
Dim_G_Input = 1
Dim_D_Input = 2
if DATASET == 'hepar2' or DATASET == 'ecoli70' or DATASET == 'heart':
    Dim_G_Input = len(V_q_not)
    Dim_D_Input = Dim_G_Input + 1

Dim_Hidden = 512  # Model dimensionality

Dim_G_Output = 1
Dim_D_Output = 1

# lr = 1e-4 # GAN WGAN
# lr = 1e-3

# lr = 0.001
# lr = 0.0015
# lr = 0.002 # GMM_in_WGAN inference
# lr = 0.0025

lr = 0.01 # GMM_in_WGAN learning
# lr = 0.1
#
ITERS = 3000  # how many generator iterations to train for
CRITIC_ITERS = 3  # How many critic iterations per generator iteration
FIXED_GENERATOR = False  # whether to hold the generator fixed at real data plus
use_cuda = True

### WGAN ###############################################
LAMBDA = .1  # Smaller lambda seems to help for toy tasks specifically
##################################################

### GMM ###############################################
# num_components = 1 # hepar2 + 1 missing: 1
# num_components = 2 # 报错：numpy.linalg.LinAlgError: When `allow_singular is False`, the input matrix must be symmetric positive definite.
num_components = 3 # GMM_in_WGAN
# num_components = 4 # GMM_in_WGAN
# num_components = 5 # GMM_in_WGAN
# num_components = 7 # GMM_in_WGAN
# num_components = 9 # GMM_in_WGAN

GMM_ITERS = 3000

### WM ###############################################
# number_clusters = 2 # hepar2 + 1 missing: 2
number_clusters = 9
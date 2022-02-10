#!/usr/bin/env python
# coding: utf-8

# ## 1. Data loading, transforming and filtering

# Start by importing the important modules

# In[5]:


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import copy
import pandas as pd
import warnings
import seaborn as sn
import scipy.io
import signal
import warnings
from multiprocessing import Pool
from mixed_linear_model import MixedLM
import operator
import xlsxwriter
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle
matplotlib.rcParams['pdf.fonttype'] = 42
from scipy.stats import hypergeom
from matplotlib.cm import ScalarMappable
import os
from statsmodels.stats.multitest import multipletests
#plotly imports for 3D plotting
import plotly.graph_objs as go
import plotly
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)


# Set a nice plotting style

# In[6]:


plt.style.use('seaborn-whitegrid')
sn.set_style("whitegrid", {'xtick.direction': 'out', 'xtick.major.size': 6.0, 'xtick.minor.size': 3.0,
    'ytick.color': '.15', 'ytick.direction': 'out', 'ytick.major.size': 6.0, 'ytick.minor.size': 3.0, 
                           'font.family': ['DejaVu Sans'],})


# First load and record the data in a dictionnary

# In[10]:


def transform_data(x):
    return np.log2(x+10**-4)-np.log2(11*10**-5)

def invert_transform(y):
    return 2**(y+np.log2(11*10**-5))-10**-4 

dic_itz = {}
dic_itz_raw = {}
dic_struc = {'rep1': ['00A','06A','12A','18A'], 'rep2': ['00B','06B','12B','18B'], 'rep3': ['00C',None,'12C',None]}
for key, val in dic_struc.items():
    for x in val:
        if x is not None:
            load_path = 'Datasets/Profiles/ZT'+x+'.mat'
            mat = scipy.io.loadmat(load_path)
        for name, data, SD in zip(mat['all_genes'], mat['MeanGeneExp'], mat['SE']):
            print("Step 1: Dictionary", flush=True)
            if name[0][0] not in dic_itz_raw:
                dic_itz_raw[name[0][0]] = {'rep1' : np.array([]), 'rep1_std' :np.array([]), 'rep2' : np.array([]), 'rep2_std' : np.array([]), 'rep3' : np.array([]), 'rep3_std' :  np.array([])}
                dic_itz[name[0][0]] = {'rep1' : np.array([]), 'rep1_std' :np.array([]), 'rep2' : np.array([]), 'rep2_std' : np.array([]), 'rep3' : np.array([]), 'rep3_std' :  np.array([])}
            if x is None:
                data = [np.nan]*8
                SD = [np.nan]*8
            if len(dic_itz_raw[name[0][0]][key])>0:
                dic_itz_raw[name[0][0]][key] = np.vstack( (dic_itz_raw[name[0][0]][key],np.array(data) ))
                dic_itz_raw[name[0][0]][key+'_std']= np.vstack((dic_itz_raw[name[0][0]][key+'_std'],np.array(SD)))
                dic_itz[name[0][0]][key]= np.vstack( (dic_itz[name[0][0]][key],transform_data(np.array(data))))
                dic_itz[name[0][0]][key+'_std']= np.vstack( (dic_itz[name[0][0]][key+'_std'],transform_data(np.array(SD))))
            else:
                dic_itz_raw[name[0][0]][key] = np.array(data) 
                dic_itz_raw[name[0][0]][key+'_std']= np.array(SD)
                dic_itz[name[0][0]][key]= transform_data(np.array(data))
                dic_itz[name[0][0]][key+'_std']= transform_data(np.array(SD))  
        
#take transpose everywhere
for key in dic_itz:
    for key2 in ['rep1' , 'rep1_std', 'rep2', 'rep2_std', 'rep3', 'rep3_std']:
        dic_itz[key][key2] = dic_itz[key][key2].T
        dic_itz_raw[key][key2] = dic_itz_raw[key][key2].T


# Define sets of reference genes

# In[11]:


l_circadian = ['arntl', 'clock', 'npas2', 'nr1d1', 'nr1d2', 'per1', 'per2', 'cry1', 'cry2', 'dbp', 'tef', 'hlf', 
               'elovl3', 'rora', 'rorc']
l_zonated = ['glul', 'ass1','asl','cyp2f2','cyp1a2','pck1','cyp2e1', 'cdh2','cdh1','cyp7a1','acly', 'alb', "oat", 
             "aldob", 'cps1']


# Look at how the replicate variance evolves with the gene expression

# In[12]:


l_names = list(dic_itz.keys())
#compute list of variance per time condition and per zone condition and then average
l_var = np.array([ np.mean(np.nanvar([dic_itz[gene_name]['rep1'], dic_itz[gene_name]['rep2'],dic_itz[gene_name]['rep3']], axis = 0))/np.nanvar(np.vstack((dic_itz[gene_name]['rep1'], dic_itz[gene_name]['rep2'],dic_itz[gene_name]['rep3']))) for gene_name in l_names])
l_var = np.array([x if not np.isnan(x) else 10**-10 for x in l_var ])
l_exp_log = [invert_transform(np.nanmax(np.vstack((dic_itz[gene_name]['rep1'], dic_itz[gene_name]['rep2'],dic_itz[gene_name]['rep3']))))  for gene_name in l_names]
l_exp = [np.nanmax(np.vstack((dic_itz[gene_name]['rep1'], dic_itz[gene_name]['rep2'],dic_itz[gene_name]['rep3']))) for gene_name in l_names]


# ## 3. Do Mixed Model linear regression

# First, create the functions needed for regression

# In[13]:


np.seterr(divide='raise')
w = 2*np.pi/24

def return_explained_variance(Y, Y_pred, dic_re = None):    
    #get model prediction
    y_formatted = np.zeros((80,40))
    for idx, val in enumerate(Y_pred):
        y_formatted[int(idx/40), int(idx%40)] = val
        
    #convert into same format as Y
    Y_p = np.zeros((8,10))
    for x in range(8):
        for t in range(4):
            Y_p[x,t] = y_formatted[x*10,t*10]
            Y_p[x,4+t] = y_formatted[x*10,t*10]
            if t==0:
                Y_p[x,8] = y_formatted[x*10,t*10]
            elif t==2:
                Y_p[x,9] = y_formatted[x*10,t*10]
                
    if dic_re is not None:
        for x in range(8):
            for idx_t in range(10):
                Y_p[x,idx_t]+=dic_re[idx_t]
            
    #compute unexplained variance
    var_res = 0
    var_tot = 0
    mean = np.mean(Y)
    var_tot = np.sum( (np.array(Y)-mean)**2 )
    var_res = np.sum( (np.array(Y)-Y_p)**2 )
    if var_tot>0:    
        #return (1-var_res/var_tot)*var_tot
        return (var_tot-var_res)/80
    else:
        return 0.
    
    
def make_2D_regression(Y, predict, force_complete = False, formula = None, force_R = False):
    #get dic of design
    Xx, Xt, dic_lm, vect_structure = return_full_design_matrix(precise = False, replicates = True)
    #add response
    flat_Y = []
    for y in Y: #dim 8*(4*2)
        flat_Y.extend(y)
    flat_Y = np.array(flat_Y)
    dic_lm['y'] = flat_Y
    data = pd.DataFrame(dic_lm)

    l_formula = return_l_formula(force_complete = force_complete, formula = formula, force_R = force_R)
    model, selected, bic, l_schwartz, dic_re = select_model_last(data, vect_structure, l_formula )
    B = model._results.params
    SE = model._results.bse

    
    if predict:
        dic_par = {x+str(i) : 0 for i in range(3) for x in ['mu', 'a', 'b']}
        for par, val in zip(selected,B):
            dic_par[par] = val
        Xx_pred, Xt_pred, dic_lm_pred, vect_structure = return_full_design_matrix(precise = True, replicates = False)
        Y_pred = np.ones(len(Xx_pred)*len(Xt_pred)) * dic_par['mu0']
        for par in dic_lm_pred:
            Y_pred += dic_lm_pred[par] * dic_par[par]
            
        var_exp = return_explained_variance(Y, Y_pred)
        var_exp_re = return_explained_variance(Y, Y_pred, dic_re)
        return selected, B, SE, bic, l_schwartz, Xx_pred, Xt_pred, Y_pred, var_exp, var_exp_re
    else:
        return selected, B, SE, bic, l_schwartz


def return_full_design_matrix(precise = False, replicates = False):
    if precise:
        factor = 10
    else:
        factor = 1
    Xx = np.linspace(0,7,8*factor,endpoint = True)
    Xt =  np.linspace(0,24,4*factor,endpoint = False)
    Xt_rep3 =  np.linspace(0,24,2*factor,endpoint = False)
    if replicates:
        Xt = np.concatenate((Xt,Xt, Xt_rep3))
    X_space = []
    X_cos = []
    X_sin = []
    vect_structure = []
    for x in Xx:
        for idx_t, t in enumerate(Xt):
            X_space.append(x)
            X_cos.append(np.cos(w*t))
            X_sin.append(np.sin(w*t))
            vect_structure+=[idx_t]

    #convert everything into array
    X_space = np.array(X_space)
    X_cos = np.array(X_cos)
    X_sin = np.array(X_sin)
    vect_structure = np.array(vect_structure)

    dic_lm = {}
    for deg in range(3):
        if deg==0:
            dic_lm['a' + str(deg)] = X_space**deg*X_cos
            dic_lm['b' + str(deg)] = X_space**deg*X_sin
        if deg==1:
            dic_lm['a' + str(deg)] = X_space**deg*X_cos
            dic_lm['b' + str(deg)] = X_space**deg*X_sin
            dic_lm['mu' + str(deg)] = X_space*deg
        elif deg==2:
            dic_lm['a' + str(deg)] = 0.5*(3*X_space**deg-1)*X_cos
            dic_lm['b' + str(deg)] = 0.5*(3*X_space**deg-1)*X_sin
            dic_lm['mu' + str(deg)] = 0.5*(3*X_space**deg-1)

    return Xx, Xt, dic_lm, vect_structure

def return_l_formula(force_complete, formula = None, force_R = False):
    if formula is not None:
        return [formula]
    elif force_complete:
        l_formula = ['y ~ 1+a0+b0+a1+b1+a2+b2+mu1+mu2']
    elif force_R:
        l_formula = ['y ~ 1+a0+b0']
    else:
        
        l_formula = ['y ~ 1',
                    'y ~ 1+mu1',
                    'y ~ 1+mu2',
                    'y ~ 1+mu1+mu2',
                    'y ~ 1+a0+b0',
                    'y ~ 1+a0+b0+a1+b1',
                    'y ~ 1+a0+b0+mu1',
                    'y ~ 1+a0+b0+mu2',
                    'y ~ 1+a0+b0+mu1+mu2',
                    'y ~ 1+a0+b0+a1+b1+mu1',
                    'y ~ 1+a0+b0+a1+b1+mu2',
                    'y ~ 1+a0+b0+a1+b1+mu1+mu2',
                 
                    ]   
    return l_formula

def select_model_last(data, vect_structure, l_formula):
    if vect_structure is None:
        print('BUG, need vect_structure')
    #print('New gene')
    l_set = [ set( formula[6:].split('+')+['mu0']) for formula in l_formula]
    try:
        l_set[0].remove('')
    except:
        pass
    
    l_bic = []
    l_llf = []
    l_model = []
    l_re = []
    params = None

    for formula, set_f in zip(l_formula, l_set):
        model = MixedLM.from_formula(formula, data, re_formula='1', groups=vect_structure)
        model.sigma0 = 0.15
        model = model.fit(do_cg = True, reml = False, start_params = None, method='nm')
        llf = model.llf
        bic = -2*llf +1*np.log(10)+len(set_f)*np.log(80)
        l_bic.append(bic)
        l_llf.append(llf)
        l_re.append(model.random_effects)
        l_model.append(model)
    selected_model_idx = np.argmin(l_bic)
    min_bic = np.min(l_bic)

    #compute shwartz weights
    sum_diff = np.sum([np.exp(-0.5*(bic-min_bic)) for bic in l_bic])
    l_schwartz = [np.exp(-0.5*(bic-min_bic))/sum_diff for bic in l_bic]
    
    #select on shwartz weights
    l_schwartz_save = copy.copy(l_schwartz)
    
    #if several models have equivalent bic, keep the most complex model
    l_bic_unsorted = copy.copy(l_bic)
    l_bic, l_idx = zip(*sorted(zip(l_bic, list(range(len(l_bic)))), reverse = False))
    best_bic = l_bic[0]
    best_idx = l_idx[0]
    l_equivalent = []
    for bic,idx in zip(l_bic, l_idx):
        if abs((best_bic-bic)/best_bic)<0.01:
            best_bic = bic
            best_idx = idx
            l_equivalent.append(idx)
        else:
            break
            
    if len(l_equivalent)>1:
        #choose the model with the hightest number of parameters in l_equivalent
        l_num_parameters = [len(l_set[idx]) for idx in l_equivalent]
        l_equivalent_selected = [idx_model for idx_model, num_parameters in zip(l_equivalent, l_num_parameters) if num_parameters == np.max(l_num_parameters)]
        #choose the model with the lowest bic among those the most complex
        l_bic_equivalent = [l_bic_unsorted[idx_model] for idx_model in l_equivalent_selected]
        #print("corresponding bic list is: ", l_bic_equivalent)
        best_idx = l_equivalent_selected[np.argmin(l_bic_equivalent)]
        selected_model_idx = best_idx
    
    return l_model[selected_model_idx], ['mu0']+[x for x in l_formula[selected_model_idx][6:].split('+') if x!=''],            l_bic[selected_model_idx], l_schwartz_save, l_re[selected_model_idx]


# Then, do the regressions and store the result in a dictionnary.

# In[14]:


def compute_regressions_mp(arg):
    print("compute")
    [name_gene, force_complete] = arg
    
    array_gene_time =np.concatenate( (dic_itz[name_gene]['rep1'], dic_itz[name_gene]['rep2'], dic_itz[name_gene]['rep3'][:,[0,2]]), axis = 1)
    selected, B, SE,  bic, l_schwartz, Xx_pred, Xt_pred, Y_pred, var_exp, var_exp_re = make_2D_regression(array_gene_time, predict = True, force_complete = force_complete)
    return [selected, B, SE, bic, l_schwartz, Xx_pred, Xt_pred, Y_pred, var_exp, var_exp_re]    

#l_names =  l_names[:100] ##kann angepasst werden für test 
dic_reg = {}
l_arg = [(x, False) for x in l_names]
n_cpu = 1   #muss angepasst werden 
warnings.simplefilter("ignore")
original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
signal.signal(signal.SIGINT, original_sigint_handler)
pool = Pool(n_cpu)
  ##extra
try:
    print("Step 2: Running Regression", flush=True)
    results = pool.map(compute_regressions_mp, l_arg) ##tqdm?
    
except Exception as e: 
    print(e)
    print("BUG")
    pool.terminate()
else:
    print("Normal termination")
pool.close()
pool.join()
for name_gene, reg in zip(l_names, results):
    dic_reg[name_gene] = reg
print("Step 3: Creating new dictionary", flush=True)

# Make a few 3D plots to check that everything worked fine. First, define the function to plot in 3D.

# In[ ]:


def compute_figure_3D_tab_3(reg_2D, array_gene_time):
    selected, B, SE,  bic, l_schwartz, Xx_pred, Xt_pred, Y_pred, var_exp, var_exp_re = reg_2D
    y_formatted = np.zeros((80,40))
    for idx, val in enumerate(Y_pred):
        y_formatted[int(idx/40), int(idx%40)] = val

    #plot gene vs prediction
    x = [i for i in range(8) for j in range(4)]
    y = [j for i in range(8) for j in range(0,24,6)]
    z_1 = [array_gene_time[i][j] for i in range(8) for j in range(4)]
    z_2 = [array_gene_time[i][j] for i in range(8) for j in range(4,8)]
    array_gene_time = np.insert(array_gene_time, 9, np.nan, axis = 1)
    array_gene_time = np.insert(array_gene_time,11, np.nan, axis = 1)
    z_3 = [array_gene_time[i][j] for i in range(8) for j in range(8,12)]

    points_1 = go.Scatter3d( x = y, y = x, z = z_1 , name = "Experimental points", mode = 'markers',
                            marker=dict(color='rgb(37, 253, 233)', size=4, symbol='circle', line=dict( color='rgba(217, 217, 217, 0.14)',  width=1 ), opacity=0.8))
    points_2 = go.Scatter3d( x = y, y = x, z = z_2 , name = "Experimental points", mode = 'markers',
                            marker=dict(color='rgb(37, 253, 233)', size=4, symbol='circle', line=dict( color='rgba(217, 217, 217, 0.14)',  width=1 ), opacity=0.8))
    points_3 = go.Scatter3d( x = y, y = x, z = z_3 , name = "Experimental points", mode = 'markers',
                            marker=dict(color='rgb(37, 253, 233)', size=4, symbol='circle', line=dict( color='rgba(217, 217, 217, 0.14)',  width=1 ), opacity=0.8))
    fit = go.Surface( x = Xt_pred, y = Xx_pred, z = y_formatted , name = "Linear fit", opacity = 0.7, showscale = False, hoverinfo = 'none',)


    zoom_factor = 6.5
    camera = dict( up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0), eye=dict(x=0.25*zoom_factor, y=0.01*zoom_factor, z=0.02*zoom_factor))
    
    layout = go.Layout(
                        scene = dict(
                        xaxis = dict(title='Time (h)'),
                        yaxis = dict(title='Spatial position (layer)'),
                        zaxis = dict(title='Expression', type = 'linear'),
                        camera = camera,
                        aspectmode = 'manual',
                        aspectratio =dict(x = 1, y = 2, z = 0.5)
                        ),
                        title = '<b>Spatiotemporal representation</b>',
                        margin=dict(r=0, l=0, b=0, t=0),
                        #width=80%,
                        showlegend=False
                        )
    
    figure = go.Figure(data=[ fit, points_1, points_2, points_3], layout=layout)

    return figure

def compute_figure_3D_mpl(reg_2D, array_gene_time):
    selected, B, SE,  bic, l_schwartz, Xx_pred, Xt_pred, Y_pred, var_exp, var_exp_re = reg_2D
    y_formatted = np.zeros((80,40))
    for idx, val in enumerate(Y_pred):
        y_formatted[int(idx/40), int(idx%40)] = val

    #plot gene vs prediction
    x = [i for i in range(8) for j in range(4)]
    y = [j for i in range(8) for j in range(0,24,6)]
    z_1 = [array_gene_time[i][j] for i in range(8) for j in range(4)]
    z_2 = [array_gene_time[i][j] for i in range(8) for j in range(4,8)]
    array_gene_time = np.insert(array_gene_time, 9, np.nan, axis = 1)
    array_gene_time = np.insert(array_gene_time,11, np.nan, axis = 1)
    z_3 = [array_gene_time[i][j] for i in range(8) for j in range(8,12)]

    # Make the plot
    # Create light source object.
    ls = LightSource(azdeg=0, altdeg=65)
    fig = plt.figure(figsize = (20,10))
    ax = fig.gca(projection='3d')
    X, Y = np.meshgrid(Xt_pred, Xx_pred)
    ax.plot_surface(X, Y, y_formatted, cmap=plt.cm.YlOrRd, linewidth=0, edgecolor='none', alpha=0.8)
    
    ax.scatter(y, x, np.array(z_1), c="C1", alpha = 1)
    ax.scatter(y, x, np.array(z_2), c="C1", alpha = 1)
    ax.scatter(y, x, np.array(z_3), c="C1", alpha = 1)
    # Rotate it
    ax.view_init(30, 30)
    
    ax.set_ylabel("Layer")
    ax.set_xlabel("Time")
    ax.set_zlabel("Expression")
    plt.show()


# Then plot a few genes, with not trivial models.

# In[ ]:


from mpl_toolkits.mplot3d import Axes3D
import plotly.io as pio
import time
from matplotlib.colors import LightSource

idx = 0
for name_gene, reg in dic_reg.items():
    if name_gene=="aldh3a2":
        [selected, B, SE, bic, l_schwartz, Xx_pred, Xt_pred, Y_pred, var_exp, var_exp_re] = reg
        if len(selected)>1:
            idx+=1
            array_gene_time =np.concatenate( (dic_itz[name_gene]['rep1'], dic_itz[name_gene]['rep2'], dic_itz[name_gene]['rep3'][:,[0,2]]), axis = 1)
            print("Step 4: Plotting selected gene", flush=True)
            fig =compute_figure_3D_tab_3(reg, array_gene_time)
            iplot(fig)
            compute_figure_3D_mpl(reg, array_gene_time)


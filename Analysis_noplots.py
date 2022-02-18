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
print("Step 1: Dictionary", flush=True)
for key, val in dic_struc.items():
    for x in val:
        if x is not None:
            load_path = 'Datasets/Profiles/ZT'+x+'.mat'
            mat = scipy.io.loadmat(load_path)
        for name, data, SD in zip(mat['all_genes'], mat['MeanGeneExp'], mat['SE']):
            
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

set_names_kept_2 = set()

#scatter plot
fig, ax = plt.subplots(figsize=(5,5))
plt.scatter(l_exp_log, l_var, s=20, alpha = 1, color = '#34495e')

#add reference genes
flag_c = True
flag_z = True
flag_u = True
for exp, var,  name in zip(l_exp_log, l_var, l_names):
    if name in l_zonated:
        if flag_z:
            plt.plot(exp, var, markersize = '20', marker = '.', lw = 0, color = "#F37F30", label = 'Reference zonated gene')
            flag_z = False
        else:
            plt.plot(exp, var, markersize = '20', marker = '.', lw = 0,color = "#F37F30")
    elif name in l_circadian:
        if flag_c:
            plt.plot(exp, var, markersize = '20', marker = '.', lw = 0,color = "#2178B4", label = 'Reference rhythmic gene')
            flag_c = False
        else:
            plt.plot(exp, var, markersize = '20', marker = '.', lw = 0,color = "#2178B4")
    if exp>10**-5 and var<0.5:
        set_names_kept_2.add(name)    


plt.xlim([10**-7,10**-1])
plt.ylim([0,1])
#plt.xscale('log', basex=10)
plt.xlabel('Profile maximal expresion', fontsize=15)
plt.ylabel('Average relative replicates variance', fontsize=15)
plt.legend()
plt.axhline(0.5, xmin = 0.335, ls='--', color = "red", alpha = 0.8)
plt.axvline(10**-5, ymax = 0.5, ls='--', color = "red", alpha = 0.8)
plt.savefig('Output/Filtering_consistency.pdf')
plt.show()

print(len(set_names_kept_2), ' genes remaining after filtering on replicates consistency')

## 2. Preliminary exploration of the data

#Look at the expression in the dataset

#plot the histogram of expression
l_exp = [ invert_transform(np.nanmax(np.vstack((dic_itz[gene_name]['rep1'], dic_itz[gene_name]['rep2'],dic_itz[gene_name]['rep3'])))) for gene_name in dic_itz]
plt.hist(l_exp, bins=np.logspace(-8,-1, 50))

#Filter dataset 

dic_itz_clean = {}
for name in set_names_kept_2:
    if 'mup' not in name and 'pisd' not in name:
        dic_itz_clean[name] = dic_itz[name]
l_names = list(dic_itz_clean.keys())

        
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
    #print("Step 2.1 First Full design Matrix", flush=True)
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
    #print("Step 2.2 - Formula and Model selection done", flush=True)

    
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
        #print("Step 2 - 2D Regression done", flush=True)    
        return selected, B, SE, bic, l_schwartz, Xx_pred, Xt_pred, Y_pred, var_exp, var_exp_re
        
    else:
        #print("Step 2 - 2D Regression done predict=false", flush=True) 
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
    #print("compute")
    [name_gene, force_complete] = arg
    
    array_gene_time =np.concatenate( (dic_itz[name_gene]['rep1'], dic_itz[name_gene]['rep2'], dic_itz[name_gene]['rep3'][:,[0,2]]), axis = 1)
    selected, B, SE,  bic, l_schwartz, Xx_pred, Xt_pred, Y_pred, var_exp, var_exp_re = make_2D_regression(array_gene_time, predict = True, force_complete = force_complete)
    return [selected, B, SE, bic, l_schwartz, Xx_pred, Xt_pred, Y_pred, var_exp, var_exp_re]    

#l_names =  l_names[:20] ##kann angepasst werden für test 
dic_reg = {}
l_arg = [(x, False) for x in l_names]
n_cpu = 2   #muss angepasst werden 
warnings.simplefilter("ignore")
original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
signal.signal(signal.SIGINT, original_sigint_handler)
pool = Pool(n_cpu) 
  ##extra
try:
    #print("Step 2: Running Regression", flush=True)
    results = pool.map(compute_regressions_mp, l_arg) ##tqdm?
    #print("Step 2: Pooled Regression done", flush=True)
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
print("Step 4: Plotting selected gene", flush=True)
for name_gene, reg in dic_reg.items():
    if name_gene=="aldh3a2":
        [selected, B, SE, bic, l_schwartz, Xx_pred, Xt_pred, Y_pred, var_exp, var_exp_re] = reg
        if len(selected)>1:
            idx+=1
            array_gene_time =np.concatenate( (dic_itz[name_gene]['rep1'], dic_itz[name_gene]['rep2'], dic_itz[name_gene]['rep3'][:,[0,2]]), axis = 1)
            fig =compute_figure_3D_tab_3(reg, array_gene_time)
            iplot(fig)
            compute_figure_3D_mpl(reg, array_gene_time)


# ## 4. Classify genes

# First, create the functions needed for classification

# In[17]:


def return_var_angle_amplitude(dic_param):
    """"""""""""""" Compute amplitude and angle deviation of a given model """""""""""""""
    X = np.arange(8)
    l_a = dic_param['a0'] + dic_param['a1']*X + dic_param['a2']*0.5*(3*X**2-1)
    l_b = dic_param['b0'] + dic_param['b1']*X + dic_param['b2']*0.5*(3*X**2-1)
    l_r = np.sqrt(l_a**2+l_b**2)
    l_angle = np.arctan2(l_b, l_a )
    l_angle = [angle for angle in l_angle if angle!=0]
    std_r =np.std(l_r)
    std_angle = scipy.stats.circstd(l_angle)
    return std_r, std_angle*24/(2*np.pi)
    
def compute_dic_cluster(dic_reg):
    """"""""""""""" Compute clusters of genes """""""""""""""
    dic_cluster = {'F' :[], 'Z' : [], 'R' : [], 'Z+R' : [],  'ZxR' : []}
    for name_gene in dic_reg:
        selected, B, SE, bic,l_schwartz, Xx_pred, Xt_pred, Y_pred, var_exp, var_exp_re = dic_reg[name_gene]
        set_selected = set(selected)
        if len(selected)==1:
            dic_cluster['F'].append(  (name_gene, selected, B,  bic, var_exp, var_exp_re)  )

        elif set_selected == set(['mu0', 'a0','b0']) or set_selected ==set(['mu0', 'a0']) or set_selected == set(['mu0', 'b0']):
            dic_cluster['R'].append( (name_gene, selected, B, bic, var_exp, var_exp_re)  )

        elif 'a0' not in selected and 'a1' not in selected and 'a2' not in selected and 'b0' not in selected and 'b1' not in selected and 'b2' not in selected:
            if 'mu1' in selected or 'mu2' in selected or 'mu3' in selected:
                dic_cluster['Z'].append(  (name_gene, selected, B, bic, var_exp, var_exp_re)  )
            else:
                print(selected)
                print('BUG in compute dic cluster')
        else:
            if 'a1' not in selected and 'a2' not in selected and 'b1' not in selected and 'b2' not in selected:
                dic_cluster['Z+R'].append([name_gene, selected, B, bic, var_exp, var_exp_re])
            else:
                dic_param = {'a0' : 0, 'a1' : 0, 'a2' : 0, 'b0' : 0, 'b1' : 0, 'b2' : 0, 'mu0' : 0, 'mu1' : 0, 'mu2' : 0}
                for name_par, par in zip(selected, B):
                    dic_param[name_par] = par
                std_r, std_angle = return_var_angle_amplitude(dic_param)
                dic_cluster['ZxR'].append([name_gene, selected, B, bic, var_exp, var_exp_re, std_r, std_angle])
            
        #sort all list per variance explained
        dic_cluster['F'] = sorted(dic_cluster['F'], key=operator.itemgetter(4), reverse=True)
        dic_cluster['Z'] = sorted(dic_cluster['Z'], key=operator.itemgetter(4), reverse=True)
        dic_cluster['R'] = sorted(dic_cluster['R'], key=operator.itemgetter(4), reverse=True)
        dic_cluster['Z+R'] = sorted(dic_cluster['Z+R'], key=operator.itemgetter(4), reverse=True)
        dic_cluster['ZxR'] = sorted(dic_cluster['ZxR'], key=operator.itemgetter(4), reverse=True)

    return dic_cluster

dic_cluster = compute_dic_cluster(dic_reg)


# Define a function to plot the fits
# 

# In[18]:


cmap = matplotlib.cm.get_cmap('rainbow')
color = ['lightblue', 'coral','yellowgreen','pink']

def plot_gene_with_fit(name_gene, title = '', color = [cmap(x) for x in np.linspace(0,1,4,endpoint = True)], pp = None):
    #get model prediction
    selected, B, SE,  bic,  l_schwartz,Xx_pred, Xt_pred, Y_pred, var_exp, var_exp_re = dic_reg[name_gene]        
    y_formatted = np.zeros((80,40))
    for idx, val in enumerate(Y_pred):
        y_formatted[int(idx/40), int(idx%40)] = val
    
    #plot experimental points
    #plt.figure(figsize=(15,5))
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))
    for t in range(4):
        ax1.plot(np.linspace(0,7,8, endpoint = True), dic_itz[name_gene]['rep1'][:,t], 'o' ,   color = color[t], markersize=10)
        ax1.plot(np.linspace(0,7,8, endpoint = True), dic_itz[name_gene]['rep2'][:,t], 'v' ,  color = color[t], markersize=10)
        ax1.plot(np.linspace(0,7,8, endpoint = True), dic_itz[name_gene]['rep3'][:,t],  'p' , color = color[t], markersize=10)
    #plot fits
    for t in range(0,40,10):
        ax1.plot(np.linspace(0,7,80, endpoint = True), y_formatted[:,t], label = 't='+str(int(t*6/10)), color = color[int(t/10)], lw = 2)
    #ax.legend(loc='best')
    #Shrink current axis by 10%
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # Put a legend to the right of the current axis
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax1.set_xlim([-0.5,7.5])
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Log2-Expression")
    ax1.set_title(title + ' (log data + fit)')   
    
    color = [cmap(x) for x in np.linspace(0,1,8,endpoint = True)]
    for x in range(8):
        ax2.plot(np.linspace(0,18,4, endpoint = True), dic_itz[name_gene]['rep1'][x,:], 'o' ,   color = color[x], markersize=10)
        ax2.plot(np.linspace(0,18,4, endpoint = True), dic_itz[name_gene]['rep2'][x,:], 'v' ,  color = color[x], markersize=10)
        ax2.plot(np.linspace(0,18,4, endpoint = True), dic_itz[name_gene]['rep3'][x,:],  'p' , color = color[x], markersize=10)
        
    for x in range(0,80,10):
        ax2.plot(np.linspace(0,24,40, endpoint = True), y_formatted[x,:], label = 'x='+str(int(x/10)), color = color[int(x/10)], lw = 2)
        
    #Shrink current axis by 10%
    box = ax2.get_position()
    ax2.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # Put a legend to the right of the current axis
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax2.set_xlim([-0.5,24])
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Log2-Expression")
    ax2.set_title(title+ ' (log data + fit)')  
    

    if pp is not None:
        pp.savefig()
    else:
        plt.show()
    plt.close()
    
color1 = sn.color_palette("husl", 4) #['#3288BD', '#ABDDA4','#FDAE61','#D53E4F']
color2 = sn.color_palette("GnBu_d",8)#sn.color_palette("Reds",8)##['#D53E4F', '#F46D43','#FDAE61','#FEE08B', "#E6F598", "#ABDDA4", "#66C2A5", "#3288BD"]
color3 = sn.color_palette("husl", 24)
def plot_gene_with_fit_alt(name_gene, title = '', annotate_phase = False, force_R = 'False'):
    true_acrophase = None
    amplitude = None
    
    #get model prediction
    selected, B, SE,  bic,  l_schwartz,Xx_pred, Xt_pred, Y_pred, var_exp, var_exp_re = dic_reg[name_gene]
    if force_R:
        array_gene_time =np.concatenate( (dic_itz[name_gene]['rep1'], dic_itz[name_gene]['rep2'], dic_itz[name_gene]['rep3'][:,[0,2]]), axis = 1)
        selected, B, SE,  bic, l_schwartz, Xx_pred, Xt_pred, Y_pred, var_exp, var_exp_re = make_2D_regression(array_gene_time, predict = True, force_complete = False, force_R = True)        
        
    y_formatted = np.zeros((80,40))
    for idx, val in enumerate(Y_pred):
        y_formatted[int(idx/40), int(idx%40)] = val
    
    #plot experimental points
    #plt.figure(figsize=(15,5))
    f, (ax1, ax2) = plt.subplots(2, 1, figsize=(6,11))
    #f, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,10))
    for t in range(4):
        ax1.plot(np.linspace(1,8,8, endpoint = True), dic_itz[name_gene]['rep1'][:,t], '.' ,   color = color1[t], markersize=10)
        ax1.plot(np.linspace(1,8,8, endpoint = True), dic_itz[name_gene]['rep2'][:,t], '.' ,  color = color1[t], markersize=10)
        ax1.plot(np.linspace(1,8,8, endpoint = True), dic_itz[name_gene]['rep3'][:,t],  '.' , color = color1[t], markersize=10)
    #plot fits
    for t in range(0,40,10):
        ax1.plot(np.linspace(1,8,80, endpoint = True), y_formatted[:,t], label = 't='+str(int(t*6/10)), color = color1[int(t/10)], lw = 2)
    #ax.legend(loc='best')
    #Shrink current axis by 10%
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # Put a legend to the right of the current axis
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax1.set_xlim([0.25,8.25])
    ax1.set_xticks([1,2,3,4,5,6,7,8])
    ax1.set_xlabel("Layer", fontsize=15)
    ax1.set_ylabel("Log2-Expression", fontsize=15)
    ax1.set_title(title, fontsize=15) 
    
    for x in range(8):
        ax2.plot(np.linspace(0,18,4, endpoint = True), dic_itz[name_gene]['rep1'][x,:], '.' ,   color = color2[x], markersize=10)
        ax2.plot(np.linspace(0,18,4, endpoint = True), dic_itz[name_gene]['rep2'][x,:], '.' ,  color = color2[x], markersize=10)
        ax2.plot(np.linspace(0,18,4, endpoint = True), dic_itz[name_gene]['rep3'][x,:],  '.' , color = color2[x], markersize=10, label = 'x='+str(x+1) if annotate_phase else None)

    if not annotate_phase:
        for x in range(0,80,10):
            ax2.plot(np.linspace(0,24,40, endpoint = False), y_formatted[x,:], label = 'x='+str(int((x+10)/10)), color = color2[int(x/10)], lw = 2)
    else:
        for x in range(0,80,10):
            time_domain = np.linspace(0,24,40, endpoint = False)
            true_acrophase = time_domain[np.argmax(y_formatted[x,:])]
            acrophase = int(round(time_domain[np.argmax(y_formatted[x,:])]))
            max_val = np.max(y_formatted[x,:])
            min_val = np.min(y_formatted[x,:])
            amplitude = (max_val-min_val)/2
            text_pos = (acrophase+12)%24
            if text_pos>18:
                text_pos = 18
            elif text_pos <2:
                text_pos = 2
            ax2.plot(time_domain, y_formatted[x,:], color = color3[acrophase], lw = 2)
            ax2.annotate(r'$\phi: $'+ str(acrophase), (text_pos, max_val*1.1), fontsize = 20)
            break
        
    #Shrink current axis by 10%
    box = ax2.get_position()
    ax2.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # Put a legend to the right of the current axis
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax2.set_xlim([-2,24])
    ax2.set_xticks([0,6,12,18])
    ax2.set_xlabel("ZT", fontsize=15)
    ax2.set_ylabel("Log2-Expression", fontsize=15)
    #ax2.set_title(title+ ' (log data + fit)') 
    #plt.tight_layout()
    plt.savefig("Output/Fits/"+title+".pdf")
    plt.show()
    plt.close()
    
    return amplitude, true_acrophase


# Plot some examples of each category

# In[19]:


#get example of flat gene
best_flat = dic_cluster['F'][0][0]
plot_gene_with_fit(best_flat, title = 'Example fit for flat gene :' + best_flat, color = color)

#get example of zonated gene
best_zonated = dic_cluster['Z'][1][0]
plot_gene_with_fit(best_zonated, title = 'Example of fit for zonated gene:' + best_zonated, color = color)

#get example of rhythmic gene
best_rhythmic = dic_cluster['R'][2][0]
plot_gene_with_fit(best_rhythmic, title = 'Example of fit for rhythmic gene:' + best_rhythmic, color = color)

#get example of zonated-rhythmic gene
best_independant = dic_cluster['Z+R'][0][0]
plot_gene_with_fit(best_independant, title = 'Example of fit for independant gene:' + best_independant, color = color)

#get example of zonated-rhythmic gene
best_zxr = dic_cluster['ZxR'][0][0]
plot_gene_with_fit(best_zxr, title = 'Example of fit forinteracting gene:' + best_zxr, color = color)

#plot clock genes
plot_gene_with_fit_alt("elovl3", title = "Elovl3", annotate_phase = False)

plot_gene_with_fit_alt("cry1", title = "Cry1", annotate_phase = False)

plot_gene_with_fit_alt("uox", title = "Uox", annotate_phase = False)

plot_gene_with_fit_alt("per1", title = "Per1", annotate_phase = False)

plot_gene_with_fit_alt("pck1", title = "pck1", annotate_phase = False)


#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import warnings

warnings.filterwarnings('ignore')

import dynamo as dyn

filename = './data/rpe1.h5ad'

rpe1 = dyn.read(filename)

dyn.convert2float(rpe1, ['Cell_cycle_possition', 'Cell_cycle_relativePos'])

rpe1.obs.exp_type.value_counts()

rpe1[rpe1.obs.exp_type == 'Chase', :].obs.time.value_counts()

rpe1[rpe1.obs.exp_type == 'Pulse', :].obs.time.value_counts()

rpe1_kinetics = rpe1[rpe1.obs.exp_type == 'Pulse', :]
rpe1_kinetics.obs['time'] = rpe1_kinetics.obs['time'].astype(str)
rpe1_kinetics.obs.loc[rpe1_kinetics.obs['time'] == 'dmso', 'time'] = -1
rpe1_kinetics.obs['time'] = rpe1_kinetics.obs['time'].astype(float)
rpe1_kinetics = rpe1_kinetics[rpe1_kinetics.obs.time != -1, :]

rpe1_genes = ['UNG', 'PCNA', 'PLK1', 'HPRT1']

rpe1_kinetics.obs.time = rpe1_kinetics.obs.time.astype('float')
rpe1_kinetics.obs.time = rpe1_kinetics.obs.time / 60  # convert minutes to hours

print(rpe1_kinetics.obs.time.value_counts())

# velocity
dyn.tl.recipe_kin_data(adata=rpe1_kinetics,
                       keep_filtered_genes=True,
                       keep_raw_layers=True,
                       del_2nd_moments=False,
                       tkey='time',
                       n_top_genes=1000,
                       est_method='CSP4ML_ICSP',
                       )

rpe1_kinetics.obsm['X_RFP_GFP'] = rpe1_kinetics.obs.loc[:,
                                  ['RFP_log10_corrected', 'GFP_log10_corrected']].values.astype('float')

# total velocity
dyn.tl.reduceDimension(rpe1_kinetics, reduction_method='umap')
dyn.tl.cell_velocities(rpe1_kinetics, enforce=True, vkey='velocity_T', ekey='M_t', basis='RFP_GFP')
dyn.pl.streamline_plot(rpe1_kinetics, color=['cell_cycle_phase'], basis='RFP_GFP')

# # spliced RNA velocity
# dyn.tl.reduceDimension(rpe1_kinetics, reduction_method='umap')
# dyn.tl.cell_velocities(rpe1_kinetics, enforce=True, vkey='velocity_S', ekey='M_s', basis='RFP_GFP')
# dyn.pl.streamline_plot(rpe1_kinetics, color=['cell_cycle_phase'], basis='RFP_GFP')

# dyn.configuration.set_figure_params(fontsize=6)
# genes = ['HMGA2']
# dyn.pl.phase_portraits(rpe1_kinetics, genes=genes, color='cell_cycle_phase', basis='RFP_GFP', vkey='velocity_T',
#                        ekey='M_t', show_arrowed_spines=False, show_quiver=True, quiver_size=5, figsize=(6*0.53, 4*0.53))
# genes = ['DCBLD2']
# dyn.pl.phase_portraits(rpe1_kinetics, genes=genes, color='cell_cycle_phase', basis='RFP_GFP', vkey='velocity_T',
#                        ekey='M_t', show_arrowed_spines=False, show_quiver=True, quiver_size=5, figsize=(6*0.53, 4*0.53))
# genes = ['HIPK2']
# dyn.pl.phase_portraits(rpe1_kinetics, genes=genes, color='cell_cycle_phase', basis='RFP_GFP', vkey='velocity_T',
#                        ekey='M_t', show_arrowed_spines=False, show_quiver=True, quiver_size=5, figsize=(6*0.53, 4*0.53))


import pylab

#SETUP PLOTTING PARAMETERS
fig_width_pt = 500.0  # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.25              # Convert pt to inches
golden_mean = (pylab.sqrt(5)-1.0)/2.0         # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # width in inches
fig_height =fig_width*golden_mean       # height in inches
fig_size = [fig_width,fig_height]
legend_padding = 0.05
params = {
	'backend': 'ps',
	'ps.usedistiller': 'xpdf',
	'font.family'  : 'serif',#'sans-serif',
	'font.style'   : 'normal',
	'font.variant' : 'normal',
	'font.weight'  : 'normal', #bold
	'font.stretch' : 'normal',
	#'font.size'    :  'large', #large,'normal'
	'axes.labelsize': 16,
	'text.fontsize': 14,
	'title.fontsize':12,
	'legend.fontsize':14,
	'xtick.labelsize': 16,
	'ytick.labelsize': 16,
	'lines.markersize':8,
	'text.usetex': True,
	'figure.figsize': fig_size
}
#pylab.rcParams.update(params) 

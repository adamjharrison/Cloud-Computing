import os
import awkward as ak  # to represent nested data in columnar format
from matplotlib.ticker import AutoMinorLocator  # for minor ticks
import matplotlib.pyplot as plt  # for plotting
import numpy as np  # for numerical calculations such as histogramming
import pika # pyright: ignore[reportMissingModuleSource]
import json
import atlasopenmagic as atom
from atlasopenmagic import install_from_environment

install_from_environment()
atom.set_release('2025e-13tev-beta')
os.environ["ATOM_CACHE"] = "/data"

MeV = 0.001
GeV = 1.0
xmin = 80 * GeV
xmax = 250 * GeV
step_size = 2.5 * GeV
lumi = 36.6
fraction=1.0

bin_edges = np.arange(start=xmin,  # The interval includes this value
                      stop=xmax+step_size,  # The interval doesn't include this value
                      step=step_size)  # Spacing between values
bin_centres = np.arange(start=xmin+step_size/2,  # The interval includes this value
                        stop=xmax+step_size/2,  # The interval doesn't include this value
                        step=step_size)  # Spacing between values

skim = "exactly4lep"
# Define empty dictionary to hold awkward arrays
all_data = {}

defs = {
    r'Data': {'dids': ['data']},
    r'Background $Z,t\bar{t},t\bar{t}+V,VVV$': {'dids': [410470, 410155, 410218,
                                                         410219, 412043, 364243,
                                                         364242, 364246, 364248,
                                                         700320, 700321, 700322,
                                                         700323, 700324, 700325], 'color': "#6b59d3"},  # purple
    r'Background $ZZ^{*}$':     {'dids': [700600], 'color': "#ff0000"},  # red
    r'Signal ($m_H$ = 125 GeV)':  {'dids': [345060, 346228, 346310, 346311, 346312,
                                            346340, 346341, 346342], 'color': "#00cdff"},  # light blue
}

samples = atom.build_dataset(defs, skim=skim, protocol='https', cache=True)

params = pika.ConnectionParameters('rabbitmq',heartbeat=600,port=5672)

connection = pika.BlockingConnection(params)
channel = connection.channel()

channel.queue_declare(queue='file_index',durable=True)
channel.queue_declare(queue='data',durable=True)

def process(ch, method, properties, body):
    global recv,count
    print("Received processed file")
    message = json.loads(body.decode())
    s = message['s']
    hist = np.array(message['hist'])
    weights = np.array(message['weights'])
    all_data[s][0].append(hist)
    all_data[s][1].append(weights)
    recv+=1 
    ch.basic_ack(delivery_tag = method.delivery_tag)
    if count==recv:
        print(f'Recieved all data')  
        channel.stop_consuming()

# Loop over samples
count = 0
recv = 0
all_data = {}
for s in samples:

    # Print which sample is being processed
    print('Processing '+s+' samples')

    # Define empty list to hold data
    all_data[s] = [[],[]]
    # Loop over each file
    for val in samples[s]['list']:
        message = json.dumps({"val":val,"s":s})
        channel.basic_publish(exchange='',
                        routing_key='file_index',
                        body=message,
                        properties=pika.BasicProperties(
                        delivery_mode = pika.DeliveryMode.Persistent))
        count+=1
print('Sent all data')

channel.basic_consume(queue='data', on_message_callback=process)
channel.start_consuming()
connection.close()

mc_x = [] # define list to hold the Monte Carlo histogram entries
mc_weights = [] # define list to hold the Monte Carlo weights
mc_colors = [] # define list to hold the colors of the Monte Carlo bars
mc_labels = [] # define list to hold the legend labels of the Monte Carlo bars

for s in samples:
    if s == 'Data':
        data_x,_ = np.histogram(np.concatenate(all_data[s][0]),bins=bin_edges)
        data_x_errors = np.sqrt( data_x )
    elif s == r'Signal ($m_H$ = 125 GeV)':
        signal_x = np.concatenate(all_data[s][0])
        signal_color = samples[r'Signal ($m_H$ = 125 GeV)']['color']
        signal_weights = np.concatenate(all_data[s][1])
    else:
        mc_x.append(np.concatenate(all_data[s][0])) # append to the list of Monte Carlo histogram entries
        mc_weights.append(np.concatenate(all_data[s][1])) # append to the list of Monte Carlo weights
        mc_colors.append(samples[s]['color']) # append to the list of Monte Carlo bar colors
        mc_labels.append(s) # append to the list of Monte Carlo legend labels
# *************
# Main plot
# *************
fig, main_axes = plt.subplots(figsize=(12, 8))

# plot the data points
main_axes.errorbar(x=bin_centres, y=data_x, yerr=data_x_errors,
                   fmt='ko',  # 'k' means black and 'o' is for circles
                   label='Data')

# plot the Monte Carlo bars
mc_heights = main_axes.hist(mc_x, bins=bin_edges,
                            weights=mc_weights, stacked=True,
                            color=mc_colors, label=mc_labels)

mc_x_tot = mc_heights[0][-1]  # stacked background MC y-axis value

# calculate MC statistical uncertainty: sqrt(sum w^2)
mc_x_err = np.sqrt(np.histogram(
    np.hstack(mc_x), bins=bin_edges, weights=np.hstack(mc_weights)**2)[0])

# plot the signal bar
signal_heights = main_axes.hist(signal_x, bins=bin_edges, bottom=mc_x_tot,
                                weights=signal_weights, color=signal_color,
                                label=r'Signal ($m_H$ = 125 GeV)')

# plot the statistical uncertainty
main_axes.bar(bin_centres,  # x
              2*mc_x_err,  # heights
              alpha=0.5,  # half transparency
              bottom=mc_x_tot-mc_x_err, color='none',
              hatch="////", width=step_size, label='Stat. Unc.')

# set the x-limit of the main axes
main_axes.set_xlim(left=xmin, right=xmax)

# separation of x axis minor ticks
main_axes.xaxis.set_minor_locator(AutoMinorLocator())

# set the axis tick parameters for the main axes
main_axes.tick_params(which='both',  # ticks on both x and y axes
                      direction='in',  # Put ticks inside and outside the axes
                      top=True,  # draw ticks on the top axis
                      right=True)  # draw ticks on right axis

# x-axis label
main_axes.set_xlabel(r'4-lepton invariant mass $\mathrm{m_{4l}}$ [GeV]',
                     fontsize=13, x=1, horizontalalignment='right')

# write y-axis label for main axes
main_axes.set_ylabel('Events / '+str(step_size)+' GeV',
                     y=1, horizontalalignment='right')

# set y-axis limits for main axes
main_axes.set_ylim(bottom=0, top=np.amax(data_x)*2.0)

# add minor ticks on y-axis for main axes
main_axes.yaxis.set_minor_locator(AutoMinorLocator())

# Add text 'ATLAS Open Data' on plot
plt.text(0.1,  # x
         0.93,  # y
         'ATLAS Open Data',  # text
         transform=main_axes.transAxes,  # coordinate system used is that of main_axes
         fontsize=16)

# Add text 'for education' on plot
plt.text(0.1,  # x
         0.88,  # y
         'for education',  # text
         transform=main_axes.transAxes,  # coordinate system used is that of main_axes
         style='italic',
         fontsize=12)

# Add energy and luminosity
lumi_used = str(lumi*fraction)  # luminosity to write on the plot
plt.text(0.1,  # x
         0.82,  # y
         r'$\sqrt{s}$=13 TeV,$\int$L dt = '+lumi_used+' fb$^{-1}$',  # text
         # coordinate system used is that of main_axes
         transform=main_axes.transAxes, fontsize=16)

# Add a label for the analysis carried out
plt.text(0.1,  # x
         0.76,  # y
         r'$H \rightarrow ZZ^* \rightarrow 4\ell$',  # text
         # coordinate system used is that of main_axes
         transform=main_axes.transAxes, fontsize=16)

# draw the legend
# no box around the legend
my_legend = main_axes.legend(frameon=False, fontsize=16)
plt.savefig('/plots/plot.png')

# Signal stacked height
signal_tot = signal_heights[0] + mc_x_tot

# Peak of signal
print(signal_tot[18])

# Neighbouring bins
print(signal_tot[17:20])

# Signal and background events
N_sig = signal_tot[17:20].sum()
N_bg = mc_x_tot[17:20].sum()

# Signal significance calculation
signal_significance = N_sig/np.sqrt(N_bg + 0.3 * N_bg**2)  # EXPLAIN THE 0.3
print(f"\nResults:\n{N_sig=}\n{N_bg=}\n{signal_significance=}\n")
import os
import numpy as np  # for numerical calculations such as histogramming
import pika # pyright: ignore[reportMissingModuleSource]
import json
import atlasopenmagic as atom

#Sets the cached directory and release
dir = "/data/openatlas"
os.environ["ATOM_CACHE"] = dir

#installs data to cache
from atlasopenmagic import install_from_environment
install_from_environment()
atom.set_release('2025e-13tev-beta')

#constants
GeV = 1.0
xmin = 80 * GeV
xmax = 250 * GeV
step_size = 2.5 * GeV
fraction=1.0

bin_edges = np.arange(start=xmin,  # The interval includes this value
                      stop=xmax+step_size,  # The interval doesn't include this value
                      step=step_size)  # Spacing between values
bin_centres = np.arange(start=xmin+step_size/2,  # The interval includes this value
                        stop=xmax+step_size/2,  # The interval doesn't include this value
                        step=step_size)  # Spacing between values

skim = "exactly4lep"

#definition of samples
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

#connects to RabbitMQ Service
params = pika.ConnectionParameters('rabbitmq',heartbeat=600,port=5672)
while True: #loop to wait until RabbitMQ Service starts
    connected = False
    try:
        connection = pika.BlockingConnection(params)
        channel = connection.channel()
        connected = True
    except pika.exceptions.AMQPConnectionError:
        print("waiting for RabbitMQ to start")
    if connected:
        break

#declare queues to connect to
channel.queue_declare(queue='file_index',durable=True)

# Loop over samples
for s in samples:

    # Print which sample is being processed
    print('Processing '+s+' samples')

    # Loop over each file
    for val in samples[s]['list']:#send the file index to the file_index queue
        message = json.dumps({"val":val,"s":s})
        channel.basic_publish(exchange='',
                        routing_key='file_index',
                        body=message,
                        properties=pika.BasicProperties(
                        delivery_mode = pika.DeliveryMode.Persistent))
print('Sent all data')
import vector  # for 4-momentum calculations
import awkward as ak  # to represent nested data in columnar format
import pika
import time
import uproot
import json
import atlasopenmagic as atom

from atlasopenmagic import install_from_environment
install_from_environment()


atom.available_releases()
atom.set_release('2025e-13tev-beta')
lumi = 36.6

variables = ['lep_pt', 'lep_eta', 'lep_phi', 'lep_e', 'lep_charge', 'lep_type', 'trigE', 'trigM', 'lep_isTrigMatched',
             'lep_isLooseID', 'lep_isMediumID', 'lep_isLooseIso', 'lep_type']

# Cut lepton type (electron type is 11,  muon type is 13)
# reduce this is if you want quicker runtime (implemented in the loop over the tree)
fraction = 1.0

def cut_lep_type(lep_type):
    sum_lep_type = lep_type[:, 0] + lep_type[:, 1] + \
        lep_type[:, 2] + lep_type[:, 3]
    lep_type_cut_bool = (sum_lep_type != 44) & (
        sum_lep_type != 48) & (sum_lep_type != 52)
    # True means we should remove this entry (lepton type does not match)
    return lep_type_cut_bool

# Cut lepton charge


def cut_lep_charge(lep_charge):
    # first lepton in each event is [:, 0], 2nd lepton is [:, 1] etc
    sum_lep_charge = lep_charge[:, 0] + lep_charge[:,
                                                   1] + lep_charge[:, 2] + lep_charge[:, 3] != 0
    # True means we should remove this entry (sum of lepton charges is not equal to 0)
    return sum_lep_charge

# Calculate invariant mass of the 4-lepton state
# [:, i] selects the i-th lepton in each event


def calc_mass(lep_pt, lep_eta, lep_phi, lep_e):
    p4 = vector.zip({"pt": lep_pt, "eta": lep_eta, "phi": lep_phi, "E": lep_e})
    # .M calculates the invariant mass
    invariant_mass = (p4[:, 0] + p4[:, 1] + p4[:, 2] + p4[:, 3]).M
    return invariant_mass


def cut_trig_match(lep_trigmatch):
    trigmatch = lep_trigmatch
    cut1 = ak.sum(trigmatch, axis=1) >= 1
    return cut1


def cut_trig(trigE, trigM):
    return trigE | trigM


def ID_iso_cut(IDel, IDmu, isoel, isomu, pid):
    thispid = pid
    return (ak.sum(((thispid == 13) & IDmu & isomu) | ((thispid == 11) & IDel & isoel), axis=1) == 4)


weight_variables = ["filteff", "kfac", "xsec", "mcWeight", "ScaleFactor_PILEUP",
                    "ScaleFactor_ELE", "ScaleFactor_MUON", "ScaleFactor_LepTRIGGER"]


def calc_weight(weight_variables, events):
    total_weight = lumi * 1000 / events["sum_of_weights"]
    for variable in weight_variables:
        total_weight = total_weight * abs(events[variable])
    return total_weight

def splice(val,s):
    fileString = val

    # start the clock
    start = time.time()
    print("\t"+val+":")

    # Open file
    tree = uproot.open(fileString + ":analysis")

    sample_data = []

    # Loop over data in the tree
    for data in tree.iterate(variables + weight_variables + ["sum_of_weights", "lep_n"],
                                library="ak",
                                entry_stop=tree.num_entries*fraction,#):  # , # process up to numevents*fraction
                                step_size = 10000000):

        # Number of events in this batch
        nIn = len(data)

        data = data[cut_trig(data.trigE, data.trigM)]
        data = data[cut_trig_match(data.lep_isTrigMatched)]

        # Record transverse momenta (see bonus activity for explanation)
        data['leading_lep_pt'] = data['lep_pt'][:, 0]
        data['sub_leading_lep_pt'] = data['lep_pt'][:, 1]
        data['third_leading_lep_pt'] = data['lep_pt'][:, 2]
        data['last_lep_pt'] = data['lep_pt'][:, 3]

        # Cuts on transverse momentum
        data = data[data['leading_lep_pt'] > 20]
        data = data[data['sub_leading_lep_pt'] > 15]
        data = data[data['third_leading_lep_pt'] > 10]

        data = data[ID_iso_cut(data.lep_isLooseID,
                                data.lep_isMediumID,
                                data.lep_isLooseIso,
                                data.lep_isLooseIso,
                                data.lep_type)]

        # Number Cuts
        # data = data[data['lep_n'] == 4]

        # Lepton cuts

        lep_type = data['lep_type']
        data = data[~cut_lep_type(lep_type)]
        lep_charge = data['lep_charge']
        data = data[~cut_lep_charge(lep_charge)]

        # Invariant Mass
        data['mass'] = calc_mass(
            data['lep_pt'], data['lep_eta'], data['lep_phi'], data['lep_e'])

        # Store Monte Carlo weights in the data
        if 'data' not in s:  # Only calculates weights if the data is MC
            data['totalWeight'] = calc_weight(weight_variables, data)
            # data['totalWeight'] = calc_weight(data)
        procdata = json.dumps({"s":s,"cid":i,"chunk":ak.to_json(data)})
        channel.basic_publish(exchange='',
                        routing_key='frame',
                        body=procdata,
                        properties=pika.BasicProperties(
                        delivery_mode = pika.DeliveryMode.Persistent))

        if not 'data' in val:
            # sum of weights passing cuts in this batch
            nOut = sum(data['totalWeight'])
        else:
            nOut = len(data)

        elapsed = time.time() - start  # time taken to process
        print("\t\t chunk"+str(i)+" nIn: "+str(nIn)+",\t nOut: \t"+str(nOut)+"\t in " +
                str(round(elapsed, 1))+"s")  # events before and after

params = pika.ConnectionParameters(
    'rabbitmq',
    heartbeat=0)

connection = pika.BlockingConnection(params)
channel = connection.channel()

channel.queue_declare(queue='file index', durable=True)
channel.queue_declare(queue='frame',durable=True)
def callback(ch, method, properties, body):
    print(f" [x] Received {body}")
    message = json.loads(body)
    val = message['val']
    s = message['s']
    splice(val,s)
    print("Processed data")
    procdata = json.dumps({"s":s,"frame":ak.to_json(frame)})
    ch.basic_publish(exchange='',
                        routing_key='frame',
                        body=procdata,
                        properties=pika.BasicProperties(
                        delivery_mode = pika.DeliveryMode.Persistent))
    print('Sent back')
    ch.basic_ack(delivery_tag = method.delivery_tag)
    
channel.basic_qos(prefetch_count=1)
# setup to listen for messages on queue 'messages'
channel.basic_consume(queue='file index', on_message_callback=callback)

print('Waiting for messages. To exit press CTRL+C')

# start listening
channel.start_consuming()
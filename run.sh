#!/bin/bash

#install keda
kubectl apply --server-side -f https://github.com/kedacore/keda/releases/download/v2.18.0/keda-2.18.0.yaml

#deploy
kubectl apply -f deployment/.

#waits for main job to output plot
kubectl wait --for=condition=complete --timeout=30m job/main

#cleanup
./cleanup.sh
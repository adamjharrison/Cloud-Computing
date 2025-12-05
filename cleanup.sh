#!/bin/bash

kubectl delete services --all
kubectl delete deployments --all
kubectl delete jobs --all
kubectl delete scaledobject worker
kubectl delete services --all -n keda
kubectl delete deployments --all -n keda
kubectl delete pvc --all
kubectl delete pv --all
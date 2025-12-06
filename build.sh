#!/bin/bash
docker image build -t adamjharrison/worker worker/.
docker image build -t adamjharrison/distributor distributor/.
docker image build -t adamjharrison/combiner combiner/.
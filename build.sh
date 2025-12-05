#!/bin/bash
docker image build -t adamjharrison/worker worker/.
docker image build -t adamjharrison/main main/.
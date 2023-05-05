#!/bin/bash
# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =======================================================================================


export DEVICE_NUM=8
export RANK_SIZE=8
export HCCL_WHITELIST_DISABLE=1
export GRAPH_OP_RUN=1
rm -rf ./train_parallel
mkdir ./train_parallel
cp -r ../deepxml ./train_parallel
cp -r ../*.py ./train_parallel
cp -r ../configure ./train_parallel
cd ./train_parallel || exit

mpirun --allow-run-as-root -n $RANK_SIZE --output-filename log_output --merge-stderr-to-stdout \
    python ../../train.py --is_distributed True > logs.txt 2>&1 &
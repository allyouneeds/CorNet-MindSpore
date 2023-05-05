#!/usr/bin/env bash
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

DATA_PATH=/p/realai/guangxu/2020xmtc/deep_data

DATASET=EUR-Lex
#DATASET=AmazonCat-13K
#DATASET=Wiki-500K

MODEL=XMLCNN
#MODEL=CorNetXMLCNN
#MODEL=BertXML
#MODEL=CorNetBertXML
#MODEL=MeSHProbeNet
#MODEL=CorNetMeSHProbeNet
#MODEL=AttentionXML
#MODEL=CorNetAttentionXML

python train.py --data-cnf configure/datasets/$DATASET.yaml --model-cnf configure/models/$MODEL-$DATASET.yaml

python evaluation.py \
--results $DATA_PATH/$DATASET/results/$MODEL-$DATASET-labels.npy \
--targets $DATA_PATH/$DATASET/test_labels.npy \
--train-labels $DATA_PATH/$DATASET/train_labels.npy

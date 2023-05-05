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

python ../deepxml/preprocess.py \
--text-path $DATA_PATH/$DATASET/train_texts.txt \
--label-path $DATA_PATH/$DATASET/train_labels.txt \
--vocab-path $DATA_PATH/$DATASET/vocab.npy \
--emb-path $DATA_PATH/$DATASET/emb_init.npy \
--w2v-model $DATA_PATH/glove.840B.300d.gensim

python ../deepxml/preprocess.py \
--text-path $DATA_PATH/$DATASET/test_texts.txt \
--label-path $DATA_PATH/$DATASET/test_labels.txt \
--vocab-path $DATA_PATH/$DATASET/vocab.npy

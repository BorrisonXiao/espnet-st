#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# bash /alt-arabic/speech/amir/competitions/IWSLT/brian/espnet-ml/tools/extra_path.sh

src_lang=ta
tgt_lang=en

train_set=train
train_dev=dev
test_set=test1

st_config=conf/config_mbart_fairseq.yaml
inference_config=conf/tuning/decode_st_md_ctc0.3_mbart.yaml
src_nbpe=2000

# tc: truecase
# lc: lowercase
# lc.rm: lowercase with punctuation removal
src_case=tc.rm
tgt_case=tc

./st.sh \
	--st_stats_dir exp_mbart/st_stats_raw_ta_en_fairseq \
	--expdir exp_mbart \
	--dumpdir dump \
	--num_nodes 1 \
	--datadir data8KHz \
	--bpe_predefined true \
    --ngpu 1 \
    --local_data_opts "${tgt_lang}" \
    --audio_format "flac.ark" \
	--ignore_init_mismatch true \
	--stage 10 \
    --stop_stage 10\
	--fs 8k \
    --nj 16 \
	--feats_normalize utterance_mvn \
    --inference_nj 64 \
    --audio_format "flac.ark" \
    --src_lang ${src_lang} \
    --tgt_lang ${tgt_lang} \
    --src_token_type "bpe" \
    --src_nbpe $src_nbpe \
    --src_case ${src_case} \
    --tgt_case ${tgt_case} \
    --feats_type raw \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --st_config "${st_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${train_dev}" \
	--tgt_token_type fairseq \
    --test_sets "${test_set} ${train_dev}" \
    --src_bpe_train_text "data/${train_set}/text.${src_case}.${src_lang}" \
    --tgt_bpe_train_text "data/${train_set}/text.${tgt_case}.${tgt_lang}" \
	--lm_train_text "data/${train_set}/text.${src_case}.${src_lang}"  "$@"
	#--use_src_lm true \
	#--src_lm_exp /alt-arabic/speech/amir/competitions/IWSLT/asr_tuned/exp/lm_rnn_lm_bpe1000 \
	# --tgt_token_type hugging_face \
	#--hugging_face_model_name_or_path /alt-arabic/speech/amir/competitions/IWSLT/mbart/mbart-large-50-one-to-many-mmt \
    #--lm_train_text "data/${train_set}/text.${tgt_case}.${tgt_lang}"  "$@"

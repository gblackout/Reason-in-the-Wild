# run standalone script
# <PATH TO DATASET> path to the standalone data, for example "data/standalone_test.json"
# <NAME_OF_THE_API_MODEL> the name of the api llm, should be one of the name stored in LLMRequestManager
# <PATH_TO_SAVE_TRAJS> path to the saved file, for example "logs/llm_standalone_eval.json"
python traj_infer.py \
    --datapath <PATH TO DATASET> \
    --base_model <NAME_OF_THE_API_MODEL> \
    --is_local False \
    --save_key "traj_eval" \
    --save_path <PATH_TO_SAVE_TRAJS> \
    --hyb_run False \
    --use_icl \
    --add_boseos_to_icl_prompt False \
    --skip_done \
    --use_icl_before_round 2 \
    --save_every 10 \
    --use_baby_prompt_after_icl

# uncomment below to run hyb setting
## run hyb script
## <PATH TO DATASET> path to the hybrid data, for example "data/hybrid_test.json"
## don't change use_icl_before_round 100, this is for routing traj to always use the icl prompts with routing examples
#python traj_infer.py \
#    --datapath <PATH TO DATASET> \
#    --base_model <NAME_OF_THE_API_MODEL> \
#    --is_local False \
#    --save_key "traj_eval" \
#    --save_path <PATH_TO_SAVE_TRAJS> \
#    --hyb_run True \
#    --use_icl \
#    --add_boseos_to_icl_prompt False \
#    --skip_done \
#    --use_icl_before_round 100 \
#    --save_every 10 \
#    --use_baby_prompt_after_icl
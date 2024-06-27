# tiger standalone infer
# <PATH TO DATASET> path to the standalone data, for example "data/standalone_test.json"
# <PATH TO BASE MODEL> the path to the base LLaMA-8B model
# <PATH TO PEFT> the path to the peft adapter
# <PATH TO SAVE> the path to the saved file, for example "logs/tiger_standalone_eval.json"
python traj_infer.py \
    --datapath <PATH TO DATASET> \
    --base_model <PATH TO BASE MODEL> \
    --peft_path <PATH TO PEFT> \
    --is_local \
    --save_key "traj_eval" \
    --save_path <PATH TO SAVE> \
    --hyb_run False \
    --use_icl False \
    --add_boseos_to_icl_prompt False \
    --skip_done \
    --use_icl_before_round 100 \
    --save_every 10

# uncomment below to do inference on hybrid dataset
## tiger hybrid infer
## <PATH TO DATASET> path to the hybrid data, for example "data/hybrid_test.json"
## <PATH TO ROUT PEFT> the path to the router peft adapter
#python traj_infer.py \
#    --datapath <PATH TO DATASET> \
#    --base_model <PATH TO BASE MODEL> \
#    --peft_path <PATH TO PEFT> \
#    --router_peft_path <PATH TO ROUT PEFT> \
#    --is_local \
#    --save_key "traj_eval" \
#    --save_path <PATH TO SAVE> \
#    --hyb_run \
#    --use_icl False \
#    --add_boseos_to_icl_prompt False \
#    --skip_done \
#    --use_icl_before_round 100 \
#    --save_every 10
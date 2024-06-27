from tactic_and_execution.traj_utils import *
import re
from functools import partial
from metrics.calc_code_bleu import compute_one, CodeBLEUOutput
import numpy as np
from utils import *
import evaluate
textbleu = evaluate.load("bleu")


def compute_bleu(pred_seq: str, true_seq: str):
    res = textbleu.compute(predictions=[pred_seq], references=[[true_seq]])
    return res['bleu']


def get_clean_program(pstr):
    res = split_string_with_separators(pstr, ['```python', '```'])
    code_str = res[0][-1]

    if code_str is None:
        return None

    code_str = code_str.strip()

    parts = code_str.split('\n')
    nocomment_parts = [e.split('#')[0] for e in parts]
    nocomment_parts = [e for e in nocomment_parts if (e != '') and (not e.isspace())]
    prog = '\n'.join(nocomment_parts)

    return prog


def compare_prorams(pred_prog, gt_prog):
    zero_score = CodeBLEUOutput(0, .0, .0, .0, .15, .15, .35, .35, score=.0)

    if gt_prog is None:
        return None
    if pred_prog is None:
        return zero_score

    pred_prog, gt_prog = get_clean_program(pred_prog), get_clean_program(gt_prog)

    if gt_prog is None:
        return None
    if pred_prog is None:
        return zero_score

    return compute_one(pred_prog, gt_prog, weights=[0.15, 0.15, 0.35, 0.35])


def calc_standalone_metric(samples, runner, traj_key='traj_eval', gt_traj_key='gt_traj', bleu_thres=0.15):
    right_dict = defaultdict(int)
    error_dict = defaultdict(lambda: defaultdict(int))
    prog_dict = defaultdict(list)
    aux_dict = defaultdict(lambda: defaultdict(int))

    def check_ans_valid(exp):
        valid_ans = True
        if all_exists(exp.observation) and len(exp.observation) > 0:
            for fb in exp.observation:
                if ('invalid answer' in fb.content) or ('invalid response format' in fb.content):
                    valid_ans = False
                    break
        return valid_ans

    for ind, sample in enumerate(samples):
        gt_traj = Trajectory.from_json(sample[gt_traj_key])
        traj = Trajectory.from_json(sample[traj_key])

        # ============================== eval program ==============================
        gt_p = get_program(
            gt_traj,
            runner.datasetname2dataset_traj_info[sample['dataset']].prog_action_name,
            False,
            True,
            False
        )
        pred_p = get_program(
            traj,
            runner.datasetname2dataset_traj_info[sample['dataset']].prog_action_name,
            False,
            True,
            False
        )

        res = compare_prorams(pred_p, gt_p)
        if res is None:
            print(f"{ind} none program")
        else:
            prog_dict[sample['dataset']].append(res)
            prog_dict['all'].append(res)

        # ============================== eval Acc. ==============================
        if traj.cur_status() == 'exec solved':
            right_dict[sample['dataset']] += 1
            right_dict['all'] += 1
            if sample['dataset'] == 'gsm8k':
                aux_dict['gsm8k soft'][sample['dataset']] += 1

            # cnting the hits only if there is a program
            if all_exists(pred_p):
                aux_dict['Acc w/ prog'][sample['dataset']] += 1
                aux_dict['Acc w/ prog']['all'] += 1
                if sample['dataset'] == 'gsm8k':
                    aux_dict['gsm8k soft w/ prog'][sample['dataset']] += 1

                # cnting the hits only if there is a program and program bleu > threshold
                if all_exists(res) and res.score > bleu_thres:
                    aux_dict['Acc w/ prog w/ bleu thres'][sample['dataset']] += 1
                    aux_dict['Acc w/ prog w/ bleu thres']['all'] += 1
                    if sample['dataset'] == 'gsm8k':
                        aux_dict['gsm8k soft w/ prog w/ bleu thres'][sample['dataset']] += 1

        else:
            _, ans_exp = find_exp(traj, 'Aggregate and answer', which_one='last')

            # if no ans then runtime error
            if ans_exp is None:
                error_dict[sample['dataset']]['Runtime error'] += 1
                error_dict['all']['Runtime error'] += 1

            else:
                out_content = None
                if all_exists(ans_exp.response) and all_exists(ans_exp.response.action_output):
                    out_content = ans_exp.response.action_output

                if sample['dataset'] == 'gsm8k':
                    rightformat = all_exists(out_content) and all_exists(re.fullmatch(r'[0-9\-,.$]*', out_content))
                else:
                    rightformat = check_ans_valid(ans_exp)

                if (out_content is None) or (not rightformat):
                    error_dict[sample['dataset']]['Wrong Format'] += 1
                    error_dict['all']['Wrong Format'] += 1
                else:
                    error_dict[sample['dataset']]['Wrong Ans.'] += 1
                    error_dict['all']['Wrong Ans.'] += 1

                # further taking care of gsm8k
                if all_exists(out_content) and sample['dataset'] == 'gsm8k':
                    gt_ans = sample['label'].replace('$', '').replace(',', '').replace('.', '').replace('-', '')
                    proc_out_content = out_content.replace('$', '').replace(',', '').replace('.', '').replace('-', '')
                    if gt_ans in proc_out_content:
                        aux_dict['gsm8k soft'][sample['dataset']] += 1
                        # NOTE what this means is that: the all metric is based on gsm8k soft not orig gsm8k
                        right_dict['all'] += 1
                        if all_exists(pred_p):
                            aux_dict['gsm8k soft w/ prog'][sample['dataset']] += 1
                            aux_dict['Acc w/ prog']['all'] += 1
                            if all_exists(res) and res.score > bleu_thres:
                                aux_dict['gsm8k soft w/ prog w/ bleu thres'][sample['dataset']] += 1
                                aux_dict['Acc w/ prog w/ bleu thres']['all'] += 1

    full_numbers = []
    for k in ['gsm8k', 'folio', 'proscript', 'reclor', 'all']:
        full = [e for e in samples if ((e['dataset'] == k) or (k == 'all'))]

        if len(full) == 0:
            continue

        v = right_dict[k]
        hits_withprog = aux_dict['Acc w/ prog'][k]
        hits_withprogwithbleu = aux_dict['Acc w/ prog w/ bleu thres'][k]

        acc = v / float(len(full))
        acc_withprog = hits_withprog / float(len(full))
        acc_withprogwithbleu = hits_withprogwithbleu / float(len(full))

        print(f'{k} {v} {len(full)} {acc * 100:.2f}')
        print(f'{k} {hits_withprog} {len(full)} acc_withprog {acc_withprog * 100:.2f}')
        print(f'{k} {hits_withprogwithbleu} {len(full)} acc_withprogwithbleu {acc_withprogwithbleu * 100:.2f}')

        full_numbers.append(f'{acc * 100:.2f}')
        full_numbers.append(f'{acc_withprog * 100:.2f}')
        full_numbers.append(f'{acc_withprogwithbleu * 100:.2f}')

        codebleu_ls = prog_dict[k]
        avg_codebleu = np.mean([cb.score for cb in codebleu_ls])
        print(f'{k} codebleu {avg_codebleu:.2f}')
        full_numbers.append(f'{avg_codebleu:.2f}')

        if k == 'gsm8k':
            soft_hits = aux_dict['gsm8k soft'][k]
            soft_hits_withprog = aux_dict['gsm8k soft w/ prog'][k]
            soft_hits_withprog_withbleu = aux_dict['gsm8k soft w/ prog w/ bleu thres'][k]

            acc_soft = soft_hits / float(len(full))
            acc_soft_withprog = soft_hits_withprog / float(len(full))
            acc_soft_withprog_withbleu = soft_hits_withprog_withbleu / float(len(full))

            print(f'{k} {soft_hits} {len(full)} acc_soft {acc_soft * 100:.2f}')
            print(f'{k} {soft_hits_withprog} {len(full)} acc_soft_withprog {acc_soft_withprog * 100:.2f}')
            print(
                f'{k} {soft_hits_withprog_withbleu} {len(full)} '
                f'acc_soft_withprog_withbleu {acc_soft_withprog_withbleu * 100:.2f}'
            )

            full_numbers.append(f'{acc_soft * 100:.2f}')
            full_numbers.append(f'{acc_soft_withprog * 100:.2f}')
            full_numbers.append(f'{acc_soft_withprog_withbleu * 100:.2f}')
            # re append the bleu since we treat gsm8k soft as a new dataset
            full_numbers.append(f'{avg_codebleu:.2f}')

    print(' '.join(full_numbers))

    full_numbers = []
    for k in ['gsm8k', 'folio', 'proscript', 'reclor', 'all']:
        type_dict = error_dict[k]
        for typek in ['Wrong Ans.', 'Runtime error', 'Wrong Format']:
            typecnt = type_dict[typek]
            print(f'{k} {typek} {typecnt}')
            full_numbers.append(f'{typecnt}')
    print(' '.join(full_numbers))


def eval_options(
        sample, traj, runner,
        verbose=False, hyb_error_dict=None, aux_dict=None, gt_traj_key='gt_traj', bleu_thres=0.15
):
    opt_stats = defaultdict(list)
    assert all_exists(hyb_error_dict)

    if verbose:
        print(f"{PCYAN_L}{traj.show_traj_status()}{PCYAN_R}")

    # ============================== eval every option ==============================
    for oind, option in enumerate(sample['options']):
        subp_bleu = .0
        tactic_hit = 0
        codebleu = .0
        answered = 0
        should_add_codebleu = True

        oqind = sample['option_ind2qls'][oind]
        oq = sample['qls'][oqind]
        is_correct_option = str(oind + 1) == sample['label']

        if verbose:
            print(
                f'{oind} ##############################\n'
                f"input:\n{oq['input']}"
                f"\n{PRED_L}option{PRED_R}: {option}\n"
                f"{PRED_L}is_correct_option:{PRED_R}{is_correct_option}\n"
                f"##########################"
            )

        # if model never generates subtraj for this option then assign 0 and continue
        if 'subtrajs' not in traj.metadata:
            opt_stats['subp_bleu'].append(subp_bleu)
            opt_stats['tactic_hit'].append(tactic_hit)
            opt_stats['answered'].append(answered)
            opt_stats['codebleu'].append(codebleu)

            hyb_error_dict[oq['dataset']]['Runtime error'] += 1
            hyb_error_dict[sample['shuffle']]['Runtime error'] += 1
            hyb_error_dict['all']['Runtime error'] += 1

            if verbose:
                print(f"metadata['subtrajs'] not found in oind {oind}\n")

            continue

        stls = traj.metadata['subtrajs']
        stls_oind = [st_info for st_info in stls if st_info['oind'] == oind]
        if verbose:
            print(f"stls_oind len : {len(stls_oind)}")

        # if model generates subtrajs but no subtraj found for this option then assign 0 and continue
        if len(stls_oind) == 0:
            if verbose:
                print(f'no subtraj found for oind {oind}')

        else:
            # if mutliple subtraj info found then take the last one
            st_info = stls_oind[-1]

            # if the subtraj cannot be found in the info (usually due to failed run) then assign 0 and continue
            if 'subtraj' not in st_info:
                opt_stats['subp_bleu'].append(subp_bleu)
                opt_stats['tactic_hit'].append(tactic_hit)
                opt_stats['answered'].append(answered)
                opt_stats['codebleu'].append(codebleu)

                hyb_error_dict[oq['dataset']]['Runtime error'] += 1
                hyb_error_dict[sample['shuffle']]['Runtime error'] += 1
                hyb_error_dict['all']['Runtime error'] += 1

                if verbose:
                    print(f'no subtraj found for oind {oind}')

                continue

            # get the subtraj
            st = Trajectory.from_json(st_info['subtraj'])

            # ============================== subproblem bleu ==============================
            subproblem = st_info['subproblem']
            subp_bleu = compute_bleu(subproblem, oq['input'])
            if verbose:
                print(
                    f"{PCYAN_L}==================== subproblem\n"
                    f"{subproblem}\n"
                    f"===================={PCYAN_R}\n"
                    f'subp_bleu: {subp_bleu:.2f}\n'
                    '==='
                )

            # ============================== tactic hit ==============================
            dataset_traj_info = runner.datasetname2dataset_traj_info[oq['dataset']]
            tactic_hit = int(dataset_traj_info.tactic_name == st_info['tactic_name'])
            if verbose:
                print(
                    f'chosen tac: {st_info["tactic_name"]}, gt tac: {dataset_traj_info.tactic_name}\n'
                    f'tactic_hit: {tactic_hit}\n'
                    '==='
                )

            # ============================== program bleu ==============================
            gt_traj = Trajectory.from_json(oq[gt_traj_key])
            chosen_datasetname = [
                dn for dn, info in runner.datasetname2dataset_traj_info.items() if
                info.tactic_name == st_info['tactic_name']
            ][0]

            gt_p = get_program(
                gt_traj,
                runner.datasetname2dataset_traj_info[oq['dataset']].prog_action_name,
                False,
                True,
                False
            )
            pred_p = get_program(
                st,
                runner.datasetname2dataset_traj_info[chosen_datasetname].prog_action_name,
                False,
                True,
                False
            )

            res = compare_prorams(pred_p, gt_p)
            if res is None:
                print(f"{oind} none program")
                should_add_codebleu = False
            else:
                codebleu = res.score

            # ============================== subtraj Acc ==============================
            out_content = None
            ans_ind, ans_exp = find_exp(st, 'Aggregate and answer', which_one='last')
            if all_exists(ans_exp):
                if all_exists(ans_exp.response) and all_exists(ans_exp.response.action_output):
                    out_content = ans_exp.response.action_output
            if verbose:
                print(
                    f"{PCYAN_L}{traj.show_traj_status()}{PCYAN_R}"
                    f"ans_ind: {PCYAN_L}{ans_ind}{PCYAN_R}, out_content:\n"
                    f"{PCYAN_L}{out_content}{PCYAN_R}\n"
                    f"gt answer:\n"
                    f"{PRED_L}{oq['label']}{PRED_R}\n"
                )

            hyb_error_added = False
            if ans_exp is None:
                hyb_error_dict[oq['dataset']]['Runtime error'] += 1
                hyb_error_dict[sample['shuffle']]['Runtime error'] += 1
                hyb_error_dict['all']['Runtime error'] += 1
                hyb_error_added = True

            # the case, the model has provided an answer to the option regardless of what it is
            if all_exists(out_content):
                answered = 1
                # make a temp feedback func
                obs = runner.obs_dict[oq['dataset']]
                fb_kwargs = deepcopy(dataset_traj_info.obs_feedback_kwargs)
                if fb_kwargs['gt_answer'] == 'SAMPLE_LABEL':
                    if oq['dataset'] == 'reclor':
                        label = '1' if is_correct_option else '2'
                    else:
                        label = oq['label']
                    fb_kwargs['gt_answer'] = label

                elif isinstance(fb_kwargs['gt_answer'], list):
                    for arg_ind, arg in enumerate(fb_kwargs['gt_answer']):
                        if arg == 'SAMPLE_LABEL':
                            fb_kwargs['gt_answer'][arg_ind] = oq['label']
                    fb_kwargs['gt_answer'] = fb_kwargs['gt_answer'][0](*fb_kwargs['gt_answer'][1:])
                else:
                    raise ValueError(f'i don\'t know how to handle this fb_kwargs {fb_kwargs["gt_answer"]}')
                fb_kwargs['answer_check'] = True
                fb_func = partial(obs.give_feedback, **fb_kwargs)

                fbls, _ = fb_func(st, which_exp=ans_ind)
                assert len(fbls) == 1
                fb = fbls[0]
                if fb.feedback_status == 'feedback solved':
                    aux_dict['Acc'][oq['dataset']] += 1
                    aux_dict['Acc']['all'] += 1
                    if oq['dataset'] == 'gsm8k':
                        aux_dict['gsm8k soft'][oq['dataset']] += 1

                    if all_exists(pred_p) and all_exists(res) and (res.score > bleu_thres) and (tactic_hit == 1):
                        aux_dict['Acc w/ prog w/ bleu thres w/ tac hit'][oq['dataset']] += 1
                        aux_dict['Acc w/ prog w/ bleu thres w/ tac hit']['all'] += 1
                        opt_stats['Acc opt done witheverything'].append(1)
                        if oq['dataset'] == 'gsm8k':
                            aux_dict['gsm8k soft w/ prog w/ bleu thres w/ tac hit'][oq['dataset']] += 1

                else:
                    wrong_format = False
                    if oq['dataset'] == 'gsm8k' and (re.fullmatch(r'[0-9\-,.$]*', out_content) is None):
                        wrong_format = True
                    elif ('invalid answer' in fb.content) or ('invalid response format' in fb.content):
                        wrong_format = True

                    if wrong_format:
                        # # uncomment to print wrong format case in gsm8k
                        # if oq['dataset'] == 'gsm8k':
                        #     print(
                        #         f"{oind} ans_ind: {PCYAN_L}{ans_ind}{PCYAN_R}, out_content:\n"
                        #         f"{PCYAN_L}{out_content}{PCYAN_R}\n"
                        #         f"gt answer:\n"
                        #         f"{PRED_L}{oq['label']}{PRED_R}\n"
                        #     )
                        hyb_error_dict[oq['dataset']]['Wrong Format'] += 1
                        hyb_error_dict[sample['shuffle']]['Wrong Format'] += 1
                        hyb_error_dict['all']['Wrong Format'] += 1
                    else:
                        hyb_error_dict[oq['dataset']]['Wrong Ans.'] += 1
                        hyb_error_dict[sample['shuffle']]['Wrong Ans.'] += 1
                        hyb_error_dict['all']['Wrong Ans.'] += 1

                    if oq['dataset'] == 'gsm8k':
                        gt_ans = oq['label'].replace('$', '').replace(',', '').replace('.', '').replace('-', '')
                        proc_out_content = out_content.replace('$', '').replace(',', '').replace('.', '').replace('-','')

                        if gt_ans in proc_out_content:
                            aux_dict['gsm8k soft'][oq['dataset']] += 1
                            # NOTE what this means is that: the all metric is based on gsm8k soft not orig gsm8k
                            aux_dict['Acc']['all'] += 1
                            if all_exists(pred_p) and all_exists(res) and (res.score > bleu_thres) and (tactic_hit == 1):
                                opt_stats['Acc opt done witheverything'].append(1)
                                aux_dict['gsm8k soft w/ prog w/ bleu thres w/ tac hit'][oq['dataset']] += 1
                                aux_dict['Acc w/ prog w/ bleu thres w/ tac hit']['all'] += 1

            else:
                if not hyb_error_added:
                    # # uncomment to print wrong format case in gsm8k
                    # if oq['dataset'] == 'gsm8k':
                    #     print(
                    #         f"{oind} ans_ind: {PCYAN_L}{ans_ind}{PCYAN_R}, out_content:\n"
                    #         f"{PCYAN_L}{out_content}{PCYAN_R}\n"
                    #         f"gt answer:\n"
                    #         f"{PRED_L}{oq['label']}{PRED_R}\n"
                    #     )
                    hyb_error_dict[oq['dataset']]['Wrong Format'] += 1
                    hyb_error_dict[sample['shuffle']]['Wrong Format'] += 1
                    hyb_error_dict['all']['Wrong Format'] += 1

        opt_stats['subp_bleu'].append(subp_bleu)
        opt_stats['tactic_hit'].append(tactic_hit)
        opt_stats['answered'].append(answered)
        if should_add_codebleu:
            opt_stats['codebleu'].append(codebleu)

    if verbose:
        print('!!!!!!\n')
        print(opt_stats)

    return opt_stats


def calc_hyb_metric(samples, runner, traj_key='traj_eval', gt_traj_key='gt_traj', verbose=False, bleu_thres=0.15):
    right_dict = defaultdict(int)
    right_answered_dict = defaultdict(int)
    error_dict = defaultdict(lambda: defaultdict(int))
    hyb_error_dict = defaultdict(lambda: defaultdict(int))
    opt_dict = defaultdict(lambda: defaultdict(list))
    aux_dict = defaultdict(lambda: defaultdict(int))

    def check_ans_valid(exp):
        valid_ans = True
        if all_exists(exp.observation) and len(exp.observation) > 0:
            for fb in exp.observation:
                if ('invalid answer' in fb.content) or ('invalid response format' in fb.content):
                    valid_ans = False
                    break
        return valid_ans

    for ind, sample in enumerate(samples):
        traj = Trajectory.from_json(sample[traj_key])
        sample_type_key = sample['qtype']

        if traj.cur_status() == 'exec solved':
            right_dict[sample_type_key] += 1
            right_dict[sample['shuffle']] += 1
            right_dict['all'] += 1
        else:
            _, ans_exp = find_exp(traj, 'Aggregate and answer', which_one='last')
            if ans_exp is None:
                error_dict[sample_type_key]['Runtime error'] += 1
                error_dict[sample['shuffle']]['Runtime error'] += 1
                error_dict['all']['Runtime error'] += 1
            else:
                out_content = None
                if all_exists(ans_exp.response) and all_exists(ans_exp.response.action_output):
                    out_content = ans_exp.response.action_output

                if sample['dataset'] == 'gsm8k':
                    rightformat = all_exists(out_content) and all_exists(re.fullmatch(r'[0-9\-,.$]*', out_content))
                else:
                    rightformat = check_ans_valid(ans_exp)

                if (out_content is None) or (not rightformat):
                    error_dict[sample_type_key]['Wrong Format'] += 1
                    error_dict[sample['shuffle']]['Wrong Format'] += 1
                    error_dict['all']['Wrong Format'] += 1
                else:
                    error_dict[sample_type_key]['Wrong Ans.'] += 1
                    error_dict[sample['shuffle']]['Wrong Ans.'] += 1
                    error_dict['all']['Wrong Ans.'] += 1

        opt_status = eval_options(
            sample, traj, runner,
            verbose=verbose,
            hyb_error_dict=hyb_error_dict,
            aux_dict=aux_dict,
            gt_traj_key=gt_traj_key,
            bleu_thres=bleu_thres
        )

        for k in ['subp_bleu', 'tactic_hit', 'codebleu']:
            # NOTE this is for case where gt trajs for all options not exist, in this case we skip it
            if (k == 'codebleu') and (len(opt_status[k]) == 0):
                continue
            opt_stat = np.mean(opt_status[k])

            opt_dict[sample_type_key][k].append(opt_stat)
            opt_dict[sample['shuffle']][k].append(opt_stat)
            opt_dict['all'][k].append(opt_stat)

        all_opt_answered_witheverything = sum(opt_status['Acc opt done witheverything']) == len(sample['options'])

        if (traj.cur_status() == 'exec solved') and all_opt_answered_witheverything:
            right_answered_dict[sample_type_key] += 1
            right_answered_dict[sample['shuffle']] += 1
            right_answered_dict['all'] += 1

    full_numbers = []
    for k in ['gg', 'gf', 'gfx', 'gfr', 'gfrx', 'all']:
        v = right_dict[k]
        full = [e for e in samples if ((e['qtype'] == k) or (k == 'all'))]

        acc = v / float(len(full))
        answer_acc = right_answered_dict[k] / float(len(full))

        print(f'{k} acc: {v} full: {len(full)} {acc * 100:.2f}')
        print(f'{k} answer_acc: {right_answered_dict[k]} full: {len(full)} {answer_acc * 100:.2f}')

        full_numbers.append(f'{acc * 100:.2f}')
        full_numbers.append(f'{answer_acc * 100:.2f}')

        opt_dict_g = opt_dict[k]
        for kopt in ['subp_bleu', 'tactic_hit', 'codebleu']:
            isbleu = not(kopt == 'tactic_hit')
            mul = 1 if isbleu else 100

            meanavg_stat = f'{np.mean(opt_dict_g[kopt]) * mul:.2f}'
            print(f'{k} {kopt} {meanavg_stat}')
            full_numbers.append(meanavg_stat)

    print(' '.join(full_numbers))

    full_numbers = []
    for k in [False, True]:
        v = right_dict[k]
        full = [e for e in samples if ((e['shuffle'] == k) or (k == 'all'))]
        acc = v / float(len(full))
        answer_acc = right_answered_dict[k] / float(len(full))
        print(f'{k} v: {v} full: {len(full)} {acc * 100:.2f}')
        print(f'{k} v: {right_answered_dict[k]} full: {len(full)} {answer_acc * 100:.2f}')
        full_numbers.append(f'{acc * 100:.2f}')
        full_numbers.append(f'{answer_acc * 100:.2f}')

        opt_dict_g = opt_dict[k]
        for kopt in ['subp_bleu', 'tactic_hit', 'codebleu']:
            isbleu = not (kopt == 'tactic_hit')
            mul = 1 if isbleu else 100

            meanavg_stat = f'{np.mean(opt_dict_g[kopt]) * mul:.2f}'
            print(f'{k} {kopt} {meanavg_stat}')
            full_numbers.append(meanavg_stat)

    print(' '.join(full_numbers))

    full_numbers = []
    for k in ['gg', 'gf', 'gfx', 'gfr', 'gfrx', 'all']:
        type_dict = error_dict[k]
        for typek in ['Wrong Ans.', 'Runtime error', 'Wrong Format']:
            typecnt = type_dict[typek]
            print(f'{k} {typek} {typecnt}')
            full_numbers.append(f'{typecnt}')
    print(' '.join(full_numbers))

    full_numbers = []
    for k in [False, True]:
        type_dict = error_dict[k]
        for typek in ['Wrong Ans.', 'Runtime error', 'Wrong Format']:
            typecnt = type_dict[typek]
            print(f'{k} {typek} {typecnt}')
            full_numbers.append(f'{typecnt}')
    print(' '.join(full_numbers))

    full_numbers = []
    for k in ['gsm8k', 'folio', 'reclor', 'all']:
        type_dict = hyb_error_dict[k]
        for tind, typek in enumerate(['Wrong Ans.', 'Runtime error', 'Wrong Format']):
            typecnt = type_dict[typek]
            print(f'{k} {typek} {typecnt}')
            full_numbers.append(f'{typecnt}')
    print(' '.join(full_numbers))

    full_numbers = []
    for k in [False, True]:
        type_dict = hyb_error_dict[k]
        for tind, typek in enumerate(['Wrong Ans.', 'Runtime error', 'Wrong Format']):
            typecnt = type_dict[typek]
            print(f'{k} {typek} {typecnt}')
            full_numbers.append(f'{typecnt}')
    print(' '.join(full_numbers))

    full_numbers = []
    total_num = [668, 478, 230]
    for kind, k in enumerate(['gsm8k', 'folio', 'reclor']):
        hits_withprogwithbleu = aux_dict['Acc w/ prog w/ bleu thres w/ tac hit'][k]
        fulllen = total_num[kind]

        acc_withprogwithbleu = hits_withprogwithbleu / fulllen
        print(f'{k} acc_witheverything {acc_withprogwithbleu * 100:.2f}')
    print(' '.join(full_numbers))

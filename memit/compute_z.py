from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from copy import deepcopy
from rome import repr_tools
from util import nethook
import heapq
from .memit_hparams import MEMITHyperParams
from geomloss import SamplesLoss



def k_log_softmax(nom, denom):     # torch.Size[len, 32]   torch.Size[len, 16416]
    exp_nom = torch.exp(nom)
    exp_denom = torch.exp(denom)
    denom_sum = torch.sum(exp_denom, dim=1, keepdims=True)
    sfmx = exp_nom / denom_sum
    log_sfmx = torch.log(sfmx)
    return log_sfmx


def compute_z(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: MEMITHyperParams,
    layer: int,
    context_templates: List[str],

) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the value (right) vector for the rank-1 update.
    Runs a simple optimization procedure.
    """

    # Get model parameters
    lm_w, ln_f = (
        nethook.get_parameter(model, f"{hparams.lm_head_module}.weight").T,
        nethook.get_module(model, hparams.ln_f_module),
    )
    try:
        lm_b = nethook.get_parameter(model, f"{hparams.lm_head_module}.bias")
    except LookupError as _:
        lm_b = next(model.parameters()).new_zeros(model.config.vocab_size)

    print("Computing right vector (v)")

    # Tokenize target into list of int token IDs
    target_ids = tok(request["target_new"]["str"], return_tensors="pt").to("cuda")["input_ids"][0]
    old_ids = tok(request["target_true"]["str"], return_tensors="pt").to("cuda")["input_ids"][0]


    # Compile list of rewriting and KL x/y pairs
    rewriting_prompts, kl_prompts = [
        context.format(request["prompt"]) + tok.decode(target_ids[:-1])
        for context_types in context_templates
        for context in context_types
    ], ["{} is a"]
    all_prompts = rewriting_prompts + kl_prompts

    input_tok = tok(
        [prompt.format(request["subject"]) for prompt in all_prompts],
        return_tensors="pt",
        padding=True,
    ).to("cuda")


    # Compute rewriting targets
    rewriting_targets = torch.tensor(-100, device="cuda").repeat(
        len(rewriting_prompts), *input_tok["input_ids"].shape[1:]
    )

    for i in range(len(rewriting_prompts)):
        ex_len = input_tok["attention_mask"][i].sum()
        rewriting_targets[i, ex_len - len(target_ids) : ex_len] = target_ids


    # Compute indices of the tokens where the fact is looked up
    lookup_idxs = [
        find_fact_lookup_idx(
            prompt, request["subject"], tok, hparams.fact_token, verbose=(i == 0)
        )
        for i, prompt in enumerate(all_prompts)
    ]

    # Finalize rewrite and loss layers
    loss_layer = max(hparams.v_loss_layer, layer)
    print(f"Rewrite layer is {layer}")
    print(f"Tying optimization objective to {loss_layer}")

    # Set up an optimization over a latent vector that, when output at the
    # rewrite layer, i.e. hypothesized fact lookup location, will induce the
    # target token to be predicted at the final layer.
    # delta = torch.zeros((model.config.n_embd,), requires_grad=True, device="cuda")
    delta = torch.zeros((model.config.hidden_size,), requires_grad=True, device="cuda")
    target_init, kl_distr_init = None, None
    loc_target_init, loc_kl_distr_init = None, None

    # Inserts new "delta" variable at the appropriate part of the computation
    def edit_output_fn(cur_out, cur_layer):
        nonlocal target_init

        if cur_layer == hparams.layer_module_tmp.format(layer):
            # Store initial value of the vector of interest
            if target_init is None:
                print("Recording initial value of v*")
                # Initial value is recorded for the clean sentence
                target_init = cur_out[0][0, lookup_idxs[0]].detach().clone()

            # Add intervened delta
            for i, idx in enumerate(lookup_idxs):
                cur_out[0][i, idx, :] += delta

        return cur_out

    opt = torch.optim.Adam([delta], lr=hparams.v_lr)
    nethook.set_requires_grad(False, model)
    # layer_norm = nethook.get_module(model, f"transformer.h.{layer}.ln_1")

    # sinkhorn algorithm for wasserstein distance loss
    sinkhorn = SamplesLoss(loss='sinkhorn', p=2, blur=.05)

    # Execute optimization
    for it in range(hparams.v_num_grad_steps):
        opt.zero_grad()

        # Forward propagation
        with nethook.TraceDict(
            module=model,
            layers=[
                hparams.layer_module_tmp.format(loss_layer),
                hparams.layer_module_tmp.format(layer),
            ],
            retain_input=False,
            retain_output=True,
            edit_output=edit_output_fn,
        ) as tr:
            output = model(**input_tok, output_hidden_states=True)

            logits = output.logits
            # Compute distribution for KL divergence
            kl_logits = torch.stack(
                [
                    logits[i - len(kl_prompts), idx, :]
                    for i, idx in enumerate(lookup_idxs[-len(kl_prompts):])
                ],
                dim=0,
            )
            kl_log_probs = torch.nn.functional.log_softmax(kl_logits, dim=1)
            if kl_distr_init is None:
                kl_distr_init = kl_log_probs.detach().clone()

        # Compute loss on rewriting targets
        full_repr = tr[hparams.layer_module_tmp.format(loss_layer)].output[0][
            : len(rewriting_prompts)
        ]
        log_probs = torch.log_softmax(ln_f(full_repr) @ lm_w + lm_b, dim=2)
        loss = torch.gather(
            log_probs,
            2,
            torch.where(rewriting_targets != -100, rewriting_targets, 0).unsqueeze(2),
        ).squeeze(2)
        mask = (rewriting_targets != -100).float()

        # Aggregate total losses
        nll_loss_each = -(loss * mask).sum(1) / target_ids.size(0)

        nll_loss = nll_loss_each.mean()

        kl_loss = hparams.kl_factor * torch.nn.functional.kl_div(
            kl_distr_init, kl_log_probs, log_target=True, reduction="batchmean"
        )

        weight_decay = hparams.v_weight_decay * (
            torch.norm(delta) / torch.norm(target_init) ** 2
        )
        distribution = target_init + delta
        if (target_init is not None) and (it > 0):
            # print(delta.shape, target_init.shape)
            # sinkhorn_loss = sinkhorn(delta.unsqueeze(0), target_init.unsqueeze(0))
            sinkhorn_loss = sinkhorn(distribution.unsqueeze(0), target_init.unsqueeze(0))
            print("sinkhorn_loss", sinkhorn_loss)
        else:
            sinkhorn_loss = torch.tensor(0.0)
            print("sinkhorn_loss", sinkhorn_loss)
	# dynamic scale, custom the threshold
        while sinkhorn_loss > 1:
            sinkhorn_loss = sinkhorn_loss / 10.0
        # weight_decay = hparams.v_weight_decay * torch.norm(delta) ** 2
        loss = nll_loss + kl_loss  +  weight_decay  + sinkhorn_loss # (*0.1, *0.5)



        print(
            f"loss {np.round(loss.item(), 3)} = {np.round(nll_loss.item(), 3)} + {np.round(kl_loss.item(), 3)} "
            # f"+ {np.round(loc_nll_loss.item(), 3)} + {np.round(loc_kl_loss.item(), 3)} + "
            # f" {np.round(loss_wk.item(), 3)}"
            f" avg prob of [{request['target_new']['str']}] "
            f"{torch.exp(-nll_loss_each).mean().item()}"
            # f" avg prob of [{request['target_true']['str']}] "
            # f"{torch.exp(-loc_nll_loss_each).mean().item()}"
        )
        if nll_loss + kl_loss + weight_decay < 5e-2:
        # if loss < 5e-2:
            break

        if it == hparams.v_num_grad_steps - 1:
            break

        # Backpropagate
        loss.backward()
        opt.step()

        # Project within L2 ball
        max_norm = hparams.clamp_norm_factor * target_init.norm()
        if delta.norm() > max_norm:
            with torch.no_grad():
                delta[...] = delta * max_norm / delta.norm()
    # print(target_init.shape, delta.shape) # torch.Size([4096]) torch.Size([4096])
    # print(delta)

    # sparsification 
    s_rate = 0.7
    print("rate: ",s_rate)
    slim_rate = int(delta.size(0) * s_rate)
    abs_delta = []
    for d in delta:
        abs_delta.append(abs(d))
    # # print("abs:", abs_delta)
    slim_value = heapq.nlargest(slim_rate, abs_delta)[-1]
    print("value:",slim_value)
    sparse_mask = torch.where(abs(delta) < slim_value, 0, 1)
    with torch.no_grad():
        delta = delta * sparse_mask.to(delta.device)
    print("delta:", delta)

    drop_p = 0.1  # 0.3
    drop_rand = torch.rand(delta.size())
    drop_mask = torch.where(drop_rand > drop_p, 1, 0)
    # # print("mask: ", drop_mask)
    with torch.no_grad():
    #     # drop
        delta = delta * drop_mask.to(delta.device)
    #     # rescale
        delta = delta * (1 / (1-drop_p) )
        # print(delta)
    ############################################
    target = target_init + delta     #  z_i  = h_i + delta
    # with open('delta_norm_new.txt', mode='a') as src:
    #     src.write(f'{delta.norm()}\n')
    print(
        f"Init norm {target_init.norm()} | Delta norm {delta.norm()} | Target norm {target.norm()}"
    )

    return target

def get_module_input_output_at_words(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer: int,
    context_templates: List[str],
    words: List[str],
    module_template: str,
    fact_token_strategy: str,
    track=None,
) -> Tuple[torch.Tensor]:
    """
    Retrieves detached representations for a word at the input and
    output of a particular layer module.
    """

    word_repr_args = dict(
        model=model,
        tok=tok,
        layer=layer,
        module_template=module_template,
    )
    if "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0:
        context_info = dict(
            context_templates=context_templates,
            words=words,
        )
        subtoken = fact_token_strategy[len("subject_") :]
        if track == 'out' or track == 'in':
            return repr_tools.get_reprs_at_word_tokens(
                track=track, subtoken=subtoken, **context_info, **word_repr_args
            )
        l_input, l_output = repr_tools.get_reprs_at_word_tokens(
            track="both", subtoken=subtoken, **context_info, **word_repr_args
        )
    elif fact_token_strategy == "last":
        raise Exception("This is definitely bugged, fix it.")
        context_info = dict(
            contexts=[
                tmp[i].format(words[i]) for i, tmp in enumerate(context_templates)
            ],
            idxs=[000000],
        )
        if track == 'out' or track == 'in':
            return repr_tools.get_reprs_at_word_tokens(
                track=track, subtoken=subtoken, **context_info, **word_repr_args
            )
        l_input, l_output = repr_tools.get_reprs_at_idxs(
            track="both", **context_info, **word_repr_args
        )
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    return l_input.detach(), l_output.detach()


def find_fact_lookup_idx(
    prompt: str,
    subject: str,
    tok: AutoTokenizer,
    fact_token_strategy: str,
    verbose=True,
) -> int:
    """
    Computes hypothesized fact lookup index given a sentence and subject.
    """

    ret = None
    if fact_token_strategy == "last":
        ret = -1
    elif (
        "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0
    ):
        ret = repr_tools.get_words_idxs_in_templates(
            tok=tok,
            context_templates=[prompt],
            words=[subject],
            subtoken=fact_token_strategy[len("subject_") :],
        )[0][0]
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    sentence = prompt.format(subject)
    if verbose:
        # print(ret)
        # print(sentence)
        # print(tok(sentence))
        # print(tok(sentence)["input_ids"])
        # print(tok(sentence)["input_ids"][ret])
        if ret < len(tok(sentence)["input_ids"]):
            print(
            f"Lookup index found: {ret} | Sentence: {sentence} | Token:",
            tok.decode(tok(sentence)["input_ids"][ret]),
            )
        else :
            print(
                f"Lookup index found: {ret} | Sentence: {sentence} | Token:",
                tok.decode(tok(sentence)["input_ids"][-1]),
            )

    return ret

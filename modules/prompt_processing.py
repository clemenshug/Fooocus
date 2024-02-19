from __future__ import annotations
from typing import List, Any, Tuple

import re
import itertools
from collections import namedtuple
import lark

from extras.expansion import safe_str
from modules.util import remove_empty_str, HWC3, resize_image, \
    get_image_shape_ceil, set_image_shape_ceil, get_shape_ceil, resample_image, erode_or_dilate
from modules.sdxl_styles import apply_style, apply_wildcards, fooocus_expansion

ScheduledPromptConditioning = namedtuple("ScheduledPromptConditioning", ["end_at_step", "cond"])

# a prompt like this: "fantasy landscape with a [mountain:lake:0.25] and [an oak:a christmas tree:0.75][ in foreground::0.6][: in background:0.25] [shoddy:masterful:0.5]"
# will be represented with prompt_schedule like this (assuming steps=100):
# [25, 'fantasy landscape with a mountain and an oak in foreground shoddy']
# [50, 'fantasy landscape with a lake and an oak in foreground in background shoddy']
# [60, 'fantasy landscape with a lake and an oak in foreground in background masterful']
# [75, 'fantasy landscape with a lake and an oak in background masterful']
# [100, 'fantasy landscape with a lake and a christmas tree in background masterful']

schedule_parser = lark.Lark(r"""
!start: (prompt | /[][():]/+)*
prompt: (emphasized | scheduled | alternate | plain | WHITESPACE)*
!emphasized: "(" prompt ")"
        | "(" prompt ":" prompt ")"
        | "[" prompt "]"
scheduled: "[" [prompt ":"] prompt ":" [WHITESPACE] NUMBER [WHITESPACE] "]"
alternate: "[" prompt ("|" [prompt])+ "]"
WHITESPACE: /\s+/
plain: /([^\\\[\]():|]|\\.)+/
%import common.SIGNED_NUMBER -> NUMBER
""")

def get_learned_conditioning_prompt_schedules(prompts, base_steps, hires_steps=None, use_old_scheduling=False) -> List[List[ScheduledPromptConditioning]]:
    """
    >>> g = lambda p: get_learned_conditioning_prompt_schedules([p], 10)[0]
    >>> g("test")
    [[10, 'test']]
    >>> g("a [b:3]")
    [[3, 'a '], [10, 'a b']]
    >>> g("a [b: 3]")
    [[3, 'a '], [10, 'a b']]
    >>> g("a [[[b]]:2]")
    [[2, 'a '], [10, 'a [[b]]']]
    >>> g("[(a:2):3]")
    [[3, ''], [10, '(a:2)']]
    >>> g("a [b : c : 1] d")
    [[1, 'a b  d'], [10, 'a  c  d']]
    >>> g("a[b:[c:d:2]:1]e")
    [[1, 'abe'], [2, 'ace'], [10, 'ade']]
    >>> g("a [unbalanced")
    [[10, 'a [unbalanced']]
    >>> g("a [b:.5] c")
    [[5, 'a  c'], [10, 'a b c']]
    >>> g("a [{b|d{:.5] c")  # not handling this right now
    [[5, 'a  c'], [10, 'a {b|d{ c']]
    >>> g("((a][:b:c [d:3]")
    [[3, '((a][:b:c '], [10, '((a][:b:c d']]
    >>> g("[a|(b:1.1)]")
    [[1, 'a'], [2, '(b:1.1)'], [3, 'a'], [4, '(b:1.1)'], [5, 'a'], [6, '(b:1.1)'], [7, 'a'], [8, '(b:1.1)'], [9, 'a'], [10, '(b:1.1)']]
    >>> g("[fe|]male")
    [[1, 'female'], [2, 'male'], [3, 'female'], [4, 'male'], [5, 'female'], [6, 'male'], [7, 'female'], [8, 'male'], [9, 'female'], [10, 'male']]
    >>> g("[fe|||]male")
    [[1, 'female'], [2, 'male'], [3, 'male'], [4, 'male'], [5, 'female'], [6, 'male'], [7, 'male'], [8, 'male'], [9, 'female'], [10, 'male']]
    >>> g = lambda p: get_learned_conditioning_prompt_schedules([p], 10, 10)[0]
    >>> g("a [b:.5] c")
    [[10, 'a b c']]
    >>> g("a [b:1.5] c")
    [[5, 'a  c'], [10, 'a b c']]
    """

    if hires_steps is None or use_old_scheduling:
        int_offset = 0
        flt_offset = 0
        steps = base_steps
    else:
        int_offset = base_steps
        flt_offset = 1.0
        steps = hires_steps

    def collect_steps(steps, tree):
        res = [steps]

        class CollectSteps(lark.Visitor):
            def scheduled(self, tree):
                s = tree.children[-2]
                v = float(s)
                if use_old_scheduling:
                    v = v*steps if v<1 else v
                else:
                    if "." in s:
                        v = (v - flt_offset) * steps
                    else:
                        v = (v - int_offset)
                tree.children[-2] = min(steps, int(v))
                if tree.children[-2] >= 1:
                    res.append(tree.children[-2])

            def alternate(self, tree):
                res.extend(range(1, steps+1))

        CollectSteps().visit(tree)
        return sorted(set(res))

    def at_step(step, tree):
        class AtStep(lark.Transformer):
            def scheduled(self, args):
                before, after, _, when, _ = args
                yield before or () if step <= when else after
            def alternate(self, args):
                args = ["" if not arg else arg for arg in args]
                yield args[(step - 1) % len(args)]
            def start(self, args):
                def flatten(x):
                    if isinstance(x, str):
                        yield x
                    else:
                        for gen in x:
                            yield from flatten(gen)
                return ''.join(flatten(args))
            def plain(self, args):
                yield args[0].value
            def __default__(self, data, children, meta):
                for child in children:
                    yield child
        return AtStep().transform(tree)

    def get_schedule(prompt):
        try:
            tree = schedule_parser.parse(prompt)
        except lark.exceptions.LarkError:
            if 0:
                import traceback
                traceback.print_exc()
            return [[steps, prompt]]
        return [[t, at_step(t, tree)] for t in collect_steps(steps, tree)]

    promptdict = {prompt: get_schedule(prompt) for prompt in set(prompts)}
    return [[ScheduledPromptConditioning(*step) for step in promptdict[prompt]] for prompt in prompts]


def raw_prompt_to_schedule(prompt: str, base_steps: int, task_rng: int) -> List[ScheduledPromptConditioning]:
    prompt_schedule = get_learned_conditioning_prompt_schedules(
        [prompt_expanded], base_steps=base_steps
    )
    return prompt_schedule[0]

def single_prompt_split(prompt: str, use_style: bool, style_selections: List[Any], pos_neg: str) -> List[str] | list:
    prompts = remove_empty_str([safe_str(p) for p in prompt.splitlines()], default='')
    prompt = prompts[0]

    if prompt == '':
        # disable expansion when empty since it is not meaningful and influences image prompt
        use_expansion = False

    extra_prompts = prompts[1:] if len(prompts) > 1 else []

    basic_workloads = []

    if use_style:
        for s in style_selections:
            p, n = apply_style(s, positive=prompt)
            if pos_neg == "positive":
                basic_workloads = basic_workloads + p
            elif pos_neg == "negative":
                basic_workloads = basic_workloads + n
            else:
                raise ValueError("pos_neg must be 'positive' or 'negative'")
    else:
        basic_workloads.append(prompt)

    basic_workloads = basic_workloads + extra_prompts

    basic_workloads = remove_empty_str(basic_workloads, default=prompt)
    return basic_workloads


def prompt_to_condition(prompt: str, use_style: bool, style_selections: List[Any], task_rng: int, pos_neg: str, base_steps: int) -> List[ScheduledPromptConditioning]:
    prompt_schedule = raw_prompt_to_schedule(prompt, base_steps, task_rng)
    prompts_final = [ScheduledPromptConditioning(p.end_at_step, single_prompt_split(p.cond, use_style, style_selections, pos_neg))  for p in prompt_schedule]
    return prompts_final


def prompt_pair_to_condition(prompt, negative_prompt, use_style, style_selections, task_rng, base_steps) -> List[ScheduledPromptConditioning]:
    positive_basic_workloads = prompt_to_condition(prompt, use_style, style_selections, task_rng, "positive", base_steps)
    negative_basic_workloads = prompt_to_condition(negative_prompt, use_style, style_selections, task_rng, "negative", base_steps)
    combined_workloads = []
    i, j = 0, 0
    all_steps = list(sorted(set(itertools.chain((x.end_at_step for x in positive_basic_workloads), (x.end_at_step for x in negative_basic_workloads)))))
    for s in all_steps:
        combined_workloads.append(ScheduledPromptConditioning(s, (positive_basic_workloads[i].cond, negative_basic_workloads[j].cond)))
        if positive_basic_workloads[i].end_at_step == s:
            i += 1
        if negative_basic_workloads[j].end_at_step == s:
            j += 1
    return combined_workloads

# def collate_schedules(schedules: List[List[ScheduledPromptConditioning]]) -> List[Tuple[int, int, str]]:
#     schedules_start_end = []
#     for schedule in schedules:
#         start_step = 0
#         for p in schedule:
#             schedules_start_end.append((start_step, p.end_at_step, p.cond))
#             start_step = p.end_at_step
#     return list(sorted(itertools.chain(*schedules_start_end)))

def collate_schedules(schedules: List[List[ScheduledPromptConditioning]]) -> List[Tuple[int, int, List[str]]]:
    collated_schedule = []
    all_steps = list(sorted(set(x.end_at_step for schedule in schedules for x in schedule)))
    indices = [0] * len(schedules)
    cur_step = 0
    for s in all_steps:
        combined = []
        for i, schedule in enumerate(schedules):
            combined.append(schedule[indices[i]].cond)
            if schedule[indices[i]].end_at_step == s:
                indices[i] += 1
        collated_schedule.append((cur_step, s, combined))
        cur_step = s
    return collated_schedule




    # tasks.append(dict(
    #     task_seed=task_seed,
    #     task_prompt=task_prompt,
    #     task_negative_prompt=task_negative_prompt,
    #     positive=positive_basic_workloads,
    #     negative=negative_basic_workloads,
    #     expansion='',
    #     c=None,
    #     uc=None,
    #     positive_top_k=len(positive_basic_workloads),
    #     negative_top_k=len(negative_basic_workloads),
    #     log_positive_prompt='\n'.join([task_prompt] + task_extra_positive_prompts),
    #     log_negative_prompt='\n'.join([task_negative_prompt] + task_extra_negative_prompts),
    # ))

    # if use_expansion:
    #     for i, t in enumerate(tasks):
    #         progressbar(async_task, 5, f'Preparing Fooocus text #{i + 1} ...')
    #         expansion = pipeline.final_expansion(t['task_prompt'], t['task_seed'])
    #         print(f'[Prompt Expansion] {expansion}')
    #         t['expansion'] = expansion
    #         t['positive'] = copy.deepcopy(t['positive']) + [expansion]  # Deep copy.

    # for i, t in enumerate(tasks):
    #     progressbar(async_task, 7, f'Encoding positive #{i + 1} ...')
    #     t['c'] = pipeline.clip_encode(texts=t['positive'], pool_top_k=t['positive_top_k'])

    # for i, t in enumerate(tasks):
    #     if abs(float(cfg_scale) - 1.0) < 1e-4:
    #         t['uc'] = pipeline.clone_cond(t['c'])
    #     else:
    #         progressbar(async_task, 10, f'Encoding negative #{i + 1} ...')
    #         t['uc'] = pipeline.clip_encode(texts=t['negative'], pool_top_k=t['negative_top_k'])



import re
from abc import ABC, abstractmethod
from typing import List, Union
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from textgrad.variable import Variable
from textgrad import logger
from textgrad.engine import EngineLM
from textgrad.config import validate_engine_or_get_default
from .optimizer_prompts import (
    construct_tgd_prompt, OPTIMIZER_SYSTEM_PROMPT, GRADIENT_TEMPLATE, GRADIENT_MULTIPART_TEMPLATE,
    CONSTRAINT_PROMPT_ADDITION,
    REFLECT_SYSTEM_PROMPT, REFLECT_PROMPT_TEMPLATE,
    LR_APPLY_PROMPT_TEMPLATE, LR_SELECT_INSTRUCTION, LR_SELECT_INSTRUCTION_AUTONOMOUS,
)
from .lr_scheduler import build_scheduler, NO_LIMIT


def get_gradient_and_context_text(variable) -> Union[str, List[Union[str, bytes]]]:
    """For the variable, aggregates and returns 
    i. the gradients 
    ii. the context for which the gradients are computed.

    This is used by the optimizer.  
    :return: A string containing the aggregated gradients and their corresponding context.
    :rtype: str
    """

    gradient_content = []
    for g in variable.gradients:
        if variable.gradients_context[g] is None:
            gradient_content.append(g.value)
        else:
            # If context is a list, we handle it differently.
            context = variable.gradients_context[g]
            if isinstance(context["context"], str):
                # The context could be all string.
                criticism_and_context = GRADIENT_TEMPLATE.format(
                    feedback=g.value, **context)
                gradient_content.append(criticism_and_context)
            elif isinstance(context["context"], list):
                # The context may have a list of images / strings. In this case, we need to handle it differently.
                context_prompt = GRADIENT_MULTIPART_TEMPLATE.format(**context, feedback=g.value)
                criticism_and_context = context["context"] + [context_prompt]
                gradient_content.extend(criticism_and_context)
            else:
                raise ValueError("Context must be either a string or a list.")
    
    # Check if all instances are string
    if all(isinstance(i, str) for i in gradient_content):
        return "\n".join(gradient_content)
    else:
        return gradient_content


class Optimizer(ABC):
    """
    Base class for all optimizers.

    :param parameters: The list of parameters to optimize.
    :type parameters: List[Variable]

    :Methods:
        - zero_grad(): Clears the gradients of all parameters.
        - step(): Performs a single optimization step.
    """

    def __init__(self, parameters: List[Variable]):
        for parameter in parameters:
            if type(parameter.value) !=  str:
                raise NotImplementedError(f"We cannot yet update multimodal content and this data type: {type(parameter.value)}. We can only evaluate gradients using multimodal models. This may change soon (looking at you, GPT-5).")
        self.parameters = parameters
        
    def zero_grad(self):
        """
        Clears the gradients of all parameters.
        """
        for p in self.parameters:
            p.gradients = set()

    @abstractmethod
    def step(self):
        """
        Performs a single optimization step.
        """
        pass


class TextualGradientDescent(Optimizer):
    def __init__(self, 
                 parameters: List[Variable], 
                 verbose: int=0, 
                 engine: Union[EngineLM, str]=None, 
                 constraints: List[str]=None,
                 new_variable_tags: List[str]=None,
                 optimizer_system_prompt: str=OPTIMIZER_SYSTEM_PROMPT,
                 in_context_examples: List[str]=None,
                 gradient_memory: int=0):
        """TextualGradientDescent optimizer

        :param engine: the engine to use for updating variables
        :type engine: EngineLM
        :param parameters: the parameters to optimize
        :type parameters: List[Variable]
        :param verbose: whether to print iterations, defaults to 0
        :type verbose: int, optional
        :param constraints: a list of natural language constraints, defaults to []
        :type constraints: List[str], optional
        :param optimizer_system_prompt: system prompt to the optimizer, defaults to textgrad.prompts.OPTIMIZER_SYSTEM_PROMPT. Needs to accept new_variable_start_tag and new_variable_end_tag
        :type optimizer_system_prompt: str, optional
        :param in_context_examples: a list of in-context examples, defaults to []
        :type in_context_examples: List[str], optional
        :param gradient_memory: the number of past gradients to store, defaults to 0
        :type gradient_memory: int, optional
        """
        super().__init__(parameters)

        if new_variable_tags is None:
            new_variable_tags = ["<IMPROVED_VARIABLE>", "</IMPROVED_VARIABLE>"]

        self.engine = validate_engine_or_get_default(engine)
        self.verbose = verbose
        self.constraints = constraints if constraints is not None else []
        self.optimizer_system_prompt = optimizer_system_prompt.format(new_variable_start_tag=new_variable_tags[0], new_variable_end_tag=new_variable_tags[1])
        self.do_constrained = (len(self.constraints) > 0)
        self.new_variable_tags = new_variable_tags
        self.in_context_examples = in_context_examples if in_context_examples is not None else []
        self.do_in_context_examples = (len(self.in_context_examples) > 0)
        self.gradient_memory = gradient_memory
        self.gradient_memory_dict = defaultdict(list)
        self.do_gradient_memory = (gradient_memory > 0)

    @property
    def constraint_text(self):
        """
        Returns a formatted string representation of the constraints.

        :return: A string containing the constraints in the format "Constraint {index}: {constraint}".
        :rtype: str
        """
        constraints_ordered = [f"Constraint {i+1}: {constraint}" for i, constraint in enumerate(self.constraints)]
        return "\n".join(constraints_ordered)
    
    def get_gradient_memory_text(self, variable: Variable):
        grad_memory = ""
        variable_grad_memory = self.gradient_memory_dict[variable][-self.gradient_memory:]
        for i, grad_info in enumerate(variable_grad_memory):
            grad_memory += f"\n<FEEDBACK-{i+1}> {grad_info['value']}</FEEDBACK-{i+1}>\n"
        return grad_memory
    
    def update_gradient_memory(self, variable: Variable):
        self.gradient_memory_dict[variable].append({"value": variable.get_gradient_text()})
    
    def _update_prompt(self, variable: Variable) -> Union[str, List[Union[str, bytes]]]:
        grad_memory = self.get_gradient_memory_text(variable)
        optimizer_information = {
            "variable_desc": variable.get_role_description(),
            "variable_value": variable.value,
            "variable_grad": get_gradient_and_context_text(variable),
            "variable_short": variable.get_short_value(),
            "constraint_text": self.constraint_text,
            "new_variable_start_tag": self.new_variable_tags[0],
            "new_variable_end_tag": self.new_variable_tags[1],
            "in_context_examples": "\n".join(self.in_context_examples),
            "gradient_memory": grad_memory
        }
        
        prompt = construct_tgd_prompt(do_constrained=self.do_constrained, 
                                      do_in_context_examples=(self.do_in_context_examples and (len(self.in_context_examples) > 0)),
                                      do_gradient_memory=(self.do_gradient_memory and (grad_memory != "")),
                                      **optimizer_information)
        
        logger.info(f"TextualGradientDescent prompt for update", extra={"prompt": prompt})
        return prompt

    def step(self):
        """
        Perform a single optimization step.
        This method updates the parameters of the optimizer by generating new text using the engine and updating the parameter values accordingly.
        It also logs the optimizer response and the updated text.
        Returns:
            None
        """
        for parameter in self.parameters:
            prompt_update_parameter = self._update_prompt(parameter)
            new_text = self.engine(prompt_update_parameter, system_prompt=self.optimizer_system_prompt)
            logger.info(f"TextualGradientDescent optimizer response", extra={"optimizer.response": new_text})
            try:
                new_value = new_text.split(self.new_variable_tags[0])[1].split(self.new_variable_tags[1])[0].strip()
            # Check if we got a cannot be indexed error
            except IndexError:
                logger.error(f"TextualGradientDescent optimizer response could not be indexed", extra={"optimizer.response": new_text})
                raise IndexError(f"TextualGradientDescent optimizer response could not be indexed. This can happen if the optimizer model cannot follow the instructions. You can try using a stronger model, or somehow reducing the context of the optimization. Response: {new_text}")
            parameter.set_value(new_value)
            logger.info(f"TextualGradientDescent updated text", extra={"parameter.value": parameter.value})
            if self.verbose:
                print("-----------------------TextualGradientDescent------------------------")
                print(parameter.value)
            
            if self.do_gradient_memory:
                self.update_gradient_memory(parameter)


class TextualGradientDescentwithMomentum(Optimizer):
    def __init__(self, 
                 engine: Union[str, EngineLM], 
                 parameters: List[Variable], 
                 momentum_window: int = 0, 
                 constraints: List[str]=None,
                 new_variable_tags: List[str]=None,
                 in_context_examples: List[str]=None,
                 optimizer_system_prompt: str=OPTIMIZER_SYSTEM_PROMPT):
        super().__init__(parameters)

        if new_variable_tags is None:
            new_variable_tags = ["<IMPROVED_VARIABLE>", "</IMPROVED_VARIABLE>"]

        self.engine = validate_engine_or_get_default(engine)
        
        if momentum_window == 0:
            return TextualGradientDescent(engine=engine, parameters=parameters, constraints=constraints)

        # Each item in the momentum storage will include past value and the criticism
        self.momentum_storage = [[] for _ in range(len(parameters))]
        self.momentum_window = momentum_window
        self.do_momentum = True
        self.constraints = constraints if constraints is not None else []
        self.do_constrained = (len(self.constraints) > 0)
        self.optimizer_system_prompt = optimizer_system_prompt.format(new_variable_start_tag=new_variable_tags[0], new_variable_end_tag=new_variable_tags[1])
        self.new_variable_tags = new_variable_tags
        self.in_context_examples = in_context_examples if in_context_examples is not None else []
        self.do_in_context_examples = (len(self.in_context_examples) > 0)

        logger.info(f"TextualGradientDescent initialized with momentum window: {momentum_window}")

    @property
    def constraint_text(self):
        constraints_ordered = [f"Constraint {i+1}: {constraint}" for i, constraint in enumerate(self.constraints)]
        return "\n".join(constraints_ordered)
    
    def _update_prompt(self, variable: Variable, momentum_storage_idx: int):
        past_values = ""
        
        past_n_steps = self.momentum_storage[momentum_storage_idx]
        for i, step_info in enumerate(past_n_steps):
            past_values += f"\n{variable.get_role_description()} at Step {i + 1}: {step_info['value']}.\n"

        optimizer_information = {
            "variable_desc": variable.get_role_description(),
            "variable_value": variable.value,
            "variable_grad": variable.get_gradient_text(),
            "variable_short": variable.get_short_value(),
            "constraint_text": self.constraint_text,
            "past_values": past_values,
            "new_variable_start_tag": self.new_variable_tags[0],
            "new_variable_end_tag": self.new_variable_tags[1],
            "in_context_examples": "\n".join(self.in_context_examples)
        }
        
        prompt = construct_tgd_prompt(do_momentum=(self.do_momentum and (past_values != "")),
                                      do_constrained=self.do_constrained,
                                      do_in_context_examples=(self.do_in_context_examples and (len(self.in_context_examples) > 0)),
                                      **optimizer_information)

        logger.info(f"TextualGradientwithMomentum prompt for update", extra={"prompt": prompt})
        return prompt


    def _update_momentum_storage(self, variable: Variable, momentum_storage_idx: int):
        if len(self.momentum_storage[momentum_storage_idx]) >= self.momentum_window:
            self.momentum_storage[momentum_storage_idx].pop(0)
        
        self.momentum_storage[momentum_storage_idx].append({"value": variable.value, "gradients": get_gradient_and_context_text(variable)})
        
    def step(self):
        for idx, parameter in enumerate(self.parameters):
            self._update_momentum_storage(parameter, momentum_storage_idx=idx)
            prompt_update_parameter = self._update_prompt(parameter, momentum_storage_idx=idx)
            new_text = self.engine(prompt_update_parameter, system_prompt=self.optimizer_system_prompt)
            logger.info(f"TextualGradientDescentwithMomentum optimizer response", extra={"optimizer.response": new_text})
            try:
                new_value = new_text.split(self.new_variable_tags[0])[1].split(self.new_variable_tags[1])[0].strip()
            # Check if we got a cannot be indexed error
            except IndexError:
                logger.error(f"TextualGradientDescent optimizer response could not be indexed", extra={"optimizer.response": new_text})
                raise IndexError(f"TextualGradientDescent optimizer response could not be indexed. This can happen if the optimizer model cannot follow the instructions. You can try using a stronger model, or somehow reducing the context of the optimization. Response: {new_text}")
            parameter.set_value(new_value)
            logger.info(f"TextualGradientDescentwithMomentum updated text", extra={"parameter.value": parameter.value})


class TextualGradientDescentwithLR(TextualGradientDescent):
    """TextualGradientDescent with a textual learning rate, in the spirit of SkillOpt.

    Each ``step()`` (one training batch) runs three stages:

    1. **Reflect**  -- one optimizer call per gradient (i.e. per training sample's
       feedback) that turns the feedback into bullet-point edit proposals.
    2. **Aggregate** -- pool all bullet edits across the batch.
    3. **Rank + clip + apply** -- one optimizer call that reasons about the most
       important edits, keeps at most ``N`` of them (``N`` = the textual learning
       rate from the scheduler), and applies only those to the variable.

    ``N`` is scheduled over training via
    :func:`textgrad.optimizer.lr_scheduler.build_scheduler`
    (constant / linear / cosine / autonomous).
    """

    def __init__(self,
                 parameters: List[Variable],
                 lr_mode: str = "cosine",
                 max_lr: int = 4,
                 min_lr: int = 2,
                 total_steps: int = 1,
                 reflect_workers: int = 4,
                 **kwargs):
        """
        :param lr_mode: schedule for the edit budget N: 'constant', 'linear', 'cosine', or 'autonomous'
        :param max_lr: largest edit budget (used at the start of training)
        :param min_lr: smallest edit budget / floor the schedule decays toward
        :param total_steps: number of optimization steps the schedule spans (set this to your
            number of ``step()`` calls; with the default of 1 the linear/cosine schedules stay at max_lr)
        :param reflect_workers: max parallel optimizer calls for the per-sample reflect stage
        :param kwargs: forwarded to :class:`TextualGradientDescent` (engine, constraints, verbose, ...)
        """
        super().__init__(parameters, **kwargs)
        self.scheduler = build_scheduler(lr_mode, max_lr=max_lr, min_lr=min_lr, total_steps=total_steps)
        self.reflect_workers = max(1, reflect_workers)

    def _format_single_gradient(self, variable: Variable, gradient: Variable) -> str:
        """Render one gradient (one sample's feedback) with its context as text."""
        context = variable.gradients_context[gradient]
        if context is None:
            return gradient.value
        if isinstance(context["context"], str):
            return GRADIENT_TEMPLATE.format(feedback=gradient.value, **context)
        # Multipart (e.g. images) -- reflect on the textual feedback only.
        return gradient.value

    def _reflect_one_sample(self, variable: Variable, gradient: Variable) -> List[str]:
        """Stage 1: turn one sample's feedback into a list of bullet-point edits."""
        prompt = REFLECT_PROMPT_TEMPLATE.format(
            variable_desc=variable.get_role_description(),
            variable_value=variable.value,
            sample_feedback=self._format_single_gradient(variable, gradient),
        )
        response = self.engine(prompt, system_prompt=REFLECT_SYSTEM_PROMPT)
        logger.info("TextualGradientDescentwithLR reflect response", extra={"reflect.response": response})
        return _parse_bullets(response)

    def _collect_batch_edits(self, variable: Variable) -> List[str]:
        """Stages 1-2: reflect on every gradient (in parallel) and pool the bullets."""
        gradients = list(variable.gradients)
        if not gradients:
            return []
        bullets: List[str] = []
        if self.reflect_workers == 1 or len(gradients) == 1:
            per_sample = [self._reflect_one_sample(variable, g) for g in gradients]
        else:
            with ThreadPoolExecutor(max_workers=min(self.reflect_workers, len(gradients))) as ex:
                per_sample = list(ex.map(lambda g: self._reflect_one_sample(variable, g), gradients))
        for i, sample_bullets in enumerate(per_sample):
            if self.verbose:
                print(f"  [1/3 reflect] sample {i + 1}/{len(gradients)} -> {len(sample_bullets)} edit(s)")
            bullets.extend(sample_bullets)
        return bullets

    def _select_and_apply(self, variable: Variable, bullets: List[str], learning_rate: int) -> str:
        """Stage 3: rank the pooled edits, keep at most N, and apply only those."""
        candidate_edits = "\n".join(f"- {b}" for b in bullets)
        if learning_rate < NO_LIMIT:
            budget_instruction = LR_SELECT_INSTRUCTION.format(learning_rate=learning_rate)
        else:
            budget_instruction = LR_SELECT_INSTRUCTION_AUTONOMOUS
        constraint_text = ""
        if self.do_constrained:
            constraint_text = CONSTRAINT_PROMPT_ADDITION.format(constraint_text=self.constraint_text)

        prompt = LR_APPLY_PROMPT_TEMPLATE.format(
            variable_desc=variable.get_role_description(),
            variable_value=variable.value,
            candidate_edits=candidate_edits,
            budget_instruction=budget_instruction,
            constraint_text=constraint_text,
            new_variable_start_tag=self.new_variable_tags[0],
            new_variable_end_tag=self.new_variable_tags[1],
        )
        new_text = self.engine(prompt, system_prompt=self.optimizer_system_prompt)
        logger.info("TextualGradientDescentwithLR apply response", extra={"optimizer.response": new_text})
        try:
            return new_text.split(self.new_variable_tags[0])[1].split(self.new_variable_tags[1])[0].strip()
        except IndexError:
            logger.error("TextualGradientDescentwithLR apply response could not be indexed", extra={"optimizer.response": new_text})
            raise IndexError(f"TextualGradientDescentwithLR apply response could not be indexed. Response: {new_text}")

    def step(self):
        for parameter in self.parameters:
            learning_rate = self.scheduler.get_lr()
            bullets = self._collect_batch_edits(parameter)
            if not bullets:
                logger.info("TextualGradientDescentwithLR: no candidate edits this step; skipping update")
                if self.verbose:
                    print("[TextualGradientDescentwithLR] no candidate edits — skipping update")
                continue
            new_value = self._select_and_apply(parameter, bullets, learning_rate)
            parameter.set_value(new_value)
            logger.info("TextualGradientDescentwithLR updated text", extra={"parameter.value": parameter.value})
            if self.verbose:
                budget = "unbounded" if learning_rate >= NO_LIMIT else str(learning_rate)
                print(f"[TextualGradientDescentwithLR] step {self.scheduler.current_step}: "
                      f"pooled {len(bullets)} edit(s), kept <= {budget}")
                print("-----------------------TextualGradientDescentwithLR------------------------")
                print(parameter.value)
        self.scheduler.step()


def _parse_bullets(text: str) -> List[str]:
    """Extract bullet-point edits from a reflect response.

    Recognizes ``- ``/``* ``/``• `` markers and ``1.``/``1)`` numbering. If the model
    returns ``NONE`` (no edits) returns an empty list; if it ignores the bullet format
    entirely, falls back to treating the whole non-empty response as one edit.
    """
    bullets: List[str] = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.upper() == "NONE":
            continue
        matched = False
        for marker in ("- ", "* ", "• "):
            if s.startswith(marker):
                bullets.append(s[len(marker):].strip())
                matched = True
                break
        if matched:
            continue
        m = re.match(r"^\d+[\.\)]\s+(.*)", s)
        if m:
            bullets.append(m.group(1).strip())
    if not bullets:
        stripped = text.strip()
        if stripped and stripped.upper() != "NONE":
            bullets.append(stripped)
    return [b for b in bullets if b]

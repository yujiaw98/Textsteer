GLOSSARY_TEXT = """
### Glossary of tags that will be sent to you:
# - <LM_SYSTEM_PROMPT>: The system prompt for the language model.
# - <LM_INPUT>: The input to the language model.
# - <LM_OUTPUT>: The output of the language model.
# - <FEEDBACK>: The feedback to the variable.
# - <CONVERSATION>: The conversation history.
# - <FOCUS>: The focus of the optimization.
# - <ROLE>: The role description of the variable."""

### Optimize Prompts

# System prompt to TGD
OPTIMIZER_SYSTEM_PROMPT = (
    "You are part of an optimization system that improves text (i.e., variable). "
    "You will be asked to creatively and critically improve prompts, solutions to problems, code, or any other text-based variable. "
    "You will receive some feedback, and use the feedback to improve the variable. "
    "The feedback may be noisy, identify what is important and what is correct. "
    "Pay attention to the role description of the variable, and the context in which it is used. "
    "This is very important: You MUST give your response by sending the improved variable between {new_variable_start_tag} {{improved variable}} {new_variable_end_tag} tags. "
    "The text you send between the tags will directly replace the variable.\n\n"
    f"{GLOSSARY_TEXT}"
)

# TGD update instruction
TGD_PROMPT_PREFIX = (
    "Here is the role of the variable you will improve: <ROLE>{variable_desc}</ROLE>.\n\n"
    "The variable is the text within the following span: <VARIABLE> {variable_short} </VARIABLE>\n\n"
    "Here is the context and feedback we got for the variable:\n\n"
    "<CONTEXT>{variable_grad}</CONTEXT>\n\n"
    "Improve the variable ({variable_desc}) using the feedback provided in <FEEDBACK> tags.\n"
)

# If the gradients are in a multi-part container
TGD_MULTIPART_PROMPT_INIT = (
    "Here is the role of the variable you will improve: <ROLE>{variable_desc}</ROLE>.\n\n"
    "The variable is the text within the following span: <VARIABLE> {variable_short} </VARIABLE>\n\n"
    "Here is the context and feedback we got for the variable:\n\n"
)

TGD_MULTIPART_PROMPT_PREFIX = (
    "Improve the variable ({variable_desc}) using the feedback provided in <FEEDBACK> tags.\n"
)

TGD_PROMPT_SUFFIX  = (
    "Send the improved variable "
    "in the following format:\n\n{new_variable_start_tag}{{the improved variable}}{new_variable_end_tag}\n\n"
    "Send ONLY the improved variable between the <IMPROVED_VARIABLE> tags, and nothing else."
)

MOMENTUM_PROMPT_ADDITION = (
    "Here are the past iterations of this variable:\n\n"
    "<PAST_ITERATIONS>{past_values}</PAST_ITERATIONS>\n\n"
    "Similar feedbacks across different steps suggests that the modifications to the variable are insufficient." 
    "If this is the case, please make more significant changes to the variable.\n\n"
)

CONSTRAINT_PROMPT_ADDITION = (
    "You must follow the following constraints:\n\n"
    "<CONSTRAINTS>{constraint_text}</CONSTRAINTS>\n\n"
)

IN_CONTEXT_EXAMPLE_PROMPT_ADDITION = (
    "You must base on the following examples when modifying the {variable_desc}:\n\n"
    "<EXAMPLES>{in_context_examples}</EXAMPLES>\n\n"
)

def construct_tgd_prompt(do_momentum: bool = False,
                         do_constrained: bool = False,
                         do_in_context_examples: bool = False,
                         **optimizer_kwargs):
    """
    Construct the textual gradient descent prompt.

    :param do_momentum: Whether to include momentum in the prompt.
    :type do_momentum: bool, optional
    :param do_constrained: Whether to include constraints in the prompt.
    :type do_constrained: bool, optional
    :param do_in_context_examples: Whether to include in-context examples in the prompt.
    :type do_in_context_examples: bool, optional
    :param optimizer_kwargs: Additional keyword arguments for formatting the prompt. These will be things like the variable description, gradient, past values, constraints, and in-context examples.
    :return: The TGD update prompt.
    :rtype: str
    """

    if isinstance(optimizer_kwargs["variable_grad"], str):
        multipart=False
        prompt = TGD_PROMPT_PREFIX.format(**optimizer_kwargs)
        
    else:
        gradient_context = optimizer_kwargs["variable_grad"]
        gradient_context = [TGD_MULTIPART_PROMPT_INIT.format(**optimizer_kwargs)] + gradient_context
        multipart=True
        prompt = TGD_MULTIPART_PROMPT_PREFIX.format(**optimizer_kwargs)
           
    if do_momentum:
        prompt += MOMENTUM_PROMPT_ADDITION.format(**optimizer_kwargs)

    if do_constrained:
        prompt += CONSTRAINT_PROMPT_ADDITION.format(**optimizer_kwargs)

    if do_in_context_examples:
        prompt += IN_CONTEXT_EXAMPLE_PROMPT_ADDITION.format(**optimizer_kwargs)

    prompt += TGD_PROMPT_SUFFIX.format(**optimizer_kwargs)

    if not multipart:
        return prompt
    
    else:
        return gradient_context + [prompt]

# This is how we save gradients to the variable.
GRADIENT_TEMPLATE = (
    "Here is a conversation:\n\n<CONVERSATION>{context}</CONVERSATION>\n\n"
    "This conversation is potentially part of a larger system. The output is used as {response_desc}\n\n"
    "Here is the feedback we got for {variable_desc} in the conversation:\n\n<FEEDBACK>{feedback}</FEEDBACK>\n\n"
)
GRADIENT_MULTIPART_TEMPLATE = (
    "Above is a conversation with a language model.\n"
    "This conversation is potentially part of a larger system. The output is used as {response_desc}\n\n"
    "Here is the feedback we got for {variable_desc} in the conversation:\n\n<FEEDBACK>{feedback}</FEEDBACK>\n\n"
)

### Learning-rate optimizer (reflect -> aggregate -> rank/clip) prompts, ported from SkillOpt.

# Stage 1 (REFLECT): turn ONE training sample's feedback into bullet-point edit proposals.
REFLECT_SYSTEM_PROMPT = (
    "You are part of an optimization system that improves a text variable (e.g., a prompt, a solution, or code). "
    "You will be shown the current variable and feedback derived from a SINGLE training example. "
    "Your job is to translate that feedback into concrete, minimal, actionable edits to the variable.\n\n"
    "Rules:\n"
    "- Output ONLY a bullet list, one edit per line starting with '- '.\n"
    "- Each bullet must be a single, self-contained change: add / remove / rephrase one specific instruction or detail.\n"
    "- Propose only changes this example actually justifies. Be specific about what to change and why.\n"
    "- Do NOT rewrite the whole variable and do NOT output the improved variable.\n"
    "- If this example reveals no needed change, output exactly: NONE"
)

REFLECT_PROMPT_TEMPLATE = (
    "Here is the role of the variable: <ROLE>{variable_desc}</ROLE>\n\n"
    "Current variable value:\n<VARIABLE>\n{variable_value}\n</VARIABLE>\n\n"
    "Here is the context and feedback from ONE training example:\n\n<FEEDBACK>\n{sample_feedback}\n</FEEDBACK>\n\n"
    "List the concrete edits to the variable that this single example implies, as a bullet list."
)

# Stage 3 (RANK + CLIP + APPLY): select the top-N edits from the pooled batch and apply only those.
# Uses OPTIMIZER_SYSTEM_PROMPT as the system prompt (it already requires the improved-variable tags).
LR_APPLY_PROMPT_TEMPLATE = (
    "Here is the role of the variable you will improve: <ROLE>{variable_desc}</ROLE>\n\n"
    "Current variable value:\n<VARIABLE>\n{variable_value}\n</VARIABLE>\n\n"
    "Below is a pool of candidate edits, gathered from every training example in the current batch. "
    "They may overlap, conflict, or vary in importance:\n\n"
    "<CANDIDATE_EDITS>\n{candidate_edits}\n</CANDIDATE_EDITS>\n\n"
    "{budget_instruction}"
    "{constraint_text}"
    "Send the improved variable in the following format:\n\n"
    "{new_variable_start_tag}{{the improved variable}}{new_variable_end_tag}\n\n"
    "Send ONLY the improved variable between the tags, and nothing else."
)

# Bounded budget: keep at most N edits (N = textual learning rate).
LR_SELECT_INSTRUCTION = (
    "The textual learning rate (edit budget) for this step is {learning_rate}. "
    "First, reason about which candidate edits matter most for the variable's role; "
    "merge duplicates and discard redundant or conflicting ones. "
    "Then select at most {learning_rate} of the most important edits. "
    "Apply ONLY those selected edits to the variable, leaving everything else unchanged. "
    "Do not introduce any change that is not among your selected edits.\n\n"
)

# Autonomous budget (no cap): the optimizer decides how many edits to apply.
LR_SELECT_INSTRUCTION_AUTONOMOUS = (
    "There is no fixed edit budget this step. Reason about which candidate edits are worth making, "
    "merge duplicates, discard redundant or conflicting ones, and apply all edits you judge genuinely "
    "beneficial to the variable, leaving everything else unchanged.\n\n"
)


# Policy-gradient-style prompt optimization. Candidate text variables are
# sampled by the editor model, evaluated externally with scalar task rewards,
# and combined using reward advantages.
CANDIDATE_PROPOSAL_PROMPT_TEMPLATE = (
    "Here is the role of the variable you will improve: <ROLE>{variable_desc}</ROLE>\n\n"
    "Current variable value:\n<VARIABLE>\n{variable_value}\n</VARIABLE>\n\n"
    "You are sampling candidate {sample_index} of {num_candidates} from a policy over improved variables. "
    "Sampling nonce: {sample_nonce}; do not mention this nonce in the sampled variable. "
    "Produce one plausible revised variable. Make a meaningful but not reckless exploration step: "
    "preserve useful behavior, change only what the feedback or task role justifies, and keep the result ready "
    "to be evaluated directly.\n\n"
    "<FEEDBACK_CONTEXT>\n{gradient_text}\n</FEEDBACK_CONTEXT>\n\n"
    "{constraint_text}"
    "{in_context_examples_text}"
    "Send the sampled variable in the following format:\n\n"
    "{new_variable_start_tag}{{the sampled variable}}{new_variable_end_tag}\n\n"
    "Send ONLY the sampled variable between the tags, and nothing else."
)

ADVANTAGE_GUIDED_UPDATE_PROMPT_TEMPLATE = (
    "Here is the role of the variable you will improve: <ROLE>{variable_desc}</ROLE>\n\n"
    "Current variable value:\n<VARIABLE>\n{variable_value}\n</VARIABLE>\n\n"
    "This optimization step used a REINFORCE-style policy-gradient estimate over text candidates. "
    "Each candidate was sampled from the optimizer policy and evaluated by the task reward. "
    "Positive advantage means the candidate performed better than the baseline and its useful changes should be reinforced. "
    "Negative advantage means the candidate performed worse than the baseline and its harmful changes should be avoided. "
    "Use the rewards as the primary signal; use textual feedback only as supporting evidence.\n\n"
    "<POLICY_ROLLOUTS>\n{rollout_text}\n</POLICY_ROLLOUTS>\n\n"
    "<FEEDBACK_CONTEXT>\n{gradient_text}\n</FEEDBACK_CONTEXT>\n\n"
    "{constraint_text}"
    "{in_context_examples_text}"
    "Synthesize the next variable by reinforcing recurring high-advantage edits, discarding low-advantage edits, "
    "and keeping stable parts of the current variable that the rollouts did not disprove.\n\n"
    "Send the improved variable in the following format:\n\n"
    "{new_variable_start_tag}{{the improved variable}}{new_variable_end_tag}\n\n"
    "Send ONLY the improved variable between the tags, and nothing else."
)

# Textual learning rate for policy-gradient synthesis. Unlike a numeric model
# learning rate, this bounds how many distinct prompt edits the editor may
# transfer from the advantage-ranked rollouts into the next prompt.
EDIT_BUDGET_INSTRUCTION = (
    "The textual learning rate (edit budget) for this step is {learning_rate}. "
    "Treat useful differences in positive-advantage candidates as proposed edits, "
    "rank those edits using reward advantage and the supporting feedback, and apply "
    "at most {learning_rate} distinct edits to the current variable. Merge duplicate "
    "edits, reject changes supported only by non-positive-advantage candidates, and "
    "leave all other parts of the current variable unchanged.\n\n"
)

EDIT_BUDGET_INSTRUCTION_AUTONOMOUS = (
    "There is no fixed textual edit budget this step. Treat useful differences in "
    "positive-advantage candidates as proposed edits, merge duplicates, reject "
    "changes supported only by non-positive-advantage candidates, and apply all "
    "remaining edits that are justified by reward advantage and supporting feedback.\n\n"
)


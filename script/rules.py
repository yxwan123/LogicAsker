# coding=utf-8
# Copyright 2023 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Inference rule definition and renderinf routines.
"""


import random

import inference_methods as im
import logic_inference_lib as lib


def generate_universal_variation(rule):
  """Generates a universally quantified version of a propositionl rule."""

  # import ipdb; ipdb.set_trace()
  bindings = {}
  for p in rule.propositions:
    bindings[p] = [p.upper(), "var x"]
  rule2 = im.apply_bindings(bindings, rule)
  rule2.premises = [["forall", "var x", x] for x in rule2.premises]
  rule2.inferences = [["forall", "var x", x] for x in rule2.inferences]
  rule2.contradictions = []
  for p in rule2.inferences:
    # negate the inferences:
    p2 = ["~", p]
    if p[0] == "forall":
      q = p[2]
      if q[0] == "~":
        q2 = q[1]
      else:
        q2 = ["~", q]
      p2 = ["exists", p[1], q2]
    rule2.contradictions.append(p2)
  rule2.unrelated = [["forall", "var x", x] for x in rule2.unrelated]  # Ignore the unrelated
  rule2.rule_name = "universal " + rule.rule_name
  return rule2


def generate_existential_variations(rule):
  """Generates existentially quantified versions of a propositionl rule."""

  rules = []
  bindings = {}
  for p in rule.propositions:
    bindings[p] = [p.upper(), "var x"]

  for i in range(len(rule.premises)):
    rule2 = im.apply_bindings(bindings, rule)
    rule2.premises = [["forall", "var x", x] for x in rule2.premises]
    # One of the premises will be existentially quantified
    rule2.premises[i][0] = "exists"
    rule2.inferences = [["exists", "var x", x] for x in rule2.inferences]
    rule2.contradictions = []
    for p in rule2.inferences:
      # negate the inferences:
      p2 = ["~", p]
      if p[0] == "exists":
        q = p[2]
        if q[0] == "~":
          q2 = q[1]
        else:
          q2 = ["~", q]
        p2 = ["forall", p[1], q2]
      rule2.contradictions.append(p2)
    rule2.unrelated = [["exists", "var x", x] for x in rule2.unrelated]  # Ignore the unrelated
    rule2.rule_name = "existential " + rule.rule_name
    rules.append(rule2)
  return rules


def capitalize(text):
  if text:
    return text[0].upper() + text[1:]
  else:
    return text


def render_logic_clause(clause):
  clause_string, _ = render_logic_clause_internal(clause)
  # Simplify double negations:
  clause_string = clause_string.replace("~~", "")
  return clause_string


def render_logic_clause_internal(predicate):
  """Converts a clause to string and also returns whether it's atomic."""

  if isinstance(predicate, str):
    if im.is_variable(predicate):
      # A variable:
      return predicate[4:], True
    else:
      return predicate, True
  if len(predicate) == 1:
    if im.is_variable(predicate[0]):
      # A variable:
      return predicate[0][4:], True
    else:
      if isinstance(predicate[0], str):
        return predicate[0], True
      else:
        raise ValueError(f"Cannot render {predicate}")
  elif len(predicate) == 2:
    if predicate[0] == "~":
      argument, atomic = render_logic_clause_internal(predicate[1])
      if atomic:
        return f"~{argument}", True
      else:
        return f"~({argument})", False
    else:
      # Something of the form "P(x)"
      argument, atomic = render_logic_clause_internal(predicate[1])
      return f"{predicate[0]}({argument})", True
  elif len(predicate) == 3:
    if predicate[0] == "forall" or predicate[0] == "exists":
      argument1, atomia = render_logic_clause_internal(predicate[1])
      argument2, atomic2 = render_logic_clause_internal(predicate[2])
      return f"{predicate[0]} {argument1}: {argument2}", False
    else:
      argument1, atomia = render_logic_clause_internal(predicate[1])
      argument2, atomic2 = render_logic_clause_internal(predicate[2])
      string1 = argument1 if atomia else f"({argument1})"
      string2 = argument2 if atomic2 else f"({argument2})"
      return f"{string1} {predicate[0]} {string2}", False
  else:
    raise ValueError(f"Cannot render: {predicate}")


def render_language_clause(clause, propositions, nl_propositions,
                           form=0):
  """Renders a clause to natural language.

  Args:
    clause: the clause to render.
    propositions: the atomic propositions in this clause.
    nl_propositions: natural language versions of the propositions in
      "propositions".
    form: 0: assertive, 1: hypothetical, 2: then, 3, 4, 5: the same but negated.

  Returns:
    The clause in natural language (a string).
  """

  if len(clause) == 1:
    idx = propositions.index(clause[0])
    return nl_propositions[idx][form]

  elif clause in propositions:
    idx = propositions.index(clause)
    return nl_propositions[idx][form]

  if clause[0] == "~":
    return render_language_clause(clause[1], propositions,
                                  nl_propositions, form=(form + 3) % 6)

  elif clause[0] == "forall":
    argument1 = render_language_clause(clause[2], propositions,
                                       nl_propositions, form=2)
    neg = "not " if form >= 3 else ""
    return f"{neg}for all {clause[1][4:]}, {argument1}"

  elif clause[0] == "exists":
    argument1 = render_language_clause(clause[2], propositions,
                                       nl_propositions, form=2)
    if form >= 3:
      targets = f"there is no {clause[1][4:]} for which {argument1}"
    else:
      targets = f"there is at least one {clause[1][4:]} for which {argument1}"
    targets = render_language_predicate_special_some_case(targets, form>=3)
    return targets

  elif len(clause) == 3:
    if clause[0] == "->":
      argument1 = render_language_clause(clause[1], propositions,
                                         nl_propositions, form=1)
      argument2 = render_language_clause(clause[2], propositions,
                                         nl_propositions, form=0)
    else:
      argument1 = render_language_clause(clause[1], propositions,
                                         nl_propositions, form=form%3)
      argument2 = render_language_clause(clause[2], propositions,
                                         nl_propositions, form=form%3)
    if form < 3:
      # positive:
      if clause[0] == "->":
        return f"if {argument1}, then {argument2}"
      elif clause[0] == "<->":
        return f"{argument1} if and only if {argument2}"
      elif clause[0] == "or":
        return f"{argument1} or {argument2}"
      elif clause[0] == "and":
        return f"{argument1} and {argument2}"
      else:
        raise ValueError(f"Cannot render to language: {clause}")
    else:
      # negated:
      if clause[0] == "->":
        return f"the fact that {argument1} does not imply that {argument2}"
      elif clause[0] == "<->":
        return f"it is not true that {argument1} if and only if {argument2}"
      elif clause[0] == "or":
        return f"neither {argument1} nor {argument2}"
      elif clause[0] == "and":
        return f"the claim that {argument1} and the claim that {argument2} cannot both be true"
      else:
        raise ValueError(f"Cannot render to language: {clause}")

  else:
    raise ValueError(f"Cannot render to language: {clause}")


def render_language_predicate_special_some_case(sentence, neg=False):
  """Reformats a special case sentence into a more readable manner."""
  pattern1 = "there is at least one " if not neg else "there is no "
  pattern2 = " for which "
  if not sentence.startswith(pattern1):
    return sentence
  rest = sentence[len(pattern1):]
  if pattern2 not in rest:
    return sentence
  variable = rest[:rest.index(pattern2)]
  if not im.is_variable("var " + variable):
    return sentence
  rest = rest[len(variable) + len(pattern2):]
  pattern3 = f"{variable} is a "
  if not rest.startswith(pattern3):
    return sentence
  rest = rest[len(pattern3):]
  pattern4a = f" and {variable} is a "
  pattern4b = f" and {variable} will "
  if pattern4a in rest:
    adjective1 = rest[:rest.index(pattern4a)]
    adjective2 = rest[len(adjective1) + len(pattern4a):]
    if " " in adjective1 or " " in adjective2:
      return sentence
    return f"some {adjective1}s are {adjective2}s"
  elif pattern4b in rest:
    adjective1 = rest[:rest.index(pattern4b)]
    adjective2 = rest[len(adjective1) + len(pattern4a) - 5:]
    if " " in adjective1:
      return sentence
    return f"some {adjective1}s {adjective2}"
  else:
    return sentence


def generate_nl_proposition(functor, arguments, bindings):
  """Generates random natural language to use as a proposition.

  Args:
    functor: the functor of the proposition to generate language for.
    arguments: a list of the arguments of the proposition, if any
    bindings: a dictionary mapping constants to concrete names already used,
    and functors to predicates.

  Returns:
    This functino will return 6 strings:
    - The string in "assertive" present mode: "it is snowing"
    - The string to be used in an "if": "it snows" (e.g., "if it snows")
    - The string to be used in a "then": "it will snow"
    - And then the same three, but negated
  """

  subjects = ["James", "Mary", "Robert", "Patricia", "John", "Jennifer",
              "Michael", "Linda", "William", "Elisabeth", "David", "Barbara",
              "Richard", "Susan", "Joseph", "Jessica", "Thomas", "Sarah",
              "Charles", "Karen"]
  predicates = ["a cashier", "a janitor", "a bartender", "a server",
                "an office clerk", "a mechanic", "a carpenter",
                "an electrician", "a nurse", "a doctor", "a police officer",
                "a taxi driver", "a soldier", "a politician", "a lawyer",
                "a scientist", "an astronaut", "a poet", "an artist",
                "a sailor", "a writer", "a musician",
                "poor", "rich", "happy", "sad", "fast",
                "curious", "excited", "bored"]
  actions = [["make tea", "makes tea", "making tea"],
             ["drink water", "drinks water", "drinking water"],
             ["read a book", "reads a book", "reading a book"],
             ["play tennis", "plays tennis", "playing tennis"],
             ["play squash", "plays squash", "playing squash"],
             ["play a game", "plays a game", "playing a game"],
             ["go running", "goes running", "running"],
             ["work", "works", "working"],
             ["sleep", "sleeps", "sleeping"],
             ["cook", "cooks", "cooking"],
             ["listen to a song", "listens to a song", "listening to a song"],
             ["write a letter", "writes a letter", "writing a letter"],
             ["drive a car", "drives a car", "driving a car"],
             ["climb a mountain", "climbs a mountain", "climbing a mountain"],
             ["take a plane", "takes a plane", "taking a plane"],]
  impersonal_candidates = [
      ["snowing", "snows", "doesn't snow", "snow"],
      ["raining", "rains", "doesn't rain", "rain"],
      ["sunny", "is sunny", "is not sunny", "be sunny"],
      ["cloudy", "is cloudy", "is not cloudy", "be cloudy"],
      ["windy", "is windy", "is not windy", "be windy"],
      ["cold", "is cold", "is not cold", "be cold"],
      ["late", "is late", "is not late", "be late"],
      ["overcast", "is overcast", "is not overcast", "be overcast"],
  ]

  # Remove already bound options:
  for c in bindings:
    if bindings[c] in subjects:
      subjects.remove(bindings[c])
    if bindings[c] in predicates:
      predicates.remove(bindings[c])
    if bindings[c] in actions:
      actions.remove(bindings[c])
    if bindings[c] in impersonal_candidates:
      impersonal_candidates.remove(bindings[c])

  if functor and (functor in bindings):
    if isinstance(bindings[functor], str):
      # a predicate:
      proposition_type = "subject-is"
      predicates = [bindings[functor]]
      actions = []
      impersonal_candidates = []
    else:
      # an action:
      proposition_type = "subject-action"
      # actions = [bindings[functor]]
      if len(bindings[functor]) == 3:
        predicates = []
        actions = [bindings[functor]]
        impersonal_candidates = []
      elif len(bindings[functor]) == 4:
        predicates = []
        actions = []
        impersonal_candidates = [bindings[functor]]
      else:
        raise ValueError("proposition bound to a value not found in either "
                         "actions nor impsersonal_candidates!")

  new_subject_binding = None  # if we need to bind the subject
  if len(arguments) == 1:
    if im.is_variable(arguments[0]):
      # Variable:
      subjects = [arguments[0][4:]]
    else:
      # Constant:
      if arguments[0] in bindings:
        subjects = [bindings[arguments[0]]]
      new_subject_binding = arguments[0]
    choices = []
    if predicates:
      choices.append("subject-is")
    if actions:
      choices.append("subject-action")
    proposition_type = random.choice(choices)
  elif not arguments:
    choices = []
    if impersonal_candidates:
      choices.append("impersonal")
    if predicates:
      choices.append("subject-is")
    if actions:
      choices.append("subject-action")
    proposition_type = random.choice(choices)
  else:
    raise ValueError("Propositions with more than 1 argument not " +
                     f"supported!: {functor} + {arguments}")

  if proposition_type == "impersonal":
    choice = random.choice(impersonal_candidates)
    bindings[functor] = choice
    return (f"it is {choice[0]}", f"it {choice[1]}", f"it will {choice[3]}",
            f"it is not {choice[0]}", f"it {choice[2]}",
            f"it will not {choice[3]}")
  elif proposition_type == "subject-is":
    subject = random.choice(subjects)
    predicate = random.choice(predicates)
    if new_subject_binding:
      bindings[new_subject_binding] = subject
    if functor:
      bindings[functor] = predicate
    return (f"{subject} is {predicate}", f"{subject} were {predicate}",
            f"{subject} is {predicate}",
            f"{subject} is not {predicate}", f"{subject} were not {predicate}",
            f"{subject} is not {predicate}")
  elif proposition_type == "subject-action":
    subject = random.choice(subjects)
    predicate = random.choice(actions)
    if new_subject_binding:
      bindings[new_subject_binding] = subject
    if functor:
      bindings[functor] = predicate
    return (f"{subject} is {predicate[2]}", f"{subject} {predicate[1]}",
            f"{subject} will {predicate[0]}",
            f"{subject} is not {predicate[2]}",
            f"{subject} doesn't {predicate[0]}",
            f"{subject} will not {predicate[0]}")
  else:
    raise ValueError(f"Unsuported proposion type: {proposition_type}")


def precompute_rules():
  """Precompute the list of rules to be used by the generation functions."""

  lib.PROPOSITIONAL_INFERENCE_RULES = [
      lib.InferenceRule(
          [["->", ["p"], ["q"]], ["p"]],  # premises
          [["q"]],                        # inferences
          [["~", ["q"]]],                 # contradictions
          [["r"], ["~", ["r"]]],          # unrelated
          ["p", "q", "r"],                # list of propositions used
          "modus ponens"),                # rule name

      lib.InferenceRule(
          [["->", ["p"], ["q"]], ["~", ["q"]]],
          [["~", ["p"]]],
          [["p"]],
          [["r"], ["~", ["r"]]],
          ["p", "q", "r"],
          "modus tollens"),

      lib.InferenceRule(
          [["->", ["p"], ["~", ["q"]]], ["q"]],
          [["~", ["p"]]],
          [["p"]],
          [["r"], ["~", ["r"]]],
          ["p", "q", "r"],
          "modus tollens"),

      lib.InferenceRule(
          [["->", ["p"], ["q"]], ["->", ["q"], ["r"]]],
          [["->", ["p"], ["r"]]],
          [["~", ["->", ["p"], ["r"]]]],
          [["p"], ["q"], ["r"]],
          ["p", "q", "r"],
          "transitivity"), # HS

      lib.InferenceRule(
          [["or", ["p"], ["q"]], ["~", ["p"]]],
          [["q"]],
          [["~", ["q"]]],
          [["r"], ["~", ["r"]]],
          ["p", "q", "r"],
          "disjunctive syllogism"),

      lib.InferenceRule(
          [["p"]],
          [["or", ["p"], ["q"]]],
          [["~", ["p"]]],
          [["r"], ["~", ["r"]]],
          ["p", "q", "r"],
          "addition"),

      lib.InferenceRule(
          [["and", ["p"], ["q"]]],
          [["p"], ["q"]],
          [["~", ["p"]], ["~", ["q"]]],
          [["r"], ["~", ["r"]]],
          ["p", "q", "r"],
          "simplification"),

      lib.InferenceRule(
          [["p"], ["q"]],
          [["and", ["p"], ["q"]]],
          [["~", ["p"]], ["~", ["q"]]],
          [["r"], ["~", ["r"]]],
          ["p", "q", "r"],
          "conjunction"),

      lib.InferenceRule(
          [["or", ["p"], ["q"]], ["or", ["~", ["p"]], ["r"]]],
          [["or", ["q"], ["r"]]],
          [["~", ["or", ["q"], ["r"]]]],
          [["q"], ["r"]],
          ["p", "q", "r"],
          "resolution"),

      # lib.InferenceRule(
      #     [["or", ["p"], ["~", ["q"]]]],
      #     [["->", ["q"], ["p"]]],
      #     [["~", ["->", ["q"], ["p"]]]],
      #     [["->", ["p"], ["q"]]],
      #     ["p", "q"],
      #     "rewriting"), 

      lib.InferenceRule(
          [["->", ["p"], ["r"]], ["->", ["q"], ["r"]], ["or", ["p"], ["q"]]],
          [["r"]],
          [["~", "r"]],
          [["s"]],
          ["p", "q", "r", "s"],
          "disjunction elimination"),

      lib.InferenceRule(
          [["->", ["p"], ["q"]], ["->", ["q"], ["p"]]],
          [["<->", ["p"], ["q"]]],
          [["~", ["<->", ["p"], ["q"]]]],
          [["p"], ["q"]],
          ["p", "q"],
          "biconditional introduction"),

      lib.InferenceRule(
          [["<->", ["p"], ["q"]]],
          [["->", ["p"], ["q"]]],
          [["~", ["->", ["p"], ["q"]]]],
          [["r"], ["~", ["r"]]],
          ["p", "q", "r"],
          "biconditional elimination"),

      lib.InferenceRule(
          [["<->", ["p"], ["q"]]],
          [["->", ["q"], ["p"]]],
          [["~", ["->", ["q"], ["p"]]]],
          [["r"], ["~", ["r"]]],
          ["p", "q", "r"],
          "biconditional elimination"),

      lib.InferenceRule(
          [["<->", ["p"], ["q"]], ["~", ["p"]]],
          [["~", ["q"]]],
          [["q"]],
          [["r"], ["~", ["r"]]],
          ["p", "q", "r"],
          "biconditional elimination"),

      lib.InferenceRule(
          [["<->", ["p"], ["q"]], ["~", ["q"]]],
          [["~", ["p"]]],
          [["p"]],
          [["r"], ["~", ["r"]]],
          ["p", "q", "r"],
          "biconditional elimination")]
  # TODO: CD, DD

  lib.QUANTIFIER_INFERENCE_RULES = [
       lib.InferenceRule(
          [["forall", "var x", ["P", "var x"]]],  # premises
          [["P", "a"]],                           # conclusions
          [["~", ["P", "a"]]],                    # contradictions
          [["Q", "a"], ["~", ["Q", "a"]]],        # irrelevant
          [["P", "var x"], ["P", "a"],            # list of propositions
           ["Q", "a"]],
          "universal instantiation"),             # rule name

      lib.InferenceRule(
          [["P", "a"]],
          [["exists", "var x", ["P", "var x"]]],
          [["forall", "var x", ["~", ["P", "var x"]]]],
          [["forall", "var x", ["P", "var x"]], ["Q", "a"], ["P", "b"]],
          [["P", "a"], ["P", "var x"],
           ["Q", "a"], ["P", "b"]],
          "existential generalization"),
          
      lib.InferenceRule(
          [["forall", "var x", ["P", "var x"]]],
          [["P", "var y"]],
          [["~", ["P", "var y"]]],
          [["Q", "a"], ["~", ["Q", "a"]], ["Q", "var x"]],
          [["P", "var x"], ["P", "var y"], ["Q", "a"], ["Q", "var x"]],
          "universal instantiation"),
      
      lib.InferenceRule(
          [["P", "var x"]],
          [["forall", "var y", ["P", "var y"]]],
          [["exists", "var y", ["~", ["P", "var y"]]]],
          [["Q", "a"], ["~", ["Q", "a"]], ["forall", "var x", ["Q", "var x"]]],
          [["P", "var x"], ["P", "var y"], ["Q", "a"], ["Q", "var x"]],
          "universal generalization"),

      lib.InferenceRule(
          [["P", "var x"]],
          [["exists", "var y", ["P", "var y"]]],
          [["forall", "var y", ["~", ["P", "var y"]]]],
          [["Q", "a"], ["~", ["Q", "a"]], ["forall", "var x", ["Q", "var x"]]],
          [["P", "var x"], ["P", "var y"], ["Q", "a"], ["Q", "var x"]],
          "universal generalization"),
      ]
  
  lib.QUANTIFIER_EQUIV_RULES = [
      lib.InferenceRule(
          [["~", ["forall", "var x", ["P", "var x"]]]],
          [["exists", "var x", ["~", ["P", "var x"]]]],
          [["~", ["exists", "var x", ["~", ["P", "var x"]]]]],
          [["Q", "a"], ["~", ["Q", "a"]], ["forall", "var x", ["Q", "var x"]]],
          [["P", "var x"], ["Q", "a"], ["Q", "var x"]],
          "Law of quantifier negation"),
      
      lib.InferenceRule(
          [["exists", "var x", ["~", ["P", "var x"]]]],
          [["~", ["forall", "var x", ["P", "var x"]]]],
          [["forall", "var x", ["P", "var x"]]],
          [["Q", "a"], ["~", ["Q", "a"]], ["forall", "var x", ["Q", "var x"]]],
          [["P", "var x"], ["Q", "a"], ["Q", "var x"]],
          "Law of quantifier negation"),
      
      lib.InferenceRule(
          [["~", ["exists", "var x", ["P", "var x"]]]],
          [["forall", "var x", ["~", ["P", "var x"]]]],
          [["~", ["forall", "var x", ["~", ["P", "var x"]]]]],
          [["Q", "a"], ["~", ["Q", "a"]], ["forall", "var x", ["Q", "var x"]]],
          [["P", "var x"], ["Q", "a"], ["Q", "var x"]],
          "Law of quantifier negation"),

      lib.InferenceRule(
          [["forall", "var x", ["~", ["P", "var x"]]]],
          [["~", ["exists", "var x", ["P", "var x"]]]],
          [["exists", "var x", ["P", "var x"]]],
          [["Q", "a"], ["~", ["Q", "a"]], ["forall", "var x", ["Q", "var x"]]],
          [["P", "var x"], ["Q", "a"], ["Q", "var x"]],
          "Law of quantifier negation"),

      # Law of quantifier distribution
      lib.InferenceRule(
          [["forall", "var x", ["and", ["P", "var x"], ["Q", "var x"]]]],
          [["and", ["forall", "var x", ["P", "var x"]], ["forall", "var x", ["Q", "var x"]]]],
          [["~", ["and", ["forall", "var x", ["P", "var x"]], ["forall", "var x", ["Q", "var x"]]]]],
          [["R", "a"], ["~", ["R", "a"]], ["forall", "var x", ["R", "var x"]]],
          [["P", "var x"], ["Q", "var x"], ["R", "a"], ["R", "var x"]],
          "Law of quantifier distribution"),
      
      lib.InferenceRule(
          [["and", ["forall", "var x", ["P", "var x"]], ["forall", "var x", ["Q", "var x"]]]],
          [["forall", "var x", ["and", ["P", "var x"], ["Q", "var x"]]]],
          [["~", ["forall", "var x", ["and", ["P", "var x"], ["Q", "var x"]]]]],
          [["R", "a"], ["~", ["R", "a"]], ["forall", "var x", ["R", "var x"]]],
          [["P", "var x"], ["Q", "var x"], ["R", "a"], ["R", "var x"]],
          "Law of quantifier distribution"),

      lib.InferenceRule(
          [["exists", "var x", ["or", ["P", "var x"], ["Q", "var x"]]]],
          [["or", ["exists", "var x", ["P", "var x"]], ["exists", "var x", ["Q", "var x"]]]],
          [["~", ["or", ["exists", "var x", ["P", "var x"]], ["exists", "var x", ["Q", "var x"]]]]],
          [["R", "a"], ["~", ["R", "a"]], ["forall", "var x", ["R", "var x"]]],
          [["P", "var x"], ["Q", "var x"], ["R", "a"], ["R", "var x"]],
          "Law of quantifier distribution"),
      
      lib.InferenceRule(
          [["or", ["exists", "var x", ["P", "var x"]], ["exists", "var x", ["Q", "var x"]]]],
          [["exists", "var x", ["or", ["P", "var x"], ["Q", "var x"]]]],
          [["~", ["exists", "var x", ["or", ["P", "var x"], ["Q", "var x"]]]]],
          [["R", "a"], ["~", ["R", "a"]], ["forall", "var x", ["R", "var x"]]],
          [["P", "var x"], ["Q", "var x"], ["R", "a"], ["R", "var x"]],
          "Law of quantifier distribution"),

      lib.InferenceRule(
          [["or", ["forall", "var x", ["P", "var x"]], ["forall", "var x", ["Q", "var x"]]]],
          [["forall", "var x", ["or", ["P", "var x"], ["Q", "var x"]]]],
          [["~", ["forall", "var x", ["or", ["P", "var x"], ["Q", "var x"]]]]],
          [["R", "a"], ["~", ["R", "a"]], ["forall", "var x", ["R", "var x"]]],
          [["P", "var x"], ["Q", "var x"], ["R", "a"], ["R", "var x"]],
          "Law of quantifier distribution"),
      
      lib.InferenceRule(
          [["exists", "var x", ["and", ["P", "var x"], ["Q", "var x"]]]],
          [["and", ["exists", "var x", ["P", "var x"]], ["exists", "var x", ["Q", "var x"]]]],
          [["~", ["and", ["exists", "var x", ["P", "var x"]], ["exists", "var x", ["Q", "var x"]]]]],
          [["R", "a"], ["~", ["R", "a"]], ["forall", "var x", ["R", "var x"]]],
          [["P", "var x"], ["P", "var y"], ["Q", "var x"], ["Q", "var y"], ["R", "a"], ["R", "var x"]],
          "Law of quantifier distribution"),

      # Law of quantifier movement
      lib.InferenceRule(
          [["->", ["p"], ["forall", "var x", ["P", "var x"]]]],
          [["forall", "var x", ["->", ["p"], ["P", "var x"]]]],
          [["~", ["forall", "var x", ["->", ["p"], ["P", "var x"]]]]],
          [["Q", "a"], ["~", ["Q", "a"]], ["forall", "var x", ["Q", "var x"]]],
          ["p", ["P", "var x"], ["Q", "a"], ["Q", "var x"]],
          "Law of quantifier movement"),
      
      lib.InferenceRule(
          [["forall", "var x", ["->", ["p"], ["P", "var x"]]]],
          [["->", ["p"], ["forall", "var x", ["P", "var x"]]]],
          [["~", ["->", ["p"], ["forall", "var x", ["P", "var x"]]]]],
          [["Q", "a"], ["~", ["Q", "a"]], ["forall", "var x", ["Q", "var x"]]],
          ["p", ["P", "var x"], ["Q", "a"], ["Q", "var x"]],
          "Law of quantifier movement"),
      
      lib.InferenceRule(
          [["->", ["p"], ["exists", "var x", ["P", "var x"]]]],
          [["exists", "var x", ["->", ["p"], ["P", "var x"]]]],
          [["~", ["exists", "var x", ["->", ["p"], ["P", "var x"]]]]],
          [["Q", "a"], ["~", ["Q", "a"]], ["forall", "var x", ["Q", "var x"]]],
          ["p", ["P", "var x"], ["Q", "a"], ["Q", "var x"]],
          "Law of quantifier movement"),
      
      lib.InferenceRule(
          [["exists", "var x", ["->", ["p"], ["P", "var x"]]]],
          [["->", ["p"], ["exists", "var x", ["P", "var x"]]]],
          [["~", ["->", ["p"], ["exists", "var x", ["P", "var x"]]]]],
          [["Q", "a"], ["~", ["Q", "a"]], ["forall", "var x", ["Q", "var x"]]],
          ["p", ["P", "var x"], ["Q", "a"], ["Q", "var x"]],
          "Law of quantifier movement"),

      lib.InferenceRule(
          [["->", ["forall", "var x", ["P", "var x"]], ["p"]]],
          [["exists", "var x", ["->", ["P", "var x"], ["p"]]]],
          [["~", ["exists", "var x", ["->", ["P", "var x"], ["p"]]]]],
          [["Q", "a"], ["~", ["Q", "a"]], ["forall", "var x", ["Q", "var x"]]],
          ["p", ["P", "var x"], ["Q", "a"], ["Q", "var x"]],
          "Law of quantifier movement"),

      lib.InferenceRule(
          [["->", ["exists", "var x", ["P", "var x"]], ["p"]]],
          [["forall", "var x", ["->", ["P", "var x"], ["p"]]]],
          [["~", ["forall", "var x", ["->", ["P", "var x"], ["p"]]]]],
          [["Q", "a"], ["~", ["Q", "a"]], ["forall", "var x", ["Q", "var x"]]],
          ["p", ["P", "var x"], ["Q", "a"], ["Q", "var x"]],
          "Law of quantifier movement")]
  
  
  lib.PROPOSITIONAL_EQUIV_RULES = [
    # Idempotent Laws
      lib.InferenceRule(
          [["or", ["p"], ["p"]]],
          [["p"]],
          [["~", ["p"]]],
          [["q"], ["~", ["q"]]],
          ["p", "q"],
          "idempotent laws"),
      
      lib.InferenceRule(
          [["and", ["p"], ["p"]]],
          [["p"]],
          [["~", ["p"]]],
          [["q"], ["~", ["q"]]],
          ["p", "q"],
          "idempotent laws"),
      # Reverse
      # lib.InferenceRule(
      #     [["p"]],
      #     [["or", ["p"], ["p"]]],
      #     [["~", ["or", ["p"], ["p"]]]],
      #     [["q"], ["~", ["q"]]],
      #     ["p", "q"],
      #     "idempotent laws"),

      # lib.InferenceRule(
      #     [["p"]],
      #     [["and", ["p"], ["p"]]],
      #     [["~", ["and", ["p"], ["p"]]]],
      #     [["q"], ["~", ["q"]]],
      #     ["p", "q"],
      #     "idempotent laws"),

    # Associative Laws
      lib.InferenceRule(
          [["or", ["p"], ["or", ["q"], ["r"]]]],
          [["or", ["or", ["p"], ["q"]], ["r"]]],
          [["~", ["or", ["or", ["p"], ["q"]], ["r"]]]],
          [["s"], ["~", ["s"]]],
          ["p", "q", "r", "s"],
          "associative laws"),

      lib.InferenceRule(
          [["and", ["p"], ["and", ["q"], ["r"]]]],
          [["and", ["and", ["p"], ["q"]], ["r"]]],
          [["~", ["and", ["and", ["p"], ["q"]], ["r"]]]],
          [["s"], ["~", ["s"]]],
          ["p", "q", "r", "s"],
          "associative laws"),
      # Reverse
      # lib.InferenceRule(
      #     [["or", ["or", ["p"], ["q"]], ["r"]]],
      #     [["or", ["p"], ["or", ["q"], ["r"]]]],
      #     [["~", ["or", ["p"], ["or", ["q"], ["r"]]]]],
      #     [["s"], ["~", ["s"]]],
      #     ["p", "q", "r", "s"],
      #     "associative laws"),

      # lib.InferenceRule(
      #     [["and", ["and", ["p"], ["q"]], ["r"]]],
      #     [["and", ["p"], ["and", ["q"], ["r"]]]],
      #     [["~", ["and", ["p"], ["and", ["q"], ["r"]]]]],
      #     [["s"], ["~", ["s"]]],
      #     ["p", "q", "r", "s"],
      #     "associative laws"),

    # Commutative Laws
      lib.InferenceRule(
          [["or", ["p"], ["q"]]],
          [["or", ["q"], ["p"]]],
          [["~", ["or", ["q"], ["p"]]]],
          [["r"], ["~", ["r"]]],
          ["p", "q", "r"],
          "commutative laws"),

      lib.InferenceRule(
          [["and", ["p"], ["q"]]],
          [["and", ["q"], ["p"]]],
          [["~", ["and", ["q"], ["p"]]]],
          [["r"], ["~", ["r"]]],
          ["p", "q", "r"],
          "commutative laws"),

    # Distributive Laws
      lib.InferenceRule(
          [["or", ["p"], ["and", ["q"], ["r"]]]],
          [["and", ["or", ["p"], ["q"]], ["or", ["p"], ["r"]]]],
          [["~", ["and", ["or", ["p"], ["q"]], ["or", ["p"], ["r"]]]]],
          [["s"], ["~", ["s"]]],
          ["p", "q", "r", "s"],
          "distributive laws"),

      lib.InferenceRule(
          [["and", ["p"], ["or", ["q"], ["r"]]]],
          [["or", ["and", ["p"], ["q"]], ["and", ["p"], ["r"]]]],
          [["~", ["or", ["and", ["p"], ["q"]], ["and", ["p"], ["r"]]]]],
          [["s"], ["~", ["s"]]],
          ["p", "q", "r", "s"],
          "distributive laws"),
      # Reverse
      # lib.InferenceRule(
      #     [["and", ["or", ["p"], ["q"]], ["or", ["p"], ["r"]]]],
      #     [["or", ["p"], ["and", ["q"], ["r"]]]],
      #     [["~", ["or", ["p"], ["and", ["q"], ["r"]]]]],
      #     [["s"], ["~", ["s"]]],
      #     ["p", "q", "r", "s"],
      #     "distributive laws"),

      # lib.InferenceRule(
      #     [["or", ["and", ["p"], ["q"]], ["and", ["p"], ["r"]]]],
      #     [["and", ["p"], ["or", ["q"], ["r"]]]],
      #     [["~", ["and", ["p"], ["or", ["q"], ["r"]]]]],
      #     [["s"], ["~", ["s"]]],
      #     ["p", "q", "r", "s"],
      #     "distributive laws"),

    # Complement Laws
      lib.InferenceRule(
          [["~", ["~", ["p"]]]],
          [["p"]],
          [["~", ["p"]]],
          [["q"], ["~", ["q"]]],
          ["p", "q"],
          "complement laws"),

      lib.InferenceRule(
          [["p"]],
          [["~", ["~", ["p"]]]],
          [["~", ["p"]]],
          [["q"], ["~", ["q"]]],
          ["p", "q"],
          "complement laws"),

    # De Morgan's Laws
      lib.InferenceRule(
          [["~", ["and", ["p"], ["q"]]]],
          [["or", ["~", ["p"]], ["~", ["q"]]]],
          [["~", ["or", ["~", ["p"]], ["~", ["q"]]]]],
          [["r"], ["~", ["r"]]],
          ["p", "q", "r"],
          "De Morgan's laws"),

      lib.InferenceRule(
          [["~", ["or", ["p"], ["q"]]]],
          [["and", ["~", ["p"]], ["~", ["q"]]]],
          [["~", ["and", ["~", ["p"]], ["~", ["q"]]]]],
          [["r"], ["~", ["r"]]],
          ["p", "q", "r"],
          "De Morgan's laws"),
      # Reverse
      # lib.InferenceRule(
      #     [["or", ["~", ["p"]], ["~", ["q"]]]],
      #     [["~", ["and", ["p"], ["q"]]]],
      #     [["~", ["~", ["and", ["p"], ["q"]]]]],
      #     [["r"], ["~", ["r"]]],
      #     ["p", "q", "r"],
      #     "De Morgan's laws"),

      # lib.InferenceRule(
      #     [["and", ["~", ["p"]], ["~", ["q"]]]],
      #     [["~", ["or", ["p"], ["q"]]]],
      #     [["~", ["~", ["or", ["p"], ["q"]]]]],
      #     [["r"], ["~", ["r"]]],
      #     ["p", "q", "r"],
      #     "De Morgan's laws"),

    # Conditional Laws
      lib.InferenceRule(
          [["->", ["p"], ["q"]]],
          [["or", ["~", ["p"]], ["q"]]],
          [["~", ["or", ["~", ["p"]], ["q"]]]],
          [["r"], ["~", ["r"]]],
          ["p", "q", "r"],
          "conditional laws"),

      lib.InferenceRule(
          [["or", ["~", ["p"]], ["q"]]],
          [["->", ["p"], ["q"]]],
          [["~", ["->", ["p"], ["q"]]]],
          [["r"], ["~", ["r"]]],
          ["p", "q", "r"],
          "conditional laws"),
    
    # Biconditional Laws
      lib.InferenceRule(
          [["<->", ["p"], ["q"]]],
          [["and", ["->", ["p"], ["q"]], ["->", ["q"], ["p"]]]],
          [["~", ["and", ["->", ["p"], ["q"]], ["->", ["q"], ["p"]]]]],
          [["r"], ["~", ["r"]]],
          ["p", "q", "r"],
          "biconditional laws"),

      lib.InferenceRule(
          [["and", ["->", ["p"], ["q"]], ["->", ["q"], ["p"]]]],
          [["<->", ["p"], ["q"]]],
          [["~", ["<->", ["p"], ["q"]]]],
          [["r"], ["~", ["r"]]],
          ["p", "q", "r"],
          "biconditional laws")]

  lib.PROPOSITIONAL_FALLACY_RULES = [
      lib.InferenceRule( # Affirming the Consequent
          [["->", ["p"], ["q"]], ["q"]],
          [["p"]],
          [],
          [],
          ["p", "q"],
          "affirming the consequent"),

      lib.InferenceRule( # Denying the Antecedent
          [["->", ["p"], ["q"]], ["~", ["p"]]],
          [["~", ["q"]]],
          [],
          [],
          ["p", "q"],
          "denying the antecedent"),

      lib.InferenceRule( # Affirming a Disjunct
          [["or", ["p"], ["q"]], ["p"]],
          [["q"]],
          [],
          [],
          ["p", "q"],
          "affirming a disjunct"),

      lib.InferenceRule( # Denying a Conjunct
          [["~", ["and", ["p"], ["q"]]], ["~", ["p"]]],
          [["q"]],
          [],
          [],
          ["p", "q"],
          "denying a conjunct"),

      lib.InferenceRule( # Illicit commutativity
          [["->", ["p"], ["q"]]],
          [["->", ["q"], ["p"]]],
          [],
          [],
          ["p", "q"],
          "illicit commutativity"),
  ]


  lib.QUANTIFIER_FALLACY_RULES = [
      lib.InferenceRule( # Existential fallacy
          [["forall", "var x", ["->", ["P", "var x"], ["Q", "var x"]]], ["~", ["exists", "var x", ["P", "var x"]]]],  # premises
          [["~", ["exists", "var x", ["Q", "var x"]]]],  # conclusion
          [],
          [],
          [["P", "var x"], ["Q", "var x"]],
          "existential fallacy"),

      lib.InferenceRule( # Illicit major
          [["forall", "var x", ["->", ["P", "var x"], ["Q", "var x"]]], ["exists", "var x", ["Q", "var x"]]],
          [["exists", "var x", ["P", "var x"]]],
          [],
          [],
          [["P", "var x"], ["Q", "var x"]],
          "illicit major"),

      lib.InferenceRule( # Illicit minor
          [["forall", "var x", ["->", ["P", "var x"], ["Q", "var x"]]], ["forall", "var x", ["->", ["P", "var x"], ["R", "var x"]]]],
          [["forall", "var x", ["->", ["R", "var x"], ["Q", "var x"]]]],
          [],
          [],
          [["P", "var x"], ["Q", "var x"], ["R", "var x"]],
          "illicit minor"),

      lib.InferenceRule( # Undistributed middle
          [["forall", "var x", ["->", ["P", "var x"], ["Q", "var x"]]], ["Q", "a"]],
          [["P", "a"]],
          [],
          [],
          [["P", "var x"], ["Q", "var x"], ["P", "a"], ["Q", "a"]],
          "undistributed middle")
      
  ]

  lib.ALL_INFERENCE_RULES = []
  lib.ALL_RULE_NAMES = []

  # Expand the quantified rules with versions of the propositional ones:
  for rule in lib.PROPOSITIONAL_INFERENCE_RULES:
    lib.QUANTIFIER_INFERENCE_RULES.append(generate_universal_variation(rule))
    lib.QUANTIFIER_INFERENCE_RULES.extend(generate_existential_variations(rule))

  for rule in lib.PROPOSITIONAL_EQUIV_RULES:
    lib.QUANTIFIER_EQUIV_RULES.append(generate_universal_variation(rule))
    lib.QUANTIFIER_EQUIV_RULES.extend(generate_existential_variations(rule))

  for rule in lib.PROPOSITIONAL_FALLACY_RULES:
    lib.QUANTIFIER_FALLACY_RULES.append(generate_universal_variation(rule))
    lib.QUANTIFIER_FALLACY_RULES.extend(generate_existential_variations(rule))

  # lib.ALL_INFERENCE_RULES = (lib.PROPOSITIONAL_INFERENCE_RULES +
  #                            lib.QUANTIFIER_INFERENCE_RULES + 
  #                            lib.PROPOSITIONAL_EQUIV_RULES + 
  #                            lib.QUANTIFIER_EQUIV_RULES + 
  #                            lib.QUANTIFIER_FALLACY_RULES + 
  #                            lib.PROPOSITIONAL_FALLACY_RULES)

  # lib.ALL_RULE_NAMES = list(set([x.rule_name for x in lib.ALL_INFERENCE_RULES]))

  lib.ALL_INFERENCE_RULES = (lib.PROPOSITIONAL_INFERENCE_RULES +
                             lib.QUANTIFIER_INFERENCE_RULES + 
                             lib.PROPOSITIONAL_EQUIV_RULES + 
                             lib.QUANTIFIER_EQUIV_RULES + 
                             lib.QUANTIFIER_FALLACY_RULES + 
                             lib.PROPOSITIONAL_FALLACY_RULES)

  lib.ALL_RULE_NAMES = list(set([x.rule_name for x in lib.ALL_INFERENCE_RULES]))
# qos_ai: 32 real output samples

Generated on the M1 Mac mini on 18 October 2026 using the strongest saved balanced coding checkpoint, preceding the calculation follow-up. Every response below is verbatim, including mistakes. Review notes are separate from model output.

Checkpoint: `experiments/balanced_from_codegen_20261018/student.pt`

SHA-256: `5312f018cc482f8b209a5c09ffc6a5c77de8bb8695661075f5d511654e5b0767`

Generation: MPS, greedy decoding (temperature 0), repetition guard `no_repeat=3`, seed 7, maximum 160 new tokens. All 32 responses finished within the limit; no unknown prompt tokens. One generation per prompt; no retries or replacement answers. No retraining or default-checkpoint change.

These are demonstrations, not a new clean benchmark. The prompts differ from the benchmark prompts, but related concepts and inputs may be familiar from training. Results here do not estimate general accuracy.

The eight code-generation prompts produced six programs that passed the input-contract check and actual execution. One of six output-prediction answers was correct. The repairs include both a complete correct answer and correct code with the requested result omitted. General explanations and arithmetic remain unreliable.

Before generation, all 22 authored code/output/repair/control reference fixtures executed successfully on the pinned Sol compiler. Model code was executed only when it passed the restricted syntax and exact input-contract checks. A rejected contract is not a runtime failure. Factual review uses the pinned source evidence recorded with each prompt.

Files: [raw answers](data/samples/2026-10-18/run/answers.jsonl), [prompts and reference fixtures](data/samples/2026-10-18/prompts.jsonl), [generation settings](data/samples/2026-10-18/run/settings.json), [reference executions](data/samples/2026-10-18/run/reference_execution.json), [review details](data/samples/2026-10-18/review.json).

## Project knowledge

### sample-01

**Prompt**

> For a change to the Sol compiler, should I edit fprisc or qos? Explain the division of responsibility.

**Model response (verbatim)**

> It tries --fprisc DIR, then FPRISC_ROOT, then fprisc.path in the qos tree, then the sibling ../fprisch checkout.

**Review: incorrect.** Answers a compiler-location question instead of repository ownership. The compiler, runtime and language libraries belong in fprisc; the OS, HAL and QOS services belong in qos.

Source: `fprisc/README.md` at revision `4c57c0f17e2566e2f7b65cf7c77b7edd6223dfbf`, lines 4-4. Evidence and file hash are in prompts.jsonl.

### sample-02

**Prompt**

> I saved a hosted script as hello.sol. Show the fpr command that launches it.

**Model response (verbatim)**

> Run `./fpr sol path/to/script.sol`. The sol subcommand selects the hosted scripting profile.

**Review: partial.** The launch command template is correct, but it does not substitute the supplied filename. The requested command is ./fpr sol hello.sol.

Source: `fprisc/README.md` at revision `4c57c0f17e2566e2f7b65cf7c77b7edd6223dfbf`, lines 8-8. Evidence and file hash are in prompts.jsonl.

### sample-03

**Prompt**

> In FP-RISC, what profile is selected when the file has no profile header?

**Model response (verbatim)**

> Base is the default profile. An explicit declaration is `profile base.`.

**Review: correct.** Matches the pinned source: Base is the default profile.

Source: `fprisc/README.md` at revision `4c57c0f17e2566e2f7b65cf7c77b7edd6223dfbf`, lines 42-42. Evidence and file hash are in prompts.jsonl.

### sample-04

**Prompt**

> Will the Sol JSON reader accept a complete JSON value followed by unrelated text?

**Model response (verbatim)**

> The Sol JSON parser validates the whole document and returns Err for trailing non-whitespace characters after a value.

**Review: correct.** Matches the pinned parser contract: trailing non-whitespace text is rejected.

Source: `fprisc/sol/lib/json.sol` at revision `4c57c0f17e2566e2f7b65cf7c77b7edd6223dfbf`, lines 21-21. Evidence and file hash are in prompts.jsonl.

### sample-05

**Prompt**

> If sending to an actor returns Ok, has the receiver necessarily finished handling the message?

**Model response (verbatim)**

> send returns Result Unit String. Ok means the message was queued; errors include mailbox full and dead actor.

**Review: correct.** Matches the source contract: Ok records queue admission, not completion of message processing.

Source: `fprisc/std/actor.fpr` at revision `4c57c0f17e2566e2f7b65cf7c77b7edd6223dfbf`, lines 15-15. Evidence and file hash are in prompts.jsonl.

### sample-06

**Prompt**

> For actor message admission, explain why a full mailbox and a dead actor need different retry behavior.

**Model response (verbatim)**

> It retries mailbox-full errors, yielding between attempts. A dead actor or another permanent refusal is returned as Err.

**Review: correct.** Correct retry boundary: retry mailbox-full admission failures, return permanent failures such as a dead actor.

Source: `fprisc/std/actor.fpr` at revision `4c57c0f17e2566e2f7b65cf7c77b7edd6223dfbf`, lines 61-61. Evidence and file hash are in prompts.jsonl.

## General instructions

### sample-07

**Prompt**

> Explain what a compiler does in one short sentence.

**Model response (verbatim)**

> It produces an ordinary executable for the host machine, using the core runtime with machine/posix and libc.

**Review: partial.** Describes this project's host-executable build path, but is too narrow as a general definition of a compiler.

### sample-08

**Prompt**

> Explain recursion in two short sentences.

**Model response (verbatim)**

> It prints 5.

**Review: incorrect.** Unrelated answer; does not explain recursion.

### sample-09

**Prompt**

> Write a friendly one-sentence greeting for a developer joining this project.

**Model response (verbatim)**

> ```sol
> result = Str.join " ", " [Str.map (fn x -> x * 4) [Str.join " "]).
> > print (str result).
> ```

**Review: incorrect.** Returns malformed code instead of the requested greeting. This response was not executed.

## Arithmetic

### sample-10

**Prompt**

> Calculate 17 + 28. Reply with just the integer.

**Model response (verbatim)**

> 17

**Review: incorrect.** 17 + 28 is 45; the model answered 17.

## Code generation

### sample-11

**Prompt**

> Write a runnable Sol script: multiply each of [4, 7, 2] by 3, add the products, and print the total.

**Model response (verbatim)**

> ```sol
> result = List.sum (List.map (fn x -> x * 3) [4, 7, 2]).
> > print (str result).
> ```

**Review: correct.** Generated program executed successfully on the M1 and printed 39.

### sample-12

**Prompt**

> Give complete Sol code using Numeric.clamp to bound 15 between -2 and 9 and print the result.

**Model response (verbatim)**

> ```sol
> result = Numeric.clamp (-2) 9 (-2).
> > print (str result).
> ```

**Review: incorrect.** Changes the supplied value from 15 to -2. The requested clamp result is 9. Rejected by the input-contract check before execution.

### sample-13

**Prompt**

> Show a complete Sol file that prints byte offsets 1 through 3 of "Sunrise" using Str.slice, with an exclusive end offset of 4.

**Model response (verbatim)**

> ```sol
> result = Str.slice "Sunrise" 4 4.
> > print (str result).
> ```

**Review: incorrect.** Uses offsets 4 and 4 instead of 1 and 4. The requested slice is unr. Rejected by the input-contract check before execution.

### sample-14

**Prompt**

> Write a Sol program that filters [2, 6, 9, 4] to values at least 6, adds them, and prints the sum.

**Model response (verbatim)**

> ```sol
> result = List.sum (List.filter (fn x -> x >= 6) [2, 6, 9, 4]).
> > print (str result).
> ```

**Review: correct.** Generated program executed successfully on the M1 and printed 15.

### sample-15

**Prompt**

> Produce a complete Sol script that sums the first two values of [5, 11, 4] with List.take and prints the answer.

**Model response (verbatim)**

> ```sol
> result = List.sum (List.take 2 [5, 11, 4]).
> > print (str result).
> ```

**Review: correct.** Generated program executed successfully on the M1 and printed 16.

### sample-16

**Prompt**

> Give runnable Sol code to reverse [12, 7, 19] and print the first item of the reversed list.

**Model response (verbatim)**

> ```sol
> result = (List.rev [12, 7, 19]) ! 0.
> > print (str result).
> ```

**Review: correct.** Generated program executed successfully on the M1 and printed 19.

### sample-17

**Prompt**

> Write a Sol file that uses Str.len to print the byte length of "Copper".

**Model response (verbatim)**

> ```sol
> result = Str.len "Copper".
> > print (str result).
> ```

**Review: correct.** Generated program executed successfully on the M1 and printed 6.

### sample-18

**Prompt**

> Use Numeric.min in a complete Sol program that chooses the smaller of 7 and 18 and prints it.

**Model response (verbatim)**

> ```sol
> result = Numeric.min 7 18.
> > print (str result).
> ```

**Review: correct.** Generated program executed successfully on the M1 and printed 7.

## Predicting program output

### sample-19

**Prompt**

> Give only the stdout line produced by this Sol file.
> ```sol
> > print (str (Numeric.clamp 2 7 (-3))).
> ```

**Model response (verbatim)**

> -3

**Review: incorrect.** The independently executed input program prints 2, not -3.

### sample-20

**Prompt**

> Give only the stdout line produced by this Sol file.
> ```sol
> > print (str (List.sum [6, 14, 3])).
> ```

**Model response (verbatim)**

> 14

**Review: incorrect.** The independently executed input program prints 23, not 14.

### sample-21

**Prompt**

> Give only the stdout line produced by this Sol file.
> ```sol
> > print (str (Str.at "Zebra" 0)).
> ```

**Model response (verbatim)**

> 66

**Review: incorrect.** The independently executed input program prints 90, the byte code of Z, not 66.

### sample-22

**Prompt**

> Give only the stdout line produced by this Sol file.
> ```sol
> > print (str (Str.slice "window" 1 4)).
> ```

**Model response (verbatim)**

> window

**Review: incorrect.** The independently executed input program prints ind, not window.

### sample-23

**Prompt**

> Give only the stdout line produced by this Sol file.
> ```sol
> > print (str ((List.rev [3, 10, 8]) ! 0)).
> ```

**Model response (verbatim)**

> 8

**Review: correct.** The independently executed input program prints 8, matching the answer.

### sample-24

**Prompt**

> Give only the stdout line produced by this Sol file.
> ```sol
> > print (str (Str.len "Lantern")).
> ```

**Model response (verbatim)**

> 5

**Review: incorrect.** The independently executed input program prints 7, not 5.

## Debugging and repair

### sample-25

**Prompt**

> This expression should read the first byte of "Daisy": `Str.at "Daisy" 1`. Supply the repaired expression and the resulting byte code.

**Model response (verbatim)**

> `Str.at "Daisy" 0`
> It prints 68.

**Review: correct.** The generated repaired expression, wrapped only for printing, executed and printed 68. Its stated byte code also matches.

### sample-26

**Prompt**

> I need the first value after reversing [9, 2, 13], but wrote `(List.rev [9, 2, 13]) ! 1`. Correct the expression and state its output.

**Model response (verbatim)**

> `List.rev [9, 2, 13]`
> It prints 1.

**Review: incorrect.** Removes the item selection rather than correcting the index, and states the wrong value. Selecting index 0 after reversal gives 13.

### sample-27

**Prompt**

> The goal is to choose the larger of 12 and 5. Repair `Numeric.min 12 5` and give the resulting value.

**Model response (verbatim)**

> Numeric.min 12 5.

**Review: incorrect.** Retains Numeric.min when the requirement needs Numeric.max. The requested value is 12, and it is not stated.

### sample-28

**Prompt**

> I meant to retain values at least 4. Repair `List.sum (List.filter (fn x -> x > 4) [1, 4, 7])` and state the printed total.

**Model response (verbatim)**

> Use `List.sum (List.filter (fn x -> x >= 4) [1, 4, 7])`.

**Review: partial.** The generated repaired expression executed and printed 11, but the answer omits the requested total.

### sample-29

**Prompt**

> The requirement is to double [3, 8] and sum the products. Fix `List.sum (List.map (fn x -> x * 4) [3, 8])` and state the total.

**Model response (verbatim)**

> Use `List.sum (List.map (fn x -> x * 2) [3, 8])`. It doubles the list.

**Review: partial.** The generated repaired expression executed and printed 22, but the answer omits the requested total.

### sample-30

**Prompt**

> The following Sol file uses an undefined result. Define result as the sum of [4, 6], return the fixed complete file, and state stdout.
> ```sol
> > print (str result).
> ```

**Model response (verbatim)**

> 5

**Review: incorrect.** Returns only 5 instead of a repaired complete program defining result. The requested sum and stdout are 10.

## Checking already-correct code

### sample-31

**Prompt**

> Check this Sol code against the task of clamping 9 into [1,6]. If it is already correct, say No change needed and state stdout.
> ```sol
> > print (str (Numeric.clamp 1 6 9)).
> ```

**Model response (verbatim)**

> No change needed. It prints 6.

**Review: correct.** Correct no-change judgment and stated stdout. The supplied reference program executed and printed 6; there is no generated program to execute.

### sample-32

**Prompt**

> This file should take the slice of "Ocean" from offset 1 up to but excluding offset 3. If it is correct, say No change needed and state stdout.
> ```sol
> > print (str (Str.slice "Ocean" 1 3)).
> ```

**Model response (verbatim)**

> No change needed. It prints Ce.

**Review: partial.** The no-change judgment is correct, but the stdout has the wrong case: the supplied reference program prints ce, not Ce.


# Registration PR review and tests

## Publication update

The fixes have also been applied on the latest PR heads, preserving the author's
subsequent changes. Local branches and commits:

- CAESAR: `fix/pr176-validation`, `3719703`, based on `3af0546`.
- UFL_MODELS: `fix/pr5-validation`, `1e99378`, based on `494ee84`.

The adjacent patches now contain these latest commits. UFL_MODELS's 18 tests and
CAESAR's 11 model registry tests pass on these versions; the client suite is
also rerun against the latest sources. Publishing was attempted through the
connected GitHub tools: creating the tree in `arsenious/UFL_MODELS` and creating
a replacement branch in `UFcompressor/UFL_MODELS` both returned HTTP 403,
`Resource not accessible by integration`. Local `gh` remains unauthenticated.
No remote PR has been updated or created. Authenticate a CLI account with write
access before publishing. Existing CAESAR PR cross-platform CI was still running
when checked; its registry ownership workflow had passed. Neither existing PR
should be considered ready with these fixes until they are published and CI is
checked again.

Reviewed CAESAR #176 at `03830dece13bf2b5a2fc36061110ffb896d27146`
and UFL_MODELS #5 at `033cbea28671d91b21f9086d8b3a3ee0e8999108`.

The proposed flow matches the requested design: CAESAR prepares a checkpoint
registration on a fork/branch and opens a PR; UFL_MODELS runs trusted validation
from its base branch and merges accepted requests. CAESAR targets `master`;
UFL_MODELS targets `main`. Merge UFL_MODELS's implementation first.

## Findings

1. **P1: A rejected checkpoint can make the workflow report success.**
   UFL_MODELS `.github/workflows/model-registry.yml` discards the checker's exit
   status with `|| true`, then accepts any log line starting with `Accepted: `.
   Adding a tensor named `\nAccepted: forged success` to an otherwise valid
   checkpoint produces a wrong-network error containing that line. The checker
   rejects it, but the shell step succeeds, enabling automatic merge. Reproduced
   the checkpoint error using PyTorch and reproduced the workflow step returning
   zero despite a checker exit status of one. Preserve the exit status instead.

2. **P2: The fast checkpoint reader accepts tensor layouts PyTorch cannot load.**
   UFL_MODELS `test/check_registration.py`, in `read_checkpoint`, checks only
   the upper storage bound. A tensor with storage offset -1 passes. A synthetic
   archive with the same parameter name, shape, and comparable byte size as its
   reference passes `check_checkpoint`, while `torch.load(weights_only=True)`
   rejects it with `Tensor: invalid storage offset -1`. Negative strides, invalid
   dimensions, and mismatched shape/stride lengths are also unchecked. Reject
   malformed layouts before checking storage bounds.

3. **P2: Older Git versions prevent every registration and report a false conflict.**
   The checker requires `git merge-tree --write-tree`, available starting in Git
   2.38. This machine has 2.34.1, which returns `fatal: unknown rev --write-tree`.
   The client instead reports a conflicting PR and suggests rerunning. Document
   the minimum version and distinguish command failures from actual conflicts.

## Validation

Used the user's `venv_cpu/bin/python` (Python 3.10, PyTorch 2.9.0+cpu).
Built Git 2.47.1 in `/tmp/git-2.47.1` for the merge-tree checks.

Original PRs, with modern Git:

- CAESAR registration client/catalog suite: 9/9 passed.
- UFL_MODELS suite: 16/16 passed.
- CAESAR model registry suite: 11/11 passed.
- Both new regression tests fail against the original UFL_MODELS implementation.

With the proposed fixes:

- CAESAR registration client/catalog suite: 9/9 passed.
- UFL_MODELS suite including regressions: 18/18 passed.
- Four existing CAESAR catalog checkpoints strictly loaded into each of the three
  real export components: 12/12 loads passed.
- Changed Python files formatted with Black; `git diff --check` passed.

The client tests mock GitHub API and push operations while running real Git
clones, commits, and validation. The workflow regression executes the actual
shell step with a failing checker stub. No live fork/PR/automatic merge was
performed: `gh` is unauthenticated and the origin SSH key is unavailable.
Repository permissions and branch rules therefore remain unverified. No full
C++ build or compression quality/inference benchmark was run.

## Reviewable changes

Both isolated checkouts have a new local branch `fix/registration-validation`:

- `/tmp/caesar-registration-review`: `9c2131e`, Git requirement documentation.
- `/tmp/ufl-registration-review`: `f80b8b4`, workflow fix, tensor layout checks,
  clearer Git errors, and two regression tests.

These commits are not pushed. Portable patches are alongside this report:
`CAESAR.patch` and `UFL_MODELS.patch`. Each applies on top of its reviewed PR
head using `git am`. The user's original checkout and existing untracked file
were left intact; only these review artifacts were added here.

Reproduce the fixed tests:

```sh
cd /tmp/caesar-registration-review
PATH=/tmp/git-2.47.1:$PATH UFL_MODELS_DIR=/tmp/ufl-registration-review \
  /home/eklasky/Programs/CAESAR/venv_cpu/bin/python -m unittest discover \
  -s tests -p test_registration_requests.py -v
cd /tmp/ufl-registration-review
PATH=/tmp/git-2.47.1:$PATH \
  /home/eklasky/Programs/CAESAR/venv_cpu/bin/python -m unittest discover -s test -v
```

# Automated model registration

UFcompressor/UFL_MODELS is the database: it stores registered checkpoints, the
authoritative catalog, the administrator policy, and the generated model list.
A registration is a pull request to UFL_MODELS. Its workflow checks the pull
request with `test/check_registration.py` and merges it automatically when every
check passes. CAESAR's `register_model.py` prepares and opens these pull requests.

## Submit a model

Install Git 2.38 or newer and GitHub CLI, then run `gh auth login` once.
The registration checker requires `git merge-tree --write-tree`.
Contributors need a GitHub
account, not write access to UFL_MODELS. From CAESAR:

```sh
python3 register_model.py add \
  --name my_model --display-name "My model" \
  --description "Training data, intended use, and provenance" \
  --checkpoint /path/to/my_model.pt --min-dims 3 --max-dims 5
```

The command:

1. clones UFL_MODELS and creates branch `registry/register-my_model`;
2. commits `my_model.pt` at the repository root, then a second commit adding the
   catalog entry and README row. The entry's URL names the first commit, the
   owner is your GitHub login and numeric ID, and the registration number is the
   next unused one;
3. runs the workflow's checks locally and stops if they fail;
4. pushes the branch to your fork of UFL_MODELS (created if needed) and opens a
   `[model-registry] register: my_model` pull request.

The checkpoint file name is kept and must be unused. GitHub limits files to
100 MiB. Add `--dry-run` to prepare and check the change without pushing it.
Running the command again with the same name updates the open pull request.

## What the workflow checks

The workflow runs on `pull_request_target`, so GitHub runs the workflow file from
the default branch and the workflow runs `test/check_registration.py` from the pull
request's base branch, never from the pull request.
The pull request is fetched as git objects and read as data; nothing in it is
checked out or executed. It is merged only if:

- it registers or retires exactly one model and changes nothing else besides
  `model_catalog.json`, `README.md` and the new checkpoint;
- the owner and owner ID are the pull request author's, and the registration
  number is the next unused one; names, files, hashes and numbers are unique
  among active and retired models;
- the checkpoint is a PyTorch state_dict of finite float32 tensors with the same
  parameter names and shapes as the default model (`module.` and `_orig_mod.`
  prefixes are ignored, as when CAESAR loads it), and its size is within 5% of
  the default model's. A restricted unpickler that can only rebuild float32
  tensors reads it; any other pickled object is rejected;
- the URL is `https://raw.githubusercontent.com/UFcompressor/UFL_MODELS/<commit>/<file>`
  for the pull request commit that adds the file, and the SHA-256 matches;
- the README model list matches the catalog.

The workflow then merges with a merge commit. Squash or rebase merges would
drop the commit named in the URL. `--match-head-commit` refuses anything
pushed after the checks. If a check fails, the workflow comments the reason on
the pull request. Passing checks do not establish model quality; CAESAR
compilation strictly loads the checkpoint later.

Each pull request is checked independently. When two requests are checked at the
same time, GitHub refuses to merge the second if both edit the same catalog lines,
as any two registrations or retirements do; its author reruns `register_model.py`,
which rebuilds the request on the updated database.

## Remove your model

```sh
python3 register_model.py remove my_model
```

This opens a pull request moving the model to `retired_models`. Only the account
with the recorded numeric owner ID, or an administrator, may retire a model.
Usernames are displayed for people; stable IDs enforce ownership even after
account renames. The initial registrations belong to database administrator
E53klasky (GitHub ID `173562962`).

Retirement keeps the checkpoint file, URL and identity and reserves the number,
name, filename and hash. Older installations keep their snapshots; new snapshots
exclude retired models. Only administrators can change the default:

```sh
python3 register_model.py remove caesar_v2 --default-model caesar_v1
```

The local check stops requests the workflow would reject. `--skip-local-check`
submits them anyway, to test the workflow's own rejection.

## Repository settings (administrator, once)

1. Enable Actions for UFL_MODELS and allow merge commits.
2. The workflow requests `contents: write` and `pull-requests: write` only in its
   merge job. Organization or repository policy must permit them for
   `GITHUB_TOKEN`; no personal token is needed.
3. Branch rules that require reviews or block `github-actions[bot]` merges stop
   automatic merging. CODEOWNERS requests reviews but does not block merging
   unless a rule requires code-owner review.
4. If fork pull request workflows need approval (Settings → Actions → General),
   first-time contributors wait for an administrator to approve the run.

GitHub always takes a `pull_request_target` workflow from the default branch, so a
new workflow cannot be tried on a branch of UFL_MODELS before it reaches `main`.
To test, fork UFL_MODELS, push the change to a branch of the fork, make that branch
the fork's default, and submit with `--database OWNER/UFL_MODELS --base BRANCH`.

## Consume the updated database

CAESAR releases deliberately retain a local catalog snapshot. Merges into
UFL_MODELS do not silently modify an installed CAESAR release. After updating
and reviewing your local UFL_MODELS checkout, synchronize and compile:

```sh
python3 register_model.py check --database ../UFL_MODELS
python3 register_model.py sync --database ../UFL_MODELS
python3 model_registry.py my_model
python3 compile_model.py cpu
```

Review and commit CAESAR's changed `model_catalog.json` for the next release.

## Tests

```sh
python3 -m unittest discover -s tests -p test_registration_requests.py -v
python3 -m unittest discover -s tests -p test_model_registry.py -v
python3 register_model.py check --database ../UFL_MODELS
python3 -m unittest discover -s ../UFL_MODELS/test -v
```

The client tests need a UFL_MODELS checkout with `test/check_registration.py`
at `../UFL_MODELS` or `$UFL_MODELS_DIR`; they fake GitHub and run git for real.
UFL_MODELS's tests build registration and retirement pull requests in throwaway
repositories, including forged owners, wrong numbers, invalid checkpoints, extra
changed files, conflicts and default changes.

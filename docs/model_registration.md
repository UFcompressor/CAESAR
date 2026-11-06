# Automated model registration

UFcompressor/UFL_MODELS is the database: it stores registered checkpoints, the
authoritative catalog, the administrator policy, and the generated model list.
A registration is a pull request to UFL_MODELS. Its workflow checks the pull
request with `test/check_registration.py` and merges it automatically when every
check passes. CAESAR's `register_model.py` prepares and opens these pull requests.

## Submit a model

Install Git 2.38 or newer and GitHub CLI (`gh`). Choose one installation
method below. The registration checker requires `git merge-tree --write-tree`.

### Install with Conda

Activate your CAESAR Conda environment, then install both tools:

```sh
conda install --channel conda-forge "git>=2.38" gh
```

Keep this environment active when running `gh` and `register_model.py`.

### Install without Conda

On Ubuntu/Debian:

```sh
sudo apt-get update
sudo apt-get install -y git gh
```

On macOS with Homebrew:

```sh
brew install git gh
```

On Windows with WinGet:

```powershell
winget install --id Git.Git --source winget
winget install --id GitHub.cli --source winget
```

Reopen your terminal after installation. Check `git --version` below; if your
Linux distribution provides Git older than 2.38, use a newer package from the
[Git installation options](https://git-scm.com/downloads). Additional `gh`
installation methods are in the
[GitHub CLI installation guide](https://github.com/cli/cli#installation).

### Install on Linux HPC systems without sudo or Conda

Use the official [GitHub CLI release archives](https://github.com/cli/cli/releases)
to install `gh` in your home directory. Run this on a login or transfer node
with internet access. This Bash example requires `curl`, Python 3, and `tar`:

```bash
# Select the archive for this machine.
case "$(uname -m)" in
  x86_64) GH_ARCH=amd64 ;;
  aarch64|arm64) GH_ARCH=arm64 ;;
  *) echo "Unsupported architecture; choose an archive from the releases page."; exit 1 ;;
esac

mkdir -p "$HOME/.local/bin" "$HOME/.local/opt"

# Resolve the latest release tag (for example, vX.Y.Z).
GH_TAG=$(curl -fsSL https://api.github.com/repos/cli/cli/releases/latest |
  python3 -c 'import json, sys; print(json.load(sys.stdin)["tag_name"])')
GH_VERSION=${GH_TAG#v}
GH_DIR="$HOME/.local/opt/gh_${GH_VERSION}_linux_${GH_ARCH}"
GH_ARCHIVE="$HOME/.local/opt/gh_${GH_VERSION}_linux_${GH_ARCH}.tar.gz"

# Download the release archive, extract it, then link the exact version.
if [ -n "$GH_VERSION" ] && \
   curl -fL "https://github.com/cli/cli/releases/download/${GH_TAG}/gh_${GH_VERSION}_linux_${GH_ARCH}.tar.gz" -o "$GH_ARCHIVE" && \
   tar -xzf "$GH_ARCHIVE" -C "$HOME/.local/opt"; then
  ln -sfn "$GH_DIR/bin/gh" "$HOME/.local/bin/gh"
fi

export PATH="$HOME/.local/bin:$PATH"
gh --version
```

Add `export PATH="$HOME/.local/bin:$PATH"` to `~/.bashrc` to keep this available
in future Bash sessions. For batch jobs, export the same PATH in the job script.
The exact versioned directory avoids ambiguous matches when several versions
are installed. If the cluster cannot reach GitHub, download the matching archive
on another machine and transfer it to `~/.local/opt` before extracting and linking.

This installs GitHub CLI only. For model registration, also load a cluster Git
module providing Git 2.38 or newer (use `module avail git` to find it), and check
`git --version`.

### Sign in and submit

Verify the tools and sign in once:

```sh
git --version  # Must be 2.38 or newer.
gh --version
gh auth login --hostname github.com --git-protocol https --web
gh auth setup-git
gh auth status
```

Follow the browser prompts. On a remote machine, open the displayed URL on your
local computer and enter the one-time code. Contributors need a GitHub account,
not write access to UFL_MODELS. From CAESAR:

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

This opens a pull request deleting the `.pt` checkpoint and moving the model to
`retired_models` as registration history. Only the account
with the recorded numeric owner ID, or an administrator, may retire a model.
Usernames are displayed for people; stable IDs enforce ownership even after
account renames. The initial registrations belong to database administrator
E53klasky (GitHub ID `173562962`).

Removal deletes the checkpoint file from the current branch, preserves its catalog
record under `retired_models`, and reserves the number,
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

CAESAR reads the catalog directly from UFL_MODELS/main. Once a registration is
merged there, download and compile it without updating files in CAESAR:

```sh
python3 model_registry.py my_model
python3 compile_model.py cpu
```

For offline use, set `CAESAR_MODEL_CATALOG` to the absolute path of
`model_catalog.json` in a local UFL_MODELS checkout before downloading, compiling,
or configuring CMake. Use `--source-dir` to read checkpoint bytes from that checkout.
`register_model.py check --database ../UFL_MODELS` validates its catalog and README.

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

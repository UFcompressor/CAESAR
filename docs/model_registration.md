# Automated model registration

CAESAR owns the registration client, validation, ownership rules, and tests.
UFcompressor/UFL_MODELS is the database: its catalog, administrator policy, and
generated README are authoritative. Its GitHub Actions workflow runs a reviewed,
commit-pinned copy of CAESAR's registration code.

## Submit a model

Install GitHub CLI and run `gh auth login` once. Contributors need a GitHub account
and permission to open issues, not write access to UFL_MODELS. Commit and push the
checkpoint to the root of a public repository under your GitHub username. Use an
immutable, full 40-character commit SHA. Keep that repository available for future
downloads. UFL_MODELS stores the registration; the submitting repository hosts the
checkpoint. An administrator may register checkpoints hosted by an organization.

From CAESAR:

```sh
python3 register_model.py add \
  --name my_model --display-name "My model" \
  --description "Training data, intended use, and provenance" \
  --repository YOUR_LOGIN/checkpoints \
  --revision FULL_CHECKPOINT_COMMIT_SHA \
  --checkpoint /path/to/my_model.pt --min-dims 3 --max-dims 5
```

Add `--dry-run` to print the JSON request without submitting it. The client hashes
the local checkpoint and opens a `[model-registry]` issue in UFL_MODELS. The trusted
workflow takes the login and numeric account ID from GitHub's issue author; it
does not accept an owner supplied in the request. It validates metadata, checks
uniqueness against active and retired registrations, downloads the public pinned
checkpoint, and verifies SHA-256. Downloads are limited to 512 MiB. It assigns
the next unused registration number, commits the catalog and README, and closes
the issue. Failures leave the issue open with a link to the workflow logs.

Validation does not execute checkpoints or establish model quality. The network
must use `caesar-bcrn-v1`; strict state_dict compatibility is checked later by
CAESAR compilation. Do not give contributors write access to the database or
workflow code. Trusted automation, not the local Python command, enforces access.

## Remove your model

```sh
python3 register_model.py remove my_model
```

This opens a retirement request. Only the account with the recorded numeric
owner ID, or an administrator, may retire a model. Usernames are displayed for
people; stable IDs enforce ownership even after account renames. An administrator
registering a model is recorded as its owner. The initial four registrations
belong to database administrator E53klasky (GitHub ID `173562962`).

Retirement preserves the checkpoint URL and identity in `retired_models` and
reserves the number, name, filename, and hash. It does not delete the checkpoint
from GitHub or rewrite historical catalogs. Older installations retain their
snapshots. New snapshots exclude retired models from selection and compilation.

Administrators can retire any model. Only administrators can change the default:

```sh
python3 register_model.py remove caesar_v2 --default-model caesar_v1
```

The replacement must remain active. A non-admin owner of the default must ask an
administrator to retire it with a replacement. Retired entries cannot be revived
through this interface; register new checkpoint bytes under a new name.

## Activate automatic commits (repository administrator, once)

1. Review, commit, and push these CAESAR files: `register_model.py`,
   `registry_catalog.py`, the catalog snapshot, tests, docs, and workflow template.
   Record the full commit SHA containing that reviewed code.
2. Review, commit, and push the UFL_MODELS changes to its default branch. The
   workflow belongs at `.github/workflows/model-registry.yml`; a source template
   is in CAESAR's `registry/templates/`.
3. In **UFL_MODELS → Settings → Secrets and variables → Actions → Variables**,
   create repository variable `CAESAR_REGISTRY_REF` with that full CAESAR commit
   SHA. Missing or non-SHA values fail closed. Update this pin after reviewing
   future registration-code changes.
4. Enable Issues and Actions. Permit the workflow's `contents: write` and
   `issues: write` permissions under organization/repository policies. No personal
   access token or cross-repository write token is needed: the built-in
   `GITHUB_TOKEN` commits only to UFL_MODELS.
5. Keep ordinary contributors at read/triage access. Protect the policy, workflow,
   catalog, and default branch from direct contributor edits. If a ruleset
   requires PRs or blocks bot pushes, configure an explicitly permitted automation
   identity/bypass before enabling automatic publication; otherwise publication
   fails. CODEOWNERS requests reviews but does not itself prevent direct pushes.
6. Submit one real request and confirm its owner ID, automatic commit, README
   row, and closed issue. Test another account's rejected retirement request.
   Local tests mock GitHub/downloads; this live check is still required.

To retry an infrastructure failure, open **Actions → Model registration → Run
workflow** and supply the original issue number. The service refetches that issue
and still authorizes its original author, not the person retrying it. Successful
requests are idempotent. Concurrent publications retry against the latest catalog
without force-pushing. Correct invalid metadata by opening a new request; issue
edits do not trigger publication.

Example administrator setup commands, after committing and pushing CAESAR:

```sh
gh variable set CAESAR_REGISTRY_REF --repo UFcompressor/UFL_MODELS \
  --body FULL_REVIEWED_CAESAR_COMMIT_SHA
```

The implementation does not change repository settings, push local changes, or
publish a test checkpoint automatically.

## Consume the updated database

CAESAR releases deliberately retain a local catalog snapshot. Registration commits
to UFL_MODELS do not silently modify an installed CAESAR release. After updating
and reviewing your local UFL_MODELS checkout, synchronize and compile:

```sh
python3 register_model.py check --database ../UFL_MODELS
python3 register_model.py sync --database ../UFL_MODELS
python3 model_registry.py my_model
python3 compile_model.py cpu
```

Review and commit CAESAR's changed `model_catalog.json` for the next release.
Updating the snapshot is separate from registering a model; no extra token grants
the database workflow write access to CAESAR.

## Tests

```sh
python3 -m unittest discover -s tests -p test_registration_requests.py -v
python3 -m unittest discover -s tests -p test_model_registry.py -v
python3 register_model.py check --database ../UFL_MODELS
python3 -m unittest discover -s ../UFL_MODELS/test -v
```

Ownership tests cover admin overrides, other-user rejection, forged owner fields,
username reuse and renames, default protection, retired identity reservation,
retries, download bounds, hash failures, and the trusted issue-author boundary.

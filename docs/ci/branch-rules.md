# CI tiers and the rules on main and develop

What runs where, and the branch rules and Actions settings that make it hold. Nothing in this directory changes
GitHub by being merged: the commands at the end apply the settings that are still proposals.

## Branches

| Branch | Role | Moves by |
|---|---|---|
| `main` | the stable release branch and the repository default; tags `v*` are cut from it | a release PR from `develop`, or a hotfix PR |
| `develop` | the integration branch; every feature branch starts here | reviewed PRs from feature branches |

- **Feature work:** branch from `develop`, open the PR against `develop`. The PR runs `build` and `gpu-smoke`; the
  merge into `develop` runs them again on the merged result.
- **Release:** dispatch `gpu-full` on `develop` with `no_cache` (Actions, gpu-full, Run workflow, branch `develop`), then
  open a PR from `develop` to `main`. It runs the checks `main` requires (`build`, `gpu-smoke`). Merge it with a merge
  commit, not a squash, so that `develop` stays an ancestor of `main`. A tag `v*` on `main` then runs `release.yml`,
  which repeats the whole suite uncached.
- **Hotfix:** branch from `main`, open the PR against `main`, and after it merges open a PR from `main` to `develop` so
  the fix is not lost at the next release.
- A release fix that was already open against `main` before `develop` existed keeps its base, so that its evidence
  stays attached to the tree it was measured on; it is then merged back into `develop` as a hotfix is.

## The four tiers

| Tier | Workflow | Runner | What it runs |
|---|---|---|---|
| 1. build | `ci.yml` | GitHub-hosted, no GPU | compiles and installs brae, brae_pimpleFoam, brae_rhoSimpleFoam, brae_interFoam, brae_rhoCentralFoam and brae-agent for sm_80, sm_90 and sm_121 (one job each), configures the whole test suite without building it, and runs the known-red checker's self-test |
| 2. gpu-smoke | `gpu-smoke.yml` | the self-hosted GB10 | the tests labelled `smoke` (`ctest --preset gpu-smoke`), cached by `tools/ctest_cached.py` |
| 3. gpu-full | `gpu-full.yml` | the self-hosted GB10 | the whole suite but `known-red` and `diagnostic` (`ctest --preset full-validation`), cached, then the six known-red gates through `tools/known_red_check.py`, then the `diagnostic` tests in a step that reports and never fails the job (`ctest --preset diagnostic`) |
| 4. release | `release.yml` | both | tier 1, then tier 3 with the cache off (every test run, whatever passed before), then the release |

| Event | build | gpu-smoke | gpu-full |
|---|---|---|---|
| Pull request from a branch of simd-ai/brae | runs | runs, unless the change is documentation only | no |
| Pull request from a fork | runs | **fails on a hosted runner without checking anything out**; a maintainer pushes the change to a branch here, whose pull request runs it | no |
| Push to main or develop (a merge) | runs | runs, unless the change is documentation only | no |
| Nightly, 01:00 UTC | no | no | runs, cached |
| Manual dispatch | no | yes | yes; `no_cache` runs every test |
| Tag `v*` (release.yml) | runs | no | runs, uncached |

**Documentation only** means every changed file is Markdown, under `docs/`, an image or `LICENSE`. No test reads
any of them (checked 2026-11-20: Markdown appears in test and tool code only inside comments and docstrings). The
`gpu-smoke` job is then skipped, which GitHub reports as success to a required check; a `paths-ignore` filter would
instead leave the check pending and block the merge. Any other change runs the tier, and `tools/ctest_cached.py`
decides test by test what the change can reach: a test runs again when the bytes its binaries load, its scripts,
its fixtures, OpenFOAM or the GPU driver changed.

**Why a fork's `gpu-smoke` fails rather than skips:** a skipped job reports success to a required check, so a skip
would let a fork merge without the GPU tier. Fork code never reaches the self-hosted runner from these workflows.
A fork can still bring its own workflow file, which is why the approval setting below matters.

## Timings (GB10, ctest -j4)

| Tier | Tests | Measured |
|---|---|---|
| gpu-smoke | 402 | 617 s of test time in the 2026-11-20 run of main 2b26cdd6, so about 3 min at four at a time, after a build of about 6 min (configure 5 s, `cmake --build -j 20` 5 min 50 s in the same run); a job of about 10 min, less when the cache holds passes |
| gpu-full | 911 selected | 805 run in 2 h 41 min, 105 held from earlier runs (2026-11-20, main 2b26cdd6, the old gpu-test.yml) |

The smoke tier is a positive list, tests/cmake/tiers.cmake: every test that passed in at most 5 s in that run and has
no `slow` or `known-red` label, from every registration group (402 tests). The four manifest drift checks and
kepsilon_production_and_blend are fast but left out while their fixes are separate pull requests; every full tier
runs them. A test joins or leaves the list by the same measurement. The build is most of the smoke job: a compiler
cache on the runner (`CMAKE_CUDA_COMPILER_LAUNCHER=ccache`) would cut it for changes that touch few files.

One test is in the smoke tier although it takes longer, decided 2026-11-21: `pimple_pressure_reference_developed`
(about 22 s on the GB10), the pressure-reference gate, bounded by OpenFOAM's own decomposed spread. It took the place
of `pimple_pressure_reference`, whose uniform start decides the limiter by rounding signs, so that its result follows
the compiler and the GPU target. That one is labelled `diagnostic`: the full tiers run it in their last step, which
never fails the job, and `ctest --preset diagnostic` runs it by hand.

## Rules on main and develop

Today (read from the API on 2026-11-24): ruleset "Protect main" (`main-ruleset.json`), active on the default branch:
no deletion, no force push, changes through a pull request with 0 approvals and every conversation resolved,
required checks `build` and `gpu-smoke` from GitHub Actions. `ci.yml` keeps a job named `build` (it waits for the
three architectures), so that rule keeps working. With one maintainer GitHub does not let an author approve their
own pull request, so 0 approvals stays until there is a second reviewer.

`develop` has no rule yet. Proposed, `develop-ruleset.json`: the same rules on `develop`, so that it too moves only
through pull requests that passed `build` and `gpu-smoke`. Also proposed: delete a pull request's branch when it
merges, so merged branches do not accumulate.

## Actions settings

| Setting | Now (2026-11-20) | Proposed | Why |
|---|---|---|---|
| Fork pull request workflows need approval | `first_time_contributors` | `all_external_contributors` | The repository is public and the GB10 runner is registered to it. A fork's pull request runs the fork's own copy of the workflow files, so after one merged contribution an outside contributor's pull request could add a workflow that runs on `[self-hosted, ARM64]`. Approval for every outside contributor puts a maintainer in front of that. |
| Default `GITHUB_TOKEN` permission | `write` | `read` | Every workflow here declares `contents: read` but release.yml, which asks for `contents: write` itself. |

## On the runner

The manifest drift checks (`manifest_*_not_drifted`) need the `ofscan` index (a private checkout) beside the brae
checkout. Off CI they skip (77) without it; under GitHub Actions they fail instead (tools/of_manifest.py), so the
GPU tiers cannot pass them by not running them. On the GB10 the runner's work directory holds `ofscan` as a link to
the local checkout, next to `brae/`; `OFSCAN_ROOT` names another place.

## Applying it

The ruleset on main is applied. The rest is not. With `gh` logged in as a repository admin:

```bash
gh api -X POST repos/simd-ai/brae/rulesets --input docs/ci/develop-ruleset.json
gh api -X PATCH repos/simd-ai/brae -F delete_branch_on_merge=true
gh api -X PUT repos/simd-ai/brae/actions/permissions/fork-pr-contributor-approval \
    -f approval_policy=all_external_contributors
gh api -X PUT repos/simd-ai/brae/actions/permissions/workflow \
    -f default_workflow_permissions=read -F can_approve_pull_request_reviews=false
```

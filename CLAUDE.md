# brae

OpenFOAM's solvers, re-ported to CUDA, validated against real OpenFOAM rather than against expectations.

Current work: the **OF-Mirror Rebuild** on `foundation/of-mirror`, issue `#26` — re-porting solvers into a
tree that mirrors OpenFOAM's own. `src/README.md` has the layout; each solver's `PORT.md` has its state.

## Rules that apply to every turn

**Never run `git commit`, `git add`, or `git push`.** Hand over a copy-pasteable command and let the user
run it. Commit messages are ONE sentence, prefixed `#26:`, no `Co-Authored-By`. Identity is
`tito <titohiera@gmail.com>`.

**OpenFOAM's source is the semantic authority.** Read the `.C`/`.H` under
`/usr/lib/openfoam/openfoam2412/src` before claiming what OpenFOAM does. Never infer it from memory, from
a comment, or from a brae file that claims to mirror it. The manifests distinguish *derived* facts
(queried from `ofscan`) from *curated* judgements for exactly this reason — see `src/README.md`.

**Refuse rather than silently substitute.** An unimplemented model, scheme, boundary condition or thermo
must throw and name itself. A default that quietly stands in for the case's own setting is the defect this
project keeps finding: `Gauss linear orthogonal` where the case said `corrected`, `rhoInlet` where
OpenFOAM uses the live rho, a frozen inlet where OpenFOAM recomputes one.

**Never weaken a test.** Bounds tighten as the code improves; they do not loosen to accommodate it. Every
gate needs a control that fails when the thing under test is broken — see the `of-gate` skill.

**Run only the tests that cover what you touched.** The full `ctest` suite runs when the user asks for it,
and being about to commit is not an exemption. Never rebuild while `ctest` is running: it invalidates the
run, and a stale binary has reported a false result here before.

**Code style** — `/home/ghost/cudafoam/code-style.md`. One parameter per line, one statement per line,
Allman braces, ASCII-only comments, no decorative banners. Comments say *why*, and carry the measurement
that justified the line where there is one.

**End every message with a status table.**

## Skills

| Skill | Use it for |
|---|---|
| `of-port` | porting an OpenFOAM component: manifest → `_cpp` → gate → end-to-end → CUDA |
| `of-gate` | building a validation gate with an oracle, a control, and a fail-proof |
| `of-instrument` | when a gap will not localise: instrument OpenFOAM's own class and read its numbers |

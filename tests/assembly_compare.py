# One comparator for "brae's assembled turbulence system against OpenFOAM's own", shared by every gate
# that makes that comparison -- so the rule about WHICH rows carry the claim is written once.
#
#   assembly_compare.py <oracle time dir> <brae dump dir> <polyMesh dir> <flavour> <match|differ> <label>
#
# The flavour is the pair of NAME MAPS: OpenFOAM's instrument writes `stage_<eqn>D`/`Src`/`DUpper`/
# `DLower` (tools/dumpKOmegaSST, tools/dumpKEpsilon) and brae's two arms write their own columns, which
# differ between the host reference and the device closure. Nothing else about the comparison changes.
#
# WHAT IT ASSERTS, and why it is split that way. A turbulence wall function constrains its near-wall
# rows through fvMatrix::setValues, which zeroes every off-diagonal touching them. On such a row the
# solve reads Src/D whatever the diagonal is, so:
#
#   * the off-diagonals, and D and Src on the rows that SURVIVE, are the discriminator -- a scheme, a
#     coefficient or a gradient shows up there;
#   * the eliminated rows' IMPLIED VALUE (Src/D) is a completeness check. It does NOT move when the
#     convection scheme changes (measured: 1.8e-14 whether the case says limitedLinear or upwind),
#     so a gate that asserted only it would be vacuous. It is here so that a change to the wall
#     constraint itself cannot pass unnoticed.
#
# `differ` is how a gate proves its own comparison can witness what it names: the same columns against
# an oracle built with a DIFFERENT scheme must miss by more than DIFFER_FLOOR.
import sys

MATCH_BOUND = 1e-10     # two orders above the worst measured agreement on either gate (8.0e-13)
DIFFER_FLOOR = 1e-3     # two orders below the smallest control measured (1.2e-01)
# ...and per flavour where the floor is a different thing. brae's HOST arm against brae's DEVICE arm is
# not brae against OpenFOAM: the two arms sum the same terms in different orders (a device gather
# against a host loop), and on a developed field that is worth 1.7e-10 -- three orders more than the
# 8.0e-13 an arm reaches against OpenFOAM's own instrument, which is one sum against one sum. The bound
# is two orders above the worst measured, and the control still misses by 4.8e-01: seven orders of
# margin. It is NOT a loosening of the bound above -- that one still holds every OpenFOAM comparison.
FLAVOUR_BOUND = {"sstHostCuda": 1e-8}

# flavour -> (OpenFOAM's eight, brae's eight), in the order
# upper, lower, D, Src, kUpper, kLower, kD, kSrc
FLAVOURS = {
    # kOmegaSST, tools/dumpKOmegaSST
    "sstHost": (["stage_sstOmDUpper", "stage_sstOmDLower", "stage_sstOmD", "stage_sstOmSrc",
                 "stage_sstKDUpper", "stage_sstKDLower", "stage_sstKD", "stage_sstKSrc"],
                ["omUpper", "omLower", "omD", "omSrc", "kUpper", "kLower", "kD", "kSrc"]),
    "sstCuda": (["stage_sstOmDUpper", "stage_sstOmDLower", "stage_sstOmD", "stage_sstOmSrc",
                 "stage_sstKDUpper", "stage_sstKDLower", "stage_sstKD", "stage_sstKSrc"],
                ["omegaSysUpper", "omegaSysLower", "omegaSysD", "omegaSysSrc",
                 "kSysUpper", "kSysLower", "kSysD", "kSysSrc"]),
    # kEpsilon, tools/dumpKEpsilon. BOTH brae arms write the same names here -- the host driver's
    # latched dump (inter_turbulence_cpp.cu) and the device closure's dumpPrefix (kEpsilon.cu).
    "ke": (["stage_epsDUpper", "stage_epsDLower", "stage_epsD", "stage_epsSrc",
            "stage_kDUpper", "stage_kDLower", "stage_kD", "stage_kSrc"],
           ["epsUpper", "epsLower", "epsD", "epsSrc", "kUpper", "kLower", "kD", "kSrc"]),
    # kOmegaSST, brae's HOST arm against brae's DEVICE arm -- both sides plain columns, no OpenFOAM
    # oracle. It exists for a mesh with a PERIODIC PAIR, where OpenFOAM's own instrument cannot be the
    # left-hand side: its stage_sstOm* are the internal faces and the folded boundary, and the pair's
    # coefficient is neither. The chain is host == OpenFOAM (fields, gated) + device == host (here).
    "sstHostCuda": (["omUpper", "omLower", "omD", "omSrc", "kUpper", "kLower", "kD", "kSrc"],
                    ["omegaSysUpper", "omegaSysLower", "omegaSysD", "omegaSysSrc",
                     "kSysUpper", "kSysLower", "kSysD", "kSysSrc"]),
}
# the flavours whose LEFT side is a brae dump (plain columns) rather than an OpenFOAM field
PLAIN_LEFT = {"sstHostCuda"}
# ...and the interface column they also compare: brae applies the pair's off-diagonal as
# Apsi[own] += ifCoeff*psi[nbr] while the host keeps OpenFOAM's boundaryCoeffs, which is its NEGATIVE
IFC = {"sstHostCuda": [("omIfc", "omegaSysIfc"), ("kIfc", "kSysIfc")]}
# the labels the report prints, in the same order
COLS = ["upper", "lower", "D", "Src", "k upper", "k lower", "k D", "k Src"]


def ofField(path, n):
    s = open(path).read()
    i = s.index('internalField')
    head = s[i:s.index('\n', i)]
    if 'nonuniform' not in head and 'uniform' in head:
        return [float(head.split('uniform')[1].strip().rstrip(';'))] * n
    k = s.index('(', i)
    return [float(x) for x in s[k + 1:s.index(')', k)].split()]


def column(path):
    return [float(x) for x in open(path).read().split()]


def ofList(path):
    s = open(path).read()
    i = s.index('(', s.index('//'))
    return [int(x) for x in s[i + 1:s.rindex(')')].split()]


def main():
    ofDir, brDir, meshDir, flavour, expect, label = sys.argv[1:7]
    ofNames, brNames = FLAVOURS[flavour]
    bound = FLAVOUR_BOUND.get(flavour, MATCH_BOUND)

    own = ofList(meshDir + "/owner")
    nei = ofList(meshDir + "/neighbour")
    nIf = len(nei)
    nC = max(own) + 1
    want = [nIf, nIf, nC, nC, nIf, nIf, nC, nC]

    if flavour in PLAIN_LEFT:
        of = [column(ofDir + "/" + n) for n in ofNames]
    else:
        of = [ofField(ofDir + "/" + n, w) for n, w in zip(ofNames, want)]
    br = [column(brDir + "/" + n) for n in brNames]

    # a column of the wrong length is a vacuous comparison, not a pass
    for i, w in enumerate(want):
        if len(of[i]) != w or len(br[i]) != w:
            print("  FAIL: %s: %s has %d values and %s has %d, against the mesh's %d"
                  % (label, ofNames[i], len(of[i]), brNames[i], len(br[i]), w))
            return 1

    # the rows setValues eliminated: every off-diagonal touching them is exactly zero
    touch = {}
    for f, (o, n) in enumerate(zip(own, nei)):
        touch.setdefault(o, []).append(f)
        touch.setdefault(n, []).append(f)
    pinned = {c for c, fs in touch.items() if all(of[0][f] == 0.0 and of[1][f] == 0.0 for f in fs)}
    free = [c for c in range(nC) if c not in pinned]
    if not pinned or not free:
        print("  FAIL: %s: %d rows eliminated and %d free -- the split cannot be right"
              % (label, len(pinned), len(free)))
        return 1

    def rel(a, b, idx):
        scale = max(abs(a[i]) for i in idx) or 1.0
        return max(abs(a[i] - b[i]) for i in idx) / scale

    faces = range(nIf)
    d = {}
    for i, name in enumerate(COLS):
        idx = faces if want[i] == nIf else free
        d[name] = rel(of[i], br[i], idx)

    worstValue = 0.0
    for i in pinned:
        if of[2][i] == 0.0 or br[2][i] == 0.0:
            continue
        va, vb = of[3][i] / of[2][i], br[3][i] / br[2][i]
        worstValue = max(worstValue, abs(va - vb) / max(abs(va), 1e-30))

    # ...and the PAIR's own off-diagonal, which is where a div scheme reaches a coupled patch and the
    # one column the internal-face arrays cannot show. Compared NEGATED, and the sign is asserted: if
    # the two agreed as-is, the convention would have changed under us.
    for left, right in IFC.get(flavour, []):
        try:
            a = column(ofDir + "/" + left)
            b = column(brDir + "/" + right)
        except OSError:
            print("  FAIL: %s: %s / %s is missing -- the pair's coefficient is not being written"
                  % (label, left, right))
            return 1
        n = min(len(a), len(b))
        if not n:
            continue
        scale = max(abs(x) for x in a) or 1.0
        neg = max(abs(a[i] + b[i]) for i in range(n)) / scale
        asis = max(abs(a[i] - b[i]) for i in range(n)) / scale
        d["pair " + left] = neg
        if expect == "match" and asis < neg:
            print("  FAIL: %s: %s matches the device's %s WITHOUT negating (%.3e against %.3e) -- the "
                  "interface sign convention has changed" % (label, left, right, asis, neg))
            return 1

    print("  %s: %s" % (label, "  ".join("%s %.3e" % (k, v) for k, v in d.items())))
    print("     %d rows eliminated by setValues, their value %.3e; %d rows carry the scheme"
          % (len(pinned), worstValue, len(free)))

    ok = True
    if expect == "match":
        for k, v in d.items():
            if not (v < bound):
                print("  FAIL: %s: %s is %.3e, above %.1e" % (label, k, v, bound))
                ok = False
        if not (worstValue < bound):
            print("  FAIL: %s: an eliminated row's value is %.3e, above %.1e"
                  % (label, worstValue, bound))
            ok = False
    else:
        moved = max(d.values())
        if not (moved > DIFFER_FLOOR):
            print("  FAIL: %s: the two schemes' systems differ by only %.3e -- this comparison cannot "
                  "witness the scheme, so the matching arms prove nothing" % (label, moved))
            ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

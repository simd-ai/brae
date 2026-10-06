# Shared by ami_store_vs_openfoam.sh and ami_rows_vs_openfoam.sh: a case of two blocks that meet at x = 1,
# meshed by real blockMesh, and OpenFOAM's own AMI of the owner patch dumped by a coded function object.
# No solver runs: `postProcess -time 0` constructs the mesh and executes the object, which asks the patch for
# its AMI. The dump: "owner <0|1>", then for the owner's faces ("src") and the neighbour's ("tgt") a line a
# face -- index, mask (a cyclicACMI's mask(); the sum of weights otherwise), sum of weights, the number of
# partners, then each partner's face and weight -- at 18 digits.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BUILD:-$ROOT/build}/test_ami_store_vs_openfoam"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
[ -x "$BIN" ]      || { echo "SKIP: $BIN not built"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }
W=${KEEP_W:-$(mktemp -d)}
[ -n "${KEEP_W:-}" ] || trap 'rm -rf "$W"' EXIT
mkdir -p "$W"
set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
for t in blockMesh postProcess; do
    command -v "$t" > /dev/null 2>&1 || { echo "SKIP: $t not on PATH"; exit 77; }
done
rc=0
say() { printf '  %-5s %s\n' "$1" "$2"; [ "$1" = "ok:" ] || rc=1; }

# shell <dir> <owner patch> <neighbour patch>: everything but the blockMeshDict, which is the caller's
shell()
{
    local C="$1" own="$2" nbr="$3"
    mkdir -p "$C/system" "$C/constant" "$C/0"
    local head='FoamFile { version 2.0; format ascii; class dictionary; object'
    {
        echo "$head controlDict; }"
        echo "application pimpleFoam; startFrom startTime; startTime 0; stopAt endTime; endTime 1; deltaT 1;"
        echo "writeControl timeStep; writeInterval 1; writeFormat ascii; writePrecision 18; timePrecision 6;"
        echo "runTimeModifiable false;"
        cat <<DICT
functions
{
    amiDump
    {
        type            coded;
        libs            (utilityFunctionObjects);
        name            amiDump;
        codeInclude
        #{
            #include "cyclicACMIPolyPatch.H"
            #include "OFstream.H"
        #};
        codeExecute
        #{
            const fvMesh& m = mesh();
            const polyBoundaryMesh& bm = m.boundaryMesh();
            const polyPatch& op = bm[bm.findPatchID("$own")];
            const polyPatch& np = bm[bm.findPatchID("$nbr")];
            const cyclicAMIPolyPatch& ap = refCast<const cyclicAMIPolyPatch>(op);
            const auto& ami = ap.AMI();
            const bool acmi = isA<cyclicACMIPolyPatch>(op);
            scalarField sm(ami.srcWeightsSum());
            scalarField tm(ami.tgtWeightsSum());
            if (acmi)
            {
                sm = refCast<const cyclicACMIPolyPatch>(op).mask();
                tm = refCast<const cyclicACMIPolyPatch>(np).mask();
            }
            OFstream os(m.time().path()/"ami_dump.txt");
            os.precision(18);
            os << "owner " << (ap.owner() ? 1 : 0) << nl;
            os << "src " << ami.srcAddress().size() << nl;
            forAll(ami.srcAddress(), i)
            {
                os << i << " " << sm[i] << " " << ami.srcWeightsSum()[i] << " " << ami.srcAddress()[i].size();
                forAll(ami.srcAddress()[i], j)
                {
                    os << " " << ami.srcAddress()[i][j] << " " << ami.srcWeights()[i][j];
                }
                os << nl;
            }
            os << "tgt " << ami.tgtAddress().size() << nl;
            forAll(ami.tgtAddress(), i)
            {
                os << i << " " << tm[i] << " " << ami.tgtWeightsSum()[i] << " " << ami.tgtAddress()[i].size();
                forAll(ami.tgtAddress()[i], j)
                {
                    os << " " << ami.tgtAddress()[i][j] << " " << ami.tgtWeights()[i][j];
                }
                os << nl;
            }
            return true;
        #};
    }
}
DICT
    } > "$C/system/controlDict"
    {
        echo "$head fvSchemes; }"
        echo "ddtSchemes { default Euler; } gradSchemes { default Gauss linear; } divSchemes { default none; }"
        echo "laplacianSchemes { default Gauss linear corrected; } interpolationSchemes { default linear; }"
        echo "snGradSchemes { default corrected; }"
    } > "$C/system/fvSchemes"
    {
        echo "$head fvSolution; }"
        echo "solvers {} PIMPLE {}"
    } > "$C/system/fvSolution"
}

# dump <dir>: blockMesh, then OpenFOAM's AMI into <dir>/ami_dump.txt
dump()
{
    local C="$1"
    ( cd "$C" && blockMesh > log.blockMesh 2>&1 ) || { echo "FAIL: blockMesh"; tail -20 "$C/log.blockMesh"; exit 1; }
    ( cd "$C" && postProcess -time 0 > log.postProcess 2>&1 ) \
        || { echo "FAIL: postProcess"; tail -20 "$C/log.postProcess"; exit 1; }
    [ -s "$C/ami_dump.txt" ] || { echo "FAIL: OpenFOAM wrote no AMI dump"; tail -20 "$C/log.postProcess"; exit 1; }
}

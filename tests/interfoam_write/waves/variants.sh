#!/usr/bin/env bash
# The write gate: three inputs the list echo meets: a size prefix, an empty entry, a quoted string.
. "$(dirname "$0")/../lib.sh"
. "$(dirname "$0")/_common.sh"
wcase streamFunction of
# the `empty` and `quoted` variants are staged on stokesI's OpenFOAM case
wcase stokesI of
# ...and three inputs the list echo meets, each against OpenFOAM:
#   sized    streamFunction's Bjs/Ejs as `10 ( ... )`: the prefix is a SIZE (ListIO.C:210-290). FAIL-PROOF
#            (2026-09-30): the reader that took every number (scalarListOr) shifted every harmonic, U 1.48e+01.
#   empty    `extra;` in stokesI's inlet: a legal zero-token entry OpenFOAM writes as `extra           ;`
#            (primitiveEntryIO.C:126-131, 280-307) -- once refused late, inside write().
#   quoted   `note "a b";`: OpenFOAM writes it back quoted; brae's tokenizer drops the quotes, so it is
#            named at start-up and the run stops at its first write with nothing written.
vin()   # vin <waveProperties> <line> -- a line at the top of the inlet entry
{
    python3 - "$1" "$2" <<'EOF_VI'
import re, sys
p, line = sys.argv[1], sys.argv[2]
s = open(p).read()
s, n = re.subn(r'(\ninlet\s*\n\{\n)', lambda m: m.group(1) + '    ' + line + '\n\n', s)
open(p, 'w').write(s)
sys.exit(0 if n == 1 else 1)
EOF_VI
}
for v in sized empty quoted; do
    src=stokesI
    [ $v = sized ] && src=streamFunction
    [ -d "$W/w_of_$src" ] || { say "ARM V  $v: $src did not run in arm W" FAIL; continue; }
    for side in of br; do
        d="$W/v_${side}_$v"
        rm -rf "$d"
        mkdir -p "$d"
        cp -r "$W/w_of_$src/0" "$W/w_of_$src/constant" "$W/w_of_$src/system" "$d/"
        wp="$d/constant/waveProperties"
        case $v in
            sized)
                python3 - "$wp" <<'EOF_SZ'
import re, sys
p = sys.argv[1]
s = open(p).read()
s, n = re.subn(r'(\n\s*(?:Bjs|Ejs)\s+)\(([^)]*)\);', lambda m: '%s%d (%s);' % (m.group(1), len(m.group(2).split()), m.group(2)), s)
open(p, 'w').write(s)
sys.exit(0 if n == 2 else 1)
EOF_SZ
                ;;
            empty)
                vin "$wp" 'extra;'
                ;;
            quoted)
                vin "$wp" 'note            "a b";'
                ;;
        esac
        [ $? -eq 0 ] || say "ARM V  $v: the input was not staged in $d" FAIL
    done
    runof "$W/v_of_$v"
    if [ $v = quoted ]; then
        first=$(timedirs "$W/v_of_$v" | awk '{print $1}')
        ( cd "$W/v_br_$v" && "$BIN" -case . > log.brae 2>&1 )
        rc=$?
        grep -q '^note *"a b";' "$W/v_of_$v/$first/uniform/waveProperties.inlet" \
            && [ $rc -ne 0 ] && grep -q "a quoted string" "$W/v_br_$v/log.brae" && [ -z "$(timedirs "$W/v_br_$v")" ] \
            && say "ARM V  quoted: OpenFOAM writes the quotes back; brae names the file at start-up, nothing written" ok \
            || { say "ARM V  quoted: OpenFOAM writes the quotes back; brae names the file at start-up, nothing written [rc $rc]" FAIL; tail -2 "$W/v_br_$v/log.brae" | sed 's/^/      /'; }
        continue
    fi
    runbrae "$W/v_br_$v" host
    python3 "$CMP" "$W/v_of_$v" "$W/v_br_$v" $(timedirs "$W/v_of_$v") > "$W/cmp_v_$v.txt" 2>&1
    bound=3e-08
    # the W host bounds: streamFunction 3e-08, stokesI 5e-07
    [ $v = empty ] && bound=5e-07
    judge "$v host" "$W/cmp_v_$v.txt" "$bound" "$W/v_of_$v/log.interFoam" \
        && vbytes "$W/v_of_$v" "$W/v_br_$v" \
        && say "ARM V  [host] $v: OpenFOAM's run within $bound, every waveProperties file byte-identical" ok \
        || say "ARM V  [host] $v: OpenFOAM's run within $bound, every waveProperties file byte-identical" FAIL
done
finish "arm V: sized, empty and quoted entries each match OpenFOAM or are named"

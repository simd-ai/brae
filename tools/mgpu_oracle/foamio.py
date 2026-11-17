"""Minimal OpenFOAM v2606 case reader (numpy only): FoamFile headers, ascii+binary lists, volFields and
surfaceFields with their boundaryField, labelLists (addressing), polyMesh/boundary, and the processor->global
maps.  Used by compare_fields.py and decomp_stats.py."""
import os
import re
import numpy as np

NCOMP = {"scalar": 1, "vector": 3, "sphericalTensor": 1, "symmTensor": 6, "tensor": 9, "label": 1}
_WS = b" \t\r\n"
_DELIM = b"(){};"


class Reader:
    def __init__(self, data, pos=0, scalar_bytes=8, label_bytes=4, binary=False):
        self.d = data
        self.p = pos
        self.sb = scalar_bytes
        self.lb = label_bytes
        self.binary = binary

    def skip_ws(self):
        d, n = self.d, len(self.d)
        while self.p < n:
            c = d[self.p:self.p + 1]
            if c in (b" ", b"\t", b"\r", b"\n"):
                self.p += 1
            elif d.startswith(b"//", self.p):
                e = d.find(b"\n", self.p)
                self.p = n if e < 0 else e + 1
            elif d.startswith(b"/*", self.p):
                e = d.find(b"*/", self.p)
                self.p = n if e < 0 else e + 2
            else:
                break

    def token(self):
        self.skip_ws()
        d, n = self.d, len(self.d)
        if self.p >= n:
            return None
        c = d[self.p:self.p + 1]
        if c in (b"(", b")", b"{", b"}", b";"):
            self.p += 1
            return c.decode()
        if c == b'"':
            e = self.p + 1
            while d[e:e + 1] != b'"':
                e += 2 if d[e:e + 1] == b"\\" else 1
            t = d[self.p:e + 1].decode()
            self.p = e + 1
            return t
        s = self.p
        depth = 0
        while self.p < n:
            ch = d[self.p:self.p + 1]
            if ch == b"<":
                depth += 1
            elif ch == b">":
                depth -= 1
            elif depth == 0 and (ch in _WS or ch in _DELIM):
                break
            self.p += 1
        return d[s:self.p].decode()

    def peek(self):
        s = self.p
        t = self.token()
        self.p = s
        return t

    def list_body(self, typ, n):
        """positioned just after the '(' of an n-element list of typ; consumes through ')'."""
        k = NCOMP[typ]
        if n == 0:
            self.expect(")")
            return np.zeros((0, k) if k > 1 else (0,))
        if self.binary:
            w = self.lb if typ == "label" else self.sb
            nb = n * k * w
            arr = np.frombuffer(self.d, dtype=("<i%d" % w if typ == "label" else "<f%d" % w),
                                count=n * k, offset=self.p)
            self.p += nb
            self.expect(")")
        else:
            vals = []
            if k == 1:
                while True:
                    t = self.token()
                    if t == ")":
                        break
                    vals.append(float(t))
            else:
                while True:
                    t = self.token()
                    if t == ")":
                        break
                    assert t == "("
                    for _ in range(k):
                        vals.append(float(self.token()))
                    self.expect(")")
            arr = np.array(vals, dtype=np.int64 if typ == "label" else np.float64)
        arr = np.array(arr, dtype=np.int64 if typ == "label" else np.float64)
        return arr.reshape(n, k) if k > 1 else arr

    def expect(self, s):
        t = self.token()
        if t != s:
            raise ValueError("expected %r got %r at %d" % (s, t, self.p))

    def value(self):
        """one entry value (up to, not including, the ';'): returns a python object or None."""
        t = self.token()
        if t == "nonuniform":
            lt = self.token()                       # List<scalar>
            m = re.match(r"List<(\w+)>", lt)
            typ = m.group(1) if m else "scalar"
            n = int(self.token())
            if n == 0 and self.peek() == ";":       # an empty list is written 'List<scalar> 0;'
                return ("nonuniform", typ, np.zeros((0, NCOMP[typ]) if NCOMP[typ] > 1 else (0,)))
            self.expect("(")
            return ("nonuniform", typ, self.list_body(typ, n))
        if t == "uniform":
            v = self.value()
            return ("uniform", v)
        if t == "(":
            vals = []
            while True:
                tt = self.peek()
                if tt == ")":
                    self.token()
                    break
                vals.append(self.value())
            return vals
        if t == "{":
            return self.dict_body()
        try:
            return float(t)
        except ValueError:
            return t

    def entry_value(self):
        """tokens of one entry until ';' (an entry may be several values e.g. 'dimensions [..]')"""
        vals = []
        while True:
            tt = self.peek()
            if tt == ";":
                self.token()
                break
            if tt == "}" or tt is None:
                break
            vals.append(self.value())
        return vals[0] if len(vals) == 1 else vals

    def dict_body(self):
        out = {}
        while True:
            t = self.token()
            if t is None or t == "}":
                return out
            nxt = self.peek()
            if nxt == "{":
                self.token()
                out[t] = self.dict_body()
            else:
                out[t] = self.entry_value()


def _header(data):
    m = re.search(rb"FoamFile\s*\{", data)
    r = Reader(data)
    r.p = m.end()
    h = r.dict_body()
    fmt = h.get("format", "ascii")
    arch = h.get("arch", '"LSB;label=32;scalar=64"')
    lb = 8 if "label=64" in str(arch) else 4
    sb = 4 if "scalar=32" in str(arch) else 8
    return h, r.p, fmt == "binary", sb, lb


def read_label_list(path):
    data = open(path, "rb").read()
    h, pos, binary, sb, lb = _header(data)
    r = Reader(data, pos, sb, lb, binary)
    n = int(r.token())
    r.expect("(")
    return r.list_body("label", n)


def _expand(v, typ_k, n):
    """uniform/nonuniform value -> (n,k) or (n,) array (None for no value)"""
    if v is None:
        return None
    if isinstance(v, tuple) and v[0] == "nonuniform":
        return v[2]
    if isinstance(v, tuple) and v[0] == "uniform":
        u = v[1]
        if isinstance(u, list):
            u = np.array([float(x) for x in u])
            return np.tile(u, (n, 1))
        return np.full(n, float(u))
    return None


class Field:
    def __init__(self, name, cls, internal, boundary, oriented):
        self.name, self.cls, self.internal, self.boundary, self.oriented = name, cls, internal, boundary, oriented


def read_field(path, patch_sizes, n_internal):
    """patch_sizes: {patchName: nFaces}; n_internal: nCells (vol) or nInternalFaces (surface)."""
    data = open(path, "rb").read()
    h, pos, binary, sb, lb = _header(data)
    cls = h["class"]
    typ = {"volScalarField": "scalar", "volVectorField": "vector", "volSymmTensorField": "symmTensor",
           "volTensorField": "tensor", "surfaceScalarField": "scalar", "surfaceVectorField": "vector",
           "volSphericalTensorField": "sphericalTensor"}[cls]
    r = Reader(data, pos, sb, lb, binary)
    body = r.dict_body()
    k = NCOMP[typ]
    iv = body["internalField"]
    nint = n_internal
    if isinstance(iv, tuple) and iv[0] == "nonuniform":
        internal = iv[2]
    else:
        internal = _expand(iv, k, nint)
    bnd = {}
    for pname, pd in body.get("boundaryField", {}).items():
        n = patch_sizes.get(pname)
        if n is None:
            continue
        val = pd.get("value") if isinstance(pd, dict) else None
        arr = _expand(val, k, n)
        bnd[pname] = (pd.get("type", "?") if isinstance(pd, dict) else "?", arr, n)
    return Field(os.path.basename(path), cls, internal, bnd, body.get("oriented") == "oriented")


def read_boundary(path):
    data = open(path, "rb").read()
    h, pos, binary, sb, lb = _header(data)
    r = Reader(data, pos, sb, lb, False)
    n = int(r.token())
    r.expect("(")
    patches = []
    for _ in range(n):
        name = r.token()
        r.expect("{")
        d = r.dict_body()
        patches.append(dict(name=name, type=d.get("type"), nFaces=int(d["nFaces"]),
                            startFace=int(d["startFace"]),
                            neighbProcNo=int(d["neighbProcNo"]) if "neighbProcNo" in d else None,
                            myProcNo=int(d["myProcNo"]) if "myProcNo" in d else None))
    return patches


def mesh_counts(polydir):
    """nCells, nPoints, nFaces, nInternalFaces from the owner header note."""
    data = open(os.path.join(polydir, "owner"), "rb").read(2000)
    m = re.search(rb"note\s+\"([^\"]*)\"", data)
    note = m.group(1).decode()
    g = dict(re.findall(r"(\w+):(\d+)", note))
    return int(g["nCells"]), int(g["nPoints"]), int(g["nFaces"]), int(g["nInternalFaces"])


def time_dirs(root):
    out = []
    for e in os.listdir(root):
        if os.path.isdir(os.path.join(root, e)):
            try:
                out.append((float(e), e))
            except ValueError:
                pass
    return sorted(out)


class Tree:
    """A result tree: a serial/reconstructed case (kind 'global') or a decomposed one with processorN dirs."""

    def __init__(self, root):
        self.root = root
        procs = sorted((d for d in os.listdir(root) if re.fullmatch(r"processor\d+", d)),
                       key=lambda s: int(s[9:]))
        self.kind = "decomposed" if procs else "global"
        if self.kind == "global":
            pd = os.path.join(root, "constant", "polyMesh")
            self.n_cells, _, self.n_faces, self.n_int = mesh_counts(pd)
            self.patches = read_boundary(os.path.join(pd, "boundary"))
            self.times = time_dirs(root)
        else:
            self.procs = procs
            self.rank = []
            for p in procs:
                pd = os.path.join(root, p, "constant", "polyMesh")
                cp = read_label_list(os.path.join(pd, "cellProcAddressing"))
                fp = read_label_list(os.path.join(pd, "faceProcAddressing"))
                nc, _, nf, ni = mesh_counts(pd)
                self.rank.append(dict(dir=p, pd=pd, cellmap=cp, facemap=fp, n_cells=nc, n_faces=nf, n_int=ni,
                                      patches=read_boundary(os.path.join(pd, "boundary"))))
            self.n_cells = int(sum(r["n_cells"] for r in self.rank))
            nbf = 0
            for r in self.rank:
                nbf += sum(p["nFaces"] for p in r["patches"] if p["type"] != "processor"
                           and p["type"] != "processorCyclic")
            self.n_faces = int(max(np.abs(r["facemap"]).max() for r in self.rank))
            self.n_int = self.n_faces - nbf
            names = []
            for r in self.rank:
                for p in r["patches"]:
                    if p["type"] not in ("processor", "processorCyclic") and p["name"] not in names:
                        names.append(p["name"])
            self.times = time_dirs(os.path.join(root, procs[0]))
            self.patch_names = names

    def patch_cells(self, ri, patchname):
        """local owner cells of the faces of a patch of rank ri"""
        r = self.rank[ri]
        if "owner" not in r:
            r["owner"] = read_label_list(os.path.join(r["pd"], "owner"))
        for p in r["patches"]:
            if p["name"] == patchname:
                return r["owner"][p["startFace"]:p["startFace"] + p["nFaces"]]
        raise KeyError(patchname)

    def field_names(self, tname):
        d = os.path.join(self.root, tname) if self.kind == "global" else os.path.join(self.root, self.procs[0], tname)
        return sorted(f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f)) and not f.startswith("."))

    def gpatches(self):
        """global boundary: [(name, start, n)] in global face numbering minus n_int"""
        if self.kind == "global":
            return [(p["name"], p["startFace"] - self.n_int, p["nFaces"]) for p in self.patches]
        raise NotImplementedError

    def load(self, tname, fname):
        """-> dict(cls, internal (cells or int faces, global order), boundary: ndarray over global boundary faces
        (NaN where a patch carries no value), oriented, extra: coherence info)"""
        if self.kind == "global":
            return self._load_global(tname, fname)
        return self._load_decomposed(tname, fname)

    def _load_global(self, tname, fname):
        sizes = {p["name"]: p["nFaces"] for p in self.patches}
        path = os.path.join(self.root, tname, fname)
        f = self._read(path, sizes)
        nb = self.n_faces - self.n_int
        k = NCOMP[_typ(f.cls)]
        b = np.full((nb, k) if k > 1 else (nb,), np.nan)
        for p in self.patches:
            ent = f.boundary.get(p["name"])
            if ent is not None and ent[1] is not None and len(ent[1]) == p["nFaces"]:
                s = p["startFace"] - self.n_int
                b[s:s + p["nFaces"]] = ent[1]
        return dict(cls=f.cls, internal=f.internal, boundary=b, oriented=f.oriented, proc_check=None, proc=None)

    def _read(self, path, sizes):
        is_surf = "surface" in _header(open(path, "rb").read(4000))[0]["class"]
        n = self.n_int if is_surf else self.n_cells
        if self.kind == "decomposed":
            raise RuntimeError
        return read_field(path, sizes, n)

    def _load_decomposed(self, tname, fname):
        nb = self.n_faces - self.n_int
        internal = None
        bnd = None
        flipped = None
        proc_vals = {}       # global internal face -> list of (rank, signed value) from processor patches
        cls = None
        oriented = False
        proc_patch = []      # (rank, patchname, neighbProcNo, global internal faces (0-based), flip, values)
        for ri, r in enumerate(self.rank):
            path = os.path.join(self.root, r["dir"], tname, fname)
            hdr = _header(open(path, "rb").read(4000))[0]
            is_surf = "surface" in hdr["class"]
            sizes = {p["name"]: p["nFaces"] for p in r["patches"]}
            f = read_field(path, sizes, r["n_int"] if is_surf else r["n_cells"])
            cls, oriented = f.cls, f.oriented
            k = NCOMP[_typ(cls)]
            shape = (lambda n: (n, k) if k > 1 else (n,))
            if internal is None:
                ni = self.n_int if is_surf else self.n_cells
                internal = np.full(shape(ni), np.nan)
                bnd = np.full(shape(nb), np.nan)
                flipped = np.zeros(self.n_int, dtype=bool) if is_surf else None
            if is_surf:
                fm = r["facemap"]
                sgn = np.where(fm[:r["n_int"]] < 0, -1.0, 1.0)
                gi = np.abs(fm[:r["n_int"]]) - 1
                v = f.internal
                if oriented:
                    v = v * (sgn if k == 1 else sgn[:, None])
                internal[gi] = v
                flipped[gi] = sgn < 0
            else:
                internal[r["cellmap"]] = f.internal
            for p in r["patches"]:
                ent = f.boundary.get(p["name"])
                if ent is None or ent[1] is None:
                    continue
                s, n = p["startFace"], p["nFaces"]
                fm = r["facemap"][s:s + n]
                gidx = np.abs(fm) - 1
                sg = np.where(fm < 0, -1.0, 1.0)
                vals = ent[1]
                if p["type"] in ("processor", "processorCyclic"):
                    if oriented and is_surf:
                        vals = vals * (sg if k == 1 else sg[:, None])
                    proc_patch.append((ri, p["name"], p["neighbProcNo"], gidx, vals))
                    if is_surf and not fname.endswith(("Upper.dump", "Lower.dump", "IntCoeffs.dump", "BouCoeffs.dump")):
                        # the cut face is an internal face of the global mesh: its value lives here
                        internal[gidx] = vals
                        flipped[gidx] = sg < 0
                    continue
                if oriented and is_surf:
                    vals = vals * (sg if k == 1 else sg[:, None])
                bnd[gidx - self.n_int] = vals
        # coherence of the two sides of every processor interface (rank pairs)
        is_surf_any = cls.startswith("surface")
        coh = 0.0
        ncoh = 0
        bymap = {}
        for (ri, nm, nb_, g, v) in proc_patch:
            bymap[(ri, nb_)] = (g, v)
        for (a, b_), (g, v) in bymap.items():
            if not is_surf_any:
                break
            if a < b_ and (b_, a) in bymap:
                g2, v2 = bymap[(b_, a)]
                o1, o2 = np.argsort(g), np.argsort(g2)
                if len(g) == len(g2) and np.array_equal(g[o1], g2[o2]):
                    d = np.abs(v[o1] - v2[o2])
                    if d.size:
                        coh = max(coh, float(np.nanmax(d)))
                    ncoh += len(g)
        return dict(cls=cls, internal=internal, boundary=bnd, oriented=oriented, flipped=flipped,
                    proc_check=(coh, ncoh), proc=proc_patch)


def _typ(cls):
    return {"volScalarField": "scalar", "volVectorField": "vector", "volSymmTensorField": "symmTensor",
            "volTensorField": "tensor", "surfaceScalarField": "scalar", "surfaceVectorField": "vector",
            "volSphericalTensorField": "sphericalTensor"}[cls]

// The geometry of this rank's processor patches, as OpenFOAM computes it, in the text format brae's
// writeProcessorGeometryDump (src/parallel/processor_geometry_dump.cuh) writes, for
// tools/mgpu_oracle/compare_proc_geometry.py. Included by interFoam.C right after the mesh is created when
// BRAE_PROC_GEOMETRY is set; the solver then exits. One file per rank: <rank's case>/procGeometry.dump.
//
//   rank <r> nCells <n> nProcPatches <k>
//   patch <name> myProcNo <a> neighbProcNo <b> size <s> start <st>
//   f <i> <cell> <nbrCell> <slot=-1> <w> <dc> <nodc> <magSf> <Sf> <delta> <nonOrthCorrectionVector>
//
// nbrCell: the NEIGHBOUR's local owner cell of the face: its patch's faceCells over PstreamBuffers.
{
    const fvBoundaryMesh& bm = mesh.boundary();

    // force the construction the solver would do
    const surfaceScalarField& pgW = mesh.surfaceInterpolation::weights();
    const surfaceScalarField& pgDc = mesh.surfaceInterpolation::deltaCoeffs();
    const surfaceScalarField& pgNdc = mesh.surfaceInterpolation::nonOrthDeltaCoeffs();
    const surfaceVectorField& pgCv = mesh.surfaceInterpolation::nonOrthCorrectionVectors();
    const surfaceVectorField& pgSf = mesh.Sf();
    const surfaceScalarField& pgMagSf = mesh.magSf();

    // the neighbour's owner cells: its patch's faceCells, face for face (processorPolyPatch's own transport)
    PstreamBuffers pgBufs(Pstream::commsTypes::nonBlocking);
    forAll(bm, patchi)
    {
        if (isA<processorFvPatch>(bm[patchi]))
        {
            const processorFvPatch& pp = refCast<const processorFvPatch>(bm[patchi]);
            UOPstream toNbr(pp.neighbProcNo(), pgBufs);
            toNbr << labelList(pp.faceCells());
        }
    }
    pgBufs.finishedSends();

    label pgNProc = 0;
    forAll(bm, patchi)
    {
        if (isA<processorFvPatch>(bm[patchi]) && bm[patchi].size() > 0)
        {
            ++pgNProc;
        }
    }

    FILE* pgf = fopen((runTime.path()/"procGeometry.dump").c_str(), "w");
    if (!pgf)
    {
        FatalErrorInFunction << "cannot write procGeometry.dump" << exit(FatalError);
    }
    fprintf(pgf, "rank %d nCells %d nProcPatches %d\n", int(Pstream::myProcNo()), int(mesh.nCells()), int(pgNProc));

    forAll(bm, patchi)
    {
        if (!isA<processorFvPatch>(bm[patchi]))
        {
            continue;
        }
        const processorFvPatch& pp = refCast<const processorFvPatch>(bm[patchi]);
        UIPstream fromNbr(pp.neighbProcNo(), pgBufs);
        const labelList nbrCells(fromNbr);
        if (pp.size() == 0)
        {
            continue;   // brae skips zero-size processor patches
        }
        const vectorField delta(pp.delta());
        const scalarField& w = pgW.boundaryField()[patchi];
        const scalarField& dc = pgDc.boundaryField()[patchi];
        const scalarField& ndc = pgNdc.boundaryField()[patchi];
        const vectorField& cv = pgCv.boundaryField()[patchi];
        const vectorField& sf = pgSf.boundaryField()[patchi];
        const scalarField& msf = pgMagSf.boundaryField()[patchi];
        const labelUList& fc = pp.faceCells();

        fprintf
        (
            pgf, "patch %s myProcNo %d neighbProcNo %d size %d start %d\n",
            pp.name().c_str(), int(pp.myProcNo()), int(pp.neighbProcNo()), int(pp.size()), int(pp.start())
        );
        forAll(pp, i)
        {
            fprintf
            (
                pgf,
                "f %d %d %d %d %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g\n",
                int(i), int(fc[i]), int(nbrCells[i]), -1, w[i], dc[i], ndc[i], msf[i], sf[i].x(), sf[i].y(),
                sf[i].z(), delta[i].x(), delta[i].y(), delta[i].z(), cv[i].x(), cv[i].y(), cv[i].z()
            );
        }
    }
    fclose(pgf);
    Info<< "procGeometry.dump written" << endl;
}

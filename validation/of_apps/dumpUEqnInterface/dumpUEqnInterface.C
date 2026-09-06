// OpenFOAM-side oracle: the REAL simpleFoam momentum matrix, with OpenFOAM's own coupled-interface
// coefficients, dumped per cell and per patch face.
//
// WHY. brae's cyclicAMI is now covered by sixteen stage gates against a _cpp reference, all
// bit-identical, and by a geometry cross-check against the separately-gated cyclic geometry. None of
// that can explain pipeCyclic's interface-localised momentum residual, and none of it ever could: every
// one of those comparisons proves brae's DEVICE equals brae's REFERENCE, and the reference is a
// transcription of what the device does. What has never been checked is brae's interface momentum
// equation against OPENFOAM's.
//
// WHAT IT DUMPS, and why these quantities. OpenFOAM assembles a coupled patch into the same two arrays
// as any other patch:
//
//   internalCoeffs[p][i]   added to the diagonal of the owner cell
//   boundaryCoeffs[p][i]   applied by the solver as  result[cell] -= boundaryCoeffs * psiNeighbour
//                          (fvMatrix::updateInterfaceMatrix), so the EFFECTIVE off-diagonal that
//                          multiplies the neighbour value is MINUS boundaryCoeffs
//
// brae holds the same two things as `diag += lap + max(phi,0)` and
// `ifCoeff = -lap + min(phi,0)` applied as `Apsi[own] += ifCoeff*interp(psi)`. So the comparison is
//
//   brae ifCoeff[i]              vs   -boundaryCoeffs[p][i]
//   brae diag contribution       vs    internalCoeffs[p][i]
//
// on the cyclicAMI patches. That is the one comparison that can settle whether brae's interface
// momentum equation IS OpenFOAM's, and it needs OpenFOAM to say so.
//
// The matrix is built exactly as simpleFoam's UEqn.H builds it -- div(phi,U) + divDevReff(U) == 0, then
// relax() -- because a matrix assembled any other way answers a different question. fvOptions and MRF
// are deliberately omitted: pipeCyclic has neither, and including them would make the dump depend on
// machinery this oracle is not testing.
#include "fvCFD.H"
#include "singlePhaseTransportModel.H"
#include "turbulentTransportModel.H"

int main(int argc, char *argv[])
{
    #include "setRootCase.H"
    #include "createTime.H"
    #include "createMesh.H"

    instantList times = runTime.times();
    runTime.setTime(times.last(), times.size() - 1);
    Info << "dumpUEqnInterface at t = " << runTime.timeName() << endl;

    volVectorField U(IOobject("U", runTime.timeName(), mesh, IOobject::MUST_READ, IOobject::NO_WRITE), mesh);
    surfaceScalarField phi(IOobject("phi", runTime.timeName(), mesh, IOobject::MUST_READ, IOobject::NO_WRITE), mesh);

    singlePhaseTransportModel laminarTransport(U, phi);
    autoPtr<incompressible::turbulenceModel> turbulence
    (
        incompressible::turbulenceModel::New(U, phi, laminarTransport)
    );
    turbulence->validate();

    // simpleFoam/UEqn.H, minus the fvOptions and MRF this case does not have.
    tmp<fvVectorMatrix> tUEqn
    (
        fvm::div(phi, U)
      + turbulence->divDevReff(U)
    );
    fvVectorMatrix& UEqn = tUEqn.ref();
    UEqn.relax();

    OFstream os(runTime.path()/"ueqn_interface.dat");
    os.precision(16);
    os << "# nCells nInternalFaces nPatches" << nl;
    os << mesh.nCells() << ' ' << mesh.nInternalFaces() << ' ' << mesh.boundary().size() << nl;

    // D() is diag() plus the component-AVERAGE of internalCoeffs, which is what OpenFOAM's segregated
    // solver actually inverts. brae keeps its boundary diagonal per component (slip and symmetry give
    // different values per component), so a comparison has to form the average on brae's side.
    const scalarField D(UEqn.D());
    os << "# D" << nl;
    forAll(D, c) os << D[c] << nl;

    os << "# diag" << nl;
    forAll(UEqn.diag(), c) os << UEqn.diag()[c] << nl;

    os << "# source" << nl;
    forAll(UEqn.source(), c)
    {
        const vector& s = UEqn.source()[c];
        os << s.x() << ' ' << s.y() << ' ' << s.z() << nl;
    }

    // source() + addBoundarySource(): the right-hand side the solver sees for a NON-coupled patch. A
    // coupled patch is deliberately excluded by OpenFOAM here (couples=false would include it), which is
    // exactly the point -- its contribution stays on the left as an interface update.
    // fvMatrix::addBoundarySource is protected, so it is reconstructed here from the same two arrays it
    // reads: for every NON-coupled patch, source[faceCell] += boundaryCoeffs. Coupled patches are skipped
    // -- which is the behaviour that matters, since it is precisely why an interface contribution stays
    // on the left-hand side as an interface update rather than folding into the right.
    vectorField bsrc(UEqn.source());
    forAll(mesh.boundary(), p)
    {
        const fvPatch& fp = mesh.boundary()[p];
        if (fp.coupled()) continue;
        forAll(fp, i) bsrc[fp.faceCells()[i]] += UEqn.boundaryCoeffs()[p][i];
    }
    os << "# sourcePlusBoundary" << nl;
    forAll(bsrc, c) os << bsrc[c].x() << ' ' << bsrc[c].y() << ' ' << bsrc[c].z() << nl;

    os << "# patches: name size coupled, then per face  iC.x iC.y iC.z  bC.x bC.y bC.z  faceCell" << nl;
    forAll(mesh.boundary(), p)
    {
        const fvPatch& fp = mesh.boundary()[p];
        os << fp.name() << ' ' << fp.size() << ' ' << (fp.coupled() ? 1 : 0) << nl;
        forAll(fp, i)
        {
            const vector& ic = UEqn.internalCoeffs()[p][i];
            const vector& bc = UEqn.boundaryCoeffs()[p][i];
            os << ic.x() << ' ' << ic.y() << ' ' << ic.z() << ' '
               << bc.x() << ' ' << bc.y() << ' ' << bc.z() << ' '
               << fp.faceCells()[i] << nl;
        }
    }

    // The interface's own geometry, so a disagreement in the coefficients can be attributed to the
    // coefficient or to the geometry underneath it without a second run.
    os << "# patch geometry: name size, then per face  deltaCoeff  weight  magSf" << nl;
    forAll(mesh.boundary(), p)
    {
        const fvPatch& fp = mesh.boundary()[p];
        if (!fp.coupled()) continue;
        // deltaCoeffs and nonOrthDeltaCoeffs live on the MESH, indexed by global face; the patch only
        // exposes its own slice of them.
        const surfaceScalarField& dcf = mesh.deltaCoeffs();
        const surfaceScalarField& nodcf = mesh.nonOrthDeltaCoeffs();
        const scalarField& dc = dcf.boundaryField()[p];
        const scalarField& nodc = nodcf.boundaryField()[p];
        const scalarField& w = fp.weights();
        os << fp.name() << ' ' << fp.size() << nl;
        forAll(fp, i)
            os << dc[i] << ' ' << nodc[i] << ' ' << w[i] << ' ' << fp.magSf()[i] << nl;
    }

    Info << "wrote " << runTime.path()/"ueqn_interface.dat" << endl;
    return 0;
}

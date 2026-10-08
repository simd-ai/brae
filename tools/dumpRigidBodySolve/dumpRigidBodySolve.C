/*---------------------------------------------------------------------------*\
    brae instrument, not an OpenFOAM application.

    OpenFOAM's own RBD::rigidBodyMotion::solve -- the articulated-body forward
    dynamics plus the joint integrator -- driven directly from a joint state and
    a spatial force, with no fluid in the loop.

    An end-to-end interFoam run cannot localise a rigid-body port: the fluid
    force, the integrator and the dynamics all move at once. This instrument
    fixes the force and the state by hand, so the only thing left to disagree is
    the dynamics.

    The model is constructed exactly as rigidBodyMeshMotion does it
    (rigidBodyMeshMotion.C:85-115): coeffDict is optionalSubDict of
    rigidBodyMotionCoeffs (the class's TypeName is "rigidBodyMotion", not its
    class name), and the state comes from <time>/uniform/rigidBodyMotionState
    when that file is there, from coeffDict when it is not.

    Prints, at setprecision(18) and prefixed [brae] so every number is
    greppable: the chain (lambda, joint type, qIndex per body), each body's mass,
    centre of mass and 6x6 spatial inertia, the integrator coefficients, and the
    post-solve q/qDot/qDdot, X0 and v.

    usage: dumpRigidBodySolve -case <dir> -time 0.005 -newTime \
               -q '(a b)' -qDot '(a b)' -qDdot '(a b)' \
               -t 0.01 -deltaT 0.005 \
               -moment '(x y z)' -force '(x y z)' [-body floatingObject]
\*---------------------------------------------------------------------------*/

#include "argList.H"
#include "Time.H"
#include "timeSelector.H"
#include "IOdictionary.H"
#include "uniformDimensionedFields.H"
#include "IOmanip.H"
#include "Function1.H"
#include "rigidBodyMotion.H"
#include "compositeJoint.H"

using namespace Foam;

// A bare space-separated line, not OpenFOAM's "2 ( a b )", so a gate can split
// on whitespace without stripping the list length and the brackets.
static void writeField
(
    const word& tag,
    const scalarField& f
)
{
    Info<< "[brae] " << tag;
    forAll(f, i)
    {
        Info<< ' ' << f[i];
    }
    Info<< endl;
}


// Row by row: the ported chain is checked term by term against these, not
// against a norm of the whole tensor.
static void writeSpatialTensor
(
    const label bodyIndex,
    const spatialTensor& st
)
{
    for (direction i = 0; i < 6; ++i)
    {
        Info<< "[brae] inertia " << bodyIndex
            << " I row " << static_cast<label>(i);
        for (direction j = 0; j < 6; ++j)
        {
            Info<< ' ' << st(i, j);
        }
        Info<< endl;
    }
}


int main(int argc, char *argv[])
{
    argList::addNote
    (
        "OpenFOAM's own rigidBodyMotion::solve for one joint state and one"
        " spatial force"
    );

    argList::addOption("body", "name", "the body the force is applied to");
    argList::addOption("q", "list", "joint position to start from, e.g. '(a b)'");
    argList::addOption("qDot", "list", "joint velocity to start from");
    argList::addOption
    (
        "qDdot",
        "list",
        "joint acceleration to start from -- what the relaxation blends against"
    );
    argList::addOption("t0", "scalar", "the time stored in the state started from");
    argList::addOption("deltaT0", "scalar", "the time-step stored in the state started from");
    argList::addOption("t", "scalar", "the time passed to solve()");
    argList::addOption("deltaT", "scalar", "the time-step passed to solve()");
    argList::addOption("moment", "vector", "the moment on the body, in the global frame");
    argList::addOption("force", "vector", "the force on the body, in the global frame");
    argList::addBoolOption
    (
        "newTime",
        "store the state as the previous time-step before solving,"
        " as rigidBodyMeshMotion does once per time index"
    );

    timeSelector::addOptions();

    #include "setRootCase.H"
    #include "createTime.H"

    // The instant the state dictionary is read from: rigidBodyMeshMotion reads
    // <time>/uniform/rigidBodyMotionState at the mesh's current time.
    const instantList timeDirs = timeSelector::select0(runTime, args);
    if (timeDirs.empty())
    {
        FatalErrorInFunction << "no time directory selected" << exit(FatalError);
    }
    runTime.setTime(timeDirs.last(), timeDirs.size() - 1);

    Info<< setprecision(18);

    IOdictionary dynamicMeshDict
    (
        IOobject
        (
            "dynamicMeshDict",
            runTime.constant(),
            runTime,
            IOobject::MUST_READ,
            IOobject::NO_WRITE,
            IOobject::NO_REGISTER
        )
    );

    // motionSolver::coeffDict() is optionalSubDict(type + "Coeffs")
    // (motionSolver.C:91), and rigidBodyMeshMotion's TypeName is
    // "rigidBodyMotion" (rigidBodyMeshMotion.H:151).
    const dictionary& coeffDict =
        dynamicMeshDict.optionalSubDict("rigidBodyMotionCoeffs");

    // typeHeaderOk is not const, so neither is this
    IOobject stateIO
    (
        "rigidBodyMotionState",
        runTime.timeName(),
        "uniform",
        runTime,
        IOobject::READ_IF_PRESENT,
        IOobject::NO_WRITE,
        IOobject::NO_REGISTER
    );

    const bool hasState = stateIO.typeHeaderOk<IOdictionary>(true);

    Info<< "[brae] stateSource "
        << (hasState ? runTime.timeName()/"uniform"/"rigidBodyMotionState" : "coeffDict")
        << endl;

    // The three-argument constructor, as rigidBodyMeshMotion.C:89-115 builds it.
    RBD::rigidBodyMotion model
    (
        runTime,
        coeffDict,
        hasState
      ? IOdictionary(stateIO)
      : coeffDict
    );

    // The body the spatial force is applied to. With no -body, the single entry
    // of the bodies dictionary -- naming it is only a convenience, never a guess
    // at which of several bodies was meant.
    const dictionary& bodiesDict = coeffDict.subDict("bodies");
    word bodyName;
    if (args.found("body"))
    {
        bodyName = args.get<word>("body");
    }
    else if (bodiesDict.size() == 1)
    {
        bodyName = bodiesDict.first()->keyword();
    }
    else
    {
        FatalErrorInFunction
            << "the case has " << bodiesDict.size() << " bodies: name one with -body"
            << exit(FatalError);
    }

    const label bodyID = model.bodyID(bodyName);
    if (bodyID < 0)
    {
        FatalErrorInFunction
            << "body " << bodyName << " has ID " << bodyID
            << ", so it has been merged into a parent and carries no external"
               " force of its own"
            << exit(FatalError);
    }

    Info<< "[brae] body " << bodyName << " bodyID " << bodyID << endl;
    Info<< "[brae] nDoF " << model.nDoF() << endl;
    Info<< "[brae] nBodies " << model.nBodies() << endl;

    // The chain, as the model built it: the parent index, the joint and where
    // the joint's degrees of freedom sit in q.
    forAll(model.lambda(), i)
    {
        const RBD::joint& j = model.joints()[i];
        const RBD::rigidBodyInertia& bodyI = model.I(i);

        Info<< "[brae] chain " << i
            << " name " << model.name(i)
            << " lambda " << model.lambda()[i]
            << " joint " << j.type()
            << " qIndex " << j.qIndex()
            << " jointDoF " << j.nDoF()
            << endl;

        // A composite joint is ONE entry in the model's joint list with the
        // summed degrees of freedom; its parts are only reachable through the
        // PtrList it also is.
        const auto* cJointPtr = isA<RBD::joints::composite>(j);
        if (cJointPtr)
        {
            const RBD::joints::composite& cJoint = *cJointPtr;
            forAll(cJoint, cj)
            {
                Info<< "[brae] chain " << i
                    << " subJoint " << cj
                    << " type " << cJoint[cj].type()
                    << " qIndex " << cJoint[cj].qIndex()
                    << " jointDoF " << cJoint[cj].nDoF()
                    << endl;
            }
        }

        Info<< "[brae] inertia " << i
            << " mass " << bodyI.m()
            << " centreOfMass " << bodyI.c()
            << endl;
        Info<< "[brae] inertia " << i << " Ic " << bodyI.Ic() << endl;

        writeSpatialTensor(i, spatialTensor(bodyI));
    }

    // The integrator and relaxation coefficients. rigidBodyMotion keeps aRelax_,
    // aDamp_ and solver_ private and the Newmark solver keeps gamma_ and beta_
    // private, so none of them can be read off the constructed objects: these are
    // re-read from the same dictionary by the same rules, and are the values in
    // force only because the rules are copied here.
    const dictionary& solverDict = coeffDict.subDict("solver");
    const word solverType(solverDict.get<word>("type"));
    Info<< "[brae] solverType " << solverType << endl;

    if (solverType == "Newmark")
    {
        // Newmark.C:56-65 -- beta is floored by the gamma-dependent stability limit
        const scalar gamma = solverDict.getOrDefault<scalar>("gamma", 0.5);
        const scalar beta =
            max(0.25*sqr(gamma + 0.5), solverDict.getOrDefault<scalar>("beta", 0.25));

        Info<< "[brae] newmarkGamma " << gamma << endl;
        Info<< "[brae] newmarkBeta " << beta << endl;
    }
    else
    {
        Info<< "[brae] newmarkGamma NOT_NEWMARK" << endl;
        Info<< "[brae] newmarkBeta NOT_NEWMARK" << endl;
    }

    const scalar aDamp = coeffDict.getOrDefault<scalar>("accelerationDamping", 1);
    Info<< "[brae] accelerationDamping " << aDamp << endl;

    autoPtr<Function1<scalar>> aRelax
    (
        Function1<scalar>::NewIfPresent
        (
            "accelerationRelaxation",
            coeffDict,
            word::null,
            &runTime
        )
    );

    autoPtr<Function1<scalar>> ramp
    (
        Function1<scalar>::NewIfPresent("ramp", coeffDict, word::null, &runTime)
    );

    // The state to start from. Without an override the state is the one the
    // constructor read; with one, it is the caller's.
    const bool overridden =
        args.found("q") || args.found("qDot") || args.found("qDdot")
     || args.found("t0") || args.found("deltaT0");

    const auto setState = [&](const word& optName, scalarField& f)
    {
        if (!args.found(optName))
        {
            return;
        }

        const List<scalar> given(args.getList<scalar>(optName));
        if (given.size() != model.nDoF())
        {
            FatalErrorInFunction
                << "-" << optName << " has " << given.size()
                << " values but the model has " << model.nDoF()
                << " degrees of freedom"
                << exit(FatalError);
        }

        f = scalarField(given);
    };

    setState("q", model.state().q());
    setState("qDot", model.state().qDot());
    setState("qDdot", model.state().qDdot());

    if (args.found("t0"))
    {
        model.state().t() = args.get<scalar>("t0");
    }
    if (args.found("deltaT0"))
    {
        model.state().deltaT() = args.get<scalar>("deltaT0");
    }

    writeField("preSolve.q", model.state().q());
    writeField("preSolve.qDot", model.state().qDot());
    writeField("preSolve.qDdot", model.state().qDdot());
    Info<< "[brae] preSolve.t " << model.state().t() << endl;
    Info<< "[brae] preSolve.deltaT " << model.state().deltaT() << endl;

    // The cached body state X0_/v_ is only refreshed by forwardDynamicsCorrection,
    // which the constructor ran on the state it read and solve() runs at its end.
    // In a run it therefore always holds motionState_ when the next solve begins,
    // and the RESTRAINTS read it (Newmark.C:85 -> linearDamper.C:73 reads v, and
    // X0 carries the force out of the body frame). After an override it would
    // hold the constructor's state instead, and a damper would act at that
    // velocity -- so it is refreshed here from the overridden state, which is
    // what a run that had reached this state would hold.
    if (overridden)
    {
        model.forwardDynamicsCorrection(model.state());
    }
    Info<< "[brae] preSolve.X0.E " << model.X0(bodyID).E() << endl;
    Info<< "[brae] preSolve.X0.r " << model.X0(bodyID).r() << endl;
    Info<< "[brae] preSolve.v " << model.v(bodyID, Zero) << endl;

    // rigidBodyMeshMotion::solve:252-256 -- once per time index, before the forces
    if (args.found("newTime"))
    {
        model.newTime();
        Info<< "[brae] newTime 1" << endl;
    }
    else
    {
        Info<< "[brae] newTime 0" << endl;
    }

    if (overridden && !args.found("newTime"))
    {
        WarningInFunction
            << "the state was overridden without -newTime, so the previous state"
               " the integrator reads (motionState0_) is still the one the"
               " constructor read. It is private, so this instrument cannot print"
               " it back." << endl;
    }

    const scalar t = args.getOrDefault<scalar>("t", model.state().t());
    const scalar deltaT = args.getOrDefault<scalar>("deltaT", model.state().deltaT());

    const scalar rampValue = (ramp ? ramp->value(t) : 1.0);
    Info<< "[brae] ramp " << rampValue << endl;
    Info<< "[brae] accelerationRelaxation "
        << (aRelax ? aRelax->value(t) : 1.0) << endl;

    // rigidBodyMeshMotion::solve:260-264 -- g comes off the Time registry, ramped
    IOobject gIO
    (
        "g",
        runTime.constant(),
        runTime,
        IOobject::MUST_READ,
        IOobject::NO_WRITE
    );

    if (gIO.typeHeaderOk<uniformDimensionedVectorField>(true))
    {
        const uniformDimensionedVectorField g(gIO);
        model.g() = rampValue*g.value();
    }
    Info<< "[brae] g " << model.g() << endl;

    Field<spatialVector> fx(model.nBodies(), Zero);
    fx[bodyID] =
        rampValue
       *spatialVector
        (
            args.getOrDefault<vector>("moment", Zero),
            args.getOrDefault<vector>("force", Zero)
        );

    forAll(fx, i)
    {
        Info<< "[brae] fx " << i << ' ' << fx[i] << endl;
    }

    Info<< "[brae] solve.t " << t << endl;
    Info<< "[brae] solve.deltaT " << deltaT << endl;

    model.solve
    (
        t,
        deltaT,
        scalarField(model.nDoF(), Zero),
        fx
    );

    writeField("q", model.state().q());
    writeField("qDot", model.state().qDot());
    writeField("qDdot", model.state().qDdot());
    Info<< "[brae] t " << model.state().t() << endl;
    Info<< "[brae] deltaT " << model.state().deltaT() << endl;

    Info<< "[brae] X0.E " << model.X0(bodyID).E() << endl;
    Info<< "[brae] X0.r " << model.X0(bodyID).r() << endl;
    Info<< "[brae] v " << model.v(bodyID, Zero) << endl;

    for (label i = 0; i < model.nBodies(); ++i)
    {
        Info<< "[brae] state " << i
            << " X0.r " << model.X0(i).r()
            << " v " << model.v(i, Zero)
            << endl;
    }

    Info<< "End\n" << endl;
    return 0;
}

// ************************************************************************* //

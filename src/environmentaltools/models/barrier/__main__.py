"""Command line of the BARRIER workflow: ``python -m environmentaltools.models.barrier <step> barrier.yaml``.

Steps:
    new           create a project: <folder>/barrier.yaml (four blocks, an idealised channel that runs) and inputs/
    base          the base case from the sections 'geometry' and 'simulation' (only what changed)
    prepare       create or update the cases of the list and the overview to review them
    run           run BARRIER in the cases without result (--case NAME for some, --again to repeat)
    postprocess   series at points, figures and comparison of the new results (--again for all)
    all           every step in a row (base only if the project has one of those sections)
"""

import argparse
import sys

from . import load_project, new_project, postprocess, prepare, prepare_base, run


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m environmentaltools.models.barrier", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["new", "base", "prepare", "run", "postprocess", "all"])
    parser.add_argument("project", help="project file (barrier.yaml); for 'new', the folder of the new project")
    parser.add_argument("--case", action="append", help="only this case (can be repeated)")
    parser.add_argument("--again", action="store_true",
                        help="write the base case again (base), repeat finished cases (run) or all of them (postprocess)")
    args = parser.parse_args(argv)

    if args.step == "new":
        try:
            new_project(args.project)
        except FileExistsError as exc:
            sys.exit(str(exc))
        return

    import matplotlib

    matplotlib.use("Agg")
    project = load_project(args.project)
    if args.step == "base" or (args.step == "all" and (project["geometry"] or project["simulation"])):
        prepare_base(project, force=args.again and args.step == "base")
    if args.step in ("prepare", "all"):
        prepare(project)
    if args.step in ("run", "all"):
        try:
            log = run(project, cases=args.case, force=args.again)
        except FileNotFoundError as exc:
            sys.exit(str(exc))
        if len(log) and not log["ok"].all():
            sys.exit("Some case failed: see barrier_console.txt in its results folder")
    if args.step in ("postprocess", "all"):
        postprocess(project, cases=args.case, force=args.again)


if __name__ == "__main__":
    main()

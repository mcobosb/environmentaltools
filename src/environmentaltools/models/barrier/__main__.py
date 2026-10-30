"""Command line of the BARRIER workflow: ``python -m environmentaltools.models.barrier <step> barrier.yaml``.

Steps:
    prepare       create or update the cases of the list and the overview to review them
    run           run BARRIER in the cases without result (--case NAME for some, --again to repeat)
    postprocess   series at points, figures and comparison of the new results (--again for all)
    all           the three steps in a row
"""

import argparse
import sys

from . import load_project, postprocess, prepare, run


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m environmentaltools.models.barrier", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["prepare", "run", "postprocess", "all"])
    parser.add_argument("project", help="project file (barrier.yaml)")
    parser.add_argument("--case", action="append", help="only this case (can be repeated)")
    parser.add_argument("--again", action="store_true", help="repeat finished cases (run) or all of them (postprocess)")
    args = parser.parse_args(argv)

    import matplotlib

    matplotlib.use("Agg")
    project = load_project(args.project)
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

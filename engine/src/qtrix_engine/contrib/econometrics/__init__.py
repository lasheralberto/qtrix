"""The causal-inference library of the bundled econometrics skill (MIT, ``agents/SKILLS/eco-skills``), importable
under a stable name from kernel cells, rebuilds included::

    from qtrix_engine.contrib.econometrics.econometric_algorithm import propensity_score_construction

The modules are not copied: this package's path is the skill's ``lib/`` directory, so updating the skill updates
them. ``econometric_algorithm`` needs linearmodels, matplotlib and scipy.
"""

from pathlib import Path

LIB_DIR = Path(__file__).resolve().parents[2] / "agents" / "SKILLS" / "eco-skills" / "econometrics-skill" / "lib"

__path__ = [str(LIB_DIR)]

from __future__ import annotations
import unittest
import numpy as np
from data_pipeline import load_yaml, prepare_inputs
from digital_twin import DigitalTwin
from poar import Validator, poar_score, security_eligible, select_validators
from consensus import pbft_quorum
from simulation import run_once


class FrameworkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg=load_yaml("config.yaml")

    def test_deterministic_inputs(self):
        a=prepare_inputs(self.cfg,"S8",101)
        b=prepare_inputs(self.cfg,"S8",101)
        self.assertTrue(a.equals(b))

    def test_normalized_bounds(self):
        d=prepare_inputs(self.cfg,"S1",101)
        for c in [x for x in d.columns if x.endswith("_n")]:
            self.assertGreaterEqual(float(d[c].min()),0.0)
            self.assertLessEqual(float(d[c].max()),1.0)

    def test_security_separated_from_poar(self):
        v=Validator(1,False,True,True,1,1,1,1,1,0,0)
        self.assertGreater(poar_score(v,self.cfg["poar"]),0.9)
        self.assertFalse(security_eligible(v,self.cfg["poar"],0))

    def test_stale_report_rejected(self):
        v=Validator(1,True,True,True,0.9,0.9,0.9,0.9,0.9,0.1,0)
        self.assertFalse(security_eligible(v,self.cfg["poar"],10))

    def test_pbft_quorum(self):
        self.assertEqual(pbft_quorum(7),5)
        self.assertEqual(pbft_quorum(4),3)

    def test_end_to_end_finite(self):
        r=run_once(self.cfg,"S8",101,{},"proposed")
        for x in [r.warning_accuracy,r.synchronization_fidelity,r.recovery_latency_s,
                  r.finality_s,r.reachability,r.validator_energy_efficiency,r.cri]:
            self.assertTrue(np.isfinite(x))
        self.assertGreaterEqual(r.cri,0)
        self.assertLessEqual(r.cri,1)

    def test_reproducible_run(self):
        a=run_once(self.cfg,"S6",211,{},"proposed")
        b=run_once(self.cfg,"S6",211,{},"proposed")
        self.assertEqual(a.__dict__,b.__dict__)


if __name__=="__main__":
    unittest.main(verbosity=2)

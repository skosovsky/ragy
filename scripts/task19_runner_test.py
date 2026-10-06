#!/usr/bin/env python3
"""AAA harness tests inject external subprocess failures, skips and timeouts."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
SCRIPTS = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load('task19_verify')
wire = load('task19_wire')


class Harness(unittest.TestCase):
    def test_external_failure_keeps_remaining_modules_and_raw_skip(self):
        # Arrange: an external executable emits genuine go-test event format,
        # a failing module, and cached/pass/skip observations without running Go.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / 'go'
            executable.write_text('''#!/usr/bin/env python3
import json,os,sys
if sys.argv[1] != 'test':
 print('test harness tool')
 sys.exit(0)
assert os.environ['GOWORK'] == 'off'
assert os.environ.get('RAGY_LIVE_GEMINI') != '1'
assert '-race' in sys.argv and '-count=1' in sys.argv
package=os.getcwd()
failed=package.endswith('/adapters/neo4j')
for event in [{'Action':'output','Package':package,'Output':'ok fake (cached)'},
 {'Action':'skip','Package':package,'Test':'TestLiveProvider'},
 {'Action':'fail' if failed else 'pass','Package':package}]:
 print(json.dumps(event))
sys.exit(1 if failed else 0)
''')
            executable.chmod(0o755)
            output = root / 'out'
            env = dict(os.environ, RAGY_LIVE_GEMINI='1')
            # Act.
            completed = subprocess.run([sys.executable, str(SCRIPTS / 'task19_verify.py'),
                                        '--go', str(executable), '--output', str(output)],
                                       env=env, capture_output=True, timeout=30)
            report = json.loads((output / 'matrix.json').read_text())
            # Assert: failure cannot erase any later module, skips or raw events.
            self.assertEqual(completed.returncode, 1)
            self.assertEqual(len(report['modules']), 14)
            self.assertEqual(report['modules'][-1]['status'], 'passed')
            self.assertEqual(report['status'], 'failed')
            self.assertEqual(sum(row['status'] == 'failed' for row in report['modules']), 1)
            self.assertTrue(all(row['cached_packages'] for row in report['modules']))
            self.assertTrue(all(any(e['status'] == 'skipped' for e in row['outcomes']) for row in report['modules']))
            self.assertTrue(all(Path(row['raw']).is_file() for row in report['modules']))

    def test_requested_pdf_skips_or_absence_and_missing_evaluation_fail_gate(self):
        # Arrange: tool readiness succeeds; module processes pass without actual
        # execution (skipped or absent), and go run exits zero with no artifact.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / 'go'
            executable.write_text('''#!/usr/bin/env python3
import json,os,sys
if sys.argv[1] == 'env':
 print('cc')
 sys.exit(0)
if sys.argv[1] != 'test':
 sys.exit(0)
package=os.getcwd()
if package.endswith('/adapters/pdf') and os.environ.get('TASK19_INJECT_SKIP') == '1':
 print(json.dumps({'Action':'skip','Package':package,'Test':'TestActualPDFLayoutParser'}))
print(json.dumps({'Action':'pass','Package':package}))
''')
            executable.chmod(0o755)
            readiness = root / 'python'
            readiness.write_text('#!/bin/sh\nexit 0\n')
            readiness.chmod(0o755)
            for mode in ('skipped', 'absent'):
                with self.subTest(mode=mode):
                    output = root / mode
                    env = dict(os.environ, TASK19_INJECT_SKIP='1' if mode == 'skipped' else '0')
                    # Act.
                    completed = subprocess.run([sys.executable, str(SCRIPTS / 'task19_verify.py'),
                                                '--go', str(executable), '--actual-pdf', str(readiness),
                                                '--evaluation', 'development', '--output', str(output)],
                                               env=env, capture_output=True, timeout=30)
                    report = json.loads((output / 'matrix.json').read_text())
                    checks = {row['name']: row for row in report['checks']}
                    profiles = {row['name']: row for row in report['profiles']}
                    # Assert: successful readiness cannot convert skipped tests
                    # or absent evaluation output to selected-profile success.
                    self.assertEqual(completed.returncode, 1)
                    self.assertTrue(all(row['status'] == 'passed' for row in report['modules']))
                    self.assertEqual(checks['actual-pdf-readiness']['status'], 'passed')
                    self.assertEqual(profiles['actual-pdf-tests']['status'], 'skipped')
                    self.assertEqual(checks['required-profile:actual-pdf']['status'], 'failed')
                    for profile in ('text', 'graph', 'tensor'):
                        self.assertEqual(checks['quality:' + profile + ':dev']['status'], 'passed')
                        self.assertEqual(checks['required-profile:evaluation:' + profile]['status'], 'failed')

    def test_incomplete_evaluation_grid_fails_even_with_valid_json(self):
        # Arrange: an independently authored single-query experiment report has
        # correct identities, denominators and all six graph observations.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus_path, split_path, output = root / 'corpus.json', root / 'split.json', root / 'report.json'
            corpus_path.write_text(json.dumps({'dataset_id': 'test-corpus', 'documents': [dict(id='d1', source_id='s1', revision='r1', scope='public', current=True)]}))
            split_path.write_text(json.dumps({'split': 'dev', 'queries': [{'id': 'q1', 'answerable': True, 'scope': 'public'}]}))
            report = dict(schema='task19-evaluation/v1', dataset_id='test-corpus', split='dev',
                          corpus_digest=runner.hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
                          split_digest=runner.hashlib.sha256(split_path.read_bytes()).hexdigest(),
                          execution_profile='deterministic-local-consumer/library-composition',
                          configuration=dict(runner.EVALUATION_LIMITS, provider='none'),
                          rows=[dict(query='q1', strategy=strategy, repeat=repeat, outcome='complete', scope='public',
                                          scope_violations=0, stale_violations=0, citation_violations=0,
                                          local_usage_known=True, retrieval_calls=1, model_calls=0,
                                          local_encoder_calls=0, local_reranker_calls=0,
                                          full_context_utf8_bytes=0, local_input_units=50, local_output_units=50,
                                          elapsed_nanos=100, dispatch_candidate_counts=[0],
                                          provider_billing_state='no-provider-invoked; monetary-price-unavailable')
                                for strategy in ('baseline', 'graph-expansion') for repeat in range(1, 4)],
                          metrics={strategy: dict(answerable_denominator=3, no_answer_denominator=0,
                                                  policy_known_compliant=True, violations=0, failures=0,
                                                  recall_at_3=0, mrr_at_3=0, ndcg_at_3=0)
                                   for strategy in ('baseline', 'graph-expansion')})
            output.write_text(json.dumps(report))
            self.assertEqual(runner.evaluation_check(output, corpus_path, split_path, 'graph')['status'], 'passed')
            # Adversarial external report files retain all raw rows. Quality
            # loss is a valid study outcome; safety/accounting failures are not.
            for name, mutate in [
                ('scope-and-citation-violations', lambda r: [row.update(scope_violations=1, citation_violations=1) for row in r['rows']]),
                ('invalid-original-despite-zero-counters', lambda r: r['rows'][0].update(candidate_ids=['missing-original'])),
                ('uncited-delivery-despite-zero-counters', lambda r: r['rows'][0].update(delivered=['d1'])),
                ('aggregate-violations', lambda r: [metric.update(violations=6) for metric in r['metrics'].values()]),
                ('overbudget-bytes', lambda r: r['rows'][0].update(local_input_units=262145)),
                ('overbudget-model-calls', lambda r: r['rows'][0].update(model_calls=4)),
                ('overbudget-dispatch-candidates', lambda r: r['rows'][0].update(dispatch_candidate_counts=[13])),
                ('overbudget-candidate-universe', lambda r: r['rows'][0].update(candidate_ids=[str(i) for i in range(37)])),
                ('unknown-local-usage', lambda r: r['rows'][0].update(local_usage_known=False)),
                ('unknown-local-compliance', lambda r: [metric.update(policy_known_compliant=False) for metric in r['metrics'].values()]),
            ]:
                with self.subTest(report=name):
                    adversarial = copy.deepcopy(report)
                    mutate(adversarial)
                    artifact = root / (name + '.json')
                    artifact.write_text(json.dumps(adversarial))
                    original = artifact.read_bytes()
                    result = runner.evaluation_check(artifact, corpus_path, split_path, 'graph')
                    self.assertEqual(result['status'], 'failed')
                    self.assertEqual(artifact.read_bytes(), original)
                    self.assertEqual(len(json.loads(artifact.read_bytes())['rows']), 6)
            # All rankings can lose or have zero recall without changing a
            # mechanical acceptance outcome or inventing provider billing.
            loss = copy.deepcopy(report)
            loss['metrics']['baseline'].update(recall_at_3=0.8, mrr_at_3=0.8, ndcg_at_3=0.8)
            loss['metrics']['graph-expansion'].update(recall_at_3=0, mrr_at_3=0, ndcg_at_3=0)
            output.write_text(json.dumps(loss))
            self.assertEqual(runner.evaluation_check(output, corpus_path, split_path, 'graph')['status'], 'passed')
            report['rows'].pop()
            output.write_text(json.dumps(report))
            # Act.
            check = runner.evaluation_check(output, corpus_path, split_path, 'graph')
            # Assert.
            self.assertEqual(check['status'], 'failed')
            self.assertIn('missing, duplicate or unexpected execution grid rows', check['errors'])

    def test_tensor_requires_baseline_and_maxsim_three_repeat_grid(self):
        # Arrange: an external report uses the frozen tensor command's two
        # profiles and three repeats, with explicit known local accounting.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus_path, split_path, output = root / 'corpus.json', root / 'split.json', root / 'tensor.json'
            corpus_path.write_text(json.dumps({'dataset_id': 'tensor-regression', 'documents': []}))
            split_path.write_text(json.dumps({'split': 'dev', 'queries': [{'id': 'q1', 'answerable': True, 'scope': 'public'}]}))
            report = dict(schema='task19-evaluation/v1', dataset_id='tensor-regression', split='dev',
                          corpus_digest=runner.hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
                          split_digest=runner.hashlib.sha256(split_path.read_bytes()).hexdigest(),
                          execution_profile='deterministic-local-consumer/library-composition',
                          configuration=dict(runner.EVALUATION_LIMITS, provider='none'),
                          rows=[dict(query='q1', strategy=strategy, repeat=repeat, outcome='complete', scope='public',
                                     scope_violations=0, stale_violations=0, citation_violations=0,
                                     local_usage_known=True, retrieval_calls=1, model_calls=0,
                                     local_encoder_calls=0, local_reranker_calls=0,
                                     full_context_utf8_bytes=0, local_input_units=50, local_output_units=50,
                                     elapsed_nanos=100, dispatch_candidate_counts=[0])
                                for strategy in ('baseline', 'tensor-candidate-maxsim') for repeat in range(1, 4)],
                          metrics={strategy: dict(answerable_denominator=3, no_answer_denominator=0,
                                                  policy_known_compliant=True, violations=0, failures=0)
                                   for strategy in ('baseline', 'tensor-candidate-maxsim')})
            output.write_text(json.dumps(report))
            # Act.
            complete = runner.evaluation_check(output, corpus_path, split_path, 'tensor')
            report['rows'] = [row for row in report['rows'] if row['strategy'] != 'baseline']
            del report['metrics']['baseline']
            output.write_text(json.dumps(report))
            missing_baseline = runner.evaluation_check(output, corpus_path, split_path, 'tensor')
            # Assert: complete two-profile observations pass; an absent baseline
            # cannot quietly convert the experiment into a one-profile study.
            self.assertEqual(complete['status'], 'passed')
            self.assertEqual(missing_baseline['status'], 'failed')
            self.assertIn('missing, duplicate or unexpected execution grid rows', missing_baseline['errors'])
            self.assertIn('incomplete strategy metrics', missing_baseline['errors'])

    def test_timeout_is_failed_and_finite(self):
        # Arrange.
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'timeout.txt'
            # Act: kill the whole isolated subprocess group after a finite bound.
            result = runner.run([sys.executable, '-c', 'import time; time.sleep(60)'],
                                runner.ROOT, os.environ, 0.1, output)
            # Assert.
            self.assertEqual(result['status'], 'failed')
            self.assertTrue(result['timeout'])
            self.assertEqual(result['exit_code'], 124)
            self.assertLess(result['elapsed_seconds'], 5)

    def test_independent_audit_detects_fixture_drift(self):
        # Arrange: clone only test sources, with a wrong positive model response.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for c in json.loads(wire.CONTRACT.read_text())['profiles']:
                for name in (c['fixture_source'], c.get('negative_source', c['fixture_source'])):
                    target = root / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text((wire.ROOT / name).read_text())
            target = root / 'adapters/openai/dense/client_test.go'
            target.write_text(target.read_text().replace('"model":"text-embedding-3-small"', '"model":"wrong"'))
            # Act.
            report = wire.verify(root)
            # Assert: no importing Go client decoders to establish correctness.
            self.assertEqual(report['status'], 'failed')
            self.assertEqual(report['outcomes'][0]['status'], 'failed')
            self.assertTrue(all(row['status'] == 'passed' for row in report['outcomes'][1:]))


if __name__ == '__main__':
    unittest.main()

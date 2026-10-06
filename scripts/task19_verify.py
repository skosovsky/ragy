#!/usr/bin/env python3
"""Bounded, continue-on-failure repository verification; standard library only."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
MODULES = ['.', 'adapters/cohere', 'adapters/elasticsearch', 'adapters/gemini',
           'adapters/jina', 'adapters/neo4j', 'adapters/observability/otel',
           'adapters/openai', 'adapters/pdf', 'adapters/pgvector', 'adapters/qdrant',
           'examples/conformance', 'examples/planner', 'examples/resilience']


def run(command, cwd, env, timeout, output):
    start = time.monotonic()
    try:
        with output.open('w') as log:
            process = subprocess.Popen(command, cwd=cwd, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = process.wait(timeout=timeout)
                timed_out = False
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                code, timed_out = 124, True
    except OSError as error:
        output.write_text(str(error) + '\n')
        code, timed_out = 127, False
    return dict(command=command, exit_code=code, timeout=timed_out,
                elapsed_seconds=round(time.monotonic() - start, 6),
                status='passed' if code == 0 else 'failed', raw=str(output))


def events(path):
    terminal, cached, errors = [], [], []
    for line in path.read_text().splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue  # compiler diagnostics may be plain text; exit status still controls outcome
        if not isinstance(item, dict):
            continue
        if item.get('Action') in ('pass', 'fail', 'skip'):
            terminal.append(dict(package=item.get('Package'), test=item.get('Test'),
                                 status={'pass': 'passed', 'fail': 'failed', 'skip': 'skipped'}[item['Action']]))
        if '(cached)' in item.get('Output', ''):
            cached.append(item.get('Package'))
        if item.get('Action') == 'build-fail':
            errors.append(item)
    return dict(outcomes=terminal, cached_packages=sorted(set(cached)), build_failures=errors)


ACTUAL_PDF_TESTS = {
    'TestActualPDFLayoutDurablePublicationReopenAndRetainedRevision',
    'TestActualParserRetainedPageCellImageDocumentResolution',
    'TestActualPDFLayoutParser',
    'TestActualPDFParserPageLimitPreservesPartialCoverage',
    'TestActualPDFParserRejectsBadPDFAndElementLimits',
    'TestActualPDFParserRejectsUnsupportedNativeGeometry',
    'TestActualParserOCRSimulationAndModalityIndexRender',
    'TestActualParserProjectionIndexRetrieveAndScopedResolve',
}
EVALUATION_STRATEGIES = {
    'text': {'baseline', 'hybrid-rerank', 'single-rewrite', 'multi-query', 'decomposition'},
    'graph': {'baseline', 'graph-expansion'},
    'tensor': {'baseline', 'tensor-candidate-maxsim'},
}


def required_pdf_check(readiness, actual_events):
    passed = {event['test'] for event in actual_events if event['status'] == 'passed'}
    missing = sorted(ACTUAL_PDF_TESTS - passed)
    incomplete = any(event['status'] != 'passed' for event in actual_events)
    return dict(name='required-profile:actual-pdf',
                status='passed' if readiness['status'] == 'passed' and not missing and not incomplete else 'failed',
                missing_or_unpassed_tests=missing,
                reason='explicit actual-PDF selection requires all pinned actual tests to execute and pass')


# Frozen local mechanical policy. These are serialized-operation byte units,
# not provider tokens or known billing. Quality gains are deliberately absent.
EVALUATION_LIMITS = {
    'top_k': 3, 'candidate_limit': 12, 'whole_attempt_candidate_universe': 36,
    'retrieval_slots': 6, 'model_slots': 3, 'full_context_utf8_bytes': 1800,
    'deadline_nanos': 5_000_000_000, 'local_input_bytes_cap': 262144,
    'local_output_bytes_cap': 65536, 'repeats': 3,
}


def mechanical_errors(report, corpus, split):
    errors = []
    policy = report['configuration']
    for key, limit in EVALUATION_LIMITS.items():
        if type(policy.get(key)) is not int or policy[key] != limit:
            errors.append('missing or changed frozen local policy: ' + key)
    if policy.get('provider') != 'none':
        errors.append('unexpected provider in local experiment')
    queries = {query['id']: query for query in split['queries']}
    documents = {document['id']: document for document in corpus['documents']}
    for index, row in enumerate(report['rows']):
        prefix = 'row ' + str(index) + ': '
        for field in ('scope_violations', 'stale_violations', 'citation_violations'):
            if type(row.get(field)) is not int or row[field] != 0:
                errors.append(prefix + field)
        if row.get('local_usage_known') is not True:
            errors.append(prefix + 'local usage/compliance unknown')
        caps = {'retrieval_calls': EVALUATION_LIMITS['retrieval_slots'],
                'full_context_utf8_bytes': EVALUATION_LIMITS['full_context_utf8_bytes'],
                'local_input_units': EVALUATION_LIMITS['local_input_bytes_cap'],
                'local_output_units': EVALUATION_LIMITS['local_output_bytes_cap'],
                'elapsed_nanos': EVALUATION_LIMITS['deadline_nanos']}
        for field, cap in caps.items():
            if type(row.get(field)) is not int or not 0 <= row[field] <= cap:
                errors.append(prefix + 'unknown/overbudget ' + field)
        model_fields = ('model_calls', 'local_encoder_calls', 'local_reranker_calls')
        if any(type(row.get(field)) is not int or row[field] < 0 for field in model_fields):
            errors.append(prefix + 'unknown model dispatch accounting')
        elif sum(row[field] for field in model_fields) > EVALUATION_LIMITS['model_slots']:
            errors.append(prefix + 'model dispatch budget exceeded')
        counts = row.get('dispatch_candidate_counts')
        if not isinstance(counts, list) or len(counts) != row.get('retrieval_calls') or any(type(n) is not int or not 0 <= n <= 12 for n in counts):
            errors.append(prefix + 'candidate dispatch bounds/accounting')
        candidates = row.get('candidate_ids') or []
        retrieved, delivered = row.get('retrieved') or [], row.get('delivered') or []
        if len(set(candidates)) > 36:
            errors.append(prefix + 'candidate universe exceeded')
        if len(retrieved) > 3 or len(delivered) > 3:
            errors.append(prefix + 'top-k bound exceeded')
        query = queries[row['query']]
        if row.get('scope') != query['scope']:
            errors.append(prefix + 'query scope mismatch')
        for identity in candidates + retrieved + delivered:
            document = documents.get(identity)
            if not document or not document['current'] or document['scope'] not in ('public', query['scope']):
                errors.append(prefix + 'ineligible/stale original: ' + identity)
        def valid_reference(reference):
            document = documents.get(reference.get('artifact'))
            if not document or not document['current'] or document['scope'] not in ('public', query['scope']):
                return False
            return reference == dict(namespace=corpus['dataset_id'], source=document['source_id'],
                                     revision=document['revision'], transformation='original',
                                     access_fingerprint=document['scope'], artifact=document['id'], representation='text')
        cited = set()
        for locator in row.get('sources') or []:
            reference = locator['reference']
            zero_document_locator = {
                'reference': reference, 'kind': 'document',
                'span': {'start': 0, 'end': 0},
                'page': {'physical_index': 0, 'printed_label': '', 'width_pt': 0, 'height_pt': 0, 'rotation_deg': 0},
                'region': {'left': 0, 'top': 0, 'right': 0, 'bottom': 0},
                'cell': {'table': '', 'element': '', 'row': 0, 'column': 0, 'row_span': 0, 'column_span': 0},
            }
            if not valid_reference(reference) or locator != zero_document_locator or reference['artifact'] not in delivered:
                errors.append(prefix + 'invalid original citation')
            else:
                cited.add(reference['artifact'])
        if set(delivered) != cited:
            errors.append(prefix + 'delivered originals lack citations')
        for reference in row.get('graph_contributors') or []:
            if not valid_reference(reference):
                errors.append(prefix + 'invalid graph contributor')
    for strategy, metric in report['metrics'].items():
        if metric.get('policy_known_compliant') is not True:
            errors.append('unknown/noncompliant local policy: ' + strategy)
        if type(metric.get('violations')) is not int or metric['violations'] != 0:
            errors.append('aggregate mechanical violations: ' + strategy)
        if type(metric.get('failures')) is not int or metric['failures'] != 0:
            errors.append('aggregate execution failures: ' + strategy)
    return errors


def evaluation_check(path, corpus_path, split_path, profile):
    errors = []
    try:
        corpus_bytes, split_bytes = corpus_path.read_bytes(), split_path.read_bytes()
        corpus, split = json.loads(corpus_bytes), json.loads(split_bytes)
        report = json.loads(path.read_bytes())
        strategies = EVALUATION_STRATEGIES[profile]
        queries = {query['id'] for query in split['queries']}
        if not queries or len(queries) != len(split['queries']):
            errors.append('empty or duplicate input queries')
        identities = {'schema': 'task19-evaluation/v1', 'dataset_id': corpus['dataset_id'],
                      'split': split['split'], 'corpus_digest': hashlib.sha256(corpus_bytes).hexdigest(),
                      'split_digest': hashlib.sha256(split_bytes).hexdigest(),
                      'execution_profile': 'deterministic-local-consumer/library-composition'}
        for key, value in identities.items():
            if report.get(key) != value:
                errors.append('report identity mismatch: ' + key)
        if report['configuration']['repeats'] != 3:
            errors.append('incorrect repeat configuration')
        expected = {(query, strategy, repeat) for query in queries for strategy in strategies for repeat in range(1, 4)}
        actual = [(row['query'], row['strategy'], row['repeat']) for row in report['rows']]
        if len(actual) != len(expected) or set(actual) != expected:
            errors.append('missing, duplicate or unexpected execution grid rows')
        if set(report['metrics']) != strategies:
            errors.append('incomplete strategy metrics')
        if any(row['outcome'] not in ('complete', 'failed') for row in report['rows']):
            errors.append('unknown execution outcome')
        if any(row['outcome'] == 'failed' for row in report['rows']):
            errors.append('execution failures retained in report')
        errors.extend(mechanical_errors(report, corpus, split))
        for strategy in strategies:
            metric = report['metrics'][strategy]
            answerable = sum(query['answerable'] for query in split['queries']) * 3
            if metric['answerable_denominator'] != answerable or metric['no_answer_denominator'] != len(queries) * 3 - answerable:
                errors.append('metric denominator mismatch: ' + strategy)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        errors.append('missing or malformed experiment data/report: ' + str(error))
    return dict(name='required-profile:evaluation:' + profile,
                status='failed' if errors else 'passed', errors=errors, artifact=str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'docs/task19/results/final')
    parser.add_argument('--go', default='go')
    parser.add_argument('--timeout', type=int, default=600, help='per subprocess seconds')
    parser.add_argument('--cache-mode', choices=['uncached-tests', 'reuse-tests'], default='uncached-tests')
    parser.add_argument('--actual-pdf', metavar='PYTHON')
    parser.add_argument('--live', action='store_true', help='allow explicit existing live opt-ins; never supplies credentials')
    parser.add_argument('--lint', metavar='EXECUTABLE')
    parser.add_argument('--evaluation', choices=['development', 'holdout'], help='fixed local deterministic consumer experiment; holdout only after strategy freeze')
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error('timeout must be positive')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, GOWORK='off')
    env.setdefault('GOCACHE', '/tmp/ragy-goal-cache')
    env.setdefault('GOPATH', '/tmp/ragy-goal-gopath')
    env.setdefault('GOMODCACHE', '/tmp/ragy-goal-modcache')
    env.setdefault('GOLANGCI_LINT_CACHE', '/tmp/ragy-task19-lint-cache')
    env.setdefault('TMPDIR', '/tmp')
    optins = ('RAGY_LIVE_GEMINI', 'RAGY_LIVE_PROVIDERS', 'RAGY_PROVIDER_SMOKE')
    if not args.live:
        for key in optins:
            env.pop(key, None)
    if args.actual_pdf:
        env['RAGY_PDF_PYTHON'] = args.actual_pdf
    else:
        env.pop('RAGY_PDF_PYTHON', None)
    report = dict(schema='ragy.task19.verification.v1', started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  cache_mode=args.cache_mode, race=True, live_allowed=args.live,
                  live_optins={key: env.get(key) == '1' for key in optins},
                  environment={key: env.get(key) for key in ('GOWORK', 'GOCACHE', 'GOPATH', 'GOMODCACHE', 'TMPDIR', 'GOLANGCI_LINT_CACHE')},
                  python=sys.version, modules=[], checks=[], profiles=[])
    for name, command in [('go-version', [args.go, 'version']), ('compiler', [args.go, 'env', 'GOOS', 'GOARCH', 'GOVERSION', 'CGO_ENABLED', 'CC']),
                          ('revision', ['git', 'rev-parse', 'HEAD']), ('working-tree', ['git', 'status', '--short'])]:
        report['checks'].append(dict(name=name, **run(command, ROOT, env, args.timeout, output / (name + '.txt'))))
    compiler_lines = (output / 'compiler.txt').read_text().splitlines()
    if compiler_lines:
        compiler_argv = shlex.split(compiler_lines[-1])
        if compiler_argv:
            report['checks'].append(dict(name='compiler-version', **run(compiler_argv + ['--version'], ROOT, env, args.timeout, output / 'compiler-version.txt')))
    discovered = sorted(str(p.parent.relative_to(ROOT)) for p in ROOT.rglob('go.mod') if not any(part.startswith('.') for part in p.relative_to(ROOT).parts))
    if sorted(MODULES) != discovered:
        report['checks'].append(dict(name='module-inventory', status='failed', expected=sorted(MODULES), actual=discovered))
    pdf = dict(name='actual-pdf', status='skipped', reason='--actual-pdf not selected')
    if args.actual_pdf:
        pdf = dict(name='actual-pdf-readiness', **run([args.actual_pdf, '-c', 'import pdfplumber,pypdf; print(pdfplumber.__version__,pypdf.__version__)'], ROOT, env, args.timeout, output / 'pdf-readiness.txt'))
        report['checks'].append(pdf)
    report['profiles'].append(pdf)
    report['checks'].append(dict(name='independent-provider-wire-fixtures', **run([sys.executable, str(ROOT / 'scripts/task19_wire.py')], ROOT, env, args.timeout, output / 'provider-wire.json')))
    if args.lint:
        report['checks'].append(dict(name='linter-version', **run([args.lint, 'version'], ROOT, env, args.timeout, output / 'lint-version.txt')))
    for module in MODULES:
        name = 'root' if module == '.' else module.replace('/', '-')
        command = [args.go, 'test', '-json', '-race', '-timeout=' + str(max(1, args.timeout - 10)) + 's']
        if args.cache_mode == 'uncached-tests':
            command.append('-count=1')
        command.append('./...')
        result = dict(module=module, **run(command, ROOT / module, env, args.timeout, output / (name + '.jsonl')))
        result.update(events(Path(result['raw'])))
        if not any(item['test'] is None for item in result['outcomes']):
            result['missing_package_outcomes'] = True
            result['status'] = 'failed'
        if any(item['status'] == 'failed' for item in result['outcomes']) or result['build_failures']:
            result['status'] = 'failed'
        report['modules'].append(result)
        if args.lint:
            report['checks'].append(dict(name='lint:' + module, **run([args.lint, 'run', '--allow-serial-runners', '--timeout=' + str(args.timeout - 1) + 's', './...'], ROOT / module, env, args.timeout, output / (name + '-lint.txt'))))
        # Persist after every module, including failures and test-level skips.
        (output / 'matrix.json').write_text(json.dumps(report, indent=2) + '\n')
        print(module + ': ' + result['status'], flush=True)
    for module in ('adapters/gemini', 'adapters/jina', 'adapters/openai', 'adapters/cohere'):
        live_events = [item for row in report['modules'] if row['module'] == module for item in row['outcomes'] if item.get('test') and (item['test'].startswith('TestLive') or item['test'] == 'TestProviderSmoke')]
        report['profiles'].append(dict(name='live:' + module, status='failed' if any(e['status'] == 'failed' for e in live_events) else 'passed' if live_events and all(e['status'] == 'passed' for e in live_events) else 'skipped', outcomes=live_events, reason='actual opt-in test events; fixtures do not count'))
    report['profiles'].append(dict(name='external-database-services', status='skipped', reason='no host provisioned live DB runner; adapter transport mocks are local evidence only'))
    if args.actual_pdf:
        actual_events = [item for row in report['modules'] if row['module'] == 'adapters/pdf' for item in row['outcomes'] if item.get('test') and item['test'].startswith('TestActual')]
        report['checks'].append(required_pdf_check(pdf, actual_events))
        report['profiles'].append(dict(name='actual-pdf-tests', status='failed' if pdf['status'] == 'failed' or any(e['status'] == 'failed' for e in actual_events) else 'passed' if actual_events and all(e['status'] == 'passed' for e in actual_events) else 'skipped', outcomes=actual_events))
    if args.evaluation:
        split = 'dev' if args.evaluation == 'development' else 'holdout'
        for profile, command_path in [('text', 'recipe_comparison'), ('graph', 'graph_comparison'), ('tensor', 'tensor_comparison')]:
            artifact = output / ('eval-' + profile + '-' + split + '.json')
            artifact.unlink(missing_ok=True)  # a selected run must not accept a stale successful artifact
            command = [args.go, 'run', './' + command_path, '-task19-corpus', 'datasets/task19/corpus.json',
                       '-task19-split', 'datasets/task19/' + split + '.json',
                       '-task19-output', str(artifact)]
            report['checks'].append(dict(name='quality:' + profile + ':' + split, **run(command, ROOT / 'examples/conformance', env, args.timeout, output / ('eval-' + profile + '-' + split + '.txt'))))
            report['checks'].append(evaluation_check(artifact, ROOT / 'examples/conformance/datasets/task19/corpus.json', ROOT / ('examples/conformance/datasets/task19/' + split + '.json'), profile))
    else:
        report['profiles'].append(dict(name='retrieval-quality', status='skipped', reason='--evaluation not selected; correctness is separate from quality'))
    report['status'] = 'failed' if any(row['status'] == 'failed' for row in report['modules'] + report['checks'] + report['profiles']) else 'passed'
    report['finished_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    (output / 'matrix.json').write_text(json.dumps(report, indent=2) + '\n')
    print(str(output / 'matrix.json'))
    return 1 if report['status'] == 'failed' else 0

if __name__ == '__main__':
    sys.exit(main())

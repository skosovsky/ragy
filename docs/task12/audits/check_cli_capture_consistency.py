"""Independent actual receipt check; no model calls or capture mutation."""
import hashlib
import json
from pathlib import Path

results = Path(__file__).resolve().parents[1] / 'results'
for name in ('text-cli-live-capture.json', 'graph-cli-live-capture.json', 'graph-cli-v2-capture.json', 'text-cli-v2-capture.json'):
    data = (results / name).read_bytes()
    capture = json.loads(data)
    receipts = []
    for sample in capture['samples']:
        calls = sample.get('cli_receipts', [])
        receipts.extend(calls)
        assert sample['model_calls'] == len(calls)
        if calls:
            assert all(call['usage'] is not None for call in calls)
            assert sample['input_tokens'] == sum(call['usage']['input_tokens'] for call in calls)
            assert sample['output_tokens'] == sum(call['usage']['output_tokens'] for call in calls)
            assert sample['provider_price_state'] == 'unavailable' and not sample['usage_known']
        for call in calls:
            assert call['exit_code'] == 0 and not call['timed_out'] and not call['tool_activity']
            settlements = [event['usage'] for event in call['events'] if event.get('type') == 'turn.completed']
            assert len(settlements) == 1
            assert call['usage']['input_tokens'] == settlements[0]['input_tokens']
            assert call['usage']['output_tokens'] == settlements[0]['output_tokens']
            assert len([event for event in call['events'] if event.get('type') == 'item.completed'
                        and event.get('item', {}).get('type') == 'agent_message']) == 1
    for extraction in capture.get('preparation', {}).get('extractions', []):
        calls = extraction.get('cli_receipts', [])
        assert len(calls) == 1
        receipts.extend(calls)
        assert extraction['model_calls'] == 1
        call = calls[0]
        assert call['usage'] is not None and call['success'] and not call['tool_activity']
        assert extraction['input_tokens'] == call['usage']['input_tokens']
        assert extraction['output_tokens'] == call['usage']['output_tokens']
    print(name, 'PASS', len(capture['samples']), 'rows', len(receipts), 'receipts',
          'sha256', hashlib.sha256(data).hexdigest())

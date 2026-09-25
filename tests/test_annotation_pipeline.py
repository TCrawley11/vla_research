"""Regression tests for motion truth, evidence scope, cache recovery and both stages."""
from pathlib import Path
import copy
import io
import json
import urllib.error

import numpy as np
import pytest

from carla_data_pipeline import annotate as local
from carla_data_pipeline import annotate as common
from carla_data_pipeline import benchmark as hosted
from scripts import prepare_annotation_eval as preparation
from scripts import run_local_annotation_eval as evaluation

ROOT = Path(__file__).resolve().parents[1]


def example():
    pool = common.load_examples(common.ExamplesConfig(path=ROOT / "configs/annotation/examples.yaml"), common.LimitsConfig())
    return pool[0]


def payload():
    gt = example().ground_truth.model_copy(update={"sample_id": "eval_000001"})
    return common.SamplePayload(gt, [], {cam: f"image-{cam}".encode() for cam in common.CAMERAS})


class RecordingClient:
    model_metadata = {"id": "quantized-27b", "max_model_len": 32768}
    def __init__(self): self.calls = []
    def call(self, model, system, content, schema, validate, gen, **kwargs):
        self.calls.append((system, kwargs, gen))
        ex = example()
        if schema["name"] == "question_set":
            obj = {"questions": [{"type": q.type, "question": q.question} for q in ex.qa_pairs]}
        else:
            obj = ex.output()
        assert validate(obj) == []
        return obj, {"attempts": 1, "usage": {"prompt_tokens": 100, "completion_tokens": 100,
                                           "reasoning_tokens": 0, "cost_usd": 0}, "provider": "fixture"}


def test_real_braking_regression():
    # run43_000285 falls 28.5%; the previous 30% threshold called this steady.
    speeds = np.array([8.0376, 7.5756, 7.0811, 6.6153, 6.1685, 5.7504])
    points = np.column_stack([np.cumsum(speeds * .5), np.zeros(6)])
    assert common.speed_profile(points, .5) == "slowing from about 8.0 m/s to about 5.8 m/s"


def test_creeping_and_low_net_displacement_do_not_mean_stationary():
    creeping = np.column_stack([np.arange(1, 7) * .1, np.zeros(6)])
    assert "creeping" in common.summarize_trajectory(creeping, "STOPPING", 3)
    looping = np.array([[2, 0], [3, 1], [2, 3], [1, 3], [.5, 2], [0, 1]])
    assert "stationary throughout" not in common.summarize_trajectory(looping, "STOPPING", 3)
    stopped = np.zeros((6, 2))
    assert "stationary throughout" in common.summarize_trajectory(stopped, "STRAIGHT", 3)


def test_stop_time_uses_last_moving_segment():
    points = np.array([[1, 0], [1, 0], [2, 0], [3, 0], [3, 0], [3, 0]])
    assert "within about 2 s" in common.speed_profile(points, .5)


def test_stop_go_stop_is_not_an_intervening_stop():
    # start stopped, move, end stopped: not "moving with an intervening stop"
    points = np.array([[0, 0], [1, 0], [2, 0], [3, 0], [3, 0], [3, 0]])
    profile = common.speed_profile(points, .5)
    assert "intervening stop" not in profile
    assert "pulled away from standstill" in profile
    assert "decelerating to a full stop" in profile
    pause_then_go = np.array([[1, 0], [2, 0], [2, 0], [2, 0], [3, 0], [5, 0]])
    assert "intervening stop" in common.speed_profile(pause_then_go, .5)


def test_waypoint_grid_must_match_time_horizon():
    gt = example().ground_truth.model_dump()
    gt['future_waypoints_ego_frame'] = [[0, 0]]
    with pytest.raises(ValueError, match="cover horizon"):
        common.GroundTruth(**gt)


@pytest.mark.parametrize('caption,answer', [
    ('The weather is foggy.', 'The road is clear of vehicles.'),
    ('FRONT shows a pedestrian on the sidewalk.', 'No pedestrians are visible in BACK.'),
    ('No pedestrians are visible.', 'Watch for pedestrians before moving.'),
    ('FRONT shows a pedestrian crossing.', 'No pedestrians are visible in FRONT.'),
])
def test_valid_claims_do_not_trigger_contradictions(caption, answer):
    assert common._contradiction_errors({'caption_short': caption}, [{'id': 'q01', 'answer': answer}]) == []


def test_same_camera_conflict_is_only_a_review_flag():
    obj = {'caption_short': 'FRONT shows a pedestrian on the sidewalk.'}
    answers = [{'id': 'q01', 'answer': 'No pedestrians are visible in FRONT.'}]
    assert any('pedestrians' in issue for issue in common._contradiction_errors(obj, answers))


def test_scene_speed_question_is_not_a_lookup():
    assert not common._is_gt_lookup('How does current speed relate to the red signal visible in FRONT?', 'behaviour')
    assert common._is_gt_lookup('What was the dominant action over the past 1.5 seconds?', 'behaviour')


def test_ego_does_not_count_as_another_agent():
    assert any('dynamic agent' in issue for issue in common._caption_content_errors(
        {'caption_detailed': 'FRONT shows the ego vehicle on a road.'}))


def test_natural_camera_names_and_agent_absence_are_recognized():
    assert common._camera_scope('front right and BACK-LEFT') == {
        'FRONT_RIGHT', 'BACK_LEFT'}
    assert common._caption_content_errors({
        'caption_detailed': 'The front right view is clear. '
                            'No other dynamic agents are visible there.'
    }) == []


def test_causal_planning_question_is_a_review_flag_only():
    questions = [{
        'id': 'q01', 'type': 'planning',
        'question': 'What visible feature justifies accelerating?',
    }]
    assert common._causal_question_errors(questions)
    assert common.validate_questions(
        {'questions': [{'type': 'planning',
                        'question': questions[0]['question']}]},
        dict(perception=0, prediction=0, planning=1, behaviour=0),
    ) == []


def test_numeric_checks_respect_current_vs_future_scope():
    gt = payload().gt
    obj = {'answers': [{'id': 'q01', 'answer': '99.0 m/s.'}]}
    questions = [{'id': 'q01', 'type': 'behaviour', 'question': 'What is the current speed?'}]
    assert common.numeric_errors(obj, questions, gt)
    questions[0]['type'] = 'prediction'
    assert common.numeric_errors(obj, questions, gt) == []
    questions[0]['type'] = 'behaviour'
    obj['answers'][0]['answer'] = f'{gt.v:.2f} m/s.'
    assert common.numeric_errors(obj, questions, gt) == []
    questions[0]['question'] = 'How does the current speed relate to the visible signal?'
    obj['answers'][0]['answer'] = 'The current speed is about 7 m/s near the signal.'
    assert common.numeric_errors(obj, questions, gt) == []


@pytest.mark.parametrize('bad', [None, 3, [], {'question': 3}])
def test_bad_questions_return_errors_instead_of_crashing(bad):
    obj = {'questions': [{'type': 'perception', 'question': bad}]}
    assert common.validate_questions(obj, dict(perception=1, prediction=0, planning=0, behaviour=0))


def test_schema_fallback_still_rejects_extra_fields():
    ex = example()
    obj = ex.output()
    obj['made_up'] = True
    assert common.validate_answers(obj, [q['id'] for q in ex.questions()], common.LimitsConfig())


@pytest.mark.parametrize('bad_id', [[], {}, None, 123])
def test_bad_answer_ids_return_errors_with_numeric_checks(bad_id):
    ex = example()
    obj = ex.output()
    obj['answers'][0]['id'] = bad_id
    assert common.validate_answers(obj, [q['id'] for q in ex.questions()], common.LimitsConfig(),
                                   ex.questions(), ex.ground_truth)


@pytest.mark.parametrize('kind', ['local', 'hosted'])
def test_both_stages_use_examples_and_recover_caches(tmp_path, kind):
    client = RecordingClient()
    p = payload()
    if kind == 'local':
        cfg = local.load_config(ROOT / 'configs/annotation/local.yaml')
        cfg.out_dir = tmp_path
        cfg.examples.path = ROOT / cfg.examples.path
        worker = local.Annotator(cfg, client)
        def run(): worker.process_payload(p, tmp_path / 'frames')
    else:
        cfg = hosted.load_config(ROOT / 'configs/annotation/benchmark.yaml')
        cfg.out_dir = tmp_path
        cfg.examples = common.ExamplesConfig(path=ROOT / 'configs/annotation/examples.yaml')
        worker = hosted.Benchmark(cfg, client)
        # Exercise the hosted runner against the same payload without HDF5 I/O.
        def run():
            from unittest.mock import patch
            with patch.object(hosted.ann, 'load_sample', return_value=p):
                worker.run(None, [1], [cfg.models[0]])
    (tmp_path / 'frames').mkdir()
    run()
    assert len(client.calls) == 2
    assert all('Examples of finished annotations' in call[0] for call in client.calls)
    run()
    assert len(client.calls) == 2  # both caches validated and reused
    # A partial question write is regenerated instead of aborting the batch.
    worker.question_path(p.gt.sample_id).write_text('{')
    run()
    assert len(client.calls) == 3
    # Input changes invalidate questions and answers, even with the same sample id.
    p.frames['BACK'] = b'changed pixels'
    run()
    assert len(client.calls) == 5
    # Generation/config changes cannot masquerade as the previous experiment.
    cfg.generation.temperature = .7
    run()
    assert len(client.calls) == 7
    result_path = worker.result_path(p.gt.sample_id) if kind == 'local' else worker.result_path(p.gt.sample_id, cfg.models[0])
    result = common.read_json(result_path)
    del result['annotation']
    common.atomic_json(result_path, result)
    run()
    assert len(client.calls) == 8
    result = common.read_json(result_path)
    assert result['schema_version'] == 2
    assert result['prompt_id'] != common.ANNOTATOR_PROMPT_ID
    assert result['examples']['scenes']
    questions = common.read_json(worker.question_path(p.gt.sample_id))
    assert questions['question_prompt_id'] != common.QUESTION_WRITER_PROMPT_ID
    assert questions['examples']['scenes']
    assert 'linear_velocity_target' not in result['annotation']['action']
    assert len(result['annotation']['qa_pairs']) == 18
    assert len({q['purpose'] for q in result['annotation']['qa_pairs']}) == 18


def test_hosted_disabled_examples_are_absent_from_prompts_and_provenance(tmp_path):
    cfg = hosted.load_config(ROOT / 'configs/annotation/benchmark.yaml')
    cfg.out_dir = tmp_path
    cfg.examples = common.ExamplesConfig(
        path=ROOT / 'configs/annotation/examples.yaml',
        questions=False,
        answers=False,
    )
    client = RecordingClient()
    worker = hosted.Benchmark(cfg, client)
    p = payload()
    from unittest.mock import patch
    with patch.object(hosted.ann, 'load_sample', return_value=p):
        worker.run(None, [1], [cfg.models[0]])
    assert all('Examples of finished annotations' not in call[0] for call in client.calls)
    questions = common.read_json(worker.question_path(p.gt.sample_id))
    result = common.read_json(worker.result_path(p.gt.sample_id, cfg.models[0]))
    assert questions['examples'] is None
    assert result['examples'] is None


def test_examples_are_real_full_sets_and_exclude_source_run():
    pool = common.load_examples(common.ExamplesConfig(path=ROOT / 'configs/annotation/examples.yaml'), common.LimitsConfig())
    assert len(pool) == 12
    for ex in pool:
        counts = {t: sum(q.type == t for q in ex.qa_pairs) for t in common.QA_TYPES}
        assert counts == common.QaCounts().as_dict()
        assert len(ex.source_revision) == 40
        assert all(len(q.question.split()) <= 22 for q in ex.qa_pairs)
    selected = common.select_examples(pool, 2, 'run43_000405')
    assert all(ex.source_run != 'run43' for ex in selected)
    assert common.select_examples(pool, 2, 'run43_000405') == selected


def test_cli_overrides_are_validated():
    cfg = local.AnnotateConfig()
    for args in [['--limit', '0'], ['--h5', 'fake.h5', '--indices', '0,0']]:
        with pytest.raises(ValueError):
            local.apply_cli_overrides(cfg, local.build_parser().parse_args(args))


def test_client_does_not_disguise_context_error_as_schema_error(monkeypatch):
    client = local.VllmClient(local.InferenceConfig())
    calls = []
    def fail(body):
        calls.append(body)
        raise urllib.error.HTTPError('http://local', 400, 'bad request', {}, io.BytesIO(b'maximum context length exceeded'))
    monkeypatch.setattr(client, '_post', fail)
    with pytest.raises(RuntimeError, match='context length'):
        client.call('model', 'system', [], {}, lambda obj: [], common.GenerationConfig())
    assert len(calls) == 1


def test_shared_contract_is_actually_shared():
    assert local.load_sample is hosted.ann.load_sample is common.load_sample
    assert local.validate_answers is hosted.ann.validate_answers is common.validate_answers
    assert local.speed_profile is hosted.ann.speed_profile is common.speed_profile


def test_local_model_identity_survives_server_restart(monkeypatch):
    created = [1]
    def response(*args, **kwargs):
        return io.BytesIO(json.dumps({'data': [{'id': 'quantized-27b', 'created': created[0],
                                              'owned_by': 'llamacpp', 'meta': {'n_ctx': 16384}}]}).encode())
    monkeypatch.setattr(local.urllib.request, 'urlopen', response)
    client = local.VllmClient(local.InferenceConfig())
    client.ping('auto')
    original = client.model_metadata
    created[0] = 2
    client.ping('auto')
    assert client.model_metadata == original
    assert client.context_length == 16384
    assert client.provider == 'llamacpp'


@pytest.mark.parametrize('field,value', [('question_set', []), ('meta', None), ('annotation', [])])
def test_malformed_result_cache_is_regenerated(tmp_path, field, value):
    cfg = local.load_config(ROOT / 'configs/annotation/local.yaml')
    cfg.out_dir = tmp_path
    worker = local.Annotator(cfg, RecordingClient())
    frames = tmp_path / 'frames'
    frames.mkdir()
    p = payload()
    worker.process_payload(p, frames)
    path = worker.result_path(p.gt.sample_id)
    result = common.read_json(path)
    result[field] = value
    common.atomic_json(path, result)
    worker.process_payload(p, frames)
    assert len(worker.client.calls) == 3


def test_dataset_revision_is_used_for_download_and_listing(monkeypatch):
    from unittest.mock import Mock
    api = Mock()
    api.list_repo_files.return_value = ['runs/run01.h5']
    assert local.list_run_paths(api, 'repo', 'pinned') == ['runs/run01.h5']
    api.list_repo_files.assert_called_once_with('repo', repo_type='dataset', revision='pinned')
    download = Mock(return_value='/tmp/run01.h5')
    monkeypatch.setattr(local, 'hf_hub_download', download)
    local.local_h5('repo', 'runs/run01.h5', None, 'pinned')
    download.assert_called_once_with('repo', 'runs/run01.h5', repo_type='dataset', revision='pinned')


def test_preparation_cache_validates_every_frozen_frame(tmp_path):
    p = payload()
    sample_dir = tmp_path / 'eval' / p.gt.sample_id
    sample_dir.mkdir(parents=True)
    for camera, data in p.frames.items():
        (sample_dir / f'{camera}.jpg').write_bytes(data)
    metadata = {
        'source': {'repo_id': 'repo', 'revision': 'pinned', 'run': 'eval'},
        'ground_truth': p.gt.record(),
        'input_id': common.input_id(p),
        'frames': {camera: f'{camera}.jpg' for camera in common.CAMERAS},
    }
    common.atomic_json(sample_dir / 'sample.json', metadata)
    entry = {
        'sample_id': p.gt.sample_id,
        'run': 'eval',
        'split': 'eval',
        'path': f'eval/{p.gt.sample_id}/sample.json',
        'input_id': common.input_id(p),
    }
    cached = {'revision': 'pinned', 'width': 800, 'count': 1, 'samples': [entry]}
    assert preparation.cached_run_samples(
        tmp_path, cached, 'repo', 'pinned', 'eval', 'eval', 1, 800
    ) == [entry]
    (sample_dir / 'BACK.jpg').unlink()
    assert preparation.cached_run_samples(
        tmp_path, cached, 'repo', 'pinned', 'eval', 'eval', 1, 800
    ) is None


def test_evaluation_variants_resume_and_preserve_report(tmp_path, monkeypatch):
    p = payload()
    class EvalClient(RecordingClient):
        def ping(self, model):
            self.served_model = self.model_metadata['id']
    client = EvalClient()
    monkeypatch.setattr(evaluation, 'VllmClient', lambda cfg: client)
    monkeypatch.setattr(evaluation, 'load_payload', lambda path: p)
    manifest = {'repo_id': 'fixture', 'revision': 'pinned', 'image_width': 800,
                'samples': [{'sample_id': p.gt.sample_id, 'run': 'eval', 'map': 'Town01',
                             'split': 'eval', 'path': 'sample.json', 'input_id': common.input_id(p)}]}
    manifest_path = tmp_path / 'manifest.json'
    common.atomic_json(manifest_path, manifest)
    cfg = local.load_config(ROOT / 'configs/annotation/local.yaml')
    out = tmp_path / 'results'
    baseline = evaluation.run_evaluation(cfg, manifest_path, out, ['baseline'])
    assert baseline['variants']['baseline']['status'] == 'complete'
    assert baseline['variants']['baseline']['config']['examples'] is None
    assert len(client.calls) == 2
    assert all('Examples of finished annotations' not in call[0] for call in client.calls)
    report = evaluation.run_evaluation(cfg, manifest_path, out, ['examples'])
    assert set(report['variants']) == {'baseline', 'examples'}
    assert report['variants']['examples']['config']['examples'] is not None
    assert all(v['completed'] == 1 for v in report['variants'].values())
    assert len(client.calls) == 4
    assert all('Examples of finished annotations' in call[0] for call in client.calls[2:])
    evaluation.run_evaluation(cfg, manifest_path, out, ['baseline', 'examples'])
    assert len(client.calls) == 4
    assert (out / 'examples' / 'inspection.html').is_file()
    manifest['samples'][0]['input_id'] = 'corrupted'
    common.atomic_json(manifest_path, manifest)
    with pytest.raises(ValueError, match='identity mismatch'):
        evaluation.run_evaluation(cfg, manifest_path, out, ['baseline'])
    assert len(client.calls) == 4


def test_evaluation_rejects_source_run_leakage_before_inference(tmp_path, monkeypatch):
    manifest = {'samples': [{'run': 'run01', 'split': 'eval'}, {'run': 'run01', 'split': 'examples'}]}
    path = tmp_path / 'manifest.json'
    common.atomic_json(path, manifest)
    def unexpected_client(cfg):
        pytest.fail('inference must not start with overlapping runs')
    monkeypatch.setattr(evaluation, 'VllmClient', unexpected_client)
    with pytest.raises(ValueError, match='overlap'):
        evaluation.run_evaluation(local.AnnotateConfig(), path, tmp_path / 'results', ['baseline'])


@pytest.mark.parametrize('failed_stage', ['questions', 'answers'])
def test_local_stage_failure_preserves_recovery(tmp_path, failed_stage):
    cfg = local.load_config(ROOT / 'configs/annotation/local.yaml')
    cfg.out_dir = tmp_path
    client = RecordingClient()
    worker = local.Annotator(cfg, client)
    p = payload()
    method = 'write_question_set' if failed_stage == 'questions' else 'annotate'
    from unittest.mock import patch
    with patch.object(worker, method, side_effect=RuntimeError('test failure')):
        worker.process_payload(p, tmp_path / 'frames')
    assert worker.failures == [(p.gt.sample_id, 'test failure')]
    assert not worker.result_path(p.gt.sample_id).exists()
    assert worker.question_path(p.gt.sample_id).exists() == (failed_stage == 'answers')
    worker.process_payload(p, tmp_path / 'frames')
    qs = worker.load_question_set(worker.question_path(p.gt.sample_id), p)
    assert worker.result_is_current(worker.result_path(p.gt.sample_id), qs, p)

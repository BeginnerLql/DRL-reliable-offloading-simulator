"""Independent v2 train/evaluate entry; old tools/run_masked_pair_ppo.py is unchanged."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import torch
from agents.context_masked_pair_ppo_agent import ContextMaskedPairPPOAgent
from config.configuration import parameters as params
from Project_main import build_pair_correlations
from diagnostics.run_masked_pair_ppo import train_one, evaluate_one, summary
from tools.paired_ppo_experiment import _agent_kwargs, generate_seed_plan, sha256_file


def checkpoint_environment():
    """Conservative snapshot: reject configuration drift instead of guessing."""
    config = {key: value for key, value in vars(params).items()
              if not key.startswith('_') and not callable(value)}
    return {
        'schema_version': 1,
        'configuration': json.loads(json.dumps(config, sort_keys=True)),
        'data_sha256': {p: sha256_file(ROOT/'data'/p)
                        for p in ['server_info.xlsx', 'task_parameters.xlsx']},
    }


def validate_checkpoint_environment(metadata, current):
    saved = metadata.get('checkpoint_environment')
    if saved is None:
        raise ValueError('Checkpoint lacks an environment snapshot; legacy checkpoint '
                         'cannot be verified. Keep the original artifacts and use its '
                         'original code/environment, or create a new checkpoint.')
    for key in ('schema_version', 'configuration', 'data_sha256'):
        if saved.get(key) != current[key]:
            raise ValueError(f'Checkpoint environment mismatch: {key}')


def load_verified_actor(agent, state):
    # load_state_dict would silently overwrite these environment-derived buffers.
    expected = agent.policy_net.state_dict()
    for key in ('pair_indices', 'pair_correlations'):
        if key not in state or not torch.equal(state[key], expected[key]):
            raise ValueError(f'Checkpoint environment mismatch: actor {key}')
    agent.policy_net.load_state_dict(state)
    agent.policy_old.load_state_dict(state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--train-episodes', type=int, default=3)
    parser.add_argument('--eval-episodes', type=int, default=3)
    parser.add_argument('--checkpoint-dir', type=Path)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--actor-mode', choices=['pair_context', 'pair_scoring'], default='pair_context')
    parser.add_argument('--credit-mode', choices=['event_interval', 'origin_task'], default='event_interval')
    parser.add_argument('--gradient-clipping', choices=['independent', 'joint'], default='independent')
    parser.add_argument('--deployment', choices=['stochastic', 'greedy'], default='stochastic')
    args = parser.parse_args()
    if args.train_episodes < 1 or args.eval_episodes < 1:
        parser.error('Episode counts must be positive')
    out = args.output_dir
    if out.exists() and any(out.iterdir()):
        parser.error('Output directory must be empty (protect prior experiments)')
    torch.set_num_threads(1)
    options = dict(actor_mode=args.actor_mode, credit_mode=args.credit_mode,
                   gradient_clipping=args.gradient_clipping)
    trial = {k: int(v) for k, v in generate_seed_plan(args.seed, 1).iloc[0].items()}
    environment = checkpoint_environment()
    if args.checkpoint_dir:
        meta = json.loads((args.checkpoint_dir/'metadata.json').read_text())
        if meta['agent_name'] != ContextMaskedPairPPOAgent.agent_name:
            parser.error('Checkpoint is not a versioned v2 agent')
        validate_checkpoint_environment(meta, environment)
        options, trial = meta['options'], meta['seed_row']
    pairs, rho = build_pair_correlations()
    kwargs = _agent_kwargs('pair_scoring', np.asarray(rho), trial['PPO_Minibatch_Seed'])
    kwargs.update(options)
    torch.manual_seed(trial['Torch_Init_Seed'])
    agent = ContextMaskedPairPPOAgent(**kwargs)
    if args.checkpoint_dir:
        load_verified_actor(agent, torch.load(args.checkpoint_dir/'actor.pt',
                                             weights_only=True, map_location='cpu'))
        agent.value_net.load_state_dict(torch.load(args.checkpoint_dir/'critic.pt', weights_only=True, map_location='cpu'))
    out.mkdir(parents=True, exist_ok=True)
    if not args.checkpoint_dir:
        train_one(agent, 'context_masked', trial, args.train_episodes, 200, out)
        torch.save(agent.policy_net.state_dict(), out/'actor.pt')
        torch.save(agent.value_net.state_dict(), out/'critic.pt')
        pd.DataFrame(agent.credit_diagnostics).to_csv(out/'credit_checks.csv', index=False)
        pd.DataFrame(agent.gradient_diagnostics).to_csv(out/'gradient_norms.csv', index=False)
    _, tasks, decisions, _ = evaluate_one(agent, 'context_masked', trial,
                                         args.eval_episodes, 200, pairs, out, args.deployment)
    result = summary(tasks, decisions, 'context_masked_v2_'+args.deployment)
    pd.DataFrame([result]).to_csv(out/'summary.csv', index=False)
    (out/'metadata.json').write_text(json.dumps({
        'checkpoint_environment': environment,
        'agent_name':agent.agent_name, 'options':options, 'seed_row':trial,
        'train_episodes': 0 if args.checkpoint_dir else args.train_episodes,
        'eval_episodes':args.eval_episodes, 'deployment':args.deployment,
        'checkpoint_source':str(args.checkpoint_dir) if args.checkpoint_dir else None,
        'data_sha256':environment['data_sha256'],
        'purpose':'functional smoke only; no effectiveness claim',
    }, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

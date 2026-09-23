"""Distribution, rollout, and legacy-isolation tests for masked Pair PPO."""
import copy
import unittest

import numpy as np
import simpy
import torch
from torch.distributions import Categorical

from agents.masked_pair_ppo_agent import (
    ReliabilityMaskedPairPPOAgent, effective_action_mask, masked_logits,
    production_reliability_vector,
)
from agents.ppo_agent import PPOAgent
from core.env_state import EnvironmentState
from core.server import Server
from core.task import Task


class TestMaskedPairPPO(unittest.TestCase):
    def setUp(self):
        self.env=simpy.Environment()
        self.state=EnvironmentState()
        rates=[.001,.002,.005,.008,.01,.02,.03,.04]
        for sid,rate in enumerate(rates,1):
            self.state.add_server_and_init_environment(Server(self.env,'Edge',sid,10.0,
                rate,-37.8,144.9))
        self.state.set_episode_effective_failure_rates(list(range(1,9)),rates)
        self.rho=np.linspace(.05,.35,28)
        self.kwargs=dict(num_states=35,num_actions=28,hidden_layers=[16],actor_mode='pair_scoring',
                         num_servers=8,pair_correlations=self.rho,min_rollout=1,batch_size=2,
                         k_epochs=2,minibatch_seed=19)

    def task(self,task_id,requirement=.9999):
        task=object.__new__(Task)
        task.id=task_id
        task.computation_demand=50.0
        task.reliability_requirement=requirement
        task.env_state=self.state
        return task

    def test_mask_probability_support_singleton_and_entropy(self):
        logits=torch.tensor([.2,.4,.6])
        distribution=Categorical(logits=masked_logits(logits,[True,False,True]))
        self.assertEqual(distribution.probs[1].item(),0)
        self.assertAlmostEqual(distribution.probs.sum().item(),1,places=6)
        single=Categorical(logits=masked_logits(logits,[False,True,False]))
        self.assertEqual(single.probs.tolist(),[0,1,0])
        self.assertAlmostEqual(single.entropy().item(),0,places=6)
        with self.assertRaises(ValueError):masked_logits(logits,[False]*3)

    def test_empty_safe_fallback_keeps_equal_best_actions(self):
        safe,effective,empty,best=effective_action_mask(np.array([.8,.9,.9,.7]),.99)
        self.assertTrue(empty)
        np.testing.assert_array_equal(safe,[False]*4)
        np.testing.assert_array_equal(effective,[False,True,True,False])
        self.assertEqual(best,.9)
        choices=set()
        for seed in range(20):
            torch.manual_seed(seed)
            dist=Categorical(logits=masked_logits(torch.tensor([1.,0.,0.,3.]),effective))
            choices.add(int(dist.sample()))
        self.assertEqual(choices,{1,2})

    def test_empty_fallback_record_has_deficit_and_actor_weighted_support(self):
        self.state.set_episode_effective_failure_rates(list(range(1,9)),[.04]*8)
        torch.manual_seed(8)
        agent=ReliabilityMaskedPairPPOAgent(**self.kwargs)
        task=self.task(1,.9999)
        state=np.zeros(35,dtype=np.float32)
        agent.prepare_action(task,self.state,1,state)
        action=agent.select_action(state)
        record=agent.selection_archive[0]
        self.assertTrue(record['safe_set_empty'])
        self.assertEqual(record['safe_set_size'],0)
        self.assertGreater(record['reliability_deficit'],0)
        self.assertEqual(sum(record['effective_mask']),28)
        self.assertIn(action,range(28))

    def test_production_feasibility_matches_selected_task(self):
        task=self.task(1)
        agent=ReliabilityMaskedPairPPOAgent(**self.kwargs)
        values=production_reliability_vector(task,self.state,agent.pairs)
        server_j,server_k=agent.pairs[0]
        task.initialize_reliability_evaluation(self.state.get_server_by_id(server_j),
                                                self.state.get_server_by_id(server_k))
        self.assertAlmostEqual(values[0],task.execution_reliability)
        safe,_,_,_=effective_action_mask(values,task.reliability_requirement)
        self.assertEqual(safe[0],task.reliability_satisfied)

    def test_old_new_logprob_ratio_and_training_update_share_mask(self):
        torch.manual_seed(27)
        agent=ReliabilityMaskedPairPPOAgent(**self.kwargs)
        task=self.task(1)
        state=np.zeros(35,dtype=np.float32)
        agent.prepare_action(task,self.state,1,state)
        torch.manual_seed(123)
        action=agent.select_action(state)
        agent.store_transition(state,action,2.0,state,delta_t=1.0,done=True,task_id=1)
        mask=np.asarray(agent.effective_masks[0],dtype=bool)
        with torch.no_grad():
            old=agent.policy_old(torch.as_tensor(state)).detach()
            new=agent.policy_net(torch.as_tensor(state)).detach()
            old_dist=Categorical(logits=masked_logits(old,mask))
            new_dist=Categorical(logits=masked_logits(new,mask))
            expected=float(old_dist.log_prob(torch.tensor(action)))
            ratio=float(torch.exp(new_dist.log_prob(torch.tensor(action))-
                                  old_dist.log_prob(torch.tensor(action))))
        self.assertAlmostEqual(agent.old_log_probs[0],expected,places=6)
        self.assertAlmostEqual(ratio,1.0,places=6)
        self.assertEqual(old_dist.probs[~torch.as_tensor(mask)].sum().item(),0.0)
        self.assertEqual(agent.selection_archive[0]['safe_set_size'],int(mask.sum()))
        agent.train_step()
        self.assertEqual(len(agent.effective_masks),0)
        self.assertLess(agent.update_diagnostics[0]['first_minibatch_ratio_max_abs_error'],1e-5)
        self.assertTrue(np.isfinite(agent.update_diagnostics[0]['mean_masked_entropy']))

    def test_different_batch_rows_keep_their_own_masks(self):
        logits=torch.zeros((2,3))
        masks=torch.tensor([[True,False,False],[False,True,True]])
        distribution=Categorical(logits=masked_logits(logits,masks))
        np.testing.assert_allclose(distribution.probs.numpy(),[[1,0,0],[0,.5,.5]])
        np.testing.assert_allclose(distribution.entropy().numpy(),[0,np.log(2)],atol=1e-6)

    def test_training_and_deployment_share_feasibility_rule(self):
        torch.manual_seed(21)
        training=ReliabilityMaskedPairPPOAgent(**self.kwargs)
        deployed=copy.deepcopy(training)
        deployed.frozen=True
        deployed.deployment_mode='stochastic'
        state=np.zeros(35,dtype=np.float32)
        for agent in (training,deployed):
            agent.prepare_action(self.task(1),self.state,1,state)
            agent.select_action(state)
        self.assertEqual(training.selection_archive[0]['safe_mask'],deployed.selection_archive[0]['safe_mask'])
        self.assertEqual(training.selection_archive[0]['effective_mask'],deployed.selection_archive[0]['effective_mask'])
        self.assertEqual(deployed.deployment_mode,'stochastic')

    def test_legacy_pair_agent_keeps_original_unmasked_path(self):
        torch.manual_seed(43)
        legacy=PPOAgent(**self.kwargs)
        torch.manual_seed(43)
        masked=ReliabilityMaskedPairPPOAgent(**self.kwargs)
        for key,tensor in legacy.policy_old.state_dict().items():
            torch.testing.assert_close(tensor,masked.policy_old.state_dict()[key],rtol=0,atol=0)
        self.assertFalse(hasattr(legacy,'prepare_action'))
        self.assertEqual(legacy.actor_mode,'pair_scoring')
        self.assertEqual(masked.agent_name,'reliability_masked_pair_ppo')

if __name__=='__main__':unittest.main()

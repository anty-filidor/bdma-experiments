from typing import Dict, List, Tuple

import networkx as nx
import network_diffusion as nd
import numpy as np

from network_diffusion.models.utils.compartmental import CompartmentalGraph


class SIR_UAModel(nd.models.BaseModel):

    def __init__(
            self,
            alpha: int,
            beta: int,
            alpha_prime: int,
            beta_prime: int,
            gamma: int,
            delta: int,
            epsilon: int,
            ill_seeds: int,
            aware_seeds: int,
    ) -> None:
        """
        A model SIR~UA.

        A spreading of two processes spreading in separate layers but dependent
        on each other is simulated with it:
            * "contagion" with states S (suspected), I (infecetd), R (removed)
            * "awareness" with states U (unaware), A (aware)
        Possible transitions can be desctibed on a graph:
            S·U---->I·U---->R·U
             |       |       |
             |       |       |
             v       v       v
            S·A---->I·A---->R·A
        All transitions except I->R are determined by interactions between 
        neighbouring nodes. Nodes can transit from I to R without any external 
        impulses. Parameters of the constructor are paobabilities of transitions
        and initial %s of infected and aware nodes. 

        :param alpha: weight probability of S->I for unaware nodes
        :param beta: probability of I->R for unaware nodes
        :param alpha_prime: probability of S->I for aware nodes
        :param beta_prime: probability of I->R for aware nodes
        :param gamma: probability of U->A for suspected nodes
        :param delta: probability of U->A for ill nodes
        :param epsilon: probability of U->A for removed nodes
        :param ill_seeds: % of initially I nodes
        :param aware_seeds: % of initially A nodes
        """
        compartments = self._create_compartments(
            alpha, beta, alpha_prime, beta_prime, gamma, delta, epsilon, ill_seeds, aware_seeds
        )
        super().__init__(compartments, nd.seeding.RandomSeedSelector())

    @staticmethod
    def _create_compartments(
        alpha: int,
        beta: int,
        alpha_prime: int,
        beta_prime: int,
        gamma: int,
        delta: int,
        epsilon: int,
        ill_seeds: int,
        aware_seeds: int,
    ) -> CompartmentalGraph:
        # define processes, allowed states and initial % of actors in that states
        phenomena = {
            "contagion": [["S", "I", "R"], [100 - ill_seeds, ill_seeds, 0]],
            "awareness": [["U", "A"], [100 - aware_seeds, aware_seeds]]
        }

        # wrap them into a compartments 
        cg = nd.models.CompartmentalGraph()
        for phenomenon, [states, budget] in phenomena.items():
            cg.add(process_name=phenomenon, states=states)  # name of process
            cg.seeding_budget.update({phenomenon: budget})  # initial %s
        cg.compile(background_weight=0)

        # set up weights of transitions for SIR and unaware
        cg.set_transition_fast("contagion.S", "contagion.I", ("awareness.U", ), alpha)
        cg.set_transition_fast("contagion.I", "contagion.R", ("awareness.U", ), beta)

        # set up weights of transitions for SIR and aware
        cg.set_transition_fast("contagion.S", "contagion.I", ("awareness.A", ), alpha_prime) 
        cg.set_transition_fast("contagion.I", "contagion.R", ("awareness.A", ), beta_prime)

        # set up weights of transitions for UA and suspected
        cg.set_transition_fast("awareness.U", "awareness.A", ("contagion.S", ), gamma)

        # set up weights of transitions for UA and infected
        cg.set_transition_fast("awareness.U", "awareness.A", ("contagion.I", ), delta)

        # set up weights of transitions for UA and removed
        cg.set_transition_fast("awareness.U", "awareness.A", ("contagion.R", ), epsilon)
        
        return cg

    def __str__(self) -> str:
        descr = f"{nd.utils.BOLD_UNDERLINE}\SIR-UA Model"
        descr += f"\n{nd.utils.THIN_UNDERLINE}\n"
        descr += self._compartmental_graph.describe()
        descr += str(self._seed_selector)
        descr += f"\n{nd.utils.BOLD_UNDERLINE}"
        return descr

    def determine_initial_states(self, net: nd.MultilayerNetwork) -> List[nd.models.NetworkUpdateBuffer]:
        if not net.is_multiplex():
            raise ValueError("This model works only with multiplex networks!")
        
        budget = self._compartmental_graph.get_seeding_budget_for_network(net)
        nodes_ranking = self._seed_selector.nodewise(net)
        initial_states = []

        # set initial states in contagion process/layer
        for node_position, node_id in enumerate(nodes_ranking["contagion"]):
            if node_position < budget["contagion"]["I"]:
                node_initial_state = "I"
            else:
                node_initial_state = "S"
            initial_states.append(
                nd.models.NetworkUpdateBuffer(node_id, "contagion", node_initial_state)
            )

        # set initial states in awareness process/layer
        for node_position, node_id in enumerate(nodes_ranking["awareness"]):
            if node_position < budget["awareness"]["A"]:
                node_initial_state = "A"
            else:
                node_initial_state = "U"
            initial_states.append(
                nd.models.NetworkUpdateBuffer(node_id, "awareness", node_initial_state)
            )
    
        return initial_states

    @staticmethod
    def flip_a_coin(prob_success: int) -> bool:
        result = np.random.choice([0, 1], p=[1 - prob_success, prob_success])
        if result == 1:
            return True
        else:
            return False

    def agent_evaluation_step(self, agent: int, layer_name: str, net: nd.MultilayerNetwork) -> str:
        layer_graph: nx.Graph = net[layer_name]

        # get possible transitions for state of the node
        current_state = layer_graph.nodes[agent]["status"]
        transitions = self._compartmental_graph.get_possible_transitions(
            net.get_actor(agent).states_as_compartmental_graph(), layer_name
        )

        # if there is no possible transition don't do anything
        if len(transitions) == 0:
            return current_state
        
        # if transition doesn't rely on interacitons with neighbours (i.e. I->R)
        if layer_name == "contagion" and current_state == "I":
            new_state = "R"
            if self.flip_a_coin(transitions[new_state]):
                return new_state

        # otherwise iterate through neighours
        else:
            for neighbour in nx.neighbors(layer_graph, agent):
                new_state = layer_graph.nodes[neighbour]["status"]
                if new_state in transitions and self.flip_a_coin(transitions[new_state]):
                        return new_state

        return current_state

    def network_evaluation_step(self, net: nd.MultilayerNetwork) -> List[nd.models.NetworkUpdateBuffer]:
        new_states = []
        for layer_name, layer_graph in net.layers.items():
            for node in layer_graph.nodes():
                new_state = self.agent_evaluation_step(node, layer_name, net)
                # print(layer_graph.nodes[node]["status"], "->", new_state)
                layer_graph.nodes[node]["status"] = new_state
        return new_states

    def get_allowed_states(self, net: nd.MultilayerNetwork) -> Dict[str, Tuple[str, ...]]:
        return self._compartmental_graph.get_compartments()

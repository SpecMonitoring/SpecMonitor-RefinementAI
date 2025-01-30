from tree_sitter import Language
import tree_sitter_java as tsjava
import tree_sitter_tlaplus as tstlaplus
from src.RefineAI import RefineAI
import torch
import pickle
import unittest
from sklearn.metrics import precision_score, recall_score, f1_score

class Evaluation:

    @staticmethod
    def compute_precision_and_recall_and_f1(y_true, y_pred):
        """Compute Precision, Recall, F1 Score."""
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')

        #print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
        return precision, recall, f1
    

class TestTLAPLUS(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            with open('tokenizer.pkl', 'rb') as f:
                cls.tokenizer = pickle.load(f)
                cls.tokenizer.parser.language = Language(tstlaplus.language())
        except FileNotFoundError:
            print("tokenizer state file has not been found.")
            return

        cls.model = RefineAI(cls.tokenizer, vocab_size=len(cls.tokenizer.vocab), num_heads=8,ff_dim = 4 * 256)
        MODEL_PATH = 'RefineAI.pth'
        state = torch.load(MODEL_PATH)  
        cls.model.load_state_dict(state['model_state_dict'], strict=False)
        cls.model.eval()
        cls.language = 'TLA'         
        cls.top_k = 5
        cls.task = 'mlm'
        cls.precisions = []
        cls.recalls = []
        cls.f1s = []
        mask_tokens = ['__MASK0__','__MASK1__','__MASK2__','__MASKVAR__']
        cls.mask_token_ids = [cls.tokenizer.vocab[token] for token in mask_tokens]
    
    @classmethod 
    def tearDownClass(cls):
        # Compute average metrics across all test cases
        avg_precision = sum(cls.precisions) / len(cls.precisions)
        avg_recall = sum(cls.recalls) / len(cls.recalls)
        avg_f1 = sum(cls.f1s) / len(cls.f1s)

        print(f"avg_precision: {avg_precision:.4f}")
        print(f"avg_recall: {avg_recall:.4f}")
        print(f"avg_f1: {avg_f1:.4f}")

    def __compute_metrics(self,logits, masked_indices, expected_predictions):
        predicted = []
        actual = []
        for masked_index in masked_indices:
            top_k_tokens_ids = torch.topk(logits[masked_index], k=self.top_k).indices.tolist()
            print(f" Masked token at index {masked_index} -  Expected: ", 
                    self.tokenizer.inverse_vocab[expected_predictions[masked_index][0]])
            print("----Prediction(s)----")
            for predicted_token_id in top_k_tokens_ids:
                print(self.tokenizer.inverse_vocab[predicted_token_id])
            # Check if any of the top-k predictions match the expected prediction
            if any(prediction in expected_predictions[masked_index] for prediction in top_k_tokens_ids):
                predicted.append(expected_predictions[masked_index][0])  # Consider correct
            else:
                predicted.append(top_k_tokens_ids[0])  #Top prediction is still considered for metrics

            actual.append(expected_predictions[masked_index][0])
        
        precision, recall, f1 = Evaluation.compute_precision_and_recall_and_f1(actual, predicted)
        self.precisions.append(precision)
        self.recalls.append(recall)
        self.f1s.append(f1)
    
    
    def test_unseen_large_spec1(self):
        spec ="""
----------------------- MODULE AbstractVeniceLeaderFollower -----------------
(***************************************************************************)
(* This module describes the behavior of the Venice Leader follower model  *)
(* in abstract terms for a single partition.  Given at least one client    *)
(* writer, a venice leader should be at some future state process all      *)
(* writes to a given partition and persist them to a queue to be consumed  *)
(* by a set of follower nodes in such a way that all replicas become       *)
(* consistent at some state. The only concrete detail we model here is     *)
(* that the transmission channel between a client and venice is that the   *)
(* channel is asynchronous and non destructive.                            *)
(***************************************************************************)

EXTENDS Integers, Sequences

VARIABLE realTimeTopic, versionTopic, nodes

vars == <<realTimeTopic, versionTopic, __MASKVAR__>>

RealTimeConsume(nodeId) ==
    /\ IF nodes[nodeId].rtOffset <= Len(realTimeTopic)
        THEN
            nodes' = [nodes EXCEPT
            ![nodeId].rtOffset = nodes[nodeId].rtOffset+1,
                ![nodeId].persistedRecords = SetValueOnReplica(nodeId,
                                            realTimeTopic[nodes[nodeId].rtOffset][1],
                                            realTimeTopic[nodes[nodeId].rtOffset][2])
                ]
            /\ versionTopic' = Append(versionTopic,
                <<realTimeTopic[nodes[nodeId].rtOffset][1],
                realTimeTopic[nodes[nodeId].rtOffset][2],
                nodes[nodeId].rtOffset>>)
        ELSE UNCHANGED vars
    /\ UNCHANGED <<realTimeTopic>>

FollowerConsume ==
    /\ \E followerNodeId \in {x \in DOMAIN nodes: nodes[x].state = FOLLOWER}:
        VersionTopicConsume(followerNodeId)

Next ==
    \/ ClientProducesToVenice
    \/ LeaderConsume
    \/ FollowerConsume
    \/ DemoteLeader
    \/ PromoteLeader

Init ==
  /\ realTimeTopic = <<>>
  /\ versionTopic = <<>>
  /\ __MASKVAR__ = [i \in nodeIds |->
    [ state |-> FOLLOWER,
    rtOffset |-> 1,
    vtOffset |-> 1,
    persistedRecords |-> {}]]

Spec == Init /\ [][Next]_vars /\ SF_vars(FollowerConsume) /\ WF_vars(LeaderConsume)

====
"""
        token_ids = self.tokenizer.encode(spec)
        masked_indices = torch.where(torch.isin(torch.tensor(token_ids, dtype=torch.long),torch.tensor(self.mask_token_ids)))[0].tolist()

        # Expected values
        expected_predictions = {
            masked_indices[0]: [self.tokenizer.vocab['nodes']],
            masked_indices[1]: [self.tokenizer.vocab['nodes']],
        }
        
        with torch.no_grad():
            logits, _ = self.model(token_ids,task= self.task, language= self.language)
            logits = logits.squeeze(0)

        print(f"Top-{self.top_k} predictions for {self._testMethodName}")
        self.__compute_metrics(logits,masked_indices,expected_predictions)
        print("============")
        print()
   
    def test_unseen_large_spec2(self):
        spec = """

------------------------------- MODULE ComplexSpec -------------------------------
EXTENDS Naturals

(* Constants *)
CONSTANT N
VARIABLE dst
vars == <<dst>>
(* Initial state *)
Init == __MASKVAR__ = 0
(* Next state *)
Next ==  dst' = dst + 2
(* Specification *)
Spec == Init /\ [][Next]_vars /\ Invariant

===============================================================================

"""

        token_ids = self.tokenizer.encode(spec)
        masked_indices = torch.where(torch.isin(torch.tensor(token_ids, dtype=torch.long),torch.tensor(self.mask_token_ids)))[0].tolist()


        # Expected values
        expected_predictions = {
            masked_indices[0]: [self.tokenizer.vocab['dst']]
        }
        
        with torch.no_grad():
            logits, _ = self.model(token_ids,task= self.task, language= self.language)
            logits = logits.squeeze(0)

        print(f"Top-{self.top_k} predictions for {self._testMethodName}")
        self.__compute_metrics(logits,masked_indices,expected_predictions)
        print("============")
        print()

    def test_operators_understanding(self):
        spec ="""--------------------------- MODULE test ---------------------------
EXTENDS WriteThroughCacheInstanced

vars __MASK0__ <<memInt, wmem, buf, ctl, cache, memQ>>

QCond == \E i \in 1 .. Len(memQ) : memQ[i][2].op = "Rd"
=============================================================================
"""
        token_ids = self.tokenizer.encode(spec)

        masked_indices = torch.where(torch.isin(torch.tensor(token_ids, dtype=torch.long),torch.tensor(self.mask_token_ids)))[0].tolist()
        
        # Expected values
        expected_predictions = {
            masked_indices[0]: [self.tokenizer.vocab['==']],
        }
        
        with torch.no_grad():
            logits, _ = self.model(token_ids,task= self.task, language= self.language)
            logits = logits.squeeze(0)

        print(f"Top-{self.top_k} predictions for {self._testMethodName}")
        self.__compute_metrics(logits,masked_indices,expected_predictions)
        print("============")
        print()    

    def test_operators_understanding2(self):
        spec ="""--------------------------- MODULE test ---------------------------
EXTENDS WriteThroughCacheInstanced

vars == <<memInt, wmem, buf, ctl, cache, memQ>>

QCond == __MASK0__ i __MASK0__ 1 .. Len(memQ) : memQ[i][2].op = "Rd"
=============================================================================
"""
        token_ids = self.tokenizer.encode(spec)
        masked_indices = torch.where(torch.isin(torch.tensor(token_ids, dtype=torch.long),torch.tensor(self.mask_token_ids)))[0].tolist()
        

        # Expected values
        expected_predictions = {
            masked_indices[0]: [self.tokenizer.vocab['\E']],
            masked_indices[1]: [self.tokenizer.vocab['\in']]
        }

        with torch.no_grad():
            logits, _ = self.model(token_ids,task= self.task, language= self.language)
            logits = logits.squeeze(0)

        print(f"Top-{self.top_k} predictions for {self._testMethodName}")
        self.__compute_metrics(logits,masked_indices,expected_predictions)
        print("============")
        print()
    
    def test_identifier_name_generalization(self):
        spec ="""--------------------------- MODULE test ---------------------------
VARIABLES b, a
vars == <<__MASKVAR__, b>>
Init == /\ b = <<>>
        /\ a = {}
=============================================================================
"""
        token_ids = self.tokenizer.encode(spec)
        masked_indices = torch.where(torch.isin(torch.tensor(token_ids, dtype=torch.long),torch.tensor(self.mask_token_ids)))[0].tolist()
        

        # Expected values
        expected_predictions = {
            masked_indices[0]: [self.tokenizer.vocab['a']],
        }
        with torch.no_grad():
            logits, _ = self.model(token_ids,task= self.task, language= self.language)
            logits = logits.squeeze(0)

        print(f"Top-{self.top_k} predictions for {self._testMethodName}")
        self.__compute_metrics(logits,masked_indices,expected_predictions)
        print("============")
        print()
    
    def test_identifier_name_generalization2(self):
        spec ="""--------------------------- MODULE test ---------------------------
VARIABLES x, buffer
vars == <<x, __MASKVAR__>>
Init == /\ x = <<>>
        /\ buffer = {}
=============================================================================
"""
        token_ids = self.tokenizer.encode(spec)
        masked_indices = torch.where(torch.isin(torch.tensor(token_ids, dtype=torch.long),torch.tensor(self.mask_token_ids)))[0].tolist()
        
        # Expected values
        expected_predictions = {
            masked_indices[0]: [self.tokenizer.vocab['buffer']],
        }

        with torch.no_grad():
            logits, _ = self.model(token_ids,task= self.task, language= self.language)
            logits = logits.squeeze(0)

        print(f"Top-{self.top_k} predictions for {self._testMethodName}")
        self.__compute_metrics(logits,masked_indices,expected_predictions)
        print("============")
        print()
    

    def test_identifier_name_generalization3(self):
        spec ="""--------------------------- MODULE test ---------------------------
CONSTANT K
VARIABLES x, b
vars == <<x, b>>
Init == x = 1 /\ b = 2
Next == __MASKVAR__' = b + 1 /\ b' = x * 3
Spec == Init /\ [][Next]_vars
=============================================================================
"""
        token_ids = self.tokenizer.encode(spec)
        masked_indices = torch.where(torch.isin(torch.tensor(token_ids, dtype=torch.long),torch.tensor(self.mask_token_ids)))[0].tolist()
        
        # Expected values
        expected_predictions = {
            masked_indices[0]: [self.tokenizer.vocab['x']],
        }

        with torch.no_grad():
            logits, _ = self.model(token_ids,task= self.task, language= self.language)
            logits = logits.squeeze(0)

        print(f"Top-{self.top_k} predictions for {self._testMethodName}")
        self.__compute_metrics(logits,masked_indices,expected_predictions)
        print("============")
        print()

    def test_keywords(self):
        spec ="""--------------------------- __MASK0__ test ---------------------------
VARIABLES buffer, waitSet
__MASK0__ == <<buffer, waitSet>>
Init == /\ buffer = <<>>
        /\ waitSet = {}
=============================================================================
"""
        token_ids = self.tokenizer.encode(spec)
        masked_indices = torch.where(torch.isin(torch.tensor(token_ids, dtype=torch.long),torch.tensor(self.mask_token_ids)))[0].tolist()
        
        # Expected values
        expected_predictions = {
            masked_indices[0]: [self.tokenizer.vocab['MODULE']],
            masked_indices[1]: [self.tokenizer.vocab['vars']],
        }

        with torch.no_grad():
            logits, _ = self.model(token_ids,task= self.task, language= self.language)
            logits = logits.squeeze(0)

        print(f"Top-{self.top_k} predictions for {self._testMethodName}")
        self.__compute_metrics(logits,masked_indices,expected_predictions)
        print("============")
        print()
    
    def test_variable_names_understanding(self):
        spec ="""--------------------------- MODULE test ---------------------------
VARIABLES buffer, waitSet
vars == <<__MASKVAR__, waitSet>>
Init == /\ buffer = <<>>
        /\ waitSet = {}
=============================================================================
"""
        token_ids = self.tokenizer.encode(spec)
        masked_indices = torch.where(torch.isin(torch.tensor(token_ids, dtype=torch.long),torch.tensor(self.mask_token_ids)))[0].tolist()
        

        # Expected values
        expected_predictions = {
            masked_indices[0]: [self.tokenizer.vocab['buffer']],
        }

        with torch.no_grad():
            logits, _ = self.model(token_ids,task= self.task, language= self.language)
            logits = logits.squeeze(0)

        print(f"Top-{self.top_k} predictions for {self._testMethodName}")
        self.__compute_metrics(logits,masked_indices,expected_predictions)
        print("============")
        print()
    
    def test_variable_names_understanding2(self):
        spec ="""--------------------------- MODULE test ---------------------------
VARIABLES buffer, waitSet
vars == <<__MASKVAR__, __MASKVAR__>>
Init == /\ buffer = <<>>
        /\ waitSet = {}
=============================================================================
"""
        token_ids = self.tokenizer.encode(spec)
        masked_indices = torch.where(torch.isin(torch.tensor(token_ids, dtype=torch.long),torch.tensor(self.mask_token_ids)))[0].tolist()
        

        # Expected values
        expected_predictions = {
            masked_indices[0]: [self.tokenizer.vocab['buffer']],
            masked_indices[1]: [self.tokenizer.vocab['waitSet']],
        }

        with torch.no_grad():
            logits, _ = self.model(token_ids,task= self.task, language= self.language)
            logits = logits.squeeze(0)

        print(f"Top-{self.top_k} predictions for {self._testMethodName}")
        self.__compute_metrics(logits,masked_indices,expected_predictions)
        print("============")
        print()
    
    def test_variable_names_understanding3(self):
        spec ="""--------------------------- MODULE test ---------------------------
EXTENDS WriteThroughCacheInstanced

vars == <<memInt, wmem, buf, ctl, cache, memQ>>

QCond == \E i \in 1 .. Len(__MASKVAR__) : memQ[i][2].op = "Rd"
=============================================================================
"""
        token_ids = self.tokenizer.encode(spec)
        masked_indices = torch.where(torch.isin(torch.tensor(token_ids, dtype=torch.long),torch.tensor(self.mask_token_ids)))[0].tolist()
        

        # Expected values
        expected_predictions = {
            masked_indices[0]: [self.tokenizer.vocab['memQ']],
        }

        with torch.no_grad():
            logits, _ = self.model(token_ids,task= self.task, language= self.language)
            logits = logits.squeeze(0)

        print(f"Top-{self.top_k} predictions for {self._testMethodName}")
        self.__compute_metrics(logits,masked_indices,expected_predictions)
        print("============")
        print()
    
    def test_initial_state_understanding(self):
        spec ="""--------------------------- MODULE test ---------------------------
VARIABLES buffer, waitSet
vars == << buffer, waitSet,BufCapacity >>
Init == /\ buffer = __MASK0__ __MASK0__
        /\ waitSet = {}
Put(t, d) ==
/\ t \notin Range(waitSeqP)
/\ \/ /\ Len(buffer) < BufCapacity
      /\ buffer' = Append(buffer, d)
      /\ NotifyOther(waitSeqC)
      /\ UNCHANGED waitSeqP
   \/ /\ Len(buffer) = BufCapacity
      /\ Wait(waitSeqP, t)
      /\ UNCHANGED waitSeqC

Get(t) ==
/\ t \notin waitSet
/\ \/ /\ buffer # <<>>
      /\ buffer' = Tail(buffer)
=============================================================================
"""
        token_ids = self.tokenizer.encode(spec)
        masked_indices = torch.where(torch.isin(torch.tensor(token_ids, dtype=torch.long),torch.tensor(self.mask_token_ids)))[0].tolist()
        
        # Expected values
        expected_predictions = {
            masked_indices[0]: [self.tokenizer.vocab['<<']],
            masked_indices[1]: [self.tokenizer.vocab['>>']]
        }

        with torch.no_grad():
            logits, _ = self.model(token_ids,task= self.task, language= self.language)
            logits = logits.squeeze(0)

        print(f"Top-{self.top_k} predictions for {self._testMethodName}")
        self.__compute_metrics(logits,masked_indices,expected_predictions)
        print("============")
        print()
            

    def test_initial_state_understanding2(self):
        spec ="""--------------------------- MODULE test ---------------------------
VARIABLES buffer, waitSet
Init == /\ buffer = <<>>
        /\ waitSet = __MASK0__ __MASK0__
RunningThreads == (Producers \cup Consumers) \ waitSet

NotifyOther(Others) == 
    IF waitSet \cap Others # {}
    THEN \E t \in waitSet \cap Others : waitSet' = waitSet \ {t}
    ELSE UNCHANGED waitSet

(* @see java.lang.Object#wait *)
Wait(t) == /\ waitSet' = waitSet \cup {t}
           /\ UNCHANGED <<buffer>>
=============================================================================
"""
        token_ids = self.tokenizer.encode(spec)
        masked_indices = torch.where(torch.isin(torch.tensor(token_ids, dtype=torch.long),torch.tensor(self.mask_token_ids)))[0].tolist()
        

        # Expected values
        expected_predictions = {
            masked_indices[0]: [self.tokenizer.vocab['{']],
            masked_indices[1]: [self.tokenizer.vocab['}']]
        }

        with torch.no_grad():
            logits, _ = self.model(token_ids,task= self.task, language= self.language)
            logits = logits.squeeze(0)

        print(f"Top-{self.top_k} predictions for {self._testMethodName}")
        self.__compute_metrics(logits,masked_indices,expected_predictions)
        print("============")
        print()

    def test_initial_state_understanding3(self):
        spec ="""--------------------------- MODULE test ---------------------------
VARIABLES buffer, waitSet
Init == /\ buffer = <<>>
        /\ waitSet = {}
RunningThreads == (Producers \cup Consumers) \ waitSet

NotifyOther(Others) == 
    IF waitSet \cap Others # {}
    THEN \E t \in waitSet \cap Others : waitSet' = waitSet \ {t}
    ELSE UNCHANGED waitSet

Put(t, d) == t \\notin waitSet /\\ Len(buffer) < BufCapacity /\\ __MASKVAR__' = Append(buffer, d)
=============================================================================
"""

        token_ids = self.tokenizer.encode(spec)
        masked_indices = torch.where(torch.isin(torch.tensor(token_ids, dtype=torch.long),torch.tensor(self.mask_token_ids)))[0].tolist()
        

        # Expected values
        expected_predictions = {
            masked_indices[0]: [self.tokenizer.vocab['buffer']]
        }

        with torch.no_grad():
            logits, _ = self.model(token_ids,task= self.task, language= self.language)
            logits = logits.squeeze(0)

        print(f"Top-{self.top_k} predictions for {self._testMethodName}")
        self.__compute_metrics(logits,masked_indices,expected_predictions)
        print("============")
        print()

    def test_condition_understanding(self):
        spec ="""--------------------------- MODULE test ---------------------------
VARIABLES buffer, waitSet
Init == /\ buffer = <<>>
        /\ waitSet = {}
RunningThreads == (Producers \cup Consumers) \ waitSet

NotifyOther(Others) == 
    IF __MASKVAR__ \cap Others # {}
    THEN \E t \in waitSet \cap Others : waitSet' = waitSet \ {t}
    ELSE UNCHANGED waitSet

(* @see java.lang.Object#wait *)
Wait(t) == /\ waitSet' = waitSet \cup {t}
           /\ UNCHANGED <<buffer>>
=============================================================================
"""
        token_ids = self.tokenizer.encode(spec)
        masked_indices = torch.where(torch.isin(torch.tensor(token_ids, dtype=torch.long),torch.tensor(self.mask_token_ids)))[0].tolist()
        

        # Expected values
        expected_predictions = {
            masked_indices[0]: [self.tokenizer.vocab['waitSet']]
        }

        with torch.no_grad():
            logits, _ = self.model(token_ids,task= self.task, language= self.language)
            logits = logits.squeeze(0)

        print(f"Top-{self.top_k} predictions for {self._testMethodName}")
        self.__compute_metrics(logits,masked_indices,expected_predictions)
        print("============")
        print()


    def test_next_state_understanding(self):
        spec ="""--------------------------- MODULE test ---------------------------
VARIABLES buffer, waitSet
Init == /\ buffer = <<>>
        /\ waitSet = {}
RunningThreads == (Producers \cup Consumers) \ waitSet

NotifyOther(Others) == 
    IF waitSet \cap Others # {}
    THEN \E t \in waitSet \cap Others : waitSet' = __MASKVAR__ \ {t}
    ELSE UNCHANGED waitSet

(* @see java.lang.Object#wait *)
Wait(t) == /\ waitSet' = waitSet \cup {t}
           /\ UNCHANGED <<buffer>>
=============================================================================
"""
        token_ids = self.tokenizer.encode(spec)
        masked_indices = torch.where(torch.isin(torch.tensor(token_ids, dtype=torch.long),torch.tensor(self.mask_token_ids)))[0].tolist()
        

        # Expected values
        expected_predictions = {
            masked_indices[0]: [self.tokenizer.vocab['waitSet']]
        }

        with torch.no_grad():
            logits, _ = self.model(token_ids,task= self.task, language= self.language)
            logits = logits.squeeze(0)

        print(f"Top-{self.top_k} predictions for {self._testMethodName}")
        self.__compute_metrics(logits,masked_indices,expected_predictions)
        print("============")
        print()


class TestSemanticJAVA(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            with open('tokenizer.pkl', 'rb') as f:
                cls.tokenizer = pickle.load(f)
                cls.tokenizer.parser.language = Language(tsjava.language())
        except FileNotFoundError:
            print("tokenizer state file has not been found.")
            return

        cls.model = RefineAI(cls.tokenizer, vocab_size=len(cls.tokenizer.vocab), num_heads=8,ff_dim = 4 * 256)
        MODEL_PATH = 'RefineAI.pth'
        state = torch.load(MODEL_PATH)  
        cls.model.load_state_dict(state['model_state_dict'])
        cls.model.eval()
        cls.language = 'Java'         
        cls.top_k = 5
        cls.task = 'mlm'
        cls.precisions = []
        cls.recalls = []
        cls.f1s = []
        mask_tokens = ['__MASK0__','__MASK1__','__MASK2__','__MASKVAR__']
        cls.mask_token_ids = [cls.tokenizer.vocab[token] for token in mask_tokens]
    
    @classmethod 
    def tearDownClass(cls):
        # Compute average metrics across all test cases
        avg_precision = sum(cls.precisions) / len(cls.precisions)
        avg_recall = sum(cls.recalls) / len(cls.recalls)
        avg_f1 = sum(cls.f1s) / len(cls.f1s)

        print(f"avg_precision: {avg_precision:.4f}")
        print(f"avg_recall: {avg_recall:.4f}")
        print(f"avg_f1: {avg_f1:.4f}")
    
    
    
    def __compute_metrics(self,logits, masked_indices, expected_predictions):
        predicted = []
        actual = []
        for masked_index in masked_indices:
            top_k_tokens_ids = torch.topk(logits[masked_index], k=self.top_k).indices.tolist()
            print(f" Masked token at index {masked_index} -  Expected: ", 
                    self.tokenizer.inverse_vocab[expected_predictions[masked_index][0]])
            print("----Prediction(s)----")
            for predicted_token_id in top_k_tokens_ids:
                print(self.tokenizer.inverse_vocab[predicted_token_id])
            # Check if any of the top-k predictions match the expected prediction
            if any(prediction in expected_predictions[masked_index] for prediction in top_k_tokens_ids):
                predicted.append(expected_predictions[masked_index][0])  # Consider correct
            else:
                predicted.append(top_k_tokens_ids[0])  #Top prediction is still considered for metrics

            actual.append(expected_predictions[masked_index][0])
        
        precision, recall, f1 = Evaluation.compute_precision_and_recall_and_f1(actual, predicted)
        self.precisions.append(precision)
        self.recalls.append(recall)
        self.f1s.append(f1)

    def test_java_keyword(self):
        spec ="""public class App2TLA {
	public __MASK0__ void main(__MASK0__ String[] args) throws IOException {
		final List<RecordedEvent> recordedEvents = RecordingFile
				.readAllEvents(Paths.get(args.length > 0 ? args[0] : "app.jfr"));
	}
}"""
        token_ids = self.tokenizer.encode(spec)
        masked_indices = torch.where(torch.isin(torch.tensor(token_ids, dtype=torch.long),torch.tensor(self.mask_token_ids)))[0].tolist()
        
        # Expected values
        expected_predictions = {
            masked_indices[0]: [self.tokenizer.vocab['Ġstatic']],
            masked_indices[1]: [self.tokenizer.vocab['Ġfinal']]
        }
        
        with torch.no_grad():
            logits, _ = self.model(token_ids,task= self.task, language= self.language)
            logits = logits.squeeze(0)

        print(f"Top-{self.top_k} predictions for {self._testMethodName}")
        self.__compute_metrics(logits,masked_indices,expected_predictions)
        print("============")
        print()
    
    def test_java_keyword2(self):
        spec ="""public __MASK0__ App2TLA {
	public static void main(final String[] args) throws IOException {
		final List<RecordedEvent> recordedEvents = RecordingFile
				.readAllEvents(Paths.get(args.length > 0 ? args[0] : "app.jfr"));
	}
}"""
        token_ids = self.tokenizer.encode(spec)
        masked_indices = torch.where(torch.isin(torch.tensor(token_ids, dtype=torch.long),torch.tensor(self.mask_token_ids)))[0].tolist()
        # Expected values
        expected_predictions = {
            masked_indices[0]: [self.tokenizer.vocab['class']],
        }

        with torch.no_grad():
            logits, _ = self.model(token_ids,task= self.task, language= self.language)
            logits = logits.squeeze(0)

        print(f"Top-{self.top_k} predictions for {self._testMethodName}")
        self.__compute_metrics(logits,masked_indices,expected_predictions)
        print("============")
        print()

    def test_java_variable(self):
        spec ="""public class APP {
    public static void main(String[] args) throws IOException {
        int __MASKVAR__ = 0;
        count = count + 1;
    }
}"""
        token_ids = self.tokenizer.encode(spec)
        masked_indices = torch.where(torch.isin(torch.tensor(token_ids, dtype=torch.long),torch.tensor(self.mask_token_ids)))[0].tolist()
        # Expected values
        expected_predictions = {
            masked_indices[0]: [self.tokenizer.vocab['count']],
        }

        with torch.no_grad():
            logits, _ = self.model(token_ids,task= self.task, language= self.language)
            logits = logits.squeeze(0)

        print(f"Top-{self.top_k} predictions for {self._testMethodName}")
        self.__compute_metrics(logits,masked_indices,expected_predictions)
        print("============")
        print()

    def test_java_variable2(self):
        spec ="""
public class Buffer {
    private ArrayList<__MASK0__> __MASKVAR__;

    public Buffer() {
        buffer = new ArrayList<>();
    }

    public void append(__MASK0__ e) {
        buffer.add(e);
    }

    public boolean invariant() {
        return buffer != null; // Check TLA+ invariant
    }
}
"""
        token_ids = self.tokenizer.encode(spec)
        masked_indices = torch.where(torch.isin(torch.tensor(token_ids, dtype=torch.long),torch.tensor(self.mask_token_ids)))[0].tolist()
        # Expected values
        expected_predictions = {
            masked_indices[0]: [self.tokenizer.vocab['Integer']],
            masked_indices[1]: [self.tokenizer.vocab['buffer']],
            masked_indices[2]: [self.tokenizer.vocab['int']]
        }

        with torch.no_grad():
            logits, _ = self.model(token_ids,task= self.task, language= self.language)
            logits = logits.squeeze(0)

        print(f"Top-{self.top_k} predictions for {self._testMethodName}")
        self.__compute_metrics(logits,masked_indices,expected_predictions)
        print("============")
        print()    
    
if __name__ == '__main__':

    unittest.main()

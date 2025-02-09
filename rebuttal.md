<!-- Thanks for your constructive feedback. We seek to address the most significatn points.
# Reviewer A
## A1. Unclear threat model
***Verifier Capability.*** We assume a verifier, who can either be a clinet or a third party, has full access of the training graph as a client is also the data owner. In contrast, they only have a black-box access to the server. In particular, a verifier does not know about the architecture of the GNNs or the unlearning method employed by the server.

## A2. Limited scope and applicability
This paper cannot be extended to other graph tasks, such as graph classification, as the definition of a token node is specifically tied to node classification. However, the concept of PANDA could be adapted to other unlearning targets, such as node unlearning. We will include a discussion on the range of tasks and scenarios in the revision.

## A3. Confusing design
* We use the surrogate graph is identical to the original graph in most experiments. However, in Section 6.4 (Table 6(b)), we also consider a scenario where the surrogate graph is a subgraph of the original graph. PANDA shows an effective transferrability as it achieves relatively high $P_{TP}$ and $P_{TN}$ under this setting. 
* The values of $p$ and $q$ are estimated prior to verification. Once these values are obtained, Algorithm 2 can be initiated for verification. Specifically, we use a random number of challenge edges along with the algorithm proposed in Section 5.1 to identify multiple token nodes. The estimation of $p$ and $q$ is performed by analyzing the proportion of token nodes whose predictions change when all or subsets of the challenge edges are removed.

## A4. Insufficient evaluation
First, we would like to clarify that GraphEraser and GNNDelete are considered mainstream approximate unlearning methods. We will also consider adding more experiments focused on approximate unlearning in the revision.

## A5. Limited transferability
waiting for the experimental results

## A6. Computational requirements
(The paper would benefit from a more detailed analysis of the time and memory requirements of the algorithm. Given that the experiments were conducted using 8 A100 GPUs, which are costly resources, please provide a thorough analysis of computational demands to better assess PANDA's practicality and reproducibility.)
* ***Time Complexity.*** 

## A7. Experimental settings and results
* For most experiments, we use $\alpha, \beta = 0.9$, which provides a reasonable guarantee. In Table 4, however, we use 0.96 ($m=3$) to explore performance when the number of token nodes is below the theoretical value. For the incompleteness ratio, $s=20%$ indicates that the server cheats on only one edge when the number of challenge edges is 5. This represents the most stringent scenario for verification.
* We will consider using absolute differences in the revision.

## Questions for authors' response
### Q1. Could you please provide a more detailed description of the verifier's capabilities and knowledge in your threat model?
Please refer to A1.
### Q2. Is it possible to extend your verification method to handle multiple unlearning requests? 
Not done yet
### Q3. Could you clarify the process of training the surrogate model in Section 5.1?
### Q4. In Algorithm 2, how are the challenge edges used in step 1 generated, and are they different from those output by the algorithm?
Please refer to A3 for both Q3 and Q4.


# Reviewer B

## B1. Unclear or confusing scenario
* ***Intialization.*** We assume that there is a target model is trained on the original graph by the server in the beginning. 
* ***Training surrogate mdoel.*** The verifier trains a surrogate model on the original graph where is identical to the original graph as verifier has full access to the training data (or in Section 6.4, we consider a smaller surrogate trained on a surrogate graph where is subsampled from the original graph.).
* ***Finding token nodes and challenge edges.*** The verifier then identifies the token nodes and inserts the cooresponding challenge edges into the training graph.
* ***Continues learning.*** Under MLaaS setting, the server may continuesly update the model if the training dataset is changed, and preduce the poisoned model. 
* ***Verfication.*** The verfier sends the unlearning requests to the server to forget the challenge edges, and verifies the unlearning by checking the prediction of token nodes.

## B2. The setting suggests that training the full model is much more involved than verification so the surrogate model is expected to be much smaller or simpler. However, the prep times in Table 2, while smaller than what I'm guessing is training time, are not that much smaller and in some cases comparable (in CS). I do not get a sense that these are truly on different levels of effort asymptotically.

waiting for the experimental results

## B3. Lack of consideration of statistical validity. 
We will provide the standard deviation in the revisoin. And here are some preliminary results on CiteSeer dataset.

## B4 (Other comments 1). The descriptions of the models used in the experiment makes them sound tiny. 
Here are the number of parameters for each models.

## B5 (Other comments 2)
Please refer to B1.

## B6 (Other comments 3)
We will revise this statement in the revision.

## B7 (Small things)
We will address these comments in the revision.


## Questions for authors' response
### Q1.
We use retraining as an unlearning method for most of experiments.

## Q2. 
No. they are different as existing adversrial attacks do not have the step that idenfies token nodes.

## Q3. What is "complete edge unlearning" as in the "Verification efficiency" paragraph? I would guess it is retraining the full model without the unlearned edges but then the follow up phrase mentions that complete unlearning is done with "white-box access to the target model" which sounds like something other than retraining is done. Please explain.

Yes, the complete edge unlearning involves retraining the entire model without the unlearned edges. However, to verify unlearning locally, the verifier requires "white-box access" to the target model. This is why we emphasize the need for this assumption in the context of local verification.

## Q4. Does using the surrogate to find more than 1 token node and challenge sets require the surrogate to be more accurate/bigger/faithful to the target model?
No. 1PF token nodes are not strongly related to the performance of surrogate. It is an inherent property of a node rely on the graph and the target model.

## Q5. Section 7.1 states "This assumption holds when the real edges are at least L hops away from the token nodes, where L is the number of layers in the target model." This seems to be making some assumptions about the architecture of the model which I don't think is true in general. Please elaborate. Does the statement only apply to GCN with RelU activation as defined in the background? Unrelated, "GNN" is used in the appendix where "GCN" was intended.

In order to derive the theorectical guarantees of unlearning both real and challenge edges, we assume the real edges are at least L hops away. However, in practice we do not have this assumption as shown in Table 8.

## Q6. Can the methodology be adapted to tasks other than classification? Feature extraction might be the basis of many other downstream tasks so it might be good to consider that as a focus of verifying unlearning.
No. The methodology relies on the definition of node tokens, which is specifically designed for node classification tasks.

## Q7. What is the significance of the verifying being a different party from the client? What is gained in the formalism with this distinction?
A different party as the verifier have more resources (e.g. computational power) compared to a normal client, enabling it to train the surrogate model and prepare for verification.

## Q8. What effect could additional challenge edges have on utility of the model before they are unlearned?
Additional challenge edges may slightly reduce the model's utility. However, as shown in Table 1 ("AccLoss"), this effect is negligible.

## Q9. The detection of challenge edges stipulates that the service could detect each challenge edge but does not seem to consider detection of an entire set of edges. What if they keep track of challenge edges and realize that edges that affect the same token node are being asked to be unlearned in sequence? They don't even need to do anything special here other than keep track of incident nodes of the unlearning edges.
It is an interesting observation. However, we assume that the server does not have the knowledge of verification mechanism.

# Reviewer C

## C1. Assumes edge behavior and model consistency that may not hold universally
We assume that the verifier has the access to the training graph as they are the client who might be the data owner. For the model consistency, we empiricall demonsrate that PANDA have good transferrability as shown in Table 6(a).

## C2. Experiments lack large-scale, real-world datasets, limiting generalizability.
We will include a large-scale dataset in the revision.

## C3. Possible detection and removal
In section 6.5, we have demonstrated that $P_{TP}$ remains high after we apply the detection methods. In the revision, we will try more to check if PANDA could withstand adaptive attacks.

# Reviewer D
## D1. Upon reading the introduction, I found myself wondering about the criteria for selecting challenge edges and token nodes, which isn't addressed until Section 5. It would improve readability to either briefly introduce this in the introduction or indicate that it will be explained later.
We will improve the wording in the revision as suggested.

## D2. There are a few edge cases that would benefit from additional discussion. For example, what if the algorithm fails to identify 1PF nodes? How would using k-PF nodes impact the verification process and reliability? While a theoretical analysis can be left for future work, any insights here would be valuable.
Using k-PF token nodes will only impact finding token nodes (Section 5.1) as the bounds (Equations 10 and 11) would be slightly different resulting in the necessity probability (q) would be getting lower. In addition, the reliabbility will remain the same since the deviation of guarantees (Equations 5, 6, and 7) won't change.

## D3. Beyond accuracy loss, is there a way to measure the computational or time overhead introduced by adding verification?
Here are some preliminary results that we compare the running w/o challenge edges. we will provide the full results in the revision.

## D4. How does the architecture affect PANDA’s performance? The current evaluation does not provide insights into why PANDA’s effectiveness varies across different models.
According to Table 6(a), different architerctures of GNN models do not affect the performance of PANDA that much. The underlying reason is that different architectures perform similarly in terms of model utility and loss as shown in the table below. As a result, they have similar boundary. Therefore, 1PF token nodes identied on GCN are highly likely 1PF token nodes to GAT, GraphSAGE, and GIN.

## D5. How do variations in challenge edges and token nodes impact verification authenticity? For instance, what happens if a subset of token nodes is used, and how does the size of this subset affect verification? Among all token nodes, are some more critical to the process than others?
If only a subset of token nodes are using, it will not satisfy the verification guarantees as suggested in our theoretical analysis (Equations 5, 6, and 7). We can obserse this in Table 4 as the $P_{TP}$ is lower than 96% when $m=2$. $P_{T}$ is getting lower alone with the size of subset decreases. Since all token nodes are 1PF, there is no more critical nodes.

##Questions for authors' response
### Q1: refer to D5.
### Q2: refer to D3.
### Q3: refer to D4. -->



Thanks for your constructive feedback. We prioritize questions in the "Questions for authors' response".

# Reviewer-A

## Q1. Verifier's capabilities
The verifier has no knowledge about the architecture of the target model and unlearning methods. (S)he has the full access to the training graph and can modify the training graph, e.g., by inserting "poisoning edges" for verification.   

## Q2. Multiple unlearning requests
<!-- To handle multiple unlearning requests, PANDA has to generate a new set of challenge edges for each unlearning request to deliver the required $(\alpha, \beta)$-unlearning guarantee. We admit that this is a weakness of PANDA, as it may not be able to provide efficient long-term verification. But our experiments have demonstrated that PANDA is effective for one-time verification. -->

To handle multiple unlearning requests, PANDA must generate new challenge edges for each request to ensure the $(\alpha, \beta)$-unlearning guarantee. While this limits its efficiency for long-term verification, experiments show PANDA is effective for one-time verification.

## Q3. Training surrogate model
<!-- The surrogate graph has the same topology structure as the original graph. But its nodes are labeled with the ones predicted by the target model.  Specifically, the verifier labels the nodes in the surrogate graph with the labels obtained by querying the target model. Next, it trains the surrogate model on the labeled surrogate graph, so that the model mimics the behaviors of the target model. -->

The surrogate graph shares the original graph's topology but uses labels predicted by the target model. The verifier queries the target model for labels, trains a surrogate model on the labeled surrogate graph, and mimics the target model's behavior.

## Q4. More details of Step 1 of Algorithm 2
<!-- We randomly select a set of token nodes $V$ first. The number of nodes in $V$ is randomly chosen. Next, we execute the FindAdvE() function of Algorithm 1 on the chosen token nodes to generate a set of challenge edges $E$.  Then we estimate $p$ and $q$ based on $V$ and $E$ by repeating 5 trials, with 50 token nodes per trial. Both $p$ and $q$ are computed as the average of 5 trials. Once both values are obtained, we compute the number of token nodes to satisfy $(\alpha, \beta)$-guarantee (Step 1 of Algorithm 2), and re-generate token nodes and challenge edges.  -->

We apply FindNode() to select a set of token nodes $V$ first. The number of nodes in $V$ is randomly chosen. Next, we execute the FindAdvE() function of Algorithm 1 on the chosen token nodes to generate a set of challenge edges $E$.  Then we estimate $p$ and $q$ based on $V$ and $E$ by repeating 5 trials, with 50 token nodes per trial. Both $p$ and $q$ are computed as the average of 5 trials. Once both values are obtained, we compute the number of token nodes to satisfy $(\alpha, \beta)$-guarantee (Step 1 of Algorithm 2), and re-generate token nodes and challenge edges. 

# Reviewer-B

## Q1
We use retraining as it is the only exact unlearning method existing in the literature. All three unlearning methods used in the experiments are approximate ones. 

## Q2
The existing adversarial attacks pick their own token nodes, which are different from ours. 

## Q3
"Complete edge unlearning" means retraining the model on the graph. The white-box access is meant for retraining. 

## Q4
Finding more token nodes does not require the surrogate model to be more accurate/bigger. Instead, it requires the surrogate model to be less "confident", as more nodes will locate close to its classification boundary.  

## Q5
The assumption of L-hop-away is only to simplify the derivation of the verification probability. Our derivation does not rely on a specific GNN architecture. 

## Q6
PANDA can be extended to other types of learning tasks and unlearning scenarios by changing the design of challenge for verification. E.g., consider the link prediction task and feature unlearning. The verification can be conducted by "poisoning" the token nodes with fake features whose insertion can change the link prediction output of those nodes. We agree that feature extraction can be an alternative verification solution. 

## Q7
Using a different party allows a party which may have more computational resources than the client to perform more efficient and effective verification. 

## Q8
Our results ("AccLoss" column in Table 1) have demonstrated that PANDA introduces negligible accuracy loss to the model before unlearning.

## Q9
We agree that a knowledgeable attacker who knows the details of the verification mechanism can conduct some adaptive attacks against PANDA to escape from verification. We will discuss these attacks, including the suggested one, in the revision. 

# Reviewer-C

## C1. Genaralizability
- **Data transferrability** We clarify that PANDA does not transfer across datasets, as the verifier has the full access to the original graph. 
- **Model transferrability** PANDA's model transferrability relies on whether those models have similar classification boundaries, as they enable the challenge edges constructed by one model to change the prediction by another model. 

## C2. Large datasets.
We will include the experiments on a large graph dataset (e.g., MAG240M) in the revision.

## C3. Possible detection
See response to Reviewer-B.Q9. 

# Reviewer D

## Q1 
-  If only a subset of token nodes is used, it may not satisfy the $(\alpha, \beta)$-completeness, as the number of token nodes is computed from $\alpha$ and $\beta$ (Eqn. 7). 

- All token nodes have the same level of criticality, as all of them are 1PF nodes. 

## Q2
A possibility is to evaluate the model training time before and after adding the challenge edges. Some preliminary results are shown below. We will provide full results in the revision.
| Setting | Training Time (s) |
|------------|------------|
| Before | 1.60 |
| After | 1.62 |

## Q3
Refer to response to Reviewer-C.C1.
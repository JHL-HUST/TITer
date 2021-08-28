import torch
import tqdm
import numpy as np

class Tester(object):
    def __init__(self, model, args, train_entities, RelEntCooccurrence):
        self.model = model
        self.args = args
        self.train_entities = train_entities
        self.RelEntCooccurrence = RelEntCooccurrence

    def get_rank(self, score, answer, entities_space, num_ent):
        """Get the location of the answer, if the answer is not in the array,
        the ranking will be the total number of entities.
        Args:
            score: list, entity score
            answer: int, the ground truth entity
            entities_space: corresponding entity with the score
            num_ent: the total number of entities
        Return: the rank of the ground truth.
        """
        if answer not in entities_space:
            rank = num_ent
        else:
            answer_prob = score[entities_space.index(answer)]
            score.sort(reverse=True)
            rank = score.index(answer_prob) + 1
        return rank

    def test(self, dataloader, ntriple, skip_dict, num_ent):
        """Get time-aware filtered metrics(MRR, Hits@1/3/10).
        Args:
            ntriple: number of the test examples.
            skip_dict: time-aware filter. Get from baseDataset
            num_ent: number of the entities.
        Return: a dict (key -> MRR/HITS@1/HITS@3/HITS@10, values -> float)
        """
        self.model.eval()
        logs = []
        with torch.no_grad():
            with tqdm.tqdm(total=ntriple, unit='ex') as bar:
                current_time = 0
                cache_IM = {}  # key -> entity, values: list, IM representations of the co-o relations.
                for src_batch, rel_batch, dst_batch, time_batch in dataloader:
                    batch_size = dst_batch.size(0)

                    if self.args.IM:
                        src = src_batch[0].item()
                        rel = rel_batch[0].item()
                        dst = dst_batch[0].item()
                        time = time_batch[0].item()

                        # representation update
                        if current_time != time:
                            current_time = time
                            for k, v in cache_IM.items():
                                ims = torch.stack(v, dim=0)
                                self.model.agent.update_entity_embedding(k, ims, self.args.mu)
                            cache_IM = {}

                        if src not in self.train_entities and rel in self.RelEntCooccurrence['subject'].keys():
                            im = self.model.agent.get_im_embedding(list(self.RelEntCooccurrence['subject'][rel]))
                            if src in cache_IM.keys():
                                cache_IM[src].append(im)
                            else:
                                cache_IM[src] = [im]

                            # prediction shift
                            self.model.agent.entities_embedding_shift(src, im, self.args.mu)

                    if self.args.cuda:
                        src_batch = src_batch.cuda()
                        rel_batch = rel_batch.cuda()
                        dst_batch = dst_batch.cuda()
                        time_batch = time_batch.cuda()

                    current_entities, beam_prob = \
                        self.model.beam_search(src_batch, time_batch, rel_batch)

                    if self.args.IM and src not in self.train_entities:
                        # We do this
                        # because events that happen at the same time in the future cannot see each other.
                        self.model.agent.back_entities_embedding(src)

                    if self.args.cuda:
                        current_entities = current_entities.cpu()
                        beam_prob = beam_prob.cpu()

                    current_entities = current_entities.numpy()
                    beam_prob = beam_prob.numpy()

                    MRR = 0
                    for i in range(batch_size):
                        candidate_answers = current_entities[i]
                        candidate_score = beam_prob[i]

                        # sort by score from largest to smallest
                        idx = np.argsort(-candidate_score)
                        candidate_answers = candidate_answers[idx]
                        candidate_score = candidate_score[idx]

                        # remove duplicate entities
                        candidate_answers, idx = np.unique(candidate_answers, return_index=True)
                        candidate_answers = list(candidate_answers)
                        candidate_score = list(candidate_score[idx])

                        src = src_batch[i].item()
                        rel = rel_batch[i].item()
                        dst = dst_batch[i].item()
                        time = time_batch[i].item()

                        # get inductive inference performance.
                        # Only count the results of the example containing new entities.
                        if self.args.test_inductive and src in self.train_entities and dst in self.train_entities:
                            continue

                        filter = skip_dict[(src, rel, time)]  # a set of ground truth entities
                        tmp_entities = candidate_answers.copy()
                        tmp_prob = candidate_score.copy()
                        # time-aware filter
                        for j in range(len(tmp_entities)):
                            if tmp_entities[j] in filter and tmp_entities[j] != dst:
                                candidate_answers.remove(tmp_entities[j])
                                candidate_score.remove(tmp_prob[j])

                        ranking_raw = self.get_rank(candidate_score, dst, candidate_answers, num_ent)

                        logs.append({
                            'MRR': 1.0 / ranking_raw,
                            'HITS@1': 1.0 if ranking_raw <= 1 else 0.0,
                            'HITS@3': 1.0 if ranking_raw <= 3 else 0.0,
                            'HITS@10': 1.0 if ranking_raw <= 10 else 0.0,
                        })
                        MRR = MRR + 1.0 / ranking_raw

                    bar.update(batch_size)
                    bar.set_postfix(MRR='{}'.format(MRR / batch_size))
        metrics = {}
        for metric in logs[0].keys():
            metrics[metric] = sum([log[metric] for log in logs]) / len(logs)
        return metrics

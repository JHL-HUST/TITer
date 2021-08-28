from collections import defaultdict
from torch.utils.data import Dataset

class baseDataset(object):
    def __init__(self, trainpath, testpath, statpath, validpath):
        """base Dataset. Read data files and preprocess.
        Args:
            trainpath: File path of train Data;
            testpath: File path of test data;
            statpath: File path of entities num and relatioins num;
            validpath: File path of valid data
        """
        self.trainQuadruples = self.load_quadruples(trainpath)
        self.testQuadruples = self.load_quadruples(testpath)
        self.validQuadruples = self.load_quadruples(validpath)
        self.allQuadruples = self.trainQuadruples + self.validQuadruples + self.testQuadruples
        self.num_e, self.num_r = self.get_total_number(statpath)  # number of entities, number of relations
        self.skip_dict = self.get_skipdict(self.allQuadruples)

        self.train_entities = set()  # Entities that have appeared in the training set
        for query in self.trainQuadruples:
            self.train_entities.add(query[0])
            self.train_entities.add(query[2])

        self.RelEntCooccurrence = self.getRelEntCooccurrence(self.trainQuadruples)  # -> dict

    def getRelEntCooccurrence(self, quadruples):
        """Used for Inductive-Mean. Get co-occurrence in the training set.
        return:
            {'subject': a dict[key -> relation, values -> a set of co-occurrence subject entities],
             'object': a dict[key -> relation, values -> a set of co-occurrence object entities],}
        """
        relation_entities_s = {}
        relation_entities_o = {}
        for ex in quadruples:
            s, r, o = ex[0], ex[1], ex[2]
            reversed_r = r + self.num_r + 1
            if r not in relation_entities_s.keys():
                relation_entities_s[r] = set()
            relation_entities_s[r].add(s)
            if r not in relation_entities_o.keys():
                relation_entities_o[r] = set()
            relation_entities_o[r].add(o)

            if reversed_r not in relation_entities_s.keys():
                relation_entities_s[reversed_r] = set()
            relation_entities_s[reversed_r].add(o)
            if reversed_r not in relation_entities_o.keys():
                relation_entities_o[reversed_r] = set()
            relation_entities_o[reversed_r].add(s)
        return {'subject': relation_entities_s, 'object': relation_entities_o}

    def get_all_timestamps(self):
        """Get all the timestamps in the dataset.
        return:
            timestamps: a set of timestamps.
        """
        timestamps = set()
        for ex in self.allQuadruples:
            timestamps.add(ex[3])
        return timestamps

    def get_skipdict(self, quadruples):
        """Used for time-dependent filtered metrics.
        return: a dict [key -> (entity, relation, timestamp),  value -> a set of ground truth entities]
        """
        filters = defaultdict(set)
        for src, rel, dst, time in quadruples:
            filters[(src, rel, time)].add(dst)
            filters[(dst, rel+self.num_r+1, time)].add(src)
        return filters

    @staticmethod
    def load_quadruples(inpath):
        """train.txt/valid.txt/test.txt reader
        inpath: File path. train.txt, valid.txt or test.txt of a dataset;
        return:
            quadrupleList: A list
            containing all quadruples([subject/headEntity, relation, object/tailEntity, timestamp]) in the file.
        """
        with open(inpath, 'r') as f:
            quadrupleList = []
            for line in f:
                try:
                    line_split = line.split()
                    head = int(line_split[0])
                    rel = int(line_split[1])
                    tail = int(line_split[2])
                    time = int(line_split[3])
                    quadrupleList.append([head, rel, tail, time])
                except:
                    print(line)
        return quadrupleList

    @staticmethod
    def get_total_number(statpath):
        """stat.txt reader
        return:
            (number of entities -> int, number of relations -> int)
        """
        with open(statpath, 'r') as fr:
            for line in fr:
                line_split = line.split()
                return int(line_split[0]), int(line_split[1])


class QuadruplesDataset(Dataset):
    def __init__(self, examples, num_r):
        """
        examples: a list of quadruples.
        num_r: number of relations
        """
        self.quadruples = examples.copy()
        for ex in examples:
            self.quadruples.append([ex[2], ex[1]+num_r+1, ex[0], ex[3]])

    def __len__(self):
        return len(self.quadruples)

    def __getitem__(self, item):
        return self.quadruples[item][0], \
               self.quadruples[item][1], \
               self.quadruples[item][2], \
               self.quadruples[item][3]
